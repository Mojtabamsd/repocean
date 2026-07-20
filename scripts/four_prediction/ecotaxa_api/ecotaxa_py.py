"""
Minimal EcoTaxa API client using only `requests`.

Logs in, resolves a sample's textual name (orig_id) to its internal numeric
id, pulls object metadata (incl. taxonomic classification) for that sample
-- optionally restricted server-side to a chosen list of taxon labels -- and
saves it to a CSV file.

EcoTaxa API base: https://ecotaxa.obs-vlfr.fr/api
Live Swagger docs (good for double-checking field names / payloads):
https://ecotaxa.obs-vlfr.fr/api/docs
"""

import csv
from typing import Optional
import os

import requests

BASE_URL = "https://ecotaxa.obs-vlfr.fr/api"


def login(username: str, password: str) -> str:
    """POST /login -> returns a JWT access token (plain string)."""
    resp = requests.post(
        f"{BASE_URL}/login",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()


def get_project(token: str, project_id: int, for_managing: bool = False):
    """GET /projects/{project_id} -> project metadata."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{BASE_URL}/projects/{project_id}",
        headers=headers,
        params={"for_managing": for_managing},
    )
    resp.raise_for_status()
    return resp.json()


def list_samples(token: str, project_id: int):
    """GET /samples/search with id_pattern='*' -> all samples in the project."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{BASE_URL}/samples/search",
        headers=headers,
        params={"project_ids": str(project_id), "id_pattern": "*"},
    )
    resp.raise_for_status()
    return resp.json()


def select_samples_by_last_token_prefix(samples: list, prefix: str) -> list:
    """
    Given a list of sample dicts (from list_samples), return only those
    whose orig_id's last underscore-separated segment starts with `prefix`.

    e.g. orig_id "ALR004_20240609_0008_0002_d0002" has last segment
    "d0002", which starts with "d" -> matches prefix="d".
    """
    matches = []
    for s in samples:
        orig_id = s.get("orig_id", "")
        last_token = orig_id.rsplit("_", 1)[-1]
        if last_token.startswith(prefix):
            matches.append(s)
    return matches


def find_sample_id(token: str, project_id: int, orig_id: str) -> int:
    """
    GET /samples/search -> find a sample's internal numeric id from its
    textual name (orig_id), e.g. "ALR004_20240609_0008_0002_d0002".

    Returns the numeric sample id of the first exact match. Raises if
    nothing or more than one exact match is found.
    """
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{BASE_URL}/samples/search",
        headers=headers,
        params={"project_ids": str(project_id), "id_pattern": orig_id},
    )
    resp.raise_for_status()
    samples = resp.json()

    if not samples:
        raise ValueError(f"No sample found matching {orig_id!r} in project {project_id}")

    # Prefer an exact match on orig_id; fall back to the first hit otherwise.
    exact = [s for s in samples if s.get("orig_id") == orig_id]
    candidates = exact or samples

    if len(candidates) > 1:
        print(f"Warning: {len(candidates)} samples matched {orig_id!r}, using the first one.")
        for s in candidates:
            print("   ", s.get("sampleid"), s.get("orig_id"))

    sample = candidates[0]
    # The numeric id key is normally 'sampleid'; print the raw dict once so
    # you can confirm/adjust if your EcoTaxa version names it differently.
    sample_id = sample.get("sampleid")
    if sample_id is None:
        print("Could not find 'sampleid' key, raw sample record was:", sample)
        raise KeyError("sampleid")
    return sample_id


def search_taxa_by_name(token: str, name: str, project_id=None):
    """
    GET /taxon_set/search -> search the (global) taxonomy tree by name.

    This is a small, lightweight call -- it searches the taxonomy
    dictionary itself, NOT project objects -- so it never downloads any
    object/image data. Passing project_id makes taxa already used in that
    project come first in the results (and flagged via 'pr'), but it does
    NOT restrict the search to that project's taxa only.

    Returns a list of dicts shaped like TaxaSearchRsp:
        {"id": 12345, "text": "larvae<Ceriantharia", "pr": 1, "renm_id": None}
    """
    headers = {"Authorization": f"Bearer {token}"}
    params = {"query": name}
    if project_id is not None:
        params["project_id"] = project_id
    resp = requests.get(
        f"{BASE_URL}/taxon_set/search",
        headers=headers,
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def resolve_taxon_ids(token: str, project_id: int, labels_to_extract: list) -> list:
    """
    Resolve a list of EXACT taxon display names (e.g. 'larvae<Ceriantharia')
    to their numeric classif_id, using the taxonomy name search endpoint.
    One lightweight request per label -- still nothing object/image related
    is ever downloaded.
    """
    resolved, missing, ambiguous = [], [], []

    for label in labels_to_extract:
        matches = search_taxa_by_name(token, label, project_id=project_id)
        exact = [m for m in matches if m.get("text") == label]

        if not exact:
            missing.append(label)
            continue
        if len(exact) > 1:
            preferred = [m for m in exact if m.get("pr") == 1]
            chosen = (preferred or exact)[0]
            ambiguous.append((label, [m["id"] for m in exact]))
        else:
            chosen = exact[0]

        resolved.append(chosen["id"])

    if ambiguous:
        for label, ids in ambiguous:
            print(f"Warning: label {label!r} matched multiple taxon ids {ids}, used one of "
                  f"them (preferring the one already used in project {project_id} if any).")

    if missing:
        raise ValueError(
            f"Could not find an exact taxon name match for: {missing}. "
            f"Check spelling/case -- names must match the EcoTaxa display "
            f"name exactly (e.g. 'larvae<Ceriantharia')."
        )
    return resolved


def get_object_set(
    token: str,
    project_id: int,
    fields: str,
    project_filters: Optional[dict] = None,
    order_field: Optional[str] = None,
    window_start: Optional[int] = None,
    window_size: Optional[int] = None,
):
    """
    POST /object_set/{project_id}/query

    `fields` is a comma-separated list of columns, e.g.:
        "obj.objid,obj.orig_id,obj.classif_qual,txo.display_name,sam.orig_id"

    `project_filters` is sent as the JSON request body. This is where ALL
    filtering happens server-side, e.g.:
        {"samples": "12345,12346"}                      # restrict to samples
        {"taxo": "5821,6003"}                            # restrict to taxa (classif_id)
        {"samples": "12345", "taxo": "5821,6003"}        # both combined (AND)
    Pass {} for "no filter" = all objects in the project.

    Response (ObjectSetQueryRsp) looks like:
        {
          "object_ids": [...],
          "details": [[<value per field>, ...], ...],
          "total_ids": N,
          ...
        }
    """
    headers = {"Authorization": f"Bearer {token}"}
    query_params = {"fields": fields}
    if order_field is not None:
        query_params["order_field"] = order_field
    if window_start is not None:
        query_params["window_start"] = window_start
    if window_size is not None:
        query_params["window_size"] = window_size

    resp = requests.post(
        f"{BASE_URL}/object_set/{project_id}/query",
        headers=headers,
        params=query_params,
        json=project_filters or {},
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_objects(
    token: str,
    project_id: int,
    fields: str,
    project_filters: dict,
    page_size: int = 2000,
):
    """Page through get_object_set until every matching object is fetched.

    Because filtering (samples, taxa, or both) is applied in
    `project_filters` and sent to the server on every page request, only
    the objects that actually match come back -- nothing unwanted is ever
    downloaded or held in memory.
    """
    all_ids, all_details = [], []
    start = 0
    while True:
        data = get_object_set(
            token,
            project_id,
            fields,
            project_filters=project_filters,
            window_start=start,
            window_size=page_size,
        )
        ids = data.get("object_ids", [])
        details = data.get("details", [])
        all_ids.extend(ids)
        all_details.extend(details)

        total = data.get("total_ids", len(all_ids))
        start += len(ids)
        if len(ids) == 0 or start >= total:
            break

    return all_ids, all_details

if __name__ == "__main__":
    # USERNAME = os.environ["ECOTAXA_USERNAME"]
    # PASSWORD = os.environ["ECOTAXA_PASSWORD"]
    PROJECT_ID = 20066   # uvp6_REF learning set
    # PROJECT_ID = 19171 # uvp6_ctd learning set
    OUT_PATH = r'C:\alr4\ai_predict\ai_predict_d'

    # Choose how to select samples:
    #   "single" -> one exact sample, matched by full orig_id
    #   "prefix" -> all samples whose orig_id's last "_segment" starts with
    #               SAMPLE_LAST_TOKEN_PREFIX (e.g. "d" matches "..._d0002")
    MODE = "prefix"
    # MODE = "single"
    SAMPLE_ORIG_ID = "ALR004_20240609_0008_0002_d0002"   # used when MODE == "single"
    SAMPLE_LAST_TOKEN_PREFIX = "d"                        # used when MODE == "prefix"

    # --- NEW: optional taxon-label filter -----------------------------
    # Leave as an empty list to keep the old behaviour (all taxa).
    # If non-empty, only objects classified into one of these EXACT
    # display names are fetched. This is resolved to classif_id and sent
    # to the server alongside the sample filter, so the server -- not
    # your machine -- does the filtering; nothing else is downloaded.
    LABELS_TO_EXTRACT = ['larvae<Ceriantharia', 'feeding', 'tentacle<larvae']
    # LABELS_TO_EXTRACT = []  # <- use this to disable taxon filtering
    # --------------------------------------------------------------------

    OUTPUT_CSV = OUT_PATH + r"\ecotaxa_sample_" \
                 + str(SAMPLE_LAST_TOKEN_PREFIX) \
                 +"_export_api_larve.csv"

    token = login(USERNAME, PASSWORD)
    print("Logged in, got token.")

    project = get_project(token, PROJECT_ID)
    print("Project title:", project.get("title"))

    samples = list_samples(token, PROJECT_ID)
    print(f"\nFound {len(samples)} samples in project {PROJECT_ID}:")
    for s in samples:
        print("   sampleid =", s.get("sampleid"), " | orig_id =", s.get("orig_id"))

    if MODE == "single":
        sample_id = find_sample_id(token, PROJECT_ID, SAMPLE_ORIG_ID)
        print(f"\nResolved sample {SAMPLE_ORIG_ID!r} -> numeric sample id {sample_id}")
        sample_ids = [sample_id]
        label = SAMPLE_ORIG_ID
    else:
        selected = select_samples_by_last_token_prefix(samples, SAMPLE_LAST_TOKEN_PREFIX)
        if not selected:
            raise ValueError(f"No samples found with last segment starting with {SAMPLE_LAST_TOKEN_PREFIX!r}")
        sample_ids = [s["sampleid"] for s in selected]
        print(f"\nSelected {len(sample_ids)} samples with last segment starting with {SAMPLE_LAST_TOKEN_PREFIX!r}:")
        for s in selected:
            print("   sampleid =", s.get("sampleid"), " | orig_id =", s.get("orig_id"))
        label = f"prefix_{SAMPLE_LAST_TOKEN_PREFIX}"

    # Full metadata set: object info, classification, taxon name, sample +
    # acquisition original ids. Add/remove obj./txo./sam./acq. fields as needed.
    fields = (
        "obj.objid,obj.orig_id,obj.classif_id,obj.classif_qual,"
        "obj.classif_auto_score,obj.classif_who,obj.classif_when,"
        "obj.latitude,obj.longitude,obj.depth_min,obj.depth_max,"
        "obj.objdate,obj.objtime,"
        "txo.display_name,sam.orig_id,acq.orig_id"
    )
    field_list = fields.split(",")

    # "samples" filter takes a comma-separated string of numeric sample ids
    project_filters = {"samples": ",".join(str(sid) for sid in sample_ids)}

    # --- NEW: fold the taxon filter into the SAME server-side request ---
    if LABELS_TO_EXTRACT:
        taxon_ids = resolve_taxon_ids(token, PROJECT_ID, LABELS_TO_EXTRACT)
        print(f"\nResolved labels {LABELS_TO_EXTRACT} -> classif_ids {taxon_ids}")
        project_filters["taxo"] = ",".join(str(t) for t in taxon_ids)
    # ----------------------------------------------------------------------

    object_ids, details = fetch_all_objects(
        token, PROJECT_ID, fields, project_filters=project_filters
    )
    print(f"\nFetched {len(object_ids)} objects for {len(sample_ids)} sample(s)"
          + (f" restricted to taxa {LABELS_TO_EXTRACT}" if LABELS_TO_EXTRACT else ""))

    # --- NEW: optional simple post-filter, keep only a given classif_qual --
    # e.g. only_validated = 'V' keeps only validated objects ('V').
    # Other EcoTaxa qualifications: 'P' predicted, 'D' dubious.
    # Leave as None/'' to keep everything (old behaviour).
    only_validated = 'V'
    # only_validated = None  # <- use this to disable the filter
    # ------------------------------------------------------------------------

    if only_validated:
        qual_idx = field_list.index("obj.classif_qual")
        filtered = [
            (obj_id, row) for obj_id, row in zip(object_ids, details)
            if row[qual_idx] == only_validated
        ]
        print(f"Keeping only obj.classif_qual == {only_validated!r}: "
              f"{len(filtered)} of {len(object_ids)} objects.")
        object_ids = [x[0] for x in filtered]
        details = [x[1] for x in filtered]

    header = ["objid"] + field_list
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for obj_id, row in zip(object_ids, details):
            writer.writerow([obj_id] + row)

    print(f"Saved {len(object_ids)} objects to {OUTPUT_CSV}")