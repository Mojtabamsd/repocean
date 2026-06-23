"""
Minimal EcoTaxa API client using only `requests`.

Logs in, resolves a sample's textual name (orig_id) to its internal numeric
id, pulls all object metadata (incl. taxonomic classification) for that
sample, and saves it to a CSV file.

EcoTaxa API base: https://ecotaxa.obs-vlfr.fr/api
Live Swagger docs (good for double-checking field names / payloads):
https://ecotaxa.obs-vlfr.fr/api/docs
"""

import csv
from typing import Optional

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

    `project_filters` is sent as the JSON request body, e.g.
        {"samples": "12345"}   # comma-separated numeric sample ids
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
    """Page through get_object_set until every matching object is fetched."""
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
    USERNAME = "masoudi.m1991@gmail.com"
    PASSWORD = "1234561qaz"
    PROJECT_ID = 19171
    OUT_PATH = r'C:\alr4\ai_predict\ai_predict_all'

    # Choose how to select samples:
    #   "single" -> one exact sample, matched by full orig_id
    #   "prefix" -> all samples whose orig_id's last "_segment" starts with
    #               SAMPLE_LAST_TOKEN_PREFIX (e.g. "d" matches "..._d0002")
    MODE = "prefix"
    # MODE = "single"
    SAMPLE_ORIG_ID = "ALR004_20240609_0008_0002_d0002"   # used when MODE == "single"
    SAMPLE_LAST_TOKEN_PREFIX = "d"                        # used when MODE == "prefix"
    OUTPUT_CSV = OUT_PATH + r"\ecotaxa_sample_d_export_api.csv"

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
    samples_filter = ",".join(str(sid) for sid in sample_ids)
    object_ids, details = fetch_all_objects(
        token, PROJECT_ID, fields, project_filters={"samples": samples_filter}
    )
    print(f"\nFetched {len(object_ids)} objects for {len(sample_ids)} sample(s)")

    header = ["objid"] + field_list
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for obj_id, row in zip(object_ids, details):
            writer.writerow([obj_id] + row)

    print(f"Saved to {OUTPUT_CSV}")