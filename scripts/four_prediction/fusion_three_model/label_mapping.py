"""
label_mapping.py
================
Builds the unified canonical label namespace from M1, M2, and M3 CSVs,
and exposes helpers for:
  - normalising any raw label to its canonical M1 form
  - computing lineage distance between two labels
  - finding the Lowest Common Ancestor (LCA) of two labels
"""

import pandas as pd
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Load source data
# ---------------------------------------------------------------------------

def load_label_tables(
    m1_path: str,
    m2_path: str,
    m3_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(m1_path)   # cols: label, ecotaxa_label, count, lineage
    df2 = pd.read_csv(m2_path)   # cols: label, count, lineage
    df3 = pd.read_csv(m3_path)   # col:  0  (label names only, lowercase)
    df3 = df3.rename(columns={"0": "label"})
    df3["label"] = df3["label"].str.strip()
    return df1, df2, df3


# ---------------------------------------------------------------------------
# 2. Canonical lineage store
#    Key:   canonical label (M1 casing, lowercase for non-living/artefact ones)
#    Value: list of nodes from root to leaf, e.g. ["living","Eukaryota",...]
# ---------------------------------------------------------------------------

# Hand-coded lineages for the 6 M3-unique labels (derived from analysis)
_M3_UNIQUE_LINEAGES: dict[str, str] = {
    "colonial<collodaria":  "living>Eukaryota>Harosa>Rhizaria>Retaria>Polycystinea>Collodaria>colonial",
    "dark<sphere":          "living>other>othertocheck>sphere>dark",
    "juvenile<lobata":      "living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Ctenophora>Cyclocoela>Lobata>juvenile",
    "t008":                 "temporary>t008",
    "themisto":             "living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Malacostraca>Eumalacostraca>Amphipoda>Hyperiidea>Hyperiidae>Themisto",
    "zoom-in<gelatinous":   "living>other>gelatinous>zoom-in",
}

# Synonym map: any of these raw labels (lowercased) → canonical M1 label
# Sources:
#   - M2 uses "child<Parent" style where M1 uses just "child"
#   - M3 is all-lowercase
#   - a few genuine taxonomic synonyms (Phyllodocida / Phyllodocidae)
_SYNONYM_MAP_RAW: list[tuple[str, str]] = [
    # (raw_lower,                  canonical_M1)
    ("copepoda<maxillopoda",       "Copepoda"),
    ("ctenophora<metazoa",         "Ctenophora"),
    ("phyllodocidae",              "Phyllodocida"),   # M2/M3 name → M1 canonical
    ("fiber<detritus",             "fiber"),
    ("filament<detritus",          "filament"),
    # M3-unique synonym
    ("dark<sphere",                "darksphere"),
    # M2 labels that are children of M1 broader labels but for fusion
    # we keep them distinct — they appear as-is in the canonical table
]

# ---------------------------------------------------------------------------
# 3. Build the unified canonical table
# ---------------------------------------------------------------------------

def build_canonical_table(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
) -> dict[str, str]:
    """
    Returns canonical_lineage: dict[canonical_label -> lineage_string]
    Canonical label = M1 label where it exists, otherwise M2 label
    (title-cased), otherwise M3 label (title-cased).
    All lookups go through normalise_label() first.
    """
    table: dict[str, str] = {}

    # M1 is ground truth — load first
    for _, row in df1.iterrows():
        label = row["label"]
        lineage = row["lineage"].split(",")[0].strip()  # take first if multi
        table[label] = lineage

    # M2 — add labels not already covered by M1
    m1_lower = {k.lower(): k for k in table}
    for _, row in df2.iterrows():
        label = row["label"]
        lineage = row["lineage"].split(",")[0].strip()
        key_lower = label.lower()
        if key_lower not in m1_lower:
            table[label] = lineage
            m1_lower[key_lower] = label

    # M3 — add labels not already covered by M1 or M2
    for _, row in df3.iterrows():
        label = row["label"]
        key_lower = label.lower()
        if key_lower not in m1_lower:
            # Use hand-coded lineage if available
            lineage = _M3_UNIQUE_LINEAGES.get(key_lower, "")
            if lineage:
                table[label] = lineage
                m1_lower[key_lower] = label

    return table


# ---------------------------------------------------------------------------
# 4. Label normalisation
# ---------------------------------------------------------------------------

def build_normaliser(
    canonical_table: dict[str, str]
) -> dict[str, str]:
    """
    Returns norm_map: raw_label_lower -> canonical_label

    Covers:
      - exact case-insensitive match
      - synonym map (Copepoda<Maxillopoda -> Copepoda, etc.)
    """
    norm: dict[str, str] = {}

    # Step 1: direct case-insensitive matches
    for canon in canonical_table:
        norm[canon.lower()] = canon

    # Step 2: apply synonym overrides
    for raw_lower, canon in _SYNONYM_MAP_RAW:
        if canon in canonical_table:          # only map if target exists
            norm[raw_lower] = canon

    return norm


def normalise_label(raw, norm_map: dict) -> Optional[str]:
    """
    Return the canonical label for any raw label string.
    Returns None if the label is completely unknown, None, NA, or non-string.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    raw = raw.strip()
    if raw.lower() in ("nan", "none", "na", "n/a", ""):
        return None
    return norm_map.get(raw.lower())


# ---------------------------------------------------------------------------
# 5. Lineage distance & LCA
# ---------------------------------------------------------------------------

def _parse_lineage(lineage_str: str) -> list[str]:
    """Split a lineage string into a list of nodes root→leaf."""
    return lineage_str.split(",")[0].strip().split(">")


def lineage_distance(
    label_a: str,
    label_b: str,
    canonical_table: dict[str, str],
    norm_map: dict[str, str],
) -> tuple[int, None]:
    """
    Compute the total lineage distance between two labels and return
    (distance, lca_node).

    Distance = (steps from A to LCA) + (steps from B to LCA).
    Distance 0  → same node (identical for fusion purposes).
    Distance 1  → one is the direct parent/child of the other.
    Distance -1 → one or both labels are unknown (can't compute).
    """
    canon_a = normalise_label(label_a, norm_map)
    canon_b = normalise_label(label_b, norm_map)

    if canon_a is None or canon_b is None:
        return -1, None

    if canon_a == canon_b:
        return 0, canon_a

    lin_a_str = canonical_table.get(canon_a)
    lin_b_str = canonical_table.get(canon_b)

    if not lin_a_str or not lin_b_str:
        return -1, None

    nodes_a = _parse_lineage(lin_a_str)
    nodes_b = _parse_lineage(lin_b_str)

    # Longest common prefix
    lca_len = 0
    for a, b in zip(nodes_a, nodes_b):
        if a == b:
            lca_len += 1
        else:
            break

    if lca_len == 0:
        return -1, None   # completely disjoint (shouldn't happen in practice)

    lca_node = nodes_a[lca_len - 1]
    dist = (len(nodes_a) - lca_len) + (len(nodes_b) - lca_len)
    return dist, lca_node


def deeper_label(
    label_a: str,
    label_b: str,
    canonical_table: dict[str, str],
    norm_map: dict[str, str],
) -> str:
    """
    Given two labels on the same lineage path (distance <= 1),
    return the more specific (deeper / longer lineage) one.
    Falls back to label_a on a tie.
    """
    canon_a = normalise_label(label_a, norm_map) or label_a
    canon_b = normalise_label(label_b, norm_map) or label_b
    lin_a = canonical_table.get(canon_a, "")
    lin_b = canonical_table.get(canon_b, "")
    return canon_a if len(lin_a) >= len(lin_b) else canon_b


# ---------------------------------------------------------------------------
# 6. Convenience: print the full canonical table
# ---------------------------------------------------------------------------

def print_canonical_table(canonical_table: dict[str, str]) -> None:
    print(f"{'Canonical label':<40} {'Lineage'}")
    print("-" * 120)
    for label, lineage in sorted(canonical_table.items()):
        print(f"{label:<40} {lineage}")