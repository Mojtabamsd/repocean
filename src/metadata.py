from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

CANDIDATE_NAME_COLS = ["img_file_name", "object_id", "object_lat", "object_lon", "object_date", "object_time",
                       "object_depth_min", "object_depth_max", "sample_id"]


def _find_name_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for k in CANDIDATE_NAME_COLS:
        if k in lower:
            return lower[k]
    for c in cols:
        cl = c.lower()
        if "name" in cl or "file" in cl:
            return c
    return None


def _normalize_image_name(x: str) -> str:
    # keep as provided; if TSV stores full paths, strip dirs to compare with H5 names
    x = str(x).replace("\\", "/")
    return x.split("/")[-1]  # filename only


def load_run_metadata(
    input_path: str | Path,
    cols: List[str],
    aliases: Optional[Dict[str, List[str]]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    input_path : folder containing *.tsv metadata files
    cols : list of canonical column names you want returned
    aliases : optional {canonical_name: [list of possible TSV names]}
    numeric_cols : list of canonical names to cast to numeric

    Returns
    -------
    DataFrame indexed by image_name with columns:
      - image_name
      - <canonical columns found>
    """
    base = Path(input_path)
    tsvs = sorted(base.glob("*.tsv"))
    frames: List[pd.DataFrame] = []

    if numeric_cols is None:
        numeric_cols = []

    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
        except Exception:
            continue

        # Try to find the image filename column
        name_col = _find_name_column(df)
        if not name_col:
            continue

        # Map canonical column -> actual column found
        canon_to_actual: Dict[str, str] = {}
        for canon in cols:
            actual = None

            # Direct match?
            if canon in df.columns:
                actual = canon
            # Alias match?
            elif aliases and canon in aliases:
                for alt in aliases[canon]:
                    if alt in df.columns:
                        actual = alt
                        break

            if actual is not None:
                canon_to_actual[canon] = actual

        # Nothing found except image_name?
        if not canon_to_actual:
            continue

        # Extract relevant columns
        sel = [name_col] + list(canon_to_actual.values())
        sub = df[sel].copy()

        # Normalise filename
        sub.rename(columns={name_col: "image_name"}, inplace=True)
        sub["image_name"] = sub["image_name"].apply(_normalize_image_name)

        # Rename actual → canonical
        for canon, actual in canon_to_actual.items():
            if canon != actual:
                sub.rename(columns={actual: canon}, inplace=True)

        # Cast numeric columns
        for c in numeric_cols:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")

        frames.append(sub)

    if not frames:
        # return empty with canonical names
        cols_all = ["image_name"] + cols
        return pd.DataFrame(columns=cols_all).set_index("image_name", drop=False)

    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset=["image_name"], keep="last")
    meta = meta.set_index("image_name", drop=False)
    return meta
