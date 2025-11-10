from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

CANDIDATE_NAME_COLS = ["image_name", "filename", "file", "img", "name"]
DEPTH_COL = "object_depth"
PROFILE_COL = "acq_id"


def _find_name_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for k in CANDIDATE_NAME_COLS:
        if k in lower:
            return lower[k]
    # heuristic: any column containing 'name' or 'file'
    for c in cols:
        cl = c.lower()
        if "name" in cl or "file" in cl:
            return c
    return None


def _normalize_image_name(x: str) -> str:
    # keep as provided; if TSV stores full paths, strip dirs to compare with H5 names
    x = str(x).replace("\\", "/")
    return x.split("/")[-1]  # filename only


def load_run_metadata(input_path: str | Path) -> pd.DataFrame:
    """
    Scan input_path for *.tsv, read minimal columns:
      - image_name (auto-detected column)
      - object_depth (float, if present)
      - acq_id (string, if present)
    Return a DataFrame indexed by normalized image_name.
    """
    base = Path(input_path)
    tsvs = sorted(base.glob("*.tsv"))
    frames: List[pd.DataFrame] = []
    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)  # read as str, cast later
        except Exception:
            continue
        name_col = _find_name_column(df)
        if not name_col:
            continue

        cols = [name_col]
        if DEPTH_COL in df.columns:
            cols.append(DEPTH_COL)
        if PROFILE_COL in df.columns:
            cols.append(PROFILE_COL)

        sub = df[cols].copy()
        sub.rename(columns={name_col: "image_name"}, inplace=True)
        sub["image_name"] = sub["image_name"].apply(_normalize_image_name)

        # cast depth if present
        if DEPTH_COL in sub.columns:
            sub[DEPTH_COL] = pd.to_numeric(sub[DEPTH_COL], errors="coerce")

        frames.append(sub)

    if not frames:
        # return empty frame with expected columns
        return pd.DataFrame(columns=["image_name", DEPTH_COL, PROFILE_COL]).set_index("image_name", drop=False)

    meta = pd.concat(frames, ignore_index=True)
    # latest rows win if duplicates
    meta = meta.drop_duplicates(subset=["image_name"], keep="last")
    meta = meta.set_index("image_name", drop=False)
    return meta
