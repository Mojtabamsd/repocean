from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

CANDIDATE_NAME_COLS = ["img_file_name", "object_id", "object_lat", "object_lon", "object_date", "object_time", ]


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
    depth_col: str = "object_depth_min",
    profile_col: str = "acq_id",
    extra_depth_aliases: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read *.tsv files under input_path and return a DF with:
      - image_name (index)
      - <depth_col> (if present)
      - <profile_col> (if present)

    You can pass your exact column names. If a chosen column is missing,
    we will try common aliases (for depth) if provided.
    """
    base = Path(input_path)
    tsvs = sorted(base.glob("*.tsv"))
    frames: List[pd.DataFrame] = []

    # depth fallback aliases (optional)
    aliases = extra_depth_aliases or ["object_depth", "object_depth_max"]

    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
        except Exception:
            continue

        name_col = _find_name_column(df)
        if not name_col:
            continue

        cols = [name_col]
        chosen_depth_col = None
        if depth_col in df.columns:
            chosen_depth_col = depth_col
            cols.append(depth_col)
        else:
            # try aliases
            for a in aliases:
                if a in df.columns:
                    chosen_depth_col = a
                    cols.append(a)
                    break

        chosen_profile_col = None
        if profile_col in df.columns:
            chosen_profile_col = profile_col
            cols.append(profile_col)

        sub = df[cols].copy()
        sub.rename(columns={name_col: "image_name"}, inplace=True)
        sub["image_name"] = sub["image_name"].apply(_normalize_image_name)

        # standardize column names after selection
        if chosen_depth_col and chosen_depth_col != depth_col:
            sub.rename(columns={chosen_depth_col: depth_col}, inplace=True)
        if chosen_profile_col and chosen_profile_col != profile_col:
            sub.rename(columns={chosen_profile_col: profile_col}, inplace=True)

        # cast depth (if present)
        if depth_col in sub.columns:
            sub[depth_col] = pd.to_numeric(sub[depth_col], errors="coerce")

        frames.append(sub)

    if not frames:
        # empty but with expected columns
        cols = ["image_name"]
        if depth_col:
            cols.append(depth_col)
        if profile_col:
            cols.append(profile_col)
        return pd.DataFrame(columns=cols).set_index("image_name", drop=False)

    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset=["image_name"], keep="last")
    meta = meta.set_index("image_name", drop=False)
    return meta
