"""
prepare_data.py
===============
Loads the merged prediction CSV and prepares it for the fusion pipeline.

Usage
-----
    python prepare_data.py
        (uses ROOT / CSV_PATH / OUT_PATH defaults below)

Or import:
    from prepare_data import prepare
    fusion_df, full_df = prepare("merge_three_prediction_all.csv")
"""

import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_mapping import (
    load_label_tables,
    build_canonical_table,
    build_normaliser,
    normalise_label,
)

# ---------------------------------------------------------------------------
# Paths — edit these to match your layout
# ---------------------------------------------------------------------------
ROOT     = r"C:\alr4\ai_predict\ai_predict_d_all"
CSV_PATH = ROOT + r"\merge_three_prediction_all.csv"
OUT_PATH = ROOT + r"\fusion_ready.csv"

M1_PATH  = "label_to_int.csv"
M2_PATH  = "label_to_int_parti.csv"
M3_PATH  = "class_unique_e.csv"

# ---------------------------------------------------------------------------
# Column names in the merged CSV
# ---------------------------------------------------------------------------
COL = {
    "id":           "object_id",
    "m1_label":     "class_p_r",
    "m1_score":     "score_p_r",
    "m2_label":     "class_p_f",
    "m2_score":     "score_p_f",
    "m3_label":     "class_e_f",
    "m3_score":     "score_e_f",
    "gt_hierarchy": "object_annotation_hierarchy",
    "gt_status":    "object_annotation_status",
}

SPATIAL_COLS    = ["object_lat", "object_lon", "object_depth_min", "object_depth_max"]
SAMPLE_COLS     = ["sample_id", "sample_cruise", "sample_ship",
                   "sample_profileid", "sample_sampledatetime", "image_name"]
ANNOTATION_COLS = ["object_annotation_status", "object_annotation_hierarchy",
                   "object_annotation_person_name", "object_annotation_date"]

# Strings that mean "no prediction" in the label columns
_NA_STRINGS = {"nan", "none", "na", "n/a", ""}


def _is_missing(x) -> bool:
    """
    Safe NA check that works regardless of whether x is:
      - float NaN  (numpy)
      - pd.NA      (pandas nullable)
      - None       (Python)
      - 'nan'      (string)
    Never calls bool() on pd.NA, so never raises TypeError.
    """
    if x is None or x is pd.NA:
        return True
    try:
        # catches float('nan') / numpy.nan — float.__eq__(nan) is False, so use pd.isna
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Step 1 — Load and clean
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> pd.DataFrame:
    print(f"Reading: {path}")
    # dtype=str forces every column to plain Python str — eliminates nullable StringDtype
    # and prevents pandas from ever storing pd.NA in label columns.
    df = pd.read_csv(path, dtype=str, low_memory=False)
    total_raw = len(df)

    # Score columns: convert to float (they were forced to str above)
    for col in [COL["m1_score"], COL["m2_score"], COL["m3_score"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label columns: strip whitespace, replace NA-like strings with None
    for col in [COL["m1_label"], COL["m2_label"], COL["m3_label"]]:
        df[col] = df[col].str.strip()
        df[col] = df[col].where(~df[col].str.lower().isin(_NA_STRINGS), other=None)

    # Drop rows where ALL three label columns are missing
    pred_cols = [COL["m1_label"], COL["m2_label"], COL["m3_label"]]
    mask_all_missing = df[pred_cols].isnull().all(axis=1)
    df = df[~mask_all_missing].reset_index(drop=True)

    dropped = total_raw - len(df)
    print(f"  {total_raw:,} rows → {len(df):,} rows with predictions "
          f"({dropped:,} empty rows dropped)")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Canonical label normalisation
# ---------------------------------------------------------------------------

def _safe_normalise(x, norm_map: dict):
    """
    Normalise a single raw label cell to its canonical form.
    Handles all possible cell types that pandas may deliver:
      - plain str  (normal case after dtype=str read)
      - float NaN  (happens when an entire column is numeric-looking)
      - None / pd.NA  (explicit missing)
    Returns None for any missing / unrecognisable value.
    """
    # reject non-string types (float NaN, None, pd.NA) without calling bool() on pd.NA
    if x is None or x is pd.NA:
        return None
    # float NaN sneaks through even with dtype=str on some pandas versions
    try:
        if not isinstance(x, str):
            x = str(x)
    except Exception:
        return None
    x = x.strip()
    if x.lower() in _NA_STRINGS:   # "nan", "none", "na", "n/a", ""
        return None
    return normalise_label(x, norm_map)


def add_canonical_labels(df: pd.DataFrame, norm_map: dict) -> pd.DataFrame:
    new_cols = {}
    for raw_col, canon_col in [
        (COL["m1_label"], "canon_m1"),
        (COL["m2_label"], "canon_m2"),
        (COL["m3_label"], "canon_m3"),
    ]:
        new_cols[canon_col] = df[raw_col].apply(_safe_normalise, norm_map=norm_map)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Step 3 — Quality flags  (NA-safe)
# ---------------------------------------------------------------------------

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    # .isnull() on object columns with None always returns plain bool Series
    m1_null = df["canon_m1"].isnull()
    m2_null = df["canon_m2"].isnull()
    m3_null = df["canon_m3"].isnull()

    # Compare raw labels and scores safely:
    # fillna("__MISSING__") so NA==NA → False, not pd.NA
    _SENT = "__MISSING__"
    label_same = (
        df[COL["m1_label"]].fillna(_SENT) == df[COL["m2_label"]].fillna(_SENT)
    )
    # For scores (float), use a tolerance-safe comparison; NaN==NaN → False by design
    s1 = df[COL["m1_score"]].to_numpy(dtype=float, na_value=float("nan"))
    s2 = df[COL["m2_score"]].to_numpy(dtype=float, na_value=float("nan"))
    import numpy as np
    score_same = pd.Series(
        (s1 == s2) & ~(np.isnan(s1) | np.isnan(s2)),
        index=df.index,
    )

    identical = label_same & score_same

    # n_active: plain int arithmetic on bool (no NA in m1_null/m2_null/m3_null)
    n_active = (~m1_null).astype(int) + (~m2_null).astype(int) + (~m3_null).astype(int)

    flags = pd.DataFrame({
        "flag_m1_null":          m1_null.astype(bool),
        "flag_m2_null":          m2_null.astype(bool),
        "flag_m3_null":          m3_null.astype(bool),
        "flag_m1_m2_identical":  identical.astype(bool),
        "n_models_active":       n_active,
    }, index=df.index)

    return pd.concat([df, flags], axis=1)


# ---------------------------------------------------------------------------
# Step 4 — Slim output
# ---------------------------------------------------------------------------

def build_fusion_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        [COL["id"]]
        + [c for c in SAMPLE_COLS     if c in df.columns]
        + [c for c in SPATIAL_COLS    if c in df.columns]
        + [c for c in ANNOTATION_COLS if c in df.columns]
        + [COL["m1_label"], COL["m1_score"],
           COL["m2_label"], COL["m2_score"],
           COL["m3_label"], COL["m3_score"],
           "canon_m1", "canon_m2", "canon_m3",
           "flag_m1_null", "flag_m2_null", "flag_m3_null",
           "flag_m1_m2_identical", "n_models_active"]
    )
    return df[[c for c in keep if c in df.columns]].copy()


# ---------------------------------------------------------------------------
# Step 5 — Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    sep = "=" * 58
    print(f"\n{sep}\nDATA SUMMARY\n{sep}")
    print(f"\n  Rows with predictions : {total:,}")
    if "sample_id" in df.columns:
        print(f"  Unique samples        : {df['sample_id'].nunique():,}")

    print("\n── Null / abstention counts ──")
    for label, flag in [("M1 (class_p_r)", "flag_m1_null"),
                        ("M2 (class_p_f)", "flag_m2_null"),
                        ("M3 (class_e_f)", "flag_m3_null")]:
        n = int(df[flag].sum())
        pct = 100 * n / total if total else 0
        print(f"  {label}: {n:,} nulls ({pct:.1f}%)")

    print("\n── M1 == M2 identical predictions ──")
    n_ident = int(df["flag_m1_m2_identical"].sum())
    pct = 100 * n_ident / total if total else 0
    print(f"  {n_ident:,} rows ({pct:.1f}%) have identical M1 label + score")
    if total > 0 and n_ident / total > 0.5:
        print("  *** WARNING: >50% identical — M1 and M2 may not be independent.")
        print("      Consider treating M2 as abstaining when flag_m1_m2_identical=True.")

    print("\n── Active models per row ──")
    for n, cnt in df["n_models_active"].value_counts().sort_index().items():
        bar = "█" * int(20 * cnt / total) if total else ""
        print(f"  {n} models: {cnt:>6,}  {bar}")

    print("\n── Raw score ranges (before normalisation) ──")
    for model, col in [("M1", COL["m1_score"]),
                       ("M2", COL["m2_score"]),
                       ("M3", COL["m3_score"])]:
        s = df[col].dropna()
        if len(s):
            print(f"  {model}: min={s.min():.3f}  p25={s.quantile(.25):.3f}  "
                  f"median={s.median():.3f}  p75={s.quantile(.75):.3f}  max={s.max():.3f}")

    print("\n── Label normalisation ──")
    for model, raw_col, canon_col in [
        ("M1", COL["m1_label"], "canon_m1"),
        ("M2", COL["m2_label"], "canon_m2"),
        ("M3", COL["m3_label"], "canon_m3"),
    ]:
        n_raw    = df[raw_col].notna().sum()
        n_mapped = df[canon_col].notna().sum()
        n_miss   = int(n_raw - n_mapped)
        print(f"  {model}: {int(n_mapped):,} mapped  |  {n_miss:,} unmapped")
        if n_miss > 0:
            bad = (df[df[canon_col].isna() & df[raw_col].notna()][raw_col]
                   .value_counts().head(5))
            for lbl, cnt in bad.items():
                print(f"      → '{lbl}'  ×{cnt}")

    print("\n── Top predicted labels per model ──")
    for model, col in [("M1", "canon_m1"), ("M2", "canon_m2"), ("M3", "canon_m3")]:
        vc = df[col].value_counts().head(5)
        print(f"  {model}:")
        for lbl, cnt in vc.items():
            pct = 100 * cnt / total if total else 0
            print(f"    {str(lbl):<35}  {cnt:>6,}  ({pct:.1f}%)")

    if COL["gt_status"] in df.columns:
        print("\n── Annotation status ──")
        for status, cnt in df[COL["gt_status"]].value_counts().items():
            print(f"  {status}: {cnt:,}")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare(
    data_path: str = CSV_PATH,
    m1_path: str = M1_PATH,
    m2_path: str = M2_PATH,
    m3_path: str = M3_PATH,
) -> tuple:
    """
    Returns (fusion_df, full_df).
    fusion_df — slim DataFrame ready for the fusion engine.
    full_df   — full DataFrame with all original columns + new flag/canon columns.
    """
    print("Loading label tables...")
    df1, df2, df3 = load_label_tables(m1_path, m2_path, m3_path)
    canonical_table = build_canonical_table(df1, df2, df3)
    norm_map        = build_normaliser(canonical_table)
    print(f"  Canonical namespace: {len(canonical_table)} labels\n")

    df = load_predictions(data_path)
    df = add_canonical_labels(df, norm_map)
    df = add_quality_flags(df)
    print_summary(df)

    fusion_df = build_fusion_df(df)
    return fusion_df, df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model predictions for fusion")
    parser.add_argument("--data",   default=CSV_PATH)
    parser.add_argument("--m1",     default=M1_PATH)
    parser.add_argument("--m2",     default=M2_PATH)
    parser.add_argument("--m3",     default=M3_PATH)
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()

    fusion_df, _ = prepare(args.data, args.m1, args.m2, args.m3)
    fusion_df.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")
    print(f"  {len(fusion_df):,} rows  |  {len(fusion_df.columns)} columns")
    print("\n  Columns:")
    for c in fusion_df.columns:
        print(f"    {c}")