# scripts/load_and_merge_one_run.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from src.io_utils import load_one_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a single run folder")
    ap.add_argument("--out_dir", default="results", help="Where to write merged outputs")
    args = ap.parse_args()

    df, feats, meta = load_one_run(args.run_dir)

    out_base = Path(args.out_dir) / meta["run_id"]
    out_base.mkdir(parents=True, exist_ok=True)

    # Save merged table
    df.to_parquet(out_base / "merged.parquet", index=False)
    df.to_csv(out_base / "merged.csv", index=False)

    # Save features (npy) and a small JSON meta
    np.save(out_base / "features.npy", feats)
    pd.Series(meta).to_json(out_base / "meta.json", indent=2)

    print(f"[INFO] Saved merged table and features under: {out_base}")
    print(f"[INFO] Rows: {len(df)} | Feature dim: {feats.shape[1]}")


if __name__ == "__main__":
    main()
