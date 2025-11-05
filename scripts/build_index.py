# scripts/build_index.py
from pathlib import Path
import pandas as pd
import argparse

from src.io_utils import find_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Parent folder containing many run subfolders")
    ap.add_argument("--out", default="data/metadata_index.csv", help="Output CSV index path")
    args = ap.parse_args()

    df = find_runs(args.parent)
    if df.empty:
        print(f"[WARN] No runs found under: {args.parent}")
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[INFO] Indexed {len(df)} runs → {args.out}")


if __name__ == "__main__":
    main()
