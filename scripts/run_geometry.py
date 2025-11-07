import argparse
from src.analyses.geometry import run_geometry_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_csv", default="results/global/geometry_metrics.csv")
    ap.add_argument("--sample_per_run", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = run_geometry_summary(
        parent_dir=args.parent,
        out_csv=args.out_csv,
        sample_per_run=args.sample_per_run,
        seed=args.seed,
    )
    print(df)
    print(f"[INFO] Wrote geometry metrics → {args.out_csv}")


if __name__ == "__main__":
    main()
