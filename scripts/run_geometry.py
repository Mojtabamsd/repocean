import argparse
from src.analyses.geometry import run_geometry_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where to write CSVs")
    ap.add_argument("--group_mode", choices=["run", "meta"], default="run")
    ap.add_argument("--group_col", default="sample_id")
    ap.add_argument("--depth_bin_size", default=10)
    ap.add_argument("--profile_col", default="sample_id")
    ap.add_argument("--sample_per_group", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = run_geometry_summary(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        group_mode=args.group_mode,
        group_col=args.group_col,
        depth_bin_size=args.depth_bin_size,
        profile_col=args.profile_col,
        sample_per_group=args.sample_per_group,
        seed=args.seed,
    )
    print(df)
    print(f"[INFO] Wrote geometry metrics → {args.out_dir}")


if __name__ == "__main__":
    main()
