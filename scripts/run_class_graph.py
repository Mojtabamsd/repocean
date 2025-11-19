import argparse
from src.analyses.class_graph import run_class_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_mode", choices=["run", "meta"], default="run")
    ap.add_argument("--group_col", default="sample_id")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--max_points_per_group", type=int, default=8000)
    ap.add_argument("--include_self", action="store_true")
    ap.add_argument("--mutual_only", action="store_true", help="Keep only mutual kNN edges")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    res = run_class_graph(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        group_mode=args.group_mode,
        group_col=args.group_col,
        k=args.k,
        pca_dim=args.pca_dim,
        max_points_per_group=args.max_points_per_group,
        include_self=bool(args.include_self),
        mutual_only=bool(args.mutual_only),
        save_plots=not args.no_plots,
    )
    print("[INFO] Class-graph outputs written under:", res["per_group_dir"])


if __name__ == "__main__":
    main()
