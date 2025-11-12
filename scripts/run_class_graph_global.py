import argparse
from src.analyses.class_graph import run_class_graph_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--max_points_total", type=int, default=20000,
                    help="Total budget across all runs")
    ap.add_argument("--per_run_cap", type=int, default=4000,
                    help="Upper cap per run (min of this and total/n_runs)")
    ap.add_argument("--include_self", action="store_true")
    ap.add_argument("--mutual_only", action="store_true", help="Keep only mutual kNN edges")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    res = run_class_graph_global(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        k=args.k,
        pca_dim=args.pca_dim,
        max_points_total=args.max_points_total,
        per_run_cap=args.per_run_cap,
        include_self=bool(args.include_self),
        mutual_only=bool(args.mutual_only),
        save_plots=not args.no_plots,
    )
    print("[INFO] Global class-graph outputs written under:", res["out_dir"])


if __name__ == "__main__":
    main()
