import argparse
from src.analyses.class_graph import run_class_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where to write class_graph outputs")
    ap.add_argument("--k", type=int, default=10, help="k for kNN")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--max_points_per_run", type=int, default=8000,
                    help="Samples per run to construct the graph (evenly spaced)")
    ap.add_argument("--include_self", action="store_true", help="Include self as neighbour (usually False)")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    res = run_class_graph(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        k=args.k,
        pca_dim=args.pca_dim,
        max_points_per_run=args.max_points_per_run,
        include_self=bool(args.include_self),
        save_plots=not args.no_plots,
    )
    print("[INFO] Class-graph outputs written under:", res["out_dir"])


if __name__ == "__main__":
    main()
