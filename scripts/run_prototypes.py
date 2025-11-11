import argparse
from src.analyses.prototypes import run_prototypes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where to write prototypes/coverage")
    ap.add_argument("--mode", choices=["per_class", "per_run"], default="per_class")
    ap.add_argument("--k", type=int, default=10, help="Number of medoids per group")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--max_per_class", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--min_points_for_group", type=int, default=50)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = run_prototypes(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        mode=args.mode,
        k=args.k,
        pca_dim=args.pca_dim,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
        metric=args.metric,
        min_points_for_group=args.min_points_for_group,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print("[INFO] Prototypes written under:", paths["out_dir"])


if __name__ == "__main__":
    main()
