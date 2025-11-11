import argparse
from src.analyses.novelty import run_novelty_inbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where to write novelty/<CSV>")
    ap.add_argument("--top_n", type=int, default=200)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--reservoir_cap", type=int, default=20000)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--w_centroid", type=float, default=0.4)
    ap.add_argument("--w_density", type=float, default=0.4)
    ap.add_argument("--w_margin", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = run_novelty_inbox(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        top_n=args.top_n,
        pca_dim=args.pca_dim,
        reservoir_cap=args.reservoir_cap,
        knn_k=args.knn_k,
        w_centroid=args.w_centroid,
        w_density=args.w_density,
        w_margin=args.w_margin,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("[INFO] Novelty inbox written under:", res["out_dir"])


if __name__ == "__main__":
    main()
