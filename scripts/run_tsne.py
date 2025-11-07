import argparse
from src.analyses.tsne import run_tsne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--sample_per_run", type=int, default=2000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--learning_rate", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = run_tsne(
        parent_dir=args.parent,
        out_csv=args.out_csv,
        out_png=args.out_png,
        sample_per_run=args.sample_per_run,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(df.head())
    print(f"[INFO] Saved t-SNE CSV → {args.out_csv}")
    if args.out_png:
        print(f"[INFO] Saved t-SNE plot → {args.out_png}")


if __name__ == "__main__":
    main()
