from __future__ import annotations
import argparse
from src.tasks.tsne_embed import tsne_across_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Parent folder with many run subfolders")
    ap.add_argument("--out_csv", required=True, help="Where to save small t-SNE CSV")
    ap.add_argument("--out_png", default=None, help="Optional PNG plot")
    ap.add_argument("--sample_per_run", type=int, default=2000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--learning_rate", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tsne_across_runs(
        parent_dir=args.parent,
        out_csv=args.out_csv,
        out_png=args.out_png,
        sample_per_run=args.sample_per_run,
        pca_dim=args.pca_dim,
        random_state=args.seed,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
    )

if __name__ == "__main__":
    main()
