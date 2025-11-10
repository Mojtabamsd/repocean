import argparse
from src.analyses.purity import run_purity_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Directory to write small CSV artifacts")
    ap.add_argument("--sample_per_run", type=int, default=2000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10], help="k values for purity@k")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = run_purity_analysis(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        sample_per_run=args.sample_per_run,
        pca_dim=args.pca_dim,
        ks=args.ks,
        seed=args.seed,
    )
    print("[INFO] Wrote:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
