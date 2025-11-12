import argparse
from src.analyses.proto_stability import run_prototype_global_stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder with run subfolders")
    ap.add_argument("--out_dir", required=True, help="Root where prototypes live and outputs go")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--min_per_class", type=int, default=3)
    ap.add_argument("--min_runs_for_sil", type=int, default=2)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    res = run_prototype_global_stability(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        pca_dim=args.pca_dim,
        min_per_class=args.min_per_class,
        min_runs_for_sil=args.min_runs_for_sil,
        save_plots=not args.no_plots,
    )
    print("[INFO] Global prototype stability written under:", res["out_dir"])


if __name__ == "__main__":
    main()
