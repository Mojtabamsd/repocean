import argparse
from src.analyses.drift import run_drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where to write drift CSVs/PNGs")
    ap.add_argument("--mode", choices=["count", "depth", "profile"], default="count")
    # count-mode params
    ap.add_argument("--window_size", type=int, default=1000)
    ap.add_argument("--hop", type=int, default=500)
    # depth/profile shared params
    ap.add_argument("--depth_bin_size", type=float, default=5.0, help="Depth bin size (same units as object_depth)")
    # common
    ap.add_argument("--sample_per_window", type=int, default=400)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--bootstrap_per_run", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    res = run_drift(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        mode=args.mode,
        window_size=args.window_size,
        hop=args.hop,
        sample_per_window=args.sample_per_window,
        pca_dim=args.pca_dim,
        bootstrap_per_run=args.bootstrap_per_run,
        depth_bin_size=args.depth_bin_size,
        seed=args.seed,
        save_plots=not args.no_plots,
    )
    print("[INFO] Wrote drift outputs under:", res["per_run_dir"])


if __name__ == "__main__":
    main()
