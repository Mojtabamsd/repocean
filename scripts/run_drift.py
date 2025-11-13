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

    # depth/profile mode config
    ap.add_argument("--depth_bin_size", type=float, default=5.0,
                    help="Bin size for depth mode (same units as depth column)")
    ap.add_argument("--depth_col", type=str, default="object_depth_min",
                    help="Column name to use as depth (default: object_depth_min)")
    ap.add_argument("--profile_col", type=str, default="acq_id",
                    help="Column name to use as profile/group id")

    # common
    ap.add_argument("--sample_per_window", type=int, default=400)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--bootstrap_per_run", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plots", action="store_true")

    ap.add_argument("--cp_value_col", type=str, default="centroid_dist_mean",
                    help="Metric column in drift_series.csv to monitor for change-points")
    ap.add_argument("--cp_k_std", type=float, default=0.5,
                    help="CUSUM allowance k as a multiple of series std (0.3–0.8 typical)")
    ap.add_argument("--cp_h_std", type=float, default=5.0,
                    help="CUSUM threshold h as a multiple of series std (4–8 typical)")
    ap.add_argument("--cp_min_gap", type=int, default=3,
                    help="Minimum index gap between detections")
    ap.add_argument("--no_changepoints", action="store_true",
                    help="Disable change-point detection/output")

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
        depth_col=args.depth_col,
        profile_col=args.profile_col,
        seed=args.seed,
        save_plots=not args.no_plots,
        cp_value_col=args.cp_value_col,
        cp_k_std=args.cp_k_std,
        cp_h_std=args.cp_h_std,
        cp_min_gap=args.cp_min_gap,
        enable_changepoints=not args.no_changepoints,
    )
    print("[INFO] Wrote drift outputs under:", res["per_run_dir"])

if __name__ == "__main__":
    main()
