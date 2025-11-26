import argparse
from src.analyses.class_mix import run_class_mix_by_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_col", default="sample_id")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--top_k_classes", type=int, default=30)
    args = ap.parse_args()

    paths = run_class_mix_by_sample(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        group_col=args.group_col,
        save_plots=not args.no_plots,
        top_k_classes=args.top_k_classes,
    )
    print(paths)


if __name__ == "__main__":
    main()
