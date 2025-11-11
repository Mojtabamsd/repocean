import argparse
from src.visual.novelty_montage import make_novelty_montage_per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where novelty_inbox.csv was written and where PNG will go")
    ap.add_argument("--top_n", type=int, default=200, help="How many top-scoring items to show")
    ap.add_argument("--thumb_w", type=int, default=256)
    ap.add_argument("--thumb_h", type=int, default=256)
    ap.add_argument("--thumbs_per_row", type=int, default=8)
    args = ap.parse_args()

    make_novelty_montage_per_run(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        top_n=args.top_n,
        thumb_size=(args.thumb_w, args.thumb_h),
        thumbs_per_row=args.thumbs_per_row,
    )


if __name__ == "__main__":
    main()
