import argparse
from src.visual.montage import make_prototype_montages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", required=True, help="Where prototypes.json lives and where montages will be written")
    ap.add_argument("--mode", choices=["per_class", "per_run"], default="per_class")
    ap.add_argument("--thumb_w", type=int, default=256)
    ap.add_argument("--thumb_h", type=int, default=256)
    ap.add_argument("--thumbs_per_row", type=int, default=8)
    args = ap.parse_args()

    make_prototype_montages(
        parent_dir=args.parent,
        out_dir=args.out_dir,
        mode=args.mode,
        thumb_size=(args.thumb_w, args.thumb_h),
        thumbs_per_row=args.thumbs_per_row,
    )


if __name__ == "__main__":
    main()
