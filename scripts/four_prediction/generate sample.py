import os
import shutil
from pathlib import Path
import pandas as pd
import re


def safe_name(name: str) -> str:
    """
    Replace Windows-incompatible characters with '_'.
    """
    name = str(name)
    # invalid chars: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def create_class_folders(
    csv_path: str,
    output_dir: str,
    exclude_classes=(""),
    image_root: str = None,
    revert_img: bool = False
):
    """
    Create folders for each class and copy images into them,
    except for classes in exclude list.

    Parameters
    ----------
    csv_path : str
        CSV with columns: image_name, class, abs_path(optional)
    output_dir : str
        Where to create the class folders
    exclude_classes : tuple
        Classes to skip (case-insensitive)
    image_root : str
        If abs_path is missing, search under this root directory
    """
    df = pd.read_csv(csv_path)

    df = df.rename(columns={"Top-1 Predicted Label": "class"})
    df = df.rename(columns={"Image Name": "image_name"})
    df = df.rename(columns={"Top-1 Confidence Score": "score"})

    # Normalize
    df["class"] = df["class"].astype(str).str.lower().str.strip()
    df["image_name"] = df["image_name"].astype(str).str.replace("\\", "/").str.strip()
    exclude = set(c.lower() for c in exclude_classes)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build a search index if needed
    search_index = {}
    if image_root:
        image_root = Path(image_root)
        for p in image_root.rglob("*.*"):
            if p.is_file():
                search_index[p.name.lower()] = p

    def resolve_path(row):
        """Return full image path from abs_path or search index."""
        if "abs_path" in row and isinstance(row["abs_path"], str) and Path(row["abs_path"]).exists():
            return Path(row["abs_path"])
        # fallback search
        name = row["image_name"].replace("\\", "/").split("/")[-1].lower()
        return search_index.get(name, None)

    # Process rows
    for _, row in df.iterrows():
        # cls = row["class"].lower()
        cls = row["bin"].lower()
        if cls in exclude:
            continue  # skip excluded classes

        img_path = resolve_path(row)
        if img_path is None or not img_path.exists():
            print(f"[WARN] image not found: {row['image_name']}")
            continue

        cls_safe = safe_name(cls)
        # Create class folder
        class_dir = out_root / cls_safe
        class_dir.mkdir(parents=True, exist_ok=True)


        # # new name:
        # score = float(row["score"])
        # orig = img_path.stem  # without extension
        # ext = img_path.suffix
        #
        # new_name = f"{score:.4f}{ext}"
        # dst = class_dir / new_name

        # Copy or symlink
        dst = class_dir / img_path.name

        if not dst.exists():
            if revert_img:
                # Create a copy with reverted colors
                from PIL import Image, ImageOps

                img = Image.open(img_path)
                img_inverted = ImageOps.invert(img.convert("RGB"))
                img_inverted.save(dst)
            else:
                shutil.copy2(img_path, dst)

    print("[INFO] Done!")


create_class_folders(
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20251125111600\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20251125111600\predictions_with_top3_scores_sample.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20251203104036\prediction_parti20251203104036\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260105155855\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260106100612\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260106124204\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260106124204\predictions_with_top3_scores_sample_euma.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260108132850\predictions_with_top3_scores.csv",
    # csv_path=r"C:\alr4\ai_predict_all\prediction_parti20260113104208\predictions_with_top3_scores.csv",
    csv_path=r"C:\alr4\binned_sample_all.csv",
    # output_dir=r"C:\alr4\analysis\out_ref_sample",
    # output_dir=r"C:\alr4\analysis\out_new_sample2",
    # output_dir=r"C:\alr4\analysis\out_fine1",
    output_dir=r"C:\alr4\analysis\binned_sample",
    # output_dir=r"C:\alr4\analysis\out_finetune",
    # exclude_classes=("detritus", "artefact"),
    # image_root=r"C:\alr4\ecodata\d",  # optional
    image_root=r"C:\alr4\ecodata",  # optional
    revert_img=True,
)
