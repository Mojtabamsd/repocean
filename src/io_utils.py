# src/io_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import h5py
import re
import yaml
from yaml.constructor import ConstructorError


# ---------- Discovery ----------

REQUIRED_FILES = [
    ("features", "features_*.h5"),
    ("preds", "predictions_with_top3_scores.csv"),
    ("model_cfg", "model_config.yaml"),
    ("run_cfg", "config.yaml"),
]


def find_runs(parent_dir: str | Path) -> pd.DataFrame:
    """
    Scan parent_dir for subfolders that look like runs (have the required files).
    Returns a DataFrame with one row per run and paths to key files.
    """
    parent = Path(parent_dir)
    rows = []

    for sub in parent.iterdir():
        if not sub.is_dir():
            continue

        # probe required files
        hit = {"run_dir": str(sub)}
        ok = True
        for key, pattern in REQUIRED_FILES:
            matches = list(sub.glob(pattern))
            if key == "features" and not matches:
                ok = False
                break
            if key != "features" and not matches:
                ok = False
                break
            hit[key] = str(matches[0]) if matches else None

        if ok:
            # also record run_id as folder name
            hit["run_id"] = sub.name
            rows.append(hit)

    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


# ---------- Loaders ----------

def load_features(h5_path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load features + image_names + metadata from features_*.h5
    Assumes datasets: 'features' and 'image_names' exist.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]  # (N, D)
        # convert byte strings to str
        image_names = f["image_names"][:]
        if image_names.dtype.kind in {"S", "O"}:
            image_names = image_names.astype(str)

        meta = {
            "encoder_name": f.attrs.get("encoder_name", "unknown"),
            "normalized": bool(f.attrs.get("normalized", False)),
            "num_features": int(features.shape[0]),
            "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
            "file_path": str(h5_path),
        }
    return features, image_names, meta


# --- helper: relaxed YAML reader that strips python/object tags ---
def _read_yaml_relaxed(yaml_path: str) -> dict:
    """
    Load YAML while ignoring Python object tags (e.g., '!!python/object:...').
    1) Try safe_load
    2) On ConstructorError, strip tags and retry
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        text = f.read()

    # First attempt: vanilla safe_load
    try:
        return yaml.safe_load(text) or {}
    except ConstructorError:
        pass  # will fall back below

    # Fallback: strip offending tags, e.g.:
    #   !!python/object:configs.config.TrainingContrastiveConfig
    #   tag:yaml.org,2002:python/object:configs.config.Configuration
    cleaned = re.sub(r"!!python/object:[^\s]+", "", text)
    cleaned = re.sub(r"tag:yaml\.org,2002:python/object:[^\s]+", "", cleaned)

    # Some dumps might include variants like !!python/object/apply:...
    cleaned = re.sub(r"!!python/[^\s]+", "", cleaned)

    # Retry with safe_load
    try:
        return yaml.safe_load(cleaned) or {}
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse YAML even after stripping tags: {yaml_path}\n{e}"
        )


def load_model_config(yaml_path: str | Path) -> Dict:
    """
    Parse model_config.yaml and extract the training_contrastive block (if present).
    Only keep the important fields you listed.
    Works even if the file has !!python/object tags.
    """
    raw = _read_yaml_relaxed(str(yaml_path))

    # Locate training_contrastive in a robust way (key may carry tags in source)
    # We check canonical name and also scan keys case-insensitively.
    tr = None
    if "training_contrastive" in raw:
        tr = raw["training_contrastive"]
    else:
        for k, v in raw.items():
            if isinstance(k, str) and k.lower().strip() == "training_contrastive":
                tr = v
                break
    if tr is None:
        tr = {}

    important = {
        "architecture_type": tr.get("architecture_type"),
        "gray": tr.get("gray"),
        "use_norm": tr.get("use_norm"),
        "loss": tr.get("loss"),
        "dataset": tr.get("dataset"),
        "temp": tr.get("temp"),
        "batch_size": tr.get("batch_size"),
        "num_epoch": tr.get("num_epoch"),
    }
    return {"training_contrastive": important}


def load_run_config(yaml_path: str | Path) -> Dict:
    """
    Parse config.yaml and return at least input_path (original images + TSV metadata).
    """
    yaml_path = Path(yaml_path)
    raw = _read_yaml_relaxed(str(yaml_path))

    # Extract and normalize input_path
    input_path = raw.get("input_path") or raw.get("input-dir") or raw.get("data_path")
    if input_path:
        # Normalize slashes so they work on any OS
        input_path = str(Path(input_path))

    return {"input_path": input_path}


def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    """
    Load predictions_with_top3_scores.csv and normalize column names.
    Expected columns (from your snippet):
      Image Name, Top-1 Predicted Label, Top-1 Confidence Score, ...
    """
    df = pd.read_csv(csv_path)
    # standardize column names
    rename = {
        "Image Name": "image_name",
        "Top-1 Predicted Label": "pred1_label",
        "Top-1 Confidence Score": "pred1_conf",
        "Top-2 Predicted Label": "pred2_label",
        "Top-2 Confidence Score": "pred2_conf",
        "Top-3 Predicted Label": "pred3_label",
        "Top-3 Confidence Score": "pred3_conf",
    }
    df = df.rename(columns=rename)
    # ensure strings
    for c in ["image_name", "pred1_label", "pred2_label", "pred3_label"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


# ---------- Merge helpers ----------

def resolve_image_paths(image_names: np.ndarray | List[str],
                        input_path: str | Path) -> List[str]:
    """
    Given the relative image names from H5/predictions and the input_path,
    build absolute paths to the original .bmp images. We keep a best-effort join.
    """
    base = Path(input_path)
    resolved = []
    for name in image_names:
        # normalize potential 'output/' prefix or mixed slashes
        rel = str(name).replace("\\", "/").replace("output/", "")
        p = base / rel
        resolved.append(str(p))
    return resolved


def merge_run(features: np.ndarray,
              image_names: np.ndarray,
              preds_df: pd.DataFrame,
              model_meta: Dict,
              encoder_meta: Dict,
              input_path: str,
              run_id: str) -> pd.DataFrame:
    """
    Create a single row-per-image table with:
      - image_name, abs_image_path
      - top-1/2/3 predictions + confidences
      - encoder/model fields
      - index pointers into feature matrix
    """
    n = features.shape[0]
    # ensure preds align by image_name
    left = pd.DataFrame({
        "image_name": image_names.astype(str),
        "feat_idx": np.arange(n, dtype=int),
    })

    df = left.merge(preds_df, on="image_name", how="left")

    df["abs_image_path"] = resolve_image_paths(df["image_name"].tolist(), input_path)

    # add meta columns
    df["run_id"] = run_id
    # df["encoder_name"] = encoder_meta.get("encoder_name", "unknown")
    # df["features_normalized"] = bool(encoder_meta.get("normalized", False))
    # df["feature_dim"] = int(encoder_meta.get("feature_dim") or features.shape[1])
    # model config (only important bits)
    tc = (model_meta or {}).get("training_contrastive", {})
    for k, v in tc.items():
        df[f"cfg_{k}"] = v

    return df


# ---------- Convenience high-level ----------

def load_one_run(run_dir: str | Path) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Load everything for a single run directory.
    Returns:
      - merged DataFrame (one row per image)
      - features matrix (N, D)
      - metadata dict with basics
    """
    run_dir = Path(run_dir)

    # resolve file paths
    feats_path = list(run_dir.glob("features_*.h5"))[0]
    preds_path = run_dir / "predictions_with_top3_scores.csv"
    model_cfg_path = run_dir / "model_config.yaml"
    run_cfg_path = run_dir / "config.yaml"

    # load components
    features, image_names, enc_meta = load_features(feats_path)
    preds_df = load_predictions(preds_path)
    model_meta = load_model_config(model_cfg_path)
    run_cfg = load_run_config(run_cfg_path)

    # merge
    df = merge_run(
        features=features,
        image_names=image_names,
        preds_df=preds_df,
        model_meta=model_meta,
        encoder_meta=enc_meta,
        input_path=run_cfg["input_path"],
        run_id=run_dir.name,
    )

    meta = {
        "run_id": run_dir.name,
        "num_images": features.shape[0],
        "feature_dim": features.shape[1],
        "encoder_name": enc_meta.get("encoder_name", "unknown"),
        "normalized": bool(enc_meta.get("normalized", False)),
        "features_path": str(feats_path),
        "preds_path": str(preds_path),
        "model_config_path": str(model_cfg_path),
        "config_path": str(run_cfg_path),
        "input_path": run_cfg["input_path"],
    }
    return df, features, meta

