from __future__ import annotations
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py


def open_h5(h5_path: str | Path) -> h5py.File:
    return h5py.File(str(h5_path), "r")


def get_h5_shapes(h5f: h5py.File) -> Tuple[int, int]:
    feats = h5f["features"]
    n, d = feats.shape[0], feats.shape[1]
    return int(n), int(d)


def iter_feature_chunks(h5f: h5py.File, batch_size: int) -> Iterator[Dict]:
    """
    Yield dictionaries with:
      - 'idx_start', 'idx_end'
      - 'features'  (numpy array [B, D])
      - 'image_names' (list[str] length B)
    Reads slices from HDF5 without loading all into memory.
    """
    feats = h5f["features"]
    names = h5f["image_names"]
    n, _ = feats.shape
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        X = feats[i:j]               # loads the slice only
        nm = names[i:j].astype(str)  # convert here per chunk
        yield {"idx_start": i, "idx_end": j, "features": X, "image_names": nm}


def sample_indices_uniform(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


def read_rows_by_indices(h5f: h5py.File, indices: np.ndarray) -> Dict:
    feats = h5f["features"]
    names = h5f["image_names"]
    X = feats[indices]                 # fancy indexing: reads only rows needed
    nm = names[indices].astype(str)
    return {"features": X, "image_names": nm, "indices": indices}


def load_predictions_map(csv_path: str | Path, cols: Optional[List[str]]=None) -> pd.DataFrame:
    """
    Read predictions CSV and index by image_name for quick join-on-demand.
    Only required columns are read to save RAM.
    """
    usecols = cols or [
        "Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score",
        "Top-2 Predicted Label", "Top-2 Confidence Score",
        "Top-3 Predicted Label", "Top-3 Confidence Score",
        "encoder_name"
    ]
    df = pd.read_csv(csv_path, usecols=[c for c in usecols if c != "encoder_name" or c in usecols])
    df = df.rename(columns={
        "Image Name": "image_name",
        "Top-1 Predicted Label": "pred1_label",
        "Top-1 Confidence Score": "pred1_conf",
        "Top-2 Predicted Label": "pred2_label",
        "Top-2 Confidence Score": "pred2_conf",
        "Top-3 Predicted Label": "pred3_label",
        "Top-3 Confidence Score": "pred3_conf",
    })
    df["image_name"] = df["image_name"].astype(str)
    df = df.set_index("image_name", drop=False)
    return df
