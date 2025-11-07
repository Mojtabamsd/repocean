from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.decomposition import PCA

from src.index import build_run_index
from src.stream import open_h5, get_h5_shapes, sample_indices_uniform, read_rows_by_indices


def _intrinsic_dim_pca(X: np.ndarray, thresholds=(0.90, 0.95)) -> Tuple[int, int]:
    ncomp = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="auto", random_state=42)
    pca.fit(X)
    csum = np.cumsum(pca.explained_variance_ratio_)
    idx90 = int(np.searchsorted(csum, thresholds[0]) + 1)
    idx95 = int(np.searchsorted(csum, thresholds[1]) + 1)
    return idx90, idx95


def run_geometry_summary(parent_dir: str,
                         out_csv: str,
                         sample_per_run: int = 2000,
                         seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    runs = build_run_index(parent_dir)
    rows = []

    for _, r in runs.iterrows():
        feats_path = r["features"]
        with open_h5(feats_path) as h5f:
            n, d = get_h5_shapes(h5f)
            if n == 0:
                continue
            idx = sample_indices_uniform(n, sample_per_run, rng)
            part = read_rows_by_indices(h5f, idx)
            X = part["features"]

        norms = np.linalg.norm(X, axis=1)
        mean_norm, std_norm = float(norms.mean()), float(norms.std())
        dim90, dim95 = _intrinsic_dim_pca(X)

        rows.append({
            "run_id": r["run_id"],
            "num_rows": n,
            "feat_dim": d,
            "sampled": X.shape[0],
            "mean_norm": round(mean_norm, 6),
            "std_norm": round(std_norm, 6),
            "pca_dim_90": int(dim90),
            "pca_dim_95": int(dim95),
        })

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
