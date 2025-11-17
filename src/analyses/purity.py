from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict

import importlib
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from src.index import build_group_index
from src.analyses.common import collect_group_samples

# -------------------------
# Utilities
# -------------------------


def _faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None or importlib.util.find_spec("faiss_cpu") is not None


def _fit_pca(X: np.ndarray, pca_dim: int, batch_size: int = 4096) -> np.ndarray:
    """Incremental PCA to D -> pca_dim (or pass-through if smaller)."""
    if X.shape[1] <= pca_dim:
        return X
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=batch_size)
    Xp = ipca.fit_transform(X)
    return Xp


def _build_knn(X: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (distances, indices) for kNN on X.
    Prefers FAISS (L2) if available; otherwise sklearn NearestNeighbors (euclidean).
    Distances are shape [N, k], indices [N, k], first neighbor is the point itself for exact kNN.
    """
    k = n_neighbors
    if _faiss_available():
        try:
            import faiss  # type: ignore
            X32 = X.astype(np.float32, copy=False)
            index = faiss.IndexFlatL2(X32.shape[1])
            index.add(X32)
            D, I = index.search(X32, k)
            return D, I
        except Exception:
            pass  # fall through to sklearn

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(X)
    D, I = nn.kneighbors(X, return_distance=True)
    return D, I


def _purity_at_k(labels: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """
    labels: shape [N], any dtype convertible to str
    knn_idx: shape [N, k], integer indices; assume knn_idx[:,0] == self
    Returns purity per row (float in [0,1]) at this k (including self neighbor).
    """
    lbl = labels.astype(str)
    self_lbl = lbl  # [N]
    nbr_lbl = lbl[knn_idx]  # [N, k]
    # match matrix
    match = (nbr_lbl == self_lbl[:, None]).astype(np.float32)
    # include self (column 0), that's fine; it just adds 1/k to every purity
    purity = match.mean(axis=1)
    return purity


def _local_density_from_dist(knn_dist: np.ndarray) -> np.ndarray:
    """
    A simple local density proxy: negative mean distance to neighbors (excluding self column 0).
    Higher density => less mean distance => larger (less negative) number when we negate distances.
    """
    if knn_dist.shape[1] <= 1:
        return np.zeros(knn_dist.shape[0], dtype=np.float32)
    return -knn_dist[:, 1:].mean(axis=1)


def _centroid_distance(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Distance from each point to its label centroid (euclidean), computed in the same space as X.
    If a label has only one sample, its distance is 0 by definition.
    """
    lbl = labels.astype(str)
    uniques, inv = np.unique(lbl, return_inverse=True)
    centroids = np.zeros((uniques.size, X.shape[1]), dtype=X.dtype)
    counts = np.bincount(inv)
    # accumulate sums
    np.add.at(centroids, inv, X)
    centroids = centroids / counts[:, None]
    # distance per point to centroid of its label
    d = np.linalg.norm(X - centroids[inv], axis=1)
    return d


# -------------------------
# Main API
# -------------------------


def run_purity_analysis(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",       # "run" | "meta"
    group_col: str = "sample_id",  # used when group_mode == "meta"
    sample_per_group: int = 2000,
    pca_dim: int = 50,
    ks: Iterable[int] = (5, 10),
    seed: int = 42,
) -> Dict[str, str]:
    """
    Global kNN purity across groups (runs or metadata groups), plus basic
    geometry–confidence alignment.

    Steps:
      1) Build run index
      2) For each run, sample up to sample_per_run rows (streamed from HDF5)
      3) Join minimal predictions (top-1 label + conf) ONLY for sampled rows
      4) PCA -> pca_dim
      5) Build kNN on concatenated sample; compute purity@k
      6) Compute local density and distance-to-centroid; correlate with pred1_conf
      7) Save small CSVs: per-point summary and per-run aggregates

    Returns a dict of output file paths.
    """
    rng = np.random.default_rng(seed)

    # --- Build group index (run-level or metadata-level) ---
    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
    )
    if groups.empty:
        raise RuntimeError(f"No groups found under {parent_dir} (mode={group_mode})")

    out_root = Path(out_dir) / "purity"
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Collect sampled features + metadata across all groups ---
    X_all, meta = collect_group_samples(
        groups=groups,
        group_mode=group_mode,
        sample_per_group=sample_per_group,
        rng=rng,
        attach_preds=True,
    )

    # --- PCA reduce for stability/efficiency ---
    X_pca = _fit_pca(X_all, pca_dim=pca_dim)

    # --- kNN once at max k ---
    ks = sorted(set(int(k) for k in ks if k >= 1))
    if not ks:
        raise ValueError("ks must contain at least one positive integer.")
    kmax = max(ks)
    knn_dist, knn_idx = _build_knn(X_pca, n_neighbors=kmax)

    # --- Per-point metrics (vectorized) ---
    labels = meta["pred1_label"].values

    # Purity at each requested k
    for k in ks:
        pur = _purity_at_k(labels, knn_idx[:, :k])
        meta[f"purity@{k}"] = pur

    # Local density proxy
    meta["local_density"] = _local_density_from_dist(knn_dist)

    # Distance to label centroid (computed on PCA space)
    meta["centroid_dist"] = _centroid_distance(X_pca, labels)

    # Confidence–geometry correlations (overall, global)
    try:
        r_density, p_density = pearsonr(
            meta["pred1_conf"].values,
            meta["local_density"].values,
        )
    except Exception:
        r_density, p_density = np.nan, np.nan

    try:
        # negative distance -> higher is better, so correlate with -dist
        r_centroid, p_centroid = pearsonr(
            meta["pred1_conf"].values,
            -meta["centroid_dist"].values,
        )
    except Exception:
        r_centroid, p_centroid = np.nan, np.nan

    # --- Aggregates, depending on grouping mode ---
    agg_cols = {f"purity@{k}": ["mean", "std"] for k in ks}
    agg_cols.update({
        "local_density": ["mean", "std"],
        "centroid_dist": ["mean", "std"],
    })

    if group_mode == "run":
        # One row per run_id
        agg = meta.groupby("run_id").agg(agg_cols)
        agg_index_cols = ["run_id"]
        out_agg_name = "purity_per_run.csv"
    else:
        # group_mode == "meta": one row per (run_id, group_id)
        if "group_id" not in meta.columns:
            raise RuntimeError("group_mode='meta' but 'group_id' column not found in metadata.")
        agg = meta.groupby(["run_id", "group_id"]).agg(agg_cols)
        agg_index_cols = ["run_id", "group_id"]
        out_agg_name = "purity_per_group.csv"

    # flatten columns
    agg.columns = [
        "_".join([c] + ([s] if isinstance(s, str) else []))
        for c, s in agg.columns.to_flat_index()
    ]
    agg = agg.reset_index()

    # attach global correlations (same for all groups)
    agg["corr_conf_local_density_r"] = r_density
    agg["corr_conf_local_density_p"] = p_density
    agg["corr_conf_centroid_dist_r"] = r_centroid
    agg["corr_conf_centroid_dist_p"] = p_centroid

    # --- Save small artifacts ---
    out_points = out_root / "purity_points.csv"
    out_agg = out_root / out_agg_name

    # Per-point CSV: always saved, includes group_id when in meta mode
    base_cols = ["run_id", "image_name", "pred1_label", "pred1_conf",
                 "local_density", "centroid_dist"]
    if group_mode == "meta" and "group_id" in meta.columns:
        base_cols.insert(1, "group_id")  # after run_id

    keep_cols = base_cols + [f"purity@{k}" for k in ks]
    meta[keep_cols].to_csv(out_points, index=False)
    agg.to_csv(out_agg, index=False)

    return {
        "points_csv": str(out_points),
        "agg_csv": str(out_agg),
        "agg_level": "run" if group_mode == "run" else "group",
    }
