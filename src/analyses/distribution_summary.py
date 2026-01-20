from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.index import build_group_index
from src.stream import (
    open_h5, get_h5_shapes, sample_indices_uniform, read_rows_by_indices
)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        # fallback to uniform if weights are degenerate
        return np.ones_like(w, dtype=np.float64) / max(len(w), 1)
    return w / s


def _weighted_mean(X: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if w is None:
        return X.mean(axis=0)
    w = _normalize_weights(w)
    return (X * w[:, None]).sum(axis=0)


def _weighted_cov_trace_and_effrank(
    X: np.ndarray,
    w: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Returns:
      trace_cov  : trace of covariance (overall variance / heterogeneity)
      eff_rank   : effective rank from eigenvalues (mode complexity)
    """
    n, d = X.shape
    if n <= 1:
        return 0.0, 0.0

    mu = _weighted_mean(X, w)
    Xc = X - mu

    if w is None:
        # Unweighted covariance: (Xc^T Xc)/(n-1)
        # trace = sum of variances = sum of squared deviations / (n-1)
        ss = float((Xc * Xc).sum())  # sum over all dims & rows
        trace_cov = ss / max(n - 1, 1)
        # eigenvalues of cov are eigenvalues of (Xc^T Xc)/(n-1)
        # use Gram matrix (n x n) if d is large; but here n is small (sampled), so either is fine
        C = (Xc.T @ Xc) / max(n - 1, 1)  # (d x d)
    else:
        w = _normalize_weights(w)
        # Weighted covariance: sum_i w_i (x_i - mu)(x_i - mu)^T
        # (This is the "population" weighted cov; consistent for comparison when weights are used.)
        # trace = sum_i w_i ||x_i - mu||^2
        trace_cov = float((w[:, None] * (Xc * Xc)).sum())
        C = (Xc.T * w) @ Xc  # (d x d)

    # Eigenvalues (clip for numerical stability)
    # For sampled n~2k and d~512/1024, this is fine.
    evals = np.linalg.eigvalsh(C)
    evals = np.clip(evals, 0.0, np.inf)

    s1 = float(evals.sum())
    s2 = float((evals * evals).sum())
    if s1 <= eps or s2 <= eps:
        eff_rank = 0.0
    else:
        eff_rank = (s1 * s1) / s2

    return float(trace_cov), float(eff_rank)


def run_distribution_summary(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",          # "run" | "meta"
    group_col: str = "sample_id",     # used when group_mode == "meta"
    sample_per_group: int = 2000,
    seed: int = 42,
    save_centroids: bool = True,
) -> pd.DataFrame:
    """
    Distribution-style summary of feature space per group (run or meta group).

    For each group we compute (on a sampled subset):
      - num_rows         : total number of feature rows in that group
      - feat_dim         : feature dimensionality
      - sampled          : number of rows used for metrics
      - mean_norm        : mean L2 norm of sampled features
      - std_norm         : std L2 norm
      - centroid_norm    : L2 norm of centroid vector (in original feature space)
      - trace_cov        : trace of covariance (overall heterogeneity)
      - eff_rank         : effective rank from covariance eigenvalues (mode complexity)

    Notes:
      - This stays in the original embedding geometry (no 2D projection).
      - Later we can add sample weights (e.g., downweight ubiquitous/detritus-like modes)
        without dropping any data by changing the weight vector per sample.
    """
    rng = np.random.default_rng(seed)

    out_root = Path(out_dir) / "distribution"
    out_root.mkdir(parents=True, exist_ok=True)

    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
    )
    if groups.empty:
        raise RuntimeError(
            f"No groups found under {parent_dir} (group_mode={group_mode}, group_col={group_col})"
        )

    rows = []
    centroids: Dict[Tuple[str, str], np.ndarray] = {}

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        features_path = g["features"]

        # decide group_id + indices
        if group_mode == "run":
            group_id = run_id
            with open_h5(features_path) as h5f:
                n_total, d = get_h5_shapes(h5f)
                if n_total == 0:
                    continue
                idx_all = np.arange(n_total, dtype=np.int64)
        else:
            group_id = g["group_id"]
            idx_group = g.get("indices", None)
            if idx_group is None:
                continue
            idx_all = np.asarray(idx_group, dtype=np.int64)
            if idx_all.size == 0:
                continue
            with open_h5(features_path) as h5f:
                n_total, d = get_h5_shapes(h5f)

        num_rows = int(idx_all.size)
        if num_rows == 0:
            continue

        k = min(sample_per_group, num_rows)
        if k <= 0:
            continue

        rel_idx = sample_indices_uniform(num_rows, k, rng)
        sel_idx = np.sort(idx_all[rel_idx])

        with open_h5(features_path) as h5f:
            part = read_rows_by_indices(h5f, sel_idx)
            X = part["features"]

        if X.size == 0:
            continue

        # ---- basic norm stats (useful sanity checks)
        norms = np.linalg.norm(X, axis=1)
        mean_norm, std_norm = float(norms.mean()), float(norms.std())

        # ---- distribution stats (unweighted v1)
        mu = X.mean(axis=0)
        centroid_norm = float(np.linalg.norm(mu))

        trace_cov, eff_rank = _weighted_cov_trace_and_effrank(X, w=None)

        rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "num_rows": num_rows,
                "feat_dim": int(d),
                "sampled": int(X.shape[0]),
                "mean_norm": round(mean_norm, 6),
                "std_norm": round(std_norm, 6),
                "centroid_norm": round(centroid_norm, 6),
                "trace_cov": round(float(trace_cov), 6),
                "eff_rank": round(float(eff_rank), 6),
            }
        )

        if save_centroids:
            centroids[(str(run_id), str(group_id))] = mu.astype(np.float32, copy=False)

    df = pd.DataFrame(rows).sort_values(["run_id", "group_id"]).reset_index(drop=True)

    out_csv = out_root / "distribution_metrics.csv"
    df.to_csv(out_csv, index=False)

    if save_centroids and centroids:
        # Save as a compact NPZ keyed by "run_id||group_id"
        keys = np.array([f"{k[0]}||{k[1]}" for k in centroids.keys()])
        mus = np.stack([centroids[k] for k in centroids.keys()], axis=0)
        np.savez_compressed(out_root / "centroids.npz", keys=keys, centroids=mus)

    return df
