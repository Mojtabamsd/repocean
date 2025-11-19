from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.decomposition import PCA

from src.index import build_group_index
from src.stream import (
    open_h5, get_h5_shapes, sample_indices_uniform, read_rows_by_indices
)


def _intrinsic_dim_pca(X: np.ndarray, thresholds=(0.90, 0.95)) -> Tuple[int, int]:
    ncomp = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="auto", random_state=42)
    pca.fit(X)
    csum = np.cumsum(pca.explained_variance_ratio_)
    idx90 = int(np.searchsorted(csum, thresholds[0]) + 1)
    idx95 = int(np.searchsorted(csum, thresholds[1]) + 1)
    return idx90, idx95


def run_geometry_summary(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",          # "run" | "meta"
    group_col: str = "sample_id",     # used when group_mode == "meta"
    sample_per_group: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Geometry summary of feature space per group (run or profile/meta).

    group_mode = "run":
        - Each run is treated as one group (original behaviour).
        - group_id column is set equal to run_id.

    group_mode = "meta":
        - Use build_group_index(..., mode="meta", group_col=group_col)
        - Each (run_id, group_id) corresponds to a subset of indices
          (e.g. one sample_id / profile_id within that run).

    For each group we compute:
        - num_rows     : total number of feature rows in that group
        - feat_dim     : feature dimensionality
        - sampled      : number of rows actually used for metrics
        - mean_norm    : mean L2 norm of features
        - std_norm     : std of L2 norm
        - pca_dim_90   : #PCs to reach 90% variance
        - pca_dim_95   : #PCs to reach 95% variance
    """
    rng = np.random.default_rng(seed)

    out_root = Path(out_dir) / "geometry"
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

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        features_path = g["features"]

        # Decide group_id + which indices belong to this group
        if group_mode == "run":
            group_id = run_id
            with open_h5(features_path) as h5f:
                n_total, d = get_h5_shapes(h5f)
                if n_total == 0:
                    continue
                # Whole run
                idx_all = np.arange(n_total, dtype=np.int64)
        else:
            # group_mode == "meta" → one profile / sample_id
            group_id = g["group_id"]
            idx_group = g.get("indices", None)
            if idx_group is None:
                continue
            idx_all = np.asarray(idx_group, dtype=np.int64)
            if idx_all.size == 0:
                continue

            # Need feature dim from file
            with open_h5(features_path) as h5f:
                n_total, d = get_h5_shapes(h5f)

        num_rows = int(idx_all.size)
        if num_rows == 0:
            continue

        # Sample within this group
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

        norms = np.linalg.norm(X, axis=1)
        mean_norm, std_norm = float(norms.mean()), float(norms.std())
        dim90, dim95 = _intrinsic_dim_pca(X)

        rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "num_rows": num_rows,
                "feat_dim": d,
                "sampled": X.shape[0],
                "mean_norm": round(mean_norm, 6),
                "std_norm": round(std_norm, 6),
                "pca_dim_90": int(dim90),
                "pca_dim_95": int(dim95),
            }
        )

    df = pd.DataFrame(rows).sort_values(["run_id", "group_id"]).reset_index(drop=True)
    out_csv = out_root / "geometry_metrics.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

