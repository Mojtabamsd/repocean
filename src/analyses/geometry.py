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
    pair_samples: int = 5000,         # how many random within-group pairs for pairwise cosine stats
) -> pd.DataFrame:
    """
    Geometry + sphere-aware summary of feature space per group (run or profile/meta).

    Additional output:
        - centroid cosine similarity between groups
          (captures whether two groups live in similar directions/regions
           of embedding space, not just whether their internal structure is similar)

    Saves:
        - geometry/geometry_metrics.csv
        - geometry/centroid_cosine_matrix.csv
        - geometry/centroid_cosine_pairs.csv
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
    centroid_dirs = []   # store unit centroid vectors for pairwise comparison
    centroid_keys = []   # store identifiers aligned with centroid_dirs

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

        n = int(X.shape[0])
        d = int(d)

        # -----------------------
        # Basic norm sanity checks
        # -----------------------
        norms = np.linalg.norm(X, axis=1)
        mean_norm = float(norms.mean())
        std_norm = float(norms.std())

        # --------------------------------------------
        # Sphere-aware: centroid + cosine-to-centroid
        # --------------------------------------------
        mu = X.mean(axis=0)
        mu_norm = float(np.linalg.norm(mu))
        centroid_norm = mu_norm

        if mu_norm > 0:
            mu_hat = mu / mu_norm
            cos_to_centroid = X @ mu_hat  # if X rows are unit, this is cosine similarity
            cos_mean = float(cos_to_centroid.mean())
            cos_std = float(cos_to_centroid.std())
            cos_p10 = float(np.percentile(cos_to_centroid, 10))
            cos_p50 = float(np.percentile(cos_to_centroid, 50))
            cos_p90 = float(np.percentile(cos_to_centroid, 90))

            # save centroid direction for between-group comparison
            centroid_dirs.append(mu_hat.astype(np.float32))
            centroid_keys.append((run_id, group_id))
        else:
            mu_hat = None
            cos_mean = cos_std = cos_p10 = cos_p50 = cos_p90 = float("nan")

        # --------------------------------------------
        # Pairwise cosine similarity (random pairs)
        # --------------------------------------------
        if n >= 2 and pair_samples > 0:
            m = min(pair_samples, n * (n - 1) // 2)
            i = rng.integers(0, n, size=m, dtype=np.int64)
            j = rng.integers(0, n, size=m, dtype=np.int64)

            bad = (i == j)
            if np.any(bad):
                j[bad] = (j[bad] + 1) % n

            pair_cos = np.einsum("ij,ij->i", X[i], X[j])
            pair_cos_mean = float(pair_cos.mean())
            pair_cos_p10 = float(np.percentile(pair_cos, 10))
            pair_cos_p50 = float(np.percentile(pair_cos, 50))
            pair_cos_p90 = float(np.percentile(pair_cos, 90))
        else:
            pair_cos_mean = pair_cos_p10 = pair_cos_p50 = pair_cos_p90 = float("nan")

        # --------------------------------------------
        # Intrinsic dims + covariance summaries
        # --------------------------------------------
        dim90, dim95 = _intrinsic_dim_pca(X)

        if n >= 2:
            Xc = X - mu
            C = (Xc.T @ Xc) / max(n - 1, 1)
            evals = np.linalg.eigvalsh(C)
            evals = np.clip(evals, 0.0, np.inf)
            trace_cov = float(evals.sum())

            s1 = float(evals.sum())
            s2 = float((evals * evals).sum())
            eff_rank = float((s1 * s1) / s2) if (s1 > 0 and s2 > 0) else 0.0
        else:
            trace_cov = 0.0
            eff_rank = 0.0

        rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "num_rows": num_rows,
                "feat_dim": d,
                "sampled": n,

                "mean_norm": round(mean_norm, 6),
                "std_norm": round(std_norm, 6),

                "centroid_norm": round(centroid_norm, 6),
                "cos_mean": round(cos_mean, 6),
                "cos_std": round(cos_std, 6),
                "cos_p10": round(cos_p10, 6),
                "cos_p50": round(cos_p50, 6),
                "cos_p90": round(cos_p90, 6),

                "pair_cos_mean": round(pair_cos_mean, 6) if np.isfinite(pair_cos_mean) else pair_cos_mean,
                "pair_cos_p10": round(pair_cos_p10, 6) if np.isfinite(pair_cos_p10) else pair_cos_p10,
                "pair_cos_p50": round(pair_cos_p50, 6) if np.isfinite(pair_cos_p50) else pair_cos_p50,
                "pair_cos_p90": round(pair_cos_p90, 6) if np.isfinite(pair_cos_p90) else pair_cos_p90,

                "trace_cov": round(trace_cov, 6),
                "eff_rank": round(eff_rank, 6),
                "pca_dim_90": int(dim90),
                "pca_dim_95": int(dim95),
            }
        )

    df = pd.DataFrame(rows).sort_values(["run_id", "group_id"]).reset_index(drop=True)

    out_csv = out_root / "geometry_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # -------------------------------------------------
    # pairwise centroid cosine similarity by group
    # -------------------------------------------------
    if len(centroid_dirs) > 0:
        U = np.stack(centroid_dirs, axis=0)  # shape: (G, D)
        sim = U @ U.T                        # cosine similarity because rows are unit vectors

        labels = [f"{run_id}::{group_id}" for run_id, group_id in centroid_keys]
        sim_df = pd.DataFrame(sim, index=labels, columns=labels)
        sim_df.to_csv(out_root / "centroid_cosine_matrix.csv")

        pair_rows = []
        G = len(centroid_keys)
        for a in range(G):
            run_a, group_a = centroid_keys[a]
            for b in range(a + 1, G):
                run_b, group_b = centroid_keys[b]
                pair_rows.append(
                    {
                        "run_id_a": run_a,
                        "group_id_a": group_a,
                        "run_id_b": run_b,
                        "group_id_b": group_b,
                        "centroid_cosine": round(float(sim[a, b]), 6),
                    }
                )

        pd.DataFrame(pair_rows).to_csv(out_root / "centroid_cosine_pairs.csv", index=False)

    return df