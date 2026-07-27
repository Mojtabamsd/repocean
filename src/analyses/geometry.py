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
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform


def _intrinsic_dim_pca(X: np.ndarray, thresholds=(0.90, 0.95)) -> Tuple[int, int]:
    ncomp = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="auto", random_state=42)
    pca.fit(X)
    csum = np.cumsum(pca.explained_variance_ratio_)
    idx90 = int(np.searchsorted(csum, thresholds[0]) + 1)
    idx95 = int(np.searchsorted(csum, thresholds[1]) + 1)
    return idx90, idx95


def _shannon_from_labels(labels: np.ndarray) -> Tuple[float, float, int]:
    labels = np.asarray(labels, dtype=object)
    labels = labels[pd.notna(labels)]
    if labels.size == 0:
        return float("nan"), float("nan"), 0

    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]

    H = float(-(p * np.log(p)).sum())
    exp_H = float(np.exp(H))
    return H, exp_H, int(len(counts))


def _counts_from_labels(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=object)
    labels = labels[pd.notna(labels)]
    if labels.size == 0:
        return {}

    vals, counts = np.unique(labels, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def _count_rows_to_long_df(
    count_rows: list[dict],
    label_source: str,
) -> pd.DataFrame:
    rows = []
    for r in count_rows:
        run_id = r["run_id"]
        group_id = r["group_id"]
        label_counts = r["label_counts"]

        total = int(sum(label_counts.values())) if label_counts else 0
        for label_name, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append(
                {
                    "run_id": run_id,
                    "group_id": group_id,
                    "label_source": label_source,
                    "label_name": str(label_name),
                    "count": int(count),
                    "proportion": (float(count) / total) if total > 0 else np.nan,
                    "total_in_group": total,
                }
            )
    return pd.DataFrame(rows)


def _count_rows_to_topk_summary(
    count_rows: list[dict],
    label_source: str,
    top_k: int = 3,
) -> pd.DataFrame:
    rows = []
    for r in count_rows:
        run_id = r["run_id"]
        group_id = r["group_id"]
        label_counts = r["label_counts"]

        items = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        total = int(sum(label_counts.values())) if label_counts else 0

        row = {
            "run_id": run_id,
            "group_id": group_id,
            "label_source": label_source,
            "n_labels_present": int(len(items)),
            "total_in_group": total,
        }

        for k in range(top_k):
            if k < len(items):
                lab, cnt = items[k]
                row[f"top{k+1}_label"] = str(lab)
                row[f"top{k+1}_count"] = int(cnt)
                row[f"top{k+1}_prop"] = (float(cnt) / total) if total > 0 else np.nan
            else:
                row[f"top{k+1}_label"] = np.nan
                row[f"top{k+1}_count"] = np.nan
                row[f"top{k+1}_prop"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _compute_nmds_from_count_rows(
    count_rows: list[dict],
    seed: int,
    prefix: str,
    out_root: Path | None = None,
) -> pd.DataFrame:
    """
    count_rows:
        [
            {"run_id": ..., "group_id": ..., "label_counts": {...}},
            ...
        ]

    returns DataFrame with:
        run_id, group_id, {prefix}_nmds1, {prefix}_nmds2
    """
    if len(count_rows) < 2:
        out = pd.DataFrame(
            [{"run_id": r["run_id"], "group_id": r["group_id"]} for r in count_rows]
        )
        out[f"{prefix}_nmds1"] = np.nan
        out[f"{prefix}_nmds2"] = np.nan
        return out

    counts_df = pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "group_id": r["group_id"],
                **r["label_counts"],
            }
            for r in count_rows
        ]
    ).fillna(0)

    meta_cols = ["run_id", "group_id"]
    label_cols = [c for c in counts_df.columns if c not in meta_cols]

    if len(label_cols) == 0:
        out = counts_df[meta_cols].copy()
        out[f"{prefix}_nmds1"] = np.nan
        out[f"{prefix}_nmds2"] = np.nan
        return out

    X_counts = counts_df[label_cols].to_numpy(dtype=float)
    D = squareform(pdist(X_counts, metric="braycurtis"))

    nmds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=10,
        max_iter=1000,
        normalized_stress="auto",
    )
    Y = nmds.fit_transform(D)

    out = counts_df[meta_cols].copy()
    out[f"{prefix}_nmds1"] = Y[:, 0]
    out[f"{prefix}_nmds2"] = Y[:, 1]

    if out_root is not None:
        counts_df.to_csv(out_root / f"{prefix}_deployment_label_counts.csv", index=False)
        pd.DataFrame(
            D,
            index=[f"{r}::{g}" for r, g in zip(counts_df["run_id"], counts_df["group_id"])],
            columns=[f"{r}::{g}" for r, g in zip(counts_df["run_id"], counts_df["group_id"])],
        ).to_csv(out_root / f"{prefix}_deployment_label_braycurtis_matrix.csv")

    return out


def _find_predictions_csv(features_path: str | Path) -> Path:
    """
    Assumes predictions_with_top3_scores.csv lives next to the feature file,
    or one level above if needed.
    """
    features_path = Path(features_path)

    candidates = [
        features_path.parent / "predictions_with_top3_scores.csv",
        features_path.parent.parent / "predictions_with_top3_scores.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find predictions_with_top3_scores.csv near {features_path}"
    )


def _compute_geometry_metrics(X: np.ndarray, pair_samples: int, rng: np.random.Generator) -> dict:
    """
    Compute the sphere-aware geometry metrics for a single (fixed-size) sample X.
    Pulled out into a helper so it can be called once per bootstrap trial and
    averaged, without duplicating logic.
    """
    n = int(X.shape[0])

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
        cos_to_centroid = X @ mu_hat
        cos_mean = float(cos_to_centroid.mean())
        cos_std = float(cos_to_centroid.std())
        cos_p10 = float(np.percentile(cos_to_centroid, 10))
        cos_p50 = float(np.percentile(cos_to_centroid, 50))
        cos_p90 = float(np.percentile(cos_to_centroid, 90))
    else:
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

    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "centroid_norm": centroid_norm,
        "cos_mean": cos_mean,
        "cos_std": cos_std,
        "cos_p10": cos_p10,
        "cos_p50": cos_p50,
        "cos_p90": cos_p90,
        "pair_cos_mean": pair_cos_mean,
        "pair_cos_p10": pair_cos_p10,
        "pair_cos_p50": pair_cos_p50,
        "pair_cos_p90": pair_cos_p90,
        "trace_cov": trace_cov,
        "eff_rank": eff_rank,
        "pca_dim_90": dim90,
        "pca_dim_95": dim95,
    }


def _average_trial_metrics(trial_metrics: list[dict]) -> dict:
    """
    Average numeric geometry metrics across bootstrap trials.
    Integer-ish fields (pca_dim_90/95) are averaged then rounded to int,
    matching their original "count of components" meaning.
    """
    keys = trial_metrics[0].keys()
    avg = {}
    for key in keys:
        vals = [m[key] for m in trial_metrics if np.isfinite(m[key])]
        if not vals:
            avg[key] = float("nan")
        else:
            avg[key] = float(np.mean(vals))

    for int_key in ("pca_dim_90", "pca_dim_95"):
        if np.isfinite(avg[int_key]):
            avg[int_key] = int(round(avg[int_key]))

    return avg


def run_geometry_summary(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",          # "run" | "meta"
    group_col: str = "sample_id",     # used when group_mode == "meta", sample_id or object_depth_min
    group_col_source: str = "meta",  # "pred" | "meta"
    depth_bin_size: float | int | None = None,
    profile_col: str = "sample_id",
    sample_per_group: int = 2000,
    n_bootstrap: int = 20,            # number of subsample trials to average when num_rows > sample_per_group
    seed: int = 42,
    pair_samples: int = 5000,
    pred_label_col: str = "Top-1 Predicted Label",
    taxonomist_label_col: str = "object_annotation_category",
    # exclude_labels: list[str] = {"detritus"}  # list[str] | set[str] | None = None, or {"detritus", "artefact"}
    exclude_labels: None = None  # list[str] | set[str] | None = None, or {"detritus", "artefact"}
) -> pd.DataFrame:
    """
    Geometry + sphere-aware summary of feature space per group (run or profile/meta).

    Sample-size handling:
        - If a group has num_rows <= sample_per_group: use all rows, single pass
          (n_trials = 1).
        - If a group has num_rows > sample_per_group: draw `n_bootstrap`
          independent random subsamples of size `sample_per_group`, compute
          geometry metrics on each, and average them.

        This means every group's geometry metrics are computed on at most
        `sample_per_group` points, which removes sample-size as a confound
        for groups above that threshold, while groups below it just use what
        they have (documented via `sampled` / `n_bootstrap_trials` columns in
        the output so you can filter/inspect afterwards).

    Additional outputs:
        - centroid cosine similarity between groups
        - prediction-label Shannon entropy from predictions_with_top3_scores.csv

    Index alignment:
        `build_group_index` returns two PARALLEL arrays per group:
          - "indices"      -> row positions in the H5 feature file
          - "pred_indices" -> row positions in the predictions CSV
        for the SAME set of images, matched by image name. They are NOT
        assumed to be numerically equal, since the CSV and H5 may not have
        identical row order or identical row counts. Always use "indices"
        to read from the H5 file and "pred_indices" to read from pred_df.
    """
    rng = np.random.default_rng(seed)
    exclude_labels = set(exclude_labels or [])

    out_root = Path(out_dir) / "geometry"
    out_root.mkdir(parents=True, exist_ok=True)

    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
        group_col_source=group_col_source,
        depth_bin_size=depth_bin_size,
        profile_col=profile_col,
    )
    if groups.empty:
        raise RuntimeError(
            f"No groups found under {parent_dir} (group_mode={group_mode}, group_col={group_col})"
        )

    rows = []
    centroid_dirs = []
    centroid_keys = []
    pred_count_rows = []
    tax_count_rows = []

    # cache so each run's predictions CSV is only loaded once
    pred_cache: dict[str, pd.DataFrame] = {}

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        features_path = g["features"]
        group_id = run_id if group_mode == "run" else g["group_id"]

        # -----------------------------------------------------
        # Aligned index pair for this group: H5 positions + CSV positions
        # -----------------------------------------------------
        idx_all = np.asarray(g["indices"], dtype=np.int64)
        pred_idx_all = np.asarray(g["pred_indices"], dtype=np.int64)

        if idx_all.size == 0:
            continue

        with open_h5(features_path) as h5f:
            n_total, d = get_h5_shapes(h5f)

        # -----------------------
        # Load / cache predictions CSV for this run
        # -----------------------
        if run_id not in pred_cache:
            pred_df = pd.read_csv(g["preds"])
            if pred_label_col not in pred_df.columns:
                raise KeyError(f"Column '{pred_label_col}' not found in {g['preds']}")
            pred_cache[run_id] = pred_df

        pred_df = pred_cache[run_id]
        has_taxonomist_col = taxonomist_label_col in pred_df.columns

        # No manual bounds-check needed here: pred_idx_all was built by
        # build_group_index via name-matching against this exact CSV, so
        # every value is guaranteed to be a valid row position in pred_df.

        # -----------------------
        # FILTER INDICES HERE
        # -----------------------
        pred_labels_full = pred_df.iloc[pred_idx_all][pred_label_col].to_numpy()

        if has_taxonomist_col:
            tax_labels_full = pred_df.iloc[pred_idx_all][taxonomist_label_col].to_numpy()
        else:
            tax_labels_full = None

        if exclude_labels:
            keep_mask = ~pd.Series(pred_labels_full).isin(exclude_labels).to_numpy()
            idx_all = idx_all[keep_mask]
            pred_idx_all = pred_idx_all[keep_mask]
            pred_labels_full = pred_labels_full[keep_mask]
            if tax_labels_full is not None:
                tax_labels_full = tax_labels_full[keep_mask]

        num_rows = int(idx_all.size)
        if num_rows == 0:
            continue

        pred_count_rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "label_counts": _counts_from_labels(pred_labels_full),
            }
        )

        pred_shannon, pred_exp_shannon, pred_num_classes_present = _shannon_from_labels(pred_labels_full)

        if tax_labels_full is not None:
            tax_count_rows.append(
                {
                    "run_id": run_id,
                    "group_id": group_id,
                    "label_counts": _counts_from_labels(tax_labels_full),
                }
            )
            tax_shannon, tax_exp_shannon, tax_num_classes_present = _shannon_from_labels(tax_labels_full)
        else:
            tax_shannon = np.nan
            tax_exp_shannon = np.nan
            tax_num_classes_present = np.nan

        # -----------------------------------------------------
        # Sample filtered rows for geometry (fixed-size + bootstrap)
        # -----------------------------------------------------
        k = min(sample_per_group, num_rows)
        if k <= 0:
            continue

        n_trials = n_bootstrap if num_rows > sample_per_group else 1

        trial_metrics = []
        last_mu_hat = None  # centroid direction used for the cross-group cosine matrix

        for _t in range(n_trials):
            rel_idx = sample_indices_uniform(num_rows, k, rng)
            sel_idx = np.sort(idx_all[rel_idx])   # H5 indices only — correct space for reading features

            with open_h5(features_path) as h5f:
                part = read_rows_by_indices(h5f, sel_idx)
                X = part["features"]

            if X.size == 0:
                continue

            m = _compute_geometry_metrics(X, pair_samples=pair_samples, rng=rng)
            trial_metrics.append(m)

            # centroid direction from the last trial (or the only trial when n_trials == 1)
            mu = X.mean(axis=0)
            mu_norm = np.linalg.norm(mu)
            if mu_norm > 0:
                last_mu_hat = (mu / mu_norm).astype(np.float32)

        if not trial_metrics:
            continue

        geo = _average_trial_metrics(trial_metrics)
        n = k  # effective sample size used per trial

        if last_mu_hat is not None:
            centroid_dirs.append(last_mu_hat)
            centroid_keys.append((run_id, group_id))

        rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "num_rows": num_rows,
                "feat_dim": d,
                "sampled": n,
                "n_bootstrap_trials": len(trial_metrics),

                "pred_num_classes_present": int(pred_num_classes_present),
                "shannon": round(pred_shannon, 6) if np.isfinite(pred_shannon) else pred_shannon,
                "exp_shannon": round(pred_exp_shannon, 6) if np.isfinite(pred_exp_shannon) else pred_exp_shannon,

                "tax_num_classes_present": (
                    int(tax_num_classes_present) if pd.notna(tax_num_classes_present) else np.nan
                ),
                "tax_shannon": round(tax_shannon, 6) if np.isfinite(tax_shannon) else tax_shannon,
                "tax_exp_shannon": round(tax_exp_shannon, 6) if np.isfinite(tax_exp_shannon) else tax_exp_shannon,

                "mean_norm": round(geo["mean_norm"], 6),
                "std_norm": round(geo["std_norm"], 6),

                "centroid_norm": round(geo["centroid_norm"], 6),
                "cos_mean": round(geo["cos_mean"], 6),
                "cos_std": round(geo["cos_std"], 6),
                "cos_p10": round(geo["cos_p10"], 6),
                "cos_p50": round(geo["cos_p50"], 6),
                "cos_p90": round(geo["cos_p90"], 6),

                "pair_cos_mean": round(geo["pair_cos_mean"], 6) if np.isfinite(geo["pair_cos_mean"]) else geo["pair_cos_mean"],
                "pair_cos_p10": round(geo["pair_cos_p10"], 6) if np.isfinite(geo["pair_cos_p10"]) else geo["pair_cos_p10"],
                "pair_cos_p50": round(geo["pair_cos_p50"], 6) if np.isfinite(geo["pair_cos_p50"]) else geo["pair_cos_p50"],
                "pair_cos_p90": round(geo["pair_cos_p90"], 6) if np.isfinite(geo["pair_cos_p90"]) else geo["pair_cos_p90"],

                "trace_cov": round(geo["trace_cov"], 6),
                "eff_rank": round(geo["eff_rank"], 6),
                "pca_dim_90": geo["pca_dim_90"],
                "pca_dim_95": geo["pca_dim_95"],
            }
        )

    # -------------------------------------------------
    # NMDS from deployment label-count vectors
    # -------------------------------------------------
    df = pd.DataFrame(rows).sort_values(["run_id", "group_id"]).reset_index(drop=True)

    # -------------------------------------------------
    # prediction-label NMDS
    # -------------------------------------------------
    pred_nmds_df = _compute_nmds_from_count_rows(
        pred_count_rows,
        seed=seed,
        prefix="pred",
        out_root=out_root,
    )
    pred_nmds_df = pred_nmds_df.rename(
        columns={
            "pred_nmds1": "nmds1",
            "pred_nmds2": "nmds2",
        }
    )
    df = df.merge(pred_nmds_df, on=["run_id", "group_id"], how="left")

    # -------------------------------------------------
    # taxonomist-label NMDS
    # -------------------------------------------------
    if len(tax_count_rows) > 0:
        tax_nmds_df = _compute_nmds_from_count_rows(
            tax_count_rows,
            seed=seed,
            prefix="tax",
            out_root=out_root,
        )
        df = df.merge(tax_nmds_df, on=["run_id", "group_id"], how="left")
    else:
        df["tax_nmds1"] = np.nan
        df["tax_nmds2"] = np.nan

    out_csv = out_root / "geometry_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # -------------------------------------------------
    # pairwise centroid cosine similarity by group
    # -------------------------------------------------
    if len(centroid_dirs) > 0:
        U = np.stack(centroid_dirs, axis=0)
        sim = U @ U.T

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

    # -------------------------------------------------
    # tidy label-count tables for inspection / plotting
    # -------------------------------------------------
    pred_long_df = _count_rows_to_long_df(pred_count_rows, label_source="pred")
    pred_long_df.to_csv(out_root / "pred_label_counts_long.csv", index=False)

    if len(tax_count_rows) > 0:
        tax_long_df = _count_rows_to_long_df(tax_count_rows, label_source="tax")
        tax_long_df.to_csv(out_root / "tax_label_counts_long.csv", index=False)
    else:
        pd.DataFrame(
            columns=["run_id", "group_id", "label_source", "label_name", "count", "proportion", "total_in_group"]
        ).to_csv(out_root / "tax_label_counts_long.csv", index=False)

    _count_rows_to_topk_summary(pred_count_rows, "pred", top_k=3).to_csv(
        out_root / "pred_group_label_summary.csv", index=False
    )

    if len(tax_count_rows) > 0:
        _count_rows_to_topk_summary(tax_count_rows, "tax", top_k=3).to_csv(
            out_root / "tax_group_label_summary.csv", index=False
        )

    return df