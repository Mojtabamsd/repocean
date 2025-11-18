from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from src.index import build_group_index
from src.stream import (
    open_h5, get_h5_shapes,
    iter_feature_chunks, load_predictions_map,
)
from src.utils.paths import _safe_slug

# ------------------------------------------------------------
# Distance utilities
# ------------------------------------------------------------


def _l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # returns squared L2 distance matrix [len(a), len(b)]
    # (faster and stable than explicit pairwise loops)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return d2


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 1 - cosine similarity, numerically safe
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = a_norm @ b_norm.T
    # prevent tiny overshoots like 1.0000001
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def _pairwise_dist(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return _l2_dist(a, b)  # squared; OK for argmin/relative comparisons
    elif metric == "cosine":
        return _cos_dist(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

# ------------------------------------------------------------
# k-medoids (PAM-lite) on a sample
# ------------------------------------------------------------


def _init_medoids_kpp(X: np.ndarray, k: int, metric: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    medoids = np.empty(k, dtype=np.int64)
    medoids[0] = rng.integers(0, n)

    # distances to nearest chosen medoid
    D = _pairwise_dist(X, X[medoids[0:1]], metric).reshape(n)
    # numerical guards
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    D = np.clip(D, 0.0, None)

    for i in range(1, k):
        tot = D.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            # all distances are zero (or bad numerically): choose uniformly
            idx = rng.integers(0, n)
        else:
            probs = D / tot
            # ensure non-negative (extra belt & braces)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / probs.sum()  # renormalize
            idx = rng.choice(n, p=probs)

        medoids[i] = idx
        Di = _pairwise_dist(X, X[idx:idx+1], metric).reshape(n)
        Di = np.nan_to_num(Di, nan=0.0, posinf=0.0, neginf=0.0)
        Di = np.clip(Di, 0.0, None)
        D = np.minimum(D, Di)

    return np.unique(medoids)[:k]


def _assign(X: np.ndarray, medoids_idx: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (assignments, dists) where assignments[i] ∈ {0..k-1}."""
    M = X[medoids_idx]
    D = _pairwise_dist(X, M, metric)  # [N, k]
    a = np.argmin(D, axis=1)
    d = D[np.arange(X.shape[0]), a]
    return a, d


def _update_medoids(X: np.ndarray, labels: np.ndarray, k: int, metric: str) -> np.ndarray:
    """For each cluster, choose point that minimizes sum of distances to others (true medoid)."""
    medoids = np.empty(k, dtype=np.int64)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            medoids[c] = -1
            continue
        # compute pairwise distances within cluster efficiently
        Xc = X[idx]
        Dc = _pairwise_dist(Xc, Xc, metric)
        sums = Dc.sum(axis=1)
        medoids[c] = idx[np.argmin(sums)]
    return medoids


def k_medoids(X: np.ndarray, k: int, metric: str = "cosine", max_iter: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (medoid_indices, assignments). Works best when X is PCA-reduced.
    """
    k = min(k, max(1, X.shape[0]))
    med = _init_medoids_kpp(X, k, metric, seed)
    if med.size < k:
        # pad with random points (rare)
        rest = np.setdiff1d(np.arange(X.shape[0]), med, assume_unique=False)
        add = rest[: (k - med.size)]
        med = np.concatenate([med, add])

    prev_med = None
    for _ in range(max_iter):
        assign, _ = _assign(X, med, metric)
        new_med = _update_medoids(X, assign, k, metric)
        # fill empty clusters by stealing farthest points
        for c in range(k):
            if new_med[c] == -1:
                # pick a point far from its current medoid
                _, d = _assign(X, med, metric)
                new_med[c] = int(np.argmax(d))
        new_med = np.unique(new_med)
        if new_med.size < k:
            # pad if collapse
            pool = np.setdiff1d(np.arange(X.shape[0]), new_med, assume_unique=False)
            need = k - new_med.size
            new_med = np.concatenate([new_med, pool[:need]])

        if prev_med is not None and np.array_equal(new_med, prev_med):
            break
        prev_med = med
        med = new_med

    final_assign, _ = _assign(X, med, metric)
    return med, final_assign

# ------------------------------------------------------------
# Sampling by class (streamed)
# ------------------------------------------------------------


def _reservoir_append(reservoir_X: List[np.ndarray], reservoir_idx: List[int],
                      Xb: np.ndarray, idxb: np.ndarray, cap: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Append new block to reservoir up to capacity cap. If overflow, keep first 'cap' (FIFO simple cap).
    (We keep it simple; for strict uniform reservoir sampling use Vitter's algo if needed.)
    """
    if Xb.size == 0 or idxb.size == 0:
        return reservoir_X, reservoir_idx
    reservoir_X.append(Xb)
    reservoir_idx.extend(idxb.tolist())
    # trim if needed
    total = sum(x.shape[0] for x in reservoir_X)
    if total > cap:
        # concatenate then take head 'cap' to avoid growing lists indefinitely
        Xall = np.vstack(reservoir_X)
        idxall = np.asarray(reservoir_idx, dtype=np.int64)
        Xall = Xall[:cap]
        idxall = idxall[:cap]
        reservoir_X = [Xall]
        reservoir_idx = idxall.tolist()
    return reservoir_X, reservoir_idx


def _collect_by_class_streamed(h5_path: str, preds_csv: str,
                               max_per_class: int, batch_size: int) -> Dict[str, Dict[str, object]]:
    """
    Stream H5 in chunks, look up each image_name in predictions, and accumulate features
    per predicted class (top-1) up to max_per_class per class.
    Returns dict: {class_label: {"X": np.ndarray, "idx": np.ndarray, "names": List[str]}}
    """
    preds = load_predictions_map(preds_csv, cols=["Image Name", "Top-1 Predicted Label"])
    buckets: Dict[str, Dict[str, object]] = {}

    with open_h5(h5_path) as h5f:
        # chunked
        for blk in iter_feature_chunks(h5f, batch_size=batch_size):
            names = blk["image_names"]  # array[str]
            X = blk["features"]
            # map labels
            sub = preds.reindex(names)
            labels = sub["pred1_label"].fillna("unknown").astype(str).values

            # group rows by label
            for lab in np.unique(labels):
                mask = (labels == lab)
                if not mask.any():
                    continue
                Xb = X[mask]
                # global row indices in H5: idx_start..idx_end (exclusive) with mask
                idxb = np.arange(blk["idx_start"], blk["idx_end"], dtype=np.int64)[mask]
                namesb = names[mask].tolist()

                if lab not in buckets:
                    buckets[lab] = {"X": [], "idx": [], "names": []}
                # append with simple capacity cap
                buckets[lab]["X"], buckets[lab]["idx"] = _reservoir_append(
                    buckets[lab]["X"], buckets[lab]["idx"], Xb, idxb, cap=max_per_class
                )
                # keep names in sync with idx capping (approximate; re-derive after concat)
                buckets[lab]["names"].extend(namesb)

    # finalize concat + truncate strictly to cap
    out: Dict[str, Dict[str, object]] = {}
    for lab, d in buckets.items():
        Xlist: List[np.ndarray] = d["X"]  # type: ignore
        if not Xlist:
            continue
        Xall = np.vstack(Xlist)
        idxall = np.asarray(d["idx"], dtype=np.int64)
        names_all = np.asarray(d["names"][: Xall.shape[0]]).astype(str)  # approximate sync
        out[lab] = {"X": Xall, "idx": idxall[: Xall.shape[0]], "names": names_all}
    return out

# ------------------------------------------------------------
# PCA fit on concatenated sample (per run)
# ------------------------------------------------------------


def _fit_project_pca_per_run(class_buckets: Dict[str, Dict[str, object]],
                             pca_dim: int) -> Dict[str, Dict[str, object]]:
    """Fit PCA on the concatenated sample across all classes in a run, project each class."""
    # concatenate to fit basis
    concat = []
    for v in class_buckets.values():
        concat.append(v["X"])
    Xbig = np.vstack(concat) if concat else None
    if Xbig is None or Xbig.size == 0:
        return class_buckets
    if Xbig.shape[1] <= pca_dim:
        # nothing to do; still center for consistency
        mu = Xbig.mean(axis=0, keepdims=True)
        for lab in class_buckets:
            class_buckets[lab]["Xp"] = class_buckets[lab]["X"] - mu
        return class_buckets

    ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
    if Xbig.shape[0] > 4096:
        for j in range(0, Xbig.shape[0], 4096):
            ipca.partial_fit(Xbig[j : j + 4096])
    else:
        ipca.partial_fit(Xbig)

    for lab in class_buckets:
        class_buckets[lab]["Xp"] = ipca.transform(class_buckets[lab]["X"])
    return class_buckets

# ------------------------------------------------------------
# Coverage metrics
# ------------------------------------------------------------


def _coverage_metrics(assignments: np.ndarray, k: int, distances: Optional[np.ndarray]=None) -> Dict[str, float]:
    """Compute entropy, normalized entropy, gini, and mean within-medoid distance (if provided)."""
    N = assignments.size
    counts = np.bincount(assignments, minlength=k).astype(float)
    p = counts / max(N, 1)
    # Shannon entropy
    eps = 1e-12
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(k) + eps)  # in [0,1]
    # Gini (probability of mismatch)
    gini = 1.0 - np.sum(p * p)
    metrics = {
        "count": float(N),
        "entropy": float(H),
        "entropy_norm": float(H_norm),
        "gini": float(gini),
    }
    if distances is not None:
        metrics["mean_within"] = float(np.mean(distances))
        metrics["median_within"] = float(np.median(distances))
    return metrics


def _collect_by_class_for_group(
    h5_path: str,
    preds_csv: str,
    indices: Optional[np.ndarray],
    max_per_class: int = 4000,
    batch_size: int = 4096,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Stream features from H5 and collect up to max_per_class samples per predicted class.

    If `indices` is None:
        use all rows in the H5 file (whole run).
    If `indices` is an array of global row indices:
        only keep those rows (e.g. for a given sample_id group).

    Returns:
      buckets: dict[label] -> {
          "X":   (N_c, D) features,
          "names": (N_c,) image names,
          "idx": (N_c,) global row indices
      }
    """
    # load preds once
    preds = load_predictions_map(
        preds_csv,
        cols=["Image Name", "Top-1 Predicted Label"],
    )

    allowed: Optional[set] = None
    if indices is not None:
        allowed = set(int(i) for i in np.asarray(indices, dtype=np.int64))

    buckets: Dict[str, Dict[str, List]] = {}

    with open_h5(h5_path) as h5f:
        global_offset = 0
        for blk in iter_feature_chunks(h5f, batch_size=batch_size):
            X = blk["features"]
            names = blk["image_names"]
            n = X.shape[0]

            sub = preds.reindex(names)
            labels = sub["pred1_label"].fillna("unknown").astype(str).values

            for j in range(n):
                gidx = global_offset + j
                if allowed is not None and gidx not in allowed:
                    continue

                cls = labels[j]
                if cls not in buckets:
                    buckets[cls] = {"X": [], "names": [], "idx": []}

                # class-level cap
                if len(buckets[cls]["X"]) >= max_per_class:
                    continue

                buckets[cls]["X"].append(X[j])
                buckets[cls]["names"].append(str(names[j]))
                buckets[cls]["idx"].append(gidx)

            global_offset += n

    # convert lists -> arrays
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cls, d in buckets.items():
        if not d["X"]:
            continue
        Xc = np.stack(d["X"], axis=0)
        names = np.asarray(d["names"])
        idxs = np.asarray(d["idx"], dtype=np.int64)
        out[cls] = {"X": Xc, "names": names, "idx": idxs}

    return out


def _fit_project_pca_for_buckets(
    buckets: Dict[str, Dict[str, np.ndarray]],
    pca_dim: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[IncrementalPCA]]:
    """
    Fit a PCA on the concatenated sample from all class-buckets
    and project each bucket into that PCA space.

    Returns updated buckets where each has 'Xp' (projected features)
    and the PCA object (or None if no reduction was applied).
    """
    if not buckets:
        return buckets, None

    Xs = [d["X"] for d in buckets.values()]
    X_all = np.vstack(Xs)

    if X_all.shape[1] > pca_dim:
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
        # single pass is fine; data is already a sample
        if X_all.shape[0] > 4096:
            for j in range(0, X_all.shape[0], 4096):
                ipca.partial_fit(X_all[j : j + 4096])
        else:
            ipca.partial_fit(X_all)

        for cls, d in buckets.items():
            buckets[cls]["Xp"] = ipca.transform(d["X"])
    else:
        ipca = None
        mean = X_all.mean(axis=0, keepdims=True)
        for cls, d in buckets.items():
            buckets[cls]["Xp"] = d["X"] - mean

    return buckets, ipca


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def run_prototypes(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",          # 'run' or 'meta'
    group_col: str = "sample_id",     # used when group_mode == 'meta'
    k: int = 10,                      # medoids per group
    pca_dim: int = 50,                # reduce before clustering
    max_per_class: int = 4000,        # cap memory per class
    batch_size: int = 4096,           # H5 chunk size for streaming
    metric: str = "cosine",           # 'cosine' or 'euclidean'
    min_points_for_group: int = 50,   # skip tiny groups
    max_iter: int = 20,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Discover medoid prototypes and coverage metrics.

    group_mode = 'run':
        - One logical group per run (all samples in that run).
    group_mode = 'meta':
        - One logical group per (run_id, group_id), where group_id is from metadata
          column `group_col` (e.g. sample_id).

    For each logical group, we compute TWO kinds of prototypes:

      1) Overall prototypes for the group (all classes mixed).
      2) Per-class prototypes inside that group.

    Outputs (for each run or (run, group_id)):

      <out_dir>/prototypes/<run_id>/[<group_id>/]prototypes_overall.json
      <out_dir>/prototypes/<run_id>/[<group_id>/]coverage_overall.csv

      <out_dir>/prototypes/<run_id>/[<group_id>/]prototypes_per_class.json
      <out_dir>/prototypes/<run_id>/[<group_id>/]coverage_per_class.csv
    """
    assert group_mode in {"run", "meta"}
    rng = np.random.default_rng(seed)

    out_root = Path(out_dir) / "prototypes"
    out_root.mkdir(parents=True, exist_ok=True)

    # Build group index
    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
    )
    if groups.empty:
        raise RuntimeError(f"No groups found under: {parent_dir} (group_mode={group_mode})")

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        group_id = g["group_id"] if group_mode == "meta" else run_id

        features_path = g["features"]
        preds_path = g["preds"]

        # indices: None => whole run, else a subset of row indices (for meta group)
        indices = None
        if group_mode == "meta":
            idx_arr = g.get("indices", None)
            if idx_arr is None or len(idx_arr) == 0:
                continue
            indices = np.asarray(idx_arr, dtype=np.int64)

        # output directory for this logical group
        if group_mode == "meta":
            safe_group = _safe_slug(str(group_id))
            group_out = out_root / run_id / safe_group
        else:
            group_out = out_root / run_id

        group_out.mkdir(parents=True, exist_ok=True)

        # --- 1) Collect per-class buckets for this group ---
        buckets = _collect_by_class_for_group(
            h5_path=features_path,
            preds_csv=preds_path,
            indices=indices,
            max_per_class=max_per_class,
            batch_size=batch_size,
        )
        if not buckets:
            continue

        # --- 2) Fit PCA on all buckets and project them ---
        buckets, ipca = _fit_project_pca_for_buckets(buckets, pca_dim=pca_dim)

        # =====================================================================
        # A) PER-CLASS PROTOTYPES (like old mode='per_class', but per group)
        # =====================================================================
        proto_records_cls: List[Dict[str, object]] = []
        cov_records_cls: List[Dict[str, object]] = []

        for cls_label in sorted(buckets.keys()):
            X = np.asarray(buckets[cls_label]["Xp"])
            names = np.asarray(buckets[cls_label]["names"])
            idxs = np.asarray(buckets[cls_label]["idx"], dtype=np.int64)

            if X.shape[0] < max(min_points_for_group, k):
                continue  # skip tiny group

            med_idx_local, assign = k_medoids(
                X, k=k, metric=metric, max_iter=max_iter, seed=seed
            )

            med_global_idx = idxs[med_idx_local].tolist()
            med_names = names[med_idx_local].tolist()

            # within distances for coverage
            M = X[med_idx_local]
            D = _pairwise_dist(X, M, metric)
            within = D[np.arange(X.shape[0]), assign]

            cov = _coverage_metrics(assign, k=len(med_idx_local), distances=within)
            counts = np.bincount(assign, minlength=len(med_idx_local)).astype(int)
            for i, c in enumerate(counts):
                cov[f"medoid_{i}_count"] = int(c)

            cov.update({
                "run_id": run_id,
                "group_id": group_id,
                "proto_group": str(cls_label),  # class group
                "n_group": int(X.shape[0]),
                "k": int(len(med_idx_local)),
                "metric": metric,
                "pca_dim": int(pca_dim),
            })
            cov_records_cls.append(cov)

            for i, (gi, nm) in enumerate(zip(med_global_idx, med_names)):
                proto_records_cls.append({
                    "run_id": run_id,
                    "group_id": group_id,
                    "proto_group": str(cls_label),
                    "medoid_rank": int(i),
                    "medoid_global_idx": int(gi),
                    "medoid_name": str(nm),
                })

        # =====================================================================
        # B) OVERALL PROTOTYPES PER GROUP (like old mode='per_run', but per group)
        # =====================================================================
        # Merge all classes into one big group for this (run, group_id)
        Xs_all, names_all, idxs_all = [], [], []
        for d in buckets.values():
            Xs_all.append(d["Xp"])
            names_all.extend(d["names"].tolist())
            idxs_all.extend(d["idx"].tolist())

        X_all = np.vstack(Xs_all)
        names_all = np.asarray(names_all)
        idxs_all = np.asarray(idxs_all, dtype=np.int64)

        proto_records_all: List[Dict[str, object]] = []
        cov_records_all: List[Dict[str, object]] = []

        if X_all.shape[0] >= max(min_points_for_group, k):
            med_idx_local, assign = k_medoids(
                X_all, k=k, metric=metric, max_iter=max_iter, seed=seed
            )

            med_global_idx = idxs_all[med_idx_local].tolist()
            med_names = names_all[med_idx_local].tolist()

            M = X_all[med_idx_local]
            D = _pairwise_dist(X_all, M, metric)
            within = D[np.arange(X_all.shape[0]), assign]
            cov = _coverage_metrics(assign, k=len(med_idx_local), distances=within)
            cov.update({
                "run_id": run_id,
                "group_id": group_id,
                "proto_group": "__ALL__",  # overall
                "n_group": int(X_all.shape[0]),
                "k": int(len(med_idx_local)),
                "metric": metric,
                "pca_dim": int(pca_dim),
            })
            cov_records_all.append(cov)

            for i, (gi, nm) in enumerate(zip(med_global_idx, med_names)):
                proto_records_all.append({
                    "run_id": run_id,
                    "group_id": group_id,
                    "proto_group": "__ALL__",
                    "medoid_rank": int(i),
                    "medoid_global_idx": int(gi),
                    "medoid_name": str(nm),
                })

        # =====================================================================
        # C) Save outputs for this (run, group_id)
        # =====================================================================
        # per-class outputs
        if proto_records_cls:
            with open(group_out / "prototypes_per_class.json", "w", encoding="utf-8") as f:
                json.dump(proto_records_cls, f, indent=2)
        if cov_records_cls:
            pd.DataFrame(cov_records_cls).to_csv(group_out / "coverage_per_class.csv", index=False)

        # overall outputs
        if proto_records_all:
            with open(group_out / "prototypes_overall.json", "w", encoding="utf-8") as f:
                json.dump(proto_records_all, f, indent=2)
        if cov_records_all:
            pd.DataFrame(cov_records_all).to_csv(group_out / "coverage_overall.csv", index=False)

    return {"out_dir": str(out_root)}
