from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from src.index import build_run_index
from src.stream import (
    open_h5, get_h5_shapes,
    iter_feature_chunks, load_predictions_map,
)

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

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def run_prototypes(
    parent_dir: str,
    out_dir: str,
    mode: str = "per_class",              # 'per_class' or 'per_run'
    k: int = 10,                          # medoids per group
    pca_dim: int = 50,                    # reduce before clustering
    max_per_class: int = 4000,            # cap memory per class (stream sampled)
    batch_size: int = 4096,               # H5 chunk size for streaming
    metric: str = "cosine",               # 'cosine' or 'euclidean'
    min_points_for_group: int = 50,       # skip tiny groups
    max_iter: int = 20,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Discover medoid prototypes and coverage metrics.
    - mode='per_class': find k medoids per predicted top-1 label
    - mode='per_run'  : treat whole run as one group and find k medoids

    Writes (per run):
      - prototypes/prototypes.json : list of {group, medoid_name, medoid_global_idx, medoid_local_row}
      - prototypes/coverage.csv    : one row per group with entropy/gini/mean distances and per-medoid sizes
    """
    assert mode in {"per_class", "per_run"}
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        run_out = out_root / run_id / "prototypes"
        run_out.mkdir(parents=True, exist_ok=True)

        if mode == "per_class":
            # 1) sample per predicted class via streaming
            buckets = _collect_by_class_streamed(
                h5_path=r["features"],
                preds_csv=r["preds"],
                max_per_class=max_per_class,
                batch_size=batch_size,
            )
            # 2) PCA per run on concatenated sample
            buckets = _fit_project_pca_per_run(buckets, pca_dim=pca_dim)
            groups = sorted(buckets.keys())

            proto_records: List[Dict[str, object]] = []
            cov_records: List[Dict[str, object]] = []

            for g in groups:
                X = buckets[g]["Xp"] if "Xp" in buckets[g] else buckets[g]["X"]
                names = buckets[g]["names"]
                idxs = buckets[g]["idx"]
                X = np.asarray(X)
                if X.shape[0] < max(min_points_for_group, k):
                    continue  # skip tiny group

                # 3) k-medoids on group
                med_idx_local, assign = k_medoids(X, k=k, metric=metric, max_iter=max_iter, seed=seed)
                # global indices & names
                med_global_idx = np.asarray(idxs)[med_idx_local].tolist()
                med_names = np.asarray(names)[med_idx_local].tolist()

                # within distances (for coverage metric)
                # reuse distance to assigned medoid
                M = X[med_idx_local]
                D = _pairwise_dist(X, M, metric)
                within = D[np.arange(X.shape[0]), assign]

                # 4) coverage metrics
                cov = _coverage_metrics(assign, k=k, distances=within)
                # per-medoid sizes
                counts = np.bincount(assign, minlength=med_idx_local.size).astype(int)
                for i, c in enumerate(counts):
                    cov[f"medoid_{i}_count"] = int(c)

                cov.update({
                    "run_id": run_id,
                    "group": g,
                    "n_group": int(X.shape[0]),
                    "k": int(med_idx_local.size),
                    "metric": metric,
                    "pca_dim": int(pca_dim),
                })
                cov_records.append(cov)

                # 5) write prototype identities
                for i, (gi, nm) in enumerate(zip(med_global_idx, med_names)):
                    proto_records.append({
                        "run_id": run_id,
                        "group": g,
                        "medoid_rank": int(i),
                        "medoid_global_idx": int(gi),
                        "medoid_name": str(nm),
                    })

            # save
            with open(run_out / "prototypes.json", "w", encoding="utf-8") as f:
                json.dump(proto_records, f, indent=2)
            pd.DataFrame(cov_records).to_csv(run_out / "coverage.csv", index=False)

        else:  # mode == 'per_run'
            # collect one large sample (balanced across classes implicitly via streaming cap)
            buckets = _collect_by_class_streamed(
                h5_path=r["features"],
                preds_csv=r["preds"],
                max_per_class=max_per_class,
                batch_size=batch_size,
            )
            # merge all classes
            if not buckets:
                continue
            Xs, names, idxs = [], [], []
            for d in buckets.values():
                Xs.append(d["X"])
                names.extend(d["names"].tolist())
                idxs.extend(d["idx"].tolist())
            Xall = np.vstack(Xs)
            names = np.asarray(names)
            idxs = np.asarray(idxs, dtype=np.int64)

            # PCA on run sample
            if Xall.shape[1] > pca_dim:
                ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
                if Xall.shape[0] > 4096:
                    for j in range(0, Xall.shape[0], 4096):
                        ipca.partial_fit(Xall[j : j + 4096])
                else:
                    ipca.partial_fit(Xall)
                Xp = ipca.transform(Xall)
            else:
                Xp = Xall - Xall.mean(axis=0, keepdims=True)

            if Xp.shape[0] < max(min_points_for_group, k):
                continue

            med_idx_local, assign = k_medoids(Xp, k=k, metric=metric, max_iter=max_iter, seed=seed)
            med_global_idx = idxs[med_idx_local].tolist()
            med_names = names[med_idx_local].tolist()

            M = Xp[med_idx_local]
            D = _pairwise_dist(Xp, M, metric)
            within = D[np.arange(Xp.shape[0]), assign]
            cov = _coverage_metrics(assign, k=len(med_idx_local), distances=within)
            cov.update({
                "run_id": run_id,
                "group": "__ALL__",
                "n_group": int(Xp.shape[0]),
                "k": int(len(med_idx_local)),
                "metric": metric,
                "pca_dim": int(pca_dim),
            })

            # outputs
            proto_records = [{
                "run_id": run_id,
                "group": "__ALL__",
                "medoid_rank": int(i),
                "medoid_global_idx": int(gi),
                "medoid_name": str(nm),
            } for i, (gi, nm) in enumerate(zip(med_global_idx, med_names))]

            with open(run_out / "prototypes.json", "w", encoding="utf-8") as f:
                json.dump(proto_records, f, indent=2)
            pd.DataFrame([cov]).to_csv(run_out / "coverage.csv", index=False)

    return {"out_dir": str(out_root)}
