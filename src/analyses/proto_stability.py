from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import math
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from src.index import build_run_index
from src.stream import open_h5
from src.utils.paths import _safe_slug
# -----------------------
# IO helpers
# -----------------------


def _load_prototypes_json(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=["run_id","group","medoid_rank","medoid_global_idx","medoid_name"])
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _load_coverage_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return df


def _collect_all_medoids(runs_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Returns a tidy table with one row per medoid:
    [run_id, group, medoid_rank, medoid_global_idx, medoid_name, features_path]
    """
    rows = []
    for _, r in runs_df.iterrows():
        run_id = r["run_id"]
        proto_p = Path(out_dir) / run_id / "prototypes" / "prototypes.json"
        dfp = _load_prototypes_json(proto_p)
        if dfp.empty:
            continue
        dfp["run_id"] = run_id
        dfp["features_path"] = r["features"]
        rows.append(dfp[["run_id","group","medoid_rank","medoid_global_idx","medoid_name","features_path"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["run_id","group","medoid_rank","medoid_global_idx","medoid_name","features_path"]
    )


def _read_rows_by_index_list(h5_path: str, idx_list: np.ndarray) -> np.ndarray:
    # Single-shot fancy indexing (indices must be sorted & int64)
    with open_h5(h5_path) as h5f:
        feats = h5f["features"][idx_list]
    return feats

# -----------------------
# math helpers
# -----------------------


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(nrm, eps, None)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: 1D unit vectors
    s = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - s  # in [0,2]


def _mean_symmetric_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Mean symmetric Hausdorff distance between two sets (L2 on unit vectors ~ cosine).
    A, B are [na,d], [nb,d], assumed L2-normalized.
    """
    if A.size == 0 or B.size == 0:
        return float("nan")
    # distances via cosine ~ Euclidean on unit sphere: 1 - dot
    # compute A->B min
    sim = A @ B.T  # [-1,1]
    d_ab = 1.0 - np.clip(sim, -1.0, 1.0)
    min_ab = d_ab.min(axis=1).mean()
    # B->A min
    sim2 = B @ A.T
    d_ba = 1.0 - np.clip(sim2, -1.0, 1.0)
    min_ba = d_ba.min(axis=1).mean()
    return float(0.5 * (min_ab + min_ba))


def _centroid(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.zeros((X.shape[1],), dtype=X.dtype)
    c = X.mean(axis=0, keepdims=True)
    return _l2_normalize(c)[0]


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two discrete distributions.
    p,q nonnegative, not necessarily same length (we pad to same).
    Returns value in [0, ln 2]; we can report sqrt(JS) if desired.
    """
    L = max(len(p), len(q))
    p2 = np.zeros(L, dtype=float); p2[:len(p)] = p
    q2 = np.zeros(L, dtype=float); q2[:len(q)] = q
    p2 = p2 / (p2.sum() + eps)
    q2 = q2 / (q2.sum() + eps)
    m = 0.5 * (p2 + q2)
    def _kl(a, b):
        mask = (a > 0)
        return np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps)))
    js = 0.5 * _kl(p2, m) + 0.5 * _kl(q2, m)
    return float(js)

# -----------------------
# core
# -----------------------


def run_prototype_stability(
    parent_dir: str,
    out_dir: str,
    pca_dim: int = 50,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Reads per-run prototypes and computes stability/coverage metrics across runs, per class.

    Outputs under <out_dir>/_global/proto_stability/ :
      - pairs.csv           : class, run_i, run_j, centroid_dist, mean_hausdorff, js_divergence (if coverage present), k_i, k_j
      - class_summary.csv   : per-class averages across pairs
      - (optional) heatmaps per class for pairwise distances
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    out_root = Path(out_dir) / "_global" / "proto_stability"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Gather all medoids across runs
    medoids = _collect_all_medoids(runs, Path(out_dir))
    if medoids.empty:
        raise RuntimeError("No prototypes.json found under runs.")

    # 2) Load medoid features per run (once) and attach to table
    feats_cache: Dict[str, np.ndarray] = {}  # key: f"{run_id}"
    idx_cache: Dict[str, np.ndarray] = {}    # the indices we used
    for run_id, sub in medoids.groupby("run_id"):
        idx = sub["medoid_global_idx"].astype(np.int64).values
        # sorted unique to read efficiently, then map back order
        idx_sorted = np.sort(np.unique(idx))
        X = _read_rows_by_index_list(sub["features_path"].iloc[0], idx_sorted)
        # map back to original row order
        pos = {v: i for i, v in enumerate(idx_sorted)}
        X_ord = np.vstack([X[pos[i]] for i in idx])
        feats_cache[run_id] = X_ord
        idx_cache[run_id] = idx

    # 3) Fit a global PCA on all medoid features, then normalize (safe for small counts)
    all_blocks = [feats_cache[rid] for rid in feats_cache]
    if not all_blocks:
        raise RuntimeError("No medoid features collected.")

    feature_dim = all_blocks[0].shape[1]
    total_medoids = int(sum(X.shape[0] for X in all_blocks))
    if total_medoids < 2:
        # Not enough data to fit PCA — use identity-ish centering later
        ipca = None
    else:
        n_components = max(2, min(pca_dim, feature_dim, total_medoids))
        ipca = IncrementalPCA(n_components=n_components, batch_size=4096)

        # Bootstrap: first partial_fit must see at least n_components rows
        buf = []
        acc = 0
        blk_idx = 0
        while blk_idx < len(all_blocks) and acc < n_components:
            Xb = all_blocks[blk_idx]
            buf.append(Xb)
            acc += Xb.shape[0]
            blk_idx += 1
        X0 = np.vstack(buf)
        ipca.partial_fit(X0)

        # Feed remaining (including unused tail of the boot batch is fine to repeat)
        for i in range(blk_idx, len(all_blocks)):
            Xb = all_blocks[i]
            if Xb.shape[0] > 4096:
                for j in range(0, Xb.shape[0], 4096):
                    ipca.partial_fit(Xb[j:j + 4096])
            else:
                ipca.partial_fit(Xb)

    # project & normalize per run
    run_proj: Dict[str, np.ndarray] = {}
    if ipca is None:
        # Not enough data for PCA: just mean-center across all medoids jointly
        # Compute a global mean for stability
        global_mean = np.mean(np.vstack(all_blocks), axis=0, keepdims=True)
        for rid, X in feats_cache.items():
            Xp = X - global_mean
            run_proj[rid] = _l2_normalize(Xp)
    else:
        for rid, X in feats_cache.items():
            if X.shape[1] > ipca.n_components:
                Xp = ipca.transform(X)
            else:
                # If feature_dim <= n_components, center using ipca.mean_
                Xp = X - ipca.mean_
            run_proj[rid] = _l2_normalize(Xp)

    # 4) Build per-run, per-class medoid matrices (projected)
    #    and (optional) coverage distributions from coverage.csv
    per = {}  # per[(run_id, class)] = {"X": [k,d], "counts": [k] or None}
    for _, r in runs.iterrows():
        run_id = r["run_id"]
        proto_p = Path(out_dir) / run_id / "prototypes" / "prototypes.json"
        cov_p   = Path(out_dir) / run_id / "prototypes" / "coverage.csv"
        dfp = _load_prototypes_json(proto_p)
        if dfp.empty:
            continue
        dfp = dfp.sort_values(["group","medoid_rank"]).reset_index(drop=True)
        Xrun = run_proj[run_id]
        # attach projected rows by original row order in medoids DF
        # medoids table has rows aligned with feats_cache ordering already
        # we reindex dfp to the order in 'medoids' subset for this run
        sub_all = medoids[medoids.run_id == run_id].reset_index(drop=True)
        # index in sub_all per (group, medoid_rank)
        key_to_pos = {(g, int(mr)): i for i, (g, mr) in enumerate(zip(sub_all["group"], sub_all["medoid_rank"]))}
        # order indices for this dfp
        pos = [key_to_pos.get((g, int(mr)), None) for g, mr in zip(dfp["group"], dfp["medoid_rank"])]
        ok_mask = [p is not None for p in pos]
        dfp = dfp.loc[ok_mask].reset_index(drop=True)
        pos = [p for p in pos if p is not None]
        Xmed = Xrun[np.array(pos, dtype=np.int64)]
        # counts (optional)
        cov = _load_coverage_csv(cov_p)
        counts_map = {}
        if not cov.empty:
            for _, row in cov.iterrows():
                g = row.get("group", None)
                if g is None:
                    continue
                # collect medoid_i_count in medoid_rank order
                counts = []
                i = 0
                while True:
                    key = f"medoid_{i}_count"
                    if key in row:
                        counts.append(int(row[key]))
                        i += 1
                    else:
                        break
                counts_map[str(g)] = np.array(counts, dtype=float)

        # stash per class
        for g in dfp["group"].unique():
            sel = dfp["group"] == g
            Xg = Xmed[sel.values]
            cts = counts_map.get(str(g), None)
            per[(run_id, str(g))] = {"X": Xg, "counts": cts}

    # 5) Pairwise comparisons per class across runs
    pairs_rows = []
    classes = sorted({g for (_, g) in per.keys()})
    for g in classes:
        # runs that have this class
        runs_g = [rid for (rid, cg) in per.keys() if cg == g]
        runs_g = sorted(set(runs_g))
        for i in range(len(runs_g)):
            for j in range(i + 1, len(runs_g)):
                ri, rj = runs_g[i], runs_g[j]
                Xi = per[(ri, g)]["X"]; Xj = per[(rj, g)]["X"]
                ci = _centroid(Xi); cj = _centroid(Xj)
                centroid_dist = _cosine_dist(ci, cj)
                mean_haus = _mean_symmetric_hausdorff(Xi, Xj)
                # coverage divergence if both have counts
                jsd = math.nan
                ci_counts = per[(ri, g)]["counts"]
                cj_counts = per[(rj, g)]["counts"]
                if ci_counts is not None and cj_counts is not None and ci_counts.sum() > 0 and cj_counts.sum() > 0:
                    jsd = _js_divergence(ci_counts, cj_counts)
                pairs_rows.append({
                    "class": g,
                    "run_i": ri, "run_j": rj,
                    "k_i": int(Xi.shape[0]), "k_j": int(Xj.shape[0]),
                    "centroid_dist": float(centroid_dist),
                    "mean_hausdorff": float(mean_haus),
                    "js_divergence": jsd,
                })

    pairs = pd.DataFrame(pairs_rows)
    pairs.to_csv(out_root / "pairs.csv", index=False)

    # 6) Per-class summary
    if not pairs.empty:
        aggs = pairs.groupby("class").agg(
            n_pairs=("class","size"),
            mean_centroid_dist=("centroid_dist","mean"),
            median_centroid_dist=("centroid_dist","median"),
            mean_hausdorff=("mean_hausdorff","mean"),
            median_hausdorff=("mean_hausdorff","median"),
            mean_js=("js_divergence","mean")
        ).reset_index()
    else:
        aggs = pd.DataFrame(columns=[
            "class","n_pairs","mean_centroid_dist","median_centroid_dist",
            "mean_hausdorff","median_hausdorff","mean_js"
        ])
    aggs.to_csv(out_root / "class_summary.csv", index=False)

    # 7) Optional per-class heatmaps (pairwise distances)
    if save_plots and not pairs.empty:
        for g, sub in pairs.groupby("class"):
            runs_g = sorted(set(sub["run_i"]).union(set(sub["run_j"])))
            idx = {rid: i for i, rid in enumerate(runs_g)}
            # fill matrices
            n = len(runs_g)
            M = np.zeros((n, n), dtype=float)
            for _, row in sub.iterrows():
                i = idx[row["run_i"]]; j = idx[row["run_j"]]
                M[i, j] = M[j, i] = row["mean_hausdorff"]
            plt.figure(figsize=(max(4, n*0.5), max(3, n*0.5)))
            plt.imshow(M, interpolation="nearest")
            plt.title(f"Proto mean-Hausdorff across runs — {g}")
            plt.xticks(range(n), runs_g, rotation=90, fontsize=8)
            plt.yticks(range(n), runs_g, fontsize=8)
            plt.colorbar(label="mean Hausdorff (cosine)")
            plt.tight_layout()
            safe_g = _safe_slug(str(g))
            plt.savefig(out_root / f"heatmap_mean_hausdorff_{safe_g}.png", dpi=160)
            plt.close()

    return {"out_dir": str(out_root)}
