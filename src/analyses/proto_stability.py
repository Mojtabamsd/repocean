from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import math
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from src.index import build_run_index
from src.stream import open_h5
from src.utils.paths import _safe_slug
from src.visual.tools import shorten_label, _shorten_for_vis
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
    Collect medoids across ALL runs and ALL groups (profile_id/sample_id).

    Looks for:
        <out_dir>/prototypes/<run_id>/**/prototypes_per_class.json
    and falls back to legacy:
        <out_dir>/prototypes/<run_id>/prototypes.json

    Returns one row per medoid with:
      [run_id, group_id, proto_group, medoid_rank, medoid_global_idx,
       medoid_name, features_path]
    """
    rows = []
    for _, r in runs_df.iterrows():
        run_id = r["run_id"]
        base_dir = out_dir / "prototypes" / run_id
        if not base_dir.exists():
            continue

        # New layout: nested per-group per-class prototypes
        proto_paths = sorted(base_dir.rglob("prototypes_per_class.json"))

        # Legacy fallback: single prototypes.json in run folder
        if not proto_paths:
            legacy = base_dir / "prototypes.json"
            if legacy.exists():
                proto_paths = [legacy]

        for p in proto_paths:
            dfp = _load_prototypes_json(p)
            if dfp.empty:
                continue

            # Ensure columns exist
            if "run_id" not in dfp.columns:
                dfp["run_id"] = run_id

            # group_id: if not present, derive from folder or fall back to run_id
            if "group_id" not in dfp.columns:
                try:
                    rel = p.parent.relative_to(base_dir)
                    group_label = "" if str(rel) == "." else str(rel)
                except ValueError:
                    group_label = ""
                dfp["group_id"] = group_label if group_label else run_id

            # proto_group: class label; legacy used "group" for that
            if "proto_group" not in dfp.columns:
                if "group" in dfp.columns:
                    dfp["proto_group"] = dfp["group"]
                else:
                    dfp["proto_group"] = "__ALL__"

            # Feature file for this run
            dfp["features_path"] = r["features"]

            rows.append(
                dfp[
                    [
                        "run_id",
                        "group_id",
                        "proto_group",
                        "medoid_rank",
                        "medoid_global_idx",
                        "medoid_name",
                        "features_path",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "group_id",
                "proto_group",
                "medoid_rank",
                "medoid_global_idx",
                "medoid_name",
                "features_path",
            ]
        )

    return pd.concat(rows, ignore_index=True)


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
    min_per_class: int = 3,       # at least this many medoids overall to evaluate a class globally
    min_profiles_for_sil: int = 2,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Prototype stability across ALL prototype folders (runs + nested group_id/profile_id).

    Uses per-class prototypes (prototypes_per_class.json where available, otherwise
    falls back to legacy prototypes.json).

    Two levels of analysis:

      1) Pairwise stability per class across profiles (run_id, group_id):

         - centroid distance between profile medoid sets
         - mean symmetric Hausdorff distance
         - Jensen–Shannon divergence of coverage (if coverage_per_class.csv exists)

         -> writes: pairs_profiles.csv, class_summary.csv

      2) Global stability per class:

         - n_profiles, n_prototypes
         - mean distance to global centroid
         - mean feature variance
         - silhouette_by_profile (profiles as labels; higher = more separated = less stable)

         -> writes: global_summary.csv
         -> optional scatter plots (first 2 PCA dims) coloured by profile

    All outputs go under:
        <out_dir>/_global/proto_stability/

    NOTE: For visualisation we also add shortened columns:
          class_short, run_*_short, group_*_short, profile_short.
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    out_root = Path(out_dir) / "_global" / "proto_stability"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Collect medoids table across all runs & groups
    medoids = _collect_all_medoids(runs, Path(out_dir))
    if medoids.empty:
        raise RuntimeError("No prototypes_per_class.json / prototypes.json found under runs.")

    # 2) Read medoid features per run into a single aligned matrix X_all
    n_rows = len(medoids)
    X_all = None
    feature_dim = None

    for run_id, sub in medoids.groupby("run_id"):
        idx = sub["medoid_global_idx"].astype(np.int64).values
        idx_sorted = np.sort(np.unique(idx))
        X = _read_rows_by_index_list(sub["features_path"].iloc[0], idx_sorted)
        if feature_dim is None:
            feature_dim = X.shape[1]
            X_all = np.zeros((n_rows, feature_dim), dtype=X.dtype)

        pos = {v: i for i, v in enumerate(idx_sorted)}
        # fill rows in X_all at positions corresponding to medoids' global indices in the DF
        for df_idx, gidx in zip(sub.index.values, idx):
            X_all[df_idx] = X[pos[gidx]]

    if X_all is None or feature_dim is None:
        raise RuntimeError("No medoid features collected.")

    # 3) Fit global PCA and normalize
    total_medoids = X_all.shape[0]
    if total_medoids < 2:
        raise RuntimeError("Not enough medoids for stability analysis.")

    feature_dim = X_all.shape[1]
    n_components = max(2, min(pca_dim, feature_dim, total_medoids))
    ipca = IncrementalPCA(n_components=n_components, batch_size=4096)

    if total_medoids >= n_components:
        if total_medoids > 4096:
            ipca.partial_fit(X_all[: max(n_components, 4096)])
            for j in range(0, total_medoids, 4096):
                ipca.partial_fit(X_all[j : j + 4096])
        else:
            ipca.partial_fit(X_all)
    else:
        ipca = None

    if ipca is None:
        global_mean = X_all.mean(axis=0, keepdims=True)
        Z_all = _l2_normalize(X_all - global_mean)
    else:
        if X_all.shape[1] > ipca.n_components:
            Z_all = _l2_normalize(ipca.transform(X_all))
        else:
            Z_all = _l2_normalize(X_all - ipca.mean_)

    # 4) Build coverage map: (run_id, group_id, proto_group) -> counts array
    coverage_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    for _, r in runs.iterrows():
        run_id = r["run_id"]
        base_dir = Path(out_dir) / "prototypes" / run_id
        if not base_dir.exists():
            continue
        for cov_path in base_dir.rglob("coverage_per_class.csv"):
            cov = _load_coverage_csv(cov_path)
            if cov.empty:
                continue
            for _, row in cov.iterrows():
                gid = str(row.get("group_id", run_id))
                cls = str(row.get("proto_group", row.get("group", "__ALL__")))
                counts = []
                i = 0
                while True:
                    key = f"medoid_{i}_count"
                    if key in row:
                        counts.append(int(row[key]))
                        i += 1
                    else:
                        break
                if counts:
                    coverage_map[(run_id, gid, cls)] = np.array(counts, dtype=float)

    # Convenience aliases (full IDs)
    cls_col = medoids["proto_group"].astype(str).values
    run_col = medoids["run_id"].astype(str).values
    grp_col = medoids["group_id"].astype(str).values

    # Short versions for visualisation only
    classes = sorted(np.unique(cls_col))
    class_to_short = {c: shorten_label(c) for c in classes}

    unique_runs = np.unique(run_col)
    run_to_short = {r: _shorten_for_vis(r) for r in unique_runs}

    unique_groups = np.unique(grp_col)
    group_to_short = {g: _shorten_for_vis(g) for g in unique_groups}

    # --------------------------------------------------------------------------
    # 5) Pairwise profile stability per class
    # --------------------------------------------------------------------------
    per_key = {}  # (run_id, group_id, class) -> indices in Z_all
    for i, (rid, gid, cls) in enumerate(zip(run_col, grp_col, cls_col)):
        per_key.setdefault((rid, gid, cls), []).append(i)

    pairs_rows = []

    for cls in classes:
        # all profiles (run_id, group_id) that have this class
        profiles = sorted({(r, g) for (r, g, c) in per_key.keys() if c == cls})
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                (ri, gi) = profiles[i]
                (rj, gj) = profiles[j]

                idx_i = per_key[(ri, gi, cls)]
                idx_j = per_key[(rj, gj, cls)]
                Xi = Z_all[idx_i]
                Xj = Z_all[idx_j]

                if Xi.size == 0 or Xj.size == 0:
                    continue

                ci = _centroid(Xi)
                cj = _centroid(Xj)
                centroid_dist = _cosine_dist(ci, cj)
                mean_haus = _mean_symmetric_hausdorff(Xi, Xj)

                # coverage divergence if both have counts
                jsd = math.nan
                ci_counts = coverage_map.get((ri, gi, cls))
                cj_counts = coverage_map.get((rj, gj, cls))
                if (
                    ci_counts is not None
                    and cj_counts is not None
                    and ci_counts.sum() > 0
                    and cj_counts.sum() > 0
                ):
                    jsd = _js_divergence(ci_counts, cj_counts)

                pairs_rows.append(
                    {
                        "class": cls,
                        "class_short": class_to_short.get(cls, cls),
                        "run_i": ri,
                        "run_i_short": run_to_short.get(ri, ri),
                        "group_i": gi,
                        "group_i_short": group_to_short.get(gi, gi),
                        "run_j": rj,
                        "run_j_short": run_to_short.get(rj, rj),
                        "group_j": gj,
                        "group_j_short": group_to_short.get(gj, gj),
                        "k_i": int(Xi.shape[0]),
                        "k_j": int(Xj.shape[0]),
                        "centroid_dist": float(centroid_dist),
                        "mean_hausdorff": float(mean_haus),
                        "js_divergence": jsd,
                    }
                )

    pairs = pd.DataFrame(pairs_rows)
    pairs.to_csv(out_root / "pairs_profiles.csv", index=False)

    # Per-class summary from pairs
    if not pairs.empty:
        class_summary = (
            pairs.groupby("class")
            .agg(
                n_pairs=("class", "size"),
                mean_centroid_dist=("centroid_dist", "mean"),
                median_centroid_dist=("centroid_dist", "median"),
                mean_hausdorff=("mean_hausdorff", "mean"),
                median_hausdorff=("mean_hausdorff", "median"),
                mean_js=("js_divergence", "mean"),
            )
            .reset_index()
        )
        class_summary["class_short"] = class_summary["class"].map(
            lambda c: class_to_short.get(c, c)
        )
    else:
        class_summary = pd.DataFrame(
            columns=[
                "class",
                "class_short",
                "n_pairs",
                "mean_centroid_dist",
                "median_centroid_dist",
                "mean_hausdorff",
                "median_hausdorff",
                "mean_js",
            ]
        )
    class_summary.to_csv(out_root / "class_summary.csv", index=False)

    # Optional per-class heatmaps (pairwise mean_hausdorff)
    if save_plots and not pairs.empty:
        for cls, sub in pairs.groupby("class"):
            cls_short = class_to_short.get(cls, cls)

            # Full profile keys for internal mapping
            profile_keys = sorted(
                {f"{ri}|{gi}" for ri, gi in zip(sub["run_i"], sub["group_i"])}
                | {f"{rj}|{gj}" for rj, gj in zip(sub["run_j"], sub["group_j"])}
            )
            idx_map = {p: i for i, p in enumerate(profile_keys)}
            n = len(profile_keys)

            # Short labels for axis tick display
            profile_labels = []
            for key in profile_keys:
                ri, gi = key.split("|", 1)
                r_short = run_to_short.get(ri, ri)
                g_short = group_to_short.get(gi, gi)
                profile_labels.append(f"{r_short}|{g_short}")

            M = np.zeros((n, n), dtype=float)
            for _, row in sub.iterrows():
                pi_full = f"{row['run_i']}|{row['group_i']}"
                pj_full = f"{row['run_j']}|{row['group_j']}"
                i = idx_map[pi_full]
                j = idx_map[pj_full]
                M[i, j] = M[j, i] = row["mean_hausdorff"]

            # plt.figure(figsize=(max(4, n * 0.5), max(3, n * 0.5)))
            plt.figure(figsize=(8, 6))
            plt.imshow(M, interpolation="nearest")
            plt.title(f"Proto mean-Hausdorff across profiles — {cls_short}")
            plt.xticks(range(n), profile_labels, rotation=90, fontsize=7)
            plt.yticks(range(n), profile_labels, fontsize=7)
            plt.colorbar(label="mean Hausdorff (cosine)")
            plt.tight_layout()
            safe_cls = _safe_slug(str(cls_short))
            plt.savefig(out_root / f"heatmap_mean_hausdorff_{safe_cls}.png", dpi=160)
            plt.close()

    # --------------------------------------------------------------------------
    # 6) Global stability per class (across all profiles)
    # --------------------------------------------------------------------------
    def _mean_dist_to_centroid(Z: np.ndarray) -> float:
        if Z.shape[0] == 0:
            return float("nan")
        c = _l2_normalize(Z.mean(axis=0, keepdims=True))[0]
        sims = Z @ c
        np.clip(sims, -1.0, 1.0, out=sims)
        d = 1.0 - sims
        return float(np.mean(d))

    def _mean_feature_variance(Z: np.ndarray) -> float:
        if Z.shape[0] <= 1:
            return 0.0
        return float(np.var(Z, axis=0).mean())

    global_rows = []
    for cls in classes:
        mask = (cls_col == cls)
        Zg = Z_all[mask]
        runs_g = run_col[mask]
        grps_g = grp_col[mask]

        n_proto = int(Zg.shape[0])
        # Full profile IDs for silhouette (no shortening → no collisions)
        profiles_full = np.array(
            [f"{r}|{g}" for r, g in zip(runs_g, grps_g)], dtype=object
        )
        n_profiles = int(len(np.unique(profiles_full)))

        cls_short = class_to_short.get(cls, cls)

        if n_proto < min_per_class:
            global_rows.append(
                {
                    "class": cls,
                    "class_short": cls_short,
                    "n_profiles": n_profiles,
                    "n_prototypes": n_proto,
                    "mean_to_centroid": np.nan,
                    "variance": np.nan,
                    "silhouette_by_profile": np.nan,
                }
            )
            continue

        mean_to_cent = _mean_dist_to_centroid(Zg)
        var_mean = _mean_feature_variance(Zg)

        # silhouette by profile: higher → profiles more separated → less stable
        if n_profiles >= min_profiles_for_sil and Zg.shape[0] >= 2:
            try:
                sil = silhouette_score(Zg, profiles_full, metric="euclidean")
            except Exception:
                sil = math.nan
        else:
            sil = math.nan

        global_rows.append(
            {
                "class": cls,
                "class_short": cls_short,
                "n_profiles": n_profiles,
                "n_prototypes": n_proto,
                "mean_to_centroid": float(mean_to_cent),
                "variance": float(var_mean),
                "silhouette_by_profile": float(sil)
                if sil == sil
                else math.nan,
            }
        )

        # optional scatter: first two dims, coloured by profile (short labels only for legend)
        if save_plots and Zg.shape[1] >= 2:
            XY = Zg[:, :2]
            # plt.figure(figsize=(5.5, 4.5))
            plt.figure(figsize=(8, 6))

            profiles_short = []
            for r, g in zip(runs_g, grps_g):
                rs = run_to_short.get(r, r)
                gs = group_to_short.get(g, g)
                profiles_short.append(f"{rs}|{gs}")
            profiles_short = np.array(profiles_short, dtype=object)

            for prof in np.unique(profiles_short):
                sel = profiles_short == prof
                plt.scatter(XY[sel, 0], XY[sel, 1], s=10, label=str(prof), alpha=0.8)

            plt.title(f"Global prototypes — {cls_short} (colored by profile)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend(markerscale=2, fontsize=6, frameon=False)
            plt.tight_layout()
            safe_cls = _safe_slug(str(cls_short))
            plt.savefig(out_root / f"global_scatter_{safe_cls}.png", dpi=160)
            plt.close()

    global_df = pd.DataFrame(global_rows)
    global_df = global_df.sort_values(
        ["silhouette_by_profile", "mean_to_centroid"], na_position="last"
    ).reset_index(drop=True)
    global_df.to_csv(out_root / "global_summary.csv", index=False)

    return {"out_dir": str(out_root)}

