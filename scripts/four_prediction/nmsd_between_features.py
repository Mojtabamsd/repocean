from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


def _prepare_feature_table(
    metrics_csv: str | Path,
    features: list[str],
    id_col: str = "group_id",
    run_id: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(Path(metrics_csv), sep=None, engine="python")
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id].copy()

    need = [id_col] + features
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {missing}. Available: {list(df.columns)}")

    # If repeated ids, aggregate (mean)
    g = df.groupby(id_col, as_index=False)[features].mean(numeric_only=True)
    g = g.dropna(subset=features).reset_index(drop=True)
    return g


def _merge_shannon(
    g: pd.DataFrame,
    shannon_csv: str | Path,
    id_col: str = "group_id",
    shannon_col: str = "Shannon",
    shannon_agg: str = "mean",
) -> pd.DataFrame:
    sh = pd.read_csv(Path(shannon_csv), sep=None, engine="python")
    if id_col not in sh.columns:
        raise ValueError(f"Shannon CSV missing '{id_col}'. Columns: {list(sh.columns)}")
    if shannon_col not in sh.columns:
        raise ValueError(f"Shannon CSV missing '{shannon_col}'. Columns: {list(sh.columns)}")

    keep = [id_col, shannon_col]
    if "Effective_species" in sh.columns:
        keep.append("Effective_species")
    sh = sh[keep].copy()

    if not sh[id_col].is_unique:
        if shannon_agg == "mean":
            sh = sh.groupby(id_col, as_index=False)[[c for c in keep if c != id_col]].mean(numeric_only=True)
        elif shannon_agg == "median":
            sh = sh.groupby(id_col, as_index=False)[[c for c in keep if c != id_col]].median(numeric_only=True)
        elif shannon_agg == "first":
            sh = sh.drop_duplicates(subset=[id_col], keep="first")
        else:
            raise ValueError("shannon_agg must be one of: 'mean', 'median', 'first'")

    out = g.merge(sh, on=id_col, how="left", validate="one_to_one")
    return out


def compute_feature_distance_matrix(
    g: pd.DataFrame,
    features: list[str],
    metric: str = "euclidean",  # try "cosine" too
) -> tuple[pd.DataFrame, np.ndarray]:
    X = g[features].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    D = pairwise_distances(Xz, metric=metric)
    dist_df = pd.DataFrame(D, index=g.iloc[:, 0].astype(str), columns=g.iloc[:, 0].astype(str))
    return dist_df, D


def save_feature_dendrogram(
    dist_df: pd.DataFrame,
    out_png: str | Path,
    method: str = "average",  # "average" is common (UPGMA); "complete"/"ward" etc.
    figsize=(12, 6),
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    condensed = squareform(dist_df.values, checks=False)
    Z = linkage(condensed, method=method)

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(Z, labels=dist_df.index.tolist(), leaf_rotation=90, ax=ax)
    ax.set_title(f"Hierarchical clustering of deployments (features distance, method={method})")
    ax.set_ylabel("Linkage distance")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return Z


def save_clustered_distance_heatmap(
    dist_df: pd.DataFrame,
    out_png: str | Path,
    method: str = "average",
    figsize=(10, 8),
    cmap="viridis",
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    condensed = squareform(dist_df.values, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    d = dist_df.iloc[order, order]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(d.values, cmap=cmap)
    ax.set_xticks(range(len(d)))
    ax.set_yticks(range(len(d)))
    ax.set_xticklabels(d.index.tolist(), rotation=90, fontsize=7)
    ax.set_yticklabels(d.index.tolist(), fontsize=7)
    ax.set_title("Clustered distance heatmap (based on feature distances)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distance")

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return d, Z, order


def save_nmds_colored_by_shannon_from_distance(
    dist_df: pd.DataFrame,
    meta: pd.DataFrame,
    out_png: str | Path,
    id_col: str = "group_id",
    shannon_col: str = "Shannon",
    label_points: bool = False,
    random_state: int = 0,
    n_init: int = 8,
    max_iter: int = 3000,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Align meta to dist_df ordering
    ids = dist_df.index.astype(str).tolist()
    m = meta.copy()
    m[id_col] = m[id_col].astype(str)

    m = m.set_index(id_col).reindex(ids).reset_index()

    # Diagnostics
    if shannon_col not in m.columns:
        raise ValueError(f"'{shannon_col}' not in meta columns: {list(m.columns)}")

    cvals = pd.to_numeric(m[shannon_col], errors="coerce").to_numpy()
    n_total = len(cvals)
    n_nan = int(np.isnan(cvals).sum())
    print(f"[diag] Shannon NaNs after reindex: {n_nan}/{n_total}")

    # Make sure distance matrix is finite
    D = dist_df.to_numpy(dtype=float)
    if not np.isfinite(D).all():
        bad = np.size(D) - int(np.isfinite(D).sum())
        raise ValueError(f"Distance matrix has {bad} non-finite entries (NaN/inf). Fix features before NMDS.")

    # NMDS
    nmds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress="auto",
    )
    Y = nmds.fit_transform(D)
    stress = getattr(nmds, "stress_", None)

    if not np.isfinite(Y).all():
        raise ValueError("NMDS produced non-finite coordinates. Try fewer points, different distance, or check D.")

    # Plot: always draw points, even if Shannon missing
    fig, ax = plt.subplots(figsize=(8.8, 6.8))

    # If all Shannon missing -> draw in grey and warn
    if n_nan == n_total:
        print("[warn] All Shannon values are NaN -> plotting points in grey (merge/join key mismatch).")
        ax.scatter(Y[:, 0], Y[:, 1], s=55, alpha=0.85)
    else:
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=cvals, s=55, alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Shannon entropy")

    if label_points:
        for i, lab in enumerate(ids):
            ax.annotate(lab, (Y[i, 0], Y[i, 1]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_title(
        f"NMDS of deployments (feature distances)\ncolored by Shannon"
        + (f" (stress={stress:.3g})" if stress is not None else ""),
        fontsize=12,
    )
    ax.set_xlabel("NMDS-1")
    ax.set_ylabel("NMDS-2")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    return Y, stress



# -------------------------
# One-call runner
# -------------------------
def run_feature_nmds_and_hierarchy(
    metrics_csv: str | Path,
    out_dir: str | Path,
    features: list[str] = ["centroid_norm", "cos_p10", "eff_rank"],
    id_col: str = "group_id",
    run_id: str | None = None,
    distance_metric: str = "euclidean",
    linkage_method: str = "average",
    shannon_csv: str | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = _prepare_feature_table(metrics_csv, features, id_col=id_col, run_id=run_id)

    if shannon_csv is not None:
        g = _merge_shannon(g, shannon_csv, id_col=id_col, shannon_col="Shannon", shannon_agg="mean")

    dist_df, _ = compute_feature_distance_matrix(g, features, metric=distance_metric)

    save_feature_dendrogram(
        dist_df,
        out_png=out_dir / "feat_dendrogram.png",
        method=linkage_method,
    )

    save_clustered_distance_heatmap(
        dist_df,
        out_png=out_dir / "feat_clustered_dist_heatmap.png",
        method=linkage_method,
    )

    if shannon_csv is not None and "Shannon" in g.columns:
        save_nmds_colored_by_shannon_from_distance(
            dist_df,
            meta=g,
            out_png=out_dir / "feat_nmds_colored_by_shannon.png",
            id_col=id_col,
            shannon_col="Shannon",
            label_points=True,
        )

    # Save the table used
    g.to_csv(out_dir / "feat_table_used.csv", index=False)
    print(f"Saved {out_dir / 'feat_table_used.csv'}")

    return g, dist_df


def nmds_geometry_colored_by_shannon(
    metrics_csv: str | Path,
    shannon_csv: str | Path,
    out_png: str | Path,
    features: list[str] = ["centroid_norm", "cos_p10", "eff_rank"],
    id_col: str = "group_id",          # join key
    run_id: str | None = None,
    shannon_col: str = "Shannon",
    shannon_agg: str = "mean",         # if Shannon table has duplicates per id
    label_points: bool = False,        # set True if you want text labels (can clutter)
    random_state: int = 0,
    n_init: int = 8,
    max_iter: int = 3000,
):
    """
    NMDS (non-metric MDS) of deployments based on geometry metrics, colored by Shannon entropy.

    Saves a PNG scatter with colorbar.
    """

    metrics_csv = Path(metrics_csv)
    shannon_csv = Path(shannon_csv)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv, sep=None, engine="python")
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id].copy()

    # ---- aggregate geometry to one row per id (if duplicates)
    missing = [c for c in [id_col] + features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing in metrics CSV: {missing}. Available: {list(df.columns)}")

    g = df.groupby(id_col, as_index=False)[features].mean(numeric_only=True)
    g = g.dropna(subset=features).reset_index(drop=True)

    # ---- read + prep Shannon
    sh = pd.read_csv(shannon_csv, sep=None, engine="python")
    if id_col not in sh.columns:
        raise ValueError(f"Shannon CSV missing '{id_col}'. Columns: {list(sh.columns)}")
    if shannon_col not in sh.columns:
        raise ValueError(f"Shannon CSV missing '{shannon_col}'. Columns: {list(sh.columns)}")

    keep_cols = [id_col, shannon_col]
    if "Effective_species" in sh.columns:
        keep_cols.append("Effective_species")
    sh = sh[keep_cols].copy()

    # aggregate Shannon if duplicates
    if not sh[id_col].is_unique:
        dup_n = int(sh.duplicated(id_col).sum())
        print(f"[info] Shannon table has {dup_n} duplicate ids on '{id_col}'. Aggregating by {shannon_agg}.")
        if shannon_agg == "mean":
            sh = sh.groupby(id_col, as_index=False)[[c for c in keep_cols if c != id_col]].mean(numeric_only=True)
        elif shannon_agg == "median":
            sh = sh.groupby(id_col, as_index=False)[[c for c in keep_cols if c != id_col]].median(numeric_only=True)
        elif shannon_agg == "first":
            sh = sh.drop_duplicates(subset=[id_col], keep="first")
        else:
            raise ValueError("shannon_agg must be one of: 'mean', 'median', 'first'")

    # ---- merge Shannon onto geometry table
    g2 = g.merge(sh, on=id_col, how="left", validate="one_to_one")

    miss = int(g2[shannon_col].isna().sum())
    if miss > 0:
        print(f"[warn] {miss}/{len(g2)} ids have missing Shannon after merge. Check id_col='{id_col}' matches.")

    # ---- NMDS
    X = g2[features].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    D = pairwise_distances(Xz, metric="euclidean")

    nmds = MDS(
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress="auto",
    )
    Y = nmds.fit_transform(D)
    stress = getattr(nmds, "stress_", None)

    # ---- Plot colored by Shannon
    cvals = pd.to_numeric(g2[shannon_col], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    sc = ax.scatter(
        Y[:, 0], Y[:, 1],
        c=cvals,
        s=55,
        alpha=0.85,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Shannon entropy (alpha diversity)")

    if label_points:
        labels = g2[id_col].astype(str).tolist()
        for i, lab in enumerate(labels):
            ax.annotate(lab, (Y[i, 0], Y[i, 1]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_title(
        f"NMDS of deployments using {', '.join(features)}\ncolored by Shannon"
        + (f" (stress={stress:.3g})" if stress is not None else ""),
        fontsize=12,
    )
    ax.set_xlabel("NMDS-1")
    ax.set_ylabel("NMDS-2")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    return g2, Y, D, stress



path = r'C:\alr4\analysis\geometry_all'
shannon_csv = r"C:\alr4\ai_predict_all\prediction_parti20260119121141\ALR4_shannon_table.csv"


# g2, Y, D, stress = nmds_geometry_colored_by_shannon(
#     metrics_csv=path + r"\geometry_metrics.csv",
#     shannon_csv=shannon_csv,
#     out_png=path + r"\geometry_viz\compare3\nmds_geom.png",
#     # features=["centroid_norm", "cos_p10", "eff_rank"],
#     features=["eff_rank"],
#     id_col="group_id",
#     label_points=True,  # set True if you really want labels
# )


g, dist_df = run_feature_nmds_and_hierarchy(
    metrics_csv=path + r"\geometry_metrics.csv",
    out_dir=path + r"\geometry_viz\compare3\feature_nmds_tree",
    features=["centroid_norm", "cos_p10", "eff_rank"],
    id_col="group_id",
    run_id=None,
    distance_metric="cosine",   # try "cosine" or "euclidean"
    linkage_method="average",
    shannon_csv=shannon_csv,
)

