
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _set_pretty_axes(ax: plt.Axes):
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))


def plot_all_timeseries_together_with_shannon(
    metrics_df: pd.DataFrame,
    shannon_csv: str | Path,
    out_path: str | Path,
    rolling: int = 7,
    shannon_join_col: str = "group_id",   # change to "run_id" if your shannon table is per-run
):
    """
    Plots the final stacked timeseries figure, plus Shannon entropy (and optional effective_species if present).

    - metrics_df: your geometry metrics dataframe (already ordered, as you said)
    - shannon_csv: path to csv that contains Shannon (and maybe Effective_species)
    - out_path: where to save (without suffix); saves both .png and .pdf
    - shannon_join_col: column to join on (must exist in BOTH tables)
    """

    out_path = Path(out_path)
    shannon_csv = Path(shannon_csv)

    # ---- read shannon table
    sh = pd.read_csv(shannon_csv, sep=None, engine="python")

    if shannon_join_col not in metrics_df.columns:
        raise ValueError(f"metrics_df missing join column '{shannon_join_col}'. Available: {list(metrics_df.columns)}")
    if shannon_join_col not in sh.columns:
        raise ValueError(f"shannon table missing join column '{shannon_join_col}'. Available: {list(sh.columns)}")

    # ---- merge shannon into metrics, preserving the existing order of metrics_df
    df = metrics_df.merge(
        sh[[shannon_join_col] + [c for c in ["Shannon", "Effective_species"] if c in sh.columns]],
        on=shannon_join_col,
        how="left",
        validate="one_to_one" if sh[shannon_join_col].is_unique else "many_to_one",
    )
    # exp(Shannon) (effective number of classes)
    df["Shannon_eff"] = np.exp(pd.to_numeric(df["Shannon"], errors="coerce"))

    # ---- metrics to plot (add Shannon)
    metrics = [
        ("centroid_norm", "Centroid norm\n(homogeneity ↑)"),
        ("Shannon", "Shannon entropy\n(alpha diversity ↑)"),
        # ("cos_p10", "Cosine p10\n(diversity ↑ ↓)"),
        # ("pair_cos_p50", "Pairwise cosine p50\n(repetitiveness ↑)"),
        ("eff_rank", "Effective rank\n(complexity ↑)"),
        # ("pca_dim_90", "PCA dim 90%\n(complexity ↑)"),
    ]

    metrics.append(("Shannon_eff", "exp(Shannon)\n(effective #classes ↑)"))

    # Optionally add effective species if present
    # if "Effective_species" in df.columns:
    #     metrics.append(("Effective_species", "exp(Shannon)\n(effective #classes ↑)"))

    # ---- sanity check required columns
    missing = [c for c, _ in metrics if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    x = np.arange(len(df), dtype=int)

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(12, 1.65 * len(metrics) + 1.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, ylabel) in zip(axes, metrics):
        y = pd.to_numeric(df[col], errors="coerce").values

        ax.plot(
            x, y,
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            alpha=0.9,
        )

        # rolling mean overlay
        if rolling and rolling >= 3 and len(y) >= rolling:
            roll = (
                pd.Series(y)
                .rolling(rolling, center=True, min_periods=max(3, rolling // 3))
                .mean()
                .values
            )
            ax.plot(x, roll, linewidth=2.2, alpha=0.9)

        _set_pretty_axes(ax)
        ax.set_ylabel(ylabel, fontsize=9)

    axes[-1].set_xlabel("Deployment index (ordered)", fontsize=10)
    fig.suptitle("Representation geometry + label diversity (Shannon) across deployments", fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.with_suffix('.png')}")
    print(f"Saved {out_path.with_suffix('.pdf')}")

    return df


def _pretty(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_rank_overlay(df: pd.DataFrame, cols: list[str], out_png: str | Path, title: str):
    x = np.arange(len(df))
    ranks = pd.DataFrame({c: df[c].rank(method="average") for c in cols})

    fig, ax = plt.subplots(figsize=(12, 4))
    for c in cols:
        ax.plot(x, ranks[c].values, linewidth=1.2, label=c)
    ax.set_title(title)
    ax.set_xlabel("Deployment index (ordered)")
    ax.set_ylabel("Rank (higher = larger value)")
    _pretty(ax)
    ax.legend(frameon=False, ncol=len(cols))
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_zscore_overlay(df: pd.DataFrame, cols: list[str], out_png: str | Path, title: str):
    x = np.arange(len(df))
    Z = pd.DataFrame(index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        Z[c] = (v - v.mean()) / (v.std(ddof=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 4))
    for c in cols:
        ax.plot(x, Z[c].values, linewidth=1.2, label=c)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Deployment index (ordered)")
    ax.set_ylabel("z-score")
    _pretty(ax)
    ax.legend(frameon=False, ncol=len(cols))
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return Z


def scatter_with_smart_labels(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_png: str | Path,
    title: str,
    id_col: str | None = "group_id",
    label_every: int = 25,
    label_outliers: int = 8,
    size_col: str | None = "sampled",
    size_min: float = 18.0,
    size_max: float = 90.0,
    size_clip_q: float = 95.0,
    show_size_legend: bool = True,
):
    x = pd.to_numeric(df[xcol], errors="coerce").values
    y = pd.to_numeric(df[ycol], errors="coerce").values

    # labels
    if id_col and id_col in df.columns:
        labels = df[id_col].astype(str).tolist()
    else:
        labels = [str(i) for i in range(len(df))]

    # marker sizes
    if size_col and (size_col in df.columns):
        sraw = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0).values
        s_pos = sraw[sraw > 0]
        if len(s_pos) > 0:
            clip_hi = np.nanpercentile(s_pos, size_clip_q)
            s = np.clip(sraw, 0, clip_hi)
            s_norm = s / (s.max() if s.max() > 0 else 1.0)
            sizes = size_min + (size_max - size_min) * s_norm
        else:
            sizes = np.full(len(df), (size_min + size_max) / 2.0)
    else:
        sizes = np.full(len(df), 30.0)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))

    # Force ONE color for everything (points + legend)
    point_color = "C0"

    ax.scatter(x, y, s=sizes, alpha=0.7, color=point_color)

    # Indices to label: extremes + every Nth
    n = len(df)
    idxs = set(range(0, n, label_every))

    good_x = np.isfinite(x)
    good_y = np.isfinite(y)
    if good_x.any():
        ix = np.argsort(x[good_x])
        gx = np.where(good_x)[0]
        pick_x = np.concatenate([gx[ix[:label_outliers]], gx[ix[-label_outliers:]]])
        idxs.update(pick_x.tolist())
    if good_y.any():
        iy = np.argsort(y[good_y])
        gy = np.where(good_y)[0]
        pick_y = np.concatenate([gy[iy[:label_outliers]], gy[iy[-label_outliers:]]])
        idxs.update(pick_y.tolist())

    for i in sorted(idxs):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        ax.annotate(labels[i], (x[i], y[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- size legend using 1st / 50th / 100th percentiles (min/median/max)
    if show_size_legend and size_col and (size_col in df.columns):
        sraw = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0).values
        s_pos = sraw[sraw > 0]
        if len(s_pos) > 0:
            # 1st, 50th, 100th percentiles
            qvals = np.percentile(s_pos, [1, 50, 100])

            # use the SAME sizing transform as points (including clip at p95)
            clip_hi = np.percentile(s_pos, size_clip_q)
            q_clip = np.clip(qvals, 0, clip_hi)

            # map qvals to marker sizes
            q_sizes = size_min + (size_max - size_min) * (q_clip / (clip_hi if clip_hi > 0 else 1.0))

            handles = [
                ax.scatter([], [], s=q_sizes[0], alpha=0.7, color=point_color,
                           label=f"{size_col}: p01={int(qvals[0])}"),
                ax.scatter([], [], s=q_sizes[1], alpha=0.7, color=point_color,
                           label=f"{size_col}: p50={int(qvals[1])}"),
                ax.scatter([], [], s=q_sizes[2], alpha=0.7, color=point_color,
                           label=f"{size_col}: p100={int(qvals[2])}"),
            ]
            ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)



def plot_zscore_heatmap(Z: pd.DataFrame, out_png: str | Path, title: str):
    # rows = metrics, cols = deployment index
    data = Z.T.values  # shape: (n_metrics, n_deployments)

    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow(data, aspect="auto")  # no explicit colors set

    ax.set_yticks(range(Z.shape[1]))
    ax.set_yticklabels(Z.columns.tolist(), fontsize=9)
    ax.set_xticks(np.linspace(0, Z.shape[0] - 1, 10).astype(int))
    ax.set_xticklabels(np.linspace(0, Z.shape[0] - 1, 10).astype(int), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Deployment index (ordered)")
    ax.set_ylabel("Metric (z-score)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score")

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# -------------------------
# RUN (example)
# -------------------------
def compare_three_metrics_visually(
    df: pd.DataFrame,
    out_dir: str | Path,
    metrics: list[tuple[str, str]] = None,
    id_col: str = "group_id",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = [
            ("Shannon", "Shannon entropy"),
            ("centroid_norm", "Centroid norm"),
            ("eff_rank", "Effective rank"),
        ]

    cols = [m[0] for m in metrics]

    # 1) rank overlay
    plot_rank_overlay(
        df, cols,
        out_dir / "rank_overlay.png",
        title="Deployment-by-deployment comparison (ranks)"
    )

    # 2) z-score overlay + heatmap
    Z = plot_zscore_overlay(
        df, cols,
        out_dir / "zscore_overlay.png",
        title="Deployment-by-deployment comparison (z-scores)"
    )
    plot_zscore_heatmap(
        Z,
        out_dir / "zscore_heatmap.png",
        title="Z-score heatmap (rows=metrics, cols=deployments)"
    )

    # 3) scatters (pairwise relationships)
    scatter_with_smart_labels(
        df, "centroid_norm", "Shannon",
        out_dir / "sc_shannon_vs_centroid.png",
        # title="Shannon vs centroid_norm (labels: outliers + every 25th)",
        title="Concentration vs Balance",
        id_col=id_col,
    )
    scatter_with_smart_labels(
        df, "eff_rank", "Shannon",
        out_dir / "sc_shannon_vs_effrank.png",
        title="Shannon vs eff_rank",
        id_col=id_col,
    )
    scatter_with_smart_labels(
        df, "centroid_norm", "eff_rank",
        out_dir / "sc_effrank_vs_centroid.png",
        title="Concentration vs complexity",
        id_col=id_col,
    )

    scatter_with_smart_labels(
        df, "centroid_norm", "cos_p10",
        out_dir / "sc_centroid_vs_cos_p10.png",
        title="Concentration vs diversity",
        id_col=id_col,
    )

    # exp(Shannon) scatters
    scatter_with_smart_labels(
        df, "centroid_norm", "Shannon_eff",
        out_dir / "sc_expShannon_vs_centroid.png",
        title="Concentration vs Effective diversity (exp(Shannon))",
        id_col=id_col,
    )

    scatter_with_smart_labels(
        df, "eff_rank", "Shannon_eff",
        out_dir / "sc_expShannon_vs_effrank.png",
        title="Complexity vs Effective diversity (exp(Shannon))",
        id_col=id_col,
    )

    print(f"Saved plots to: {out_dir}")


path = r'C:\alr4\analysis\geometry_all'
df_metrics = pd.read_csv(path + r"\geometry_metrics.csv", sep=None, engine="python")
# keep order as-is (as you mentioned you already do)
df_merged = plot_all_timeseries_together_with_shannon(
    metrics_df=df_metrics,
    shannon_csv=r"C:\alr4\ai_predict_all\prediction_parti20260119121141\ALR4_shannon_table.csv",
    out_path=path + r"\geometry_viz\ts_all_metrics_plus_shannon",
    rolling=7,
    shannon_join_col="group_id",   # change to "run_id" if needed
)

# Example call (you already have merged df with Shannon):
compare_three_metrics_visually(df_merged, out_dir=path + r"\geometry_viz\compare3", id_col="group_id")


