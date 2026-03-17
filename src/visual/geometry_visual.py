from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Optional: parse datetime from your group_id for nicer tick labels
_gid_re = re.compile(
    r"^(?P<cruise>[^_]+)_(?P<date>\d{8})_(?P<hhmm>\d{4})_(?P<seg>\d{4})_d(?P<dep>\d+)$"
)

def _maybe_add_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" in df.columns:
        return df
    dts = []
    for gid in df["group_id"].astype(str).tolist():
        m = _gid_re.match(gid)
        if m:
            dt = pd.to_datetime(m.group("date") + m.group("hhmm"), format="%Y%m%d%H%M", errors="coerce")
            dts.append(dt)
        else:
            dts.append(pd.NaT)
    out = df.copy()
    out["dt"] = dts
    return out


def _set_pretty_axes(ax: plt.Axes):
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))


def _save(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _annotate_extremes(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: list[str], k: int = 3):
    if len(y) < (2 * k + 1):
        return
    idx = np.argsort(y)
    picks = np.concatenate([idx[:k], idx[-k:]])
    for i in picks:
        ax.annotate(
            labels[i],
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            rotation=25,
        )


def _plot_series(
    df: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    add_rolling: int = 0,
    annotate: bool = True,
):
    x = np.arange(len(df), dtype=int)
    y = df[col].astype(float).values
    labels = df["group_id"].astype(str).tolist()

    fig = plt.figure(figsize=(11.5, 4.2))
    ax = fig.add_subplot(111)

    ax.plot(x, y, marker="o", linewidth=1.2, markersize=2.6)

    if add_rolling and add_rolling >= 3 and len(df) >= add_rolling:
        roll = pd.Series(y).rolling(add_rolling, center=True, min_periods=max(3, add_rolling // 3)).mean().values
        ax.plot(x, roll, linewidth=2.0, alpha=0.9, label=f"rolling mean (w={add_rolling})")
        ax.legend(frameon=False, fontsize=9)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Deployment index", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    _set_pretty_axes(ax)

    if annotate:
        _annotate_extremes(ax, x, y, labels, k=3)

    _save(fig, out_path)


def _plot_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = True,
):
    x = df[xcol].astype(float).values
    y = df[ycol].astype(float).values
    labels = df["group_id"].astype(str).tolist()

    if "sampled" in df.columns:
        s = df["sampled"].astype(float).fillna(0.0).values
        s = np.clip(s, 0, np.nanpercentile(s, 95) if np.any(s > 0) else 1.0)
        sizes = 18 + 70 * (s / (s.max() if s.max() > 0 else 1.0))
    else:
        sizes = 30

    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=sizes, alpha=0.7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    _set_pretty_axes(ax)

    if annotate and len(df) >= 8:
        idx = np.argsort(y)
        picks = np.concatenate([idx[:3], idx[-3:]])
        for i in picks:
            ax.annotate(labels[i], (x[i], y[i]), fontsize=8)

    _save(fig, out_path)


def _plot_hist(
    df: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 30,
):
    vals = df[col].astype(float).dropna().values

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)

    ax.hist(vals, bins=bins, alpha=0.9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    _set_pretty_axes(ax)
    _save(fig, out_path)


def _read_centroid_cosine_matrix(path: str | Path) -> pd.DataFrame:
    M = pd.read_csv(path, index_col=0)
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    return M


def _compute_mean_centroid_cosine_to_others(M: pd.DataFrame) -> pd.Series:
    A = M.astype(float).values.copy()
    if A.shape[0] <= 1:
        vals = np.full(A.shape[0], np.nan, dtype=float)
    else:
        np.fill_diagonal(A, np.nan)
        vals = np.nanmean(A, axis=1)
    return pd.Series(vals, index=M.index, name="mean_centroid_cosine_to_others")


def _attach_centroid_cosine_summary(df: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_key"] = out["run_id"].astype(str) + "::" + out["group_id"].astype(str)

    mean_sim = _compute_mean_centroid_cosine_to_others(M)
    out = out.merge(
        mean_sim.rename("mean_centroid_cosine_to_others"),
        left_on="group_key",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    return out


def _plot_matrix_heatmap(
    M: pd.DataFrame,
    out_path: Path,
    title: str = "Centroid cosine similarity between deployments",
):
    A = M.astype(float).values
    n = A.shape[0]

    fig_w = max(7, min(16, 0.28 * n + 4))
    fig_h = max(6, min(14, 0.28 * n + 3))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    im = ax.imshow(A, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("centroid cosine", fontsize=10)

    ax.set_title(title, fontsize=12)

    if n <= 60:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(M.columns.tolist(), rotation=90, fontsize=7)
        ax.set_yticklabels(M.index.tolist(), fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Deployments", fontsize=10)
        ax.set_ylabel("Deployments", fontsize=10)

    _save(fig, out_path)


def _plot_offdiag_hist(
    M: pd.DataFrame,
    out_path: Path,
    bins: int = 30,
):
    A = M.astype(float).values
    if A.shape[0] < 2:
        vals = np.array([], dtype=float)
    else:
        mask = ~np.eye(A.shape[0], dtype=bool)
        vals = A[mask]

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)

    ax.hist(vals, bins=bins, alpha=0.9)
    ax.set_title("Distribution of centroid cosine between deployments", fontsize=12)
    ax.set_xlabel("centroid cosine (off-diagonal)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    _set_pretty_axes(ax)
    _save(fig, out_path)


def plot_all_timeseries_together(
    df: pd.DataFrame,
    out_path: Path,
    rolling: int = 7,
):
    metrics = [
        ("centroid_norm", "Centroid norm\n(homogeneity ↑)"),
        ("mean_centroid_cosine_to_others", "Mean centroid cosine\nto others ↑"),
        ("cos_p10", "Cosine p10\n(diversity ↑ ↓)"),
        ("pair_cos_p50", "Pairwise cosine p50\n(repetitiveness ↑)"),
        ("eff_rank", "Effective rank\n(complexity ↑)"),
        ("pca_dim_90", "PCA dim 90%\n(complexity ↑)"),
    ]

    metrics = [(c, y) for c, y in metrics if c in df.columns]

    x = np.arange(len(df), dtype=int)

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, ylabel) in zip(axes, metrics):
        y = df[col].astype(float).values

        ax.plot(
            x, y,
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            alpha=0.9,
        )

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

    fig.suptitle(
        "Representation geometry summary across deployments",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def visualize_geometry_metrics(
    metrics_csv: str | Path,
    out_dir: str | Path,
    run_id: str | None = None,
    rolling: int = 7,
    centroid_cosine_matrix_csv: str | Path | None = None,
):
    """
    rolling: window for an optional rolling mean overlay on time-series plots.

    centroid_cosine_matrix_csv:
        Optional path to centroid_cosine_matrix.csv produced by your geometry summary code.
    """
    metrics_csv = Path(metrics_csv)
    out_dir = Path(out_dir) / "geometry_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv, sep=None, engine="python")
    if run_id is not None:
        df = df[df["run_id"] == run_id].copy()

    df = _maybe_add_dt(df)

    # -----------------------------------------
    # NEW: attach between-group centroid metric
    # -----------------------------------------
    M = None
    if centroid_cosine_matrix_csv is not None:
        centroid_cosine_matrix_csv = Path(centroid_cosine_matrix_csv)
        if centroid_cosine_matrix_csv.exists():
            M = _read_centroid_cosine_matrix(centroid_cosine_matrix_csv)

            if run_id is not None:
                keep_mask = [idx.startswith(f"{run_id}::") for idx in M.index]
                keep = M.index[np.array(keep_mask)]
                M = M.loc[keep, keep]

            df = _attach_centroid_cosine_summary(df, M)

            # save augmented metrics
            df.to_csv(out_dir / "geometry_metrics_used.csv", index=False)

            # matrix visualizations
            _plot_matrix_heatmap(
                M,
                out_dir / "heatmap_centroid_cosine_matrix",
                title="Centroid cosine similarity between deployments",
            )
            _plot_offdiag_hist(
                M,
                out_dir / "hist_centroid_cosine_offdiag",
                bins=32,
            )
        else:
            df.to_csv(out_dir / "geometry_metrics_used.csv", index=False)
    else:
        df.to_csv(out_dir / "geometry_metrics_used.csv", index=False)

    # ---- Time series (ordered deployments)
    _plot_series(
        df, "centroid_norm",
        out_dir / "ts_centroid_norm",
        title="Centroid norm across deployments (higher = more concentrated / homogeneous)",
        ylabel="centroid_norm",
        add_rolling=rolling,
        annotate=True,
    )

    if "mean_centroid_cosine_to_others" in df.columns:
        _plot_series(
            df, "mean_centroid_cosine_to_others",
            out_dir / "ts_mean_centroid_cosine_to_others",
            title="Mean centroid cosine to other deployments (higher = more directionally typical)",
            ylabel="mean_centroid_cosine_to_others",
            add_rolling=rolling,
            annotate=True,
        )

    _plot_series(
        df, "cos_p10",
        out_dir / "ts_cos_p10",
        title="Cosine-to-centroid p10 (lower = heavier tail / more diverse)",
        ylabel="cos_p10",
        add_rolling=rolling,
        annotate=True,
    )
    _plot_series(
        df, "cos_std",
        out_dir / "ts_cos_std",
        title="Cosine-to-centroid spread (std; higher = more variation)",
        ylabel="cos_std",
        add_rolling=rolling,
        annotate=True,
    )
    _plot_series(
        df, "pair_cos_p50",
        out_dir / "ts_pair_cos_p50",
        title="Median pairwise cosine similarity (higher = more repetitive within deployment)",
        ylabel="pair_cos_p50",
        add_rolling=rolling,
        annotate=True,
    )
    _plot_series(
        df, "eff_rank",
        out_dir / "ts_eff_rank",
        title="Effective rank (higher = more complex variability)",
        ylabel="eff_rank",
        add_rolling=rolling,
        annotate=True,
    )
    _plot_series(
        df, "pca_dim_90",
        out_dir / "ts_pca_dim_90",
        title="Intrinsic complexity proxy: #PCs for 90% variance",
        ylabel="pca_dim_90",
        add_rolling=rolling,
        annotate=True,
    )

    # ---- Scatter relationships
    if "trace_cov" in df.columns:
        _plot_scatter(
            df, "centroid_norm", "trace_cov",
            out_dir / "sc_centroid_vs_trace",
            title="Concentration vs spread (unit-sphere: should be inversely related)",
            xlabel="centroid_norm",
            ylabel="trace_cov",
            annotate=True,
        )

    _plot_scatter(
        df, "centroid_norm", "pair_cos_p50",
        out_dir / "sc_centroid_vs_pair_p50",
        title="Concentration vs repetitiveness",
        xlabel="centroid_norm",
        ylabel="pair_cos_p50",
        annotate=True,
    )
    _plot_scatter(
        df, "centroid_norm", "eff_rank",
        out_dir / "sc_centroid_vs_effrank",
        title="Concentration vs complexity",
        xlabel="centroid_norm",
        ylabel="eff_rank",
        annotate=True,
    )
    _plot_scatter(
        df, "pair_cos_p50", "eff_rank",
        out_dir / "sc_pair_p50_vs_effrank",
        title="Repetitiveness vs complexity",
        xlabel="pair_cos_p50",
        ylabel="eff_rank",
        annotate=True,
    )
    _plot_scatter(
        df, "centroid_norm", "cos_p10",
        out_dir / "sc_centroid_vs_cos_p10",
        title="Concentration vs diversity",
        xlabel="centroid_norm",
        ylabel="cos_p10",
        annotate=True,
    )

    if "mean_centroid_cosine_to_others" in df.columns:
        _plot_scatter(
            df, "mean_centroid_cosine_to_others", "centroid_norm",
            out_dir / "sc_mean_centroid_cosine_vs_centroid_norm",
            title="Directional typicality vs concentration",
            xlabel="mean_centroid_cosine_to_others",
            ylabel="centroid_norm",
            annotate=True,
        )
        _plot_scatter(
            df, "mean_centroid_cosine_to_others", "eff_rank",
            out_dir / "sc_mean_centroid_cosine_vs_effrank",
            title="Directional typicality vs complexity",
            xlabel="mean_centroid_cosine_to_others",
            ylabel="eff_rank",
            annotate=True,
        )
        _plot_scatter(
            df, "mean_centroid_cosine_to_others", "pair_cos_p50",
            out_dir / "sc_mean_centroid_cosine_vs_pair_p50",
            title="Directional typicality vs repetitiveness",
            xlabel="mean_centroid_cosine_to_others",
            ylabel="pair_cos_p50",
            annotate=True,
        )

    # ---- Histograms
    _plot_hist(
        df, "centroid_norm",
        out_dir / "hist_centroid_norm",
        title="Distribution of centroid_norm across deployments",
        xlabel="centroid_norm",
        bins=28,
    )
    _plot_hist(
        df, "pair_cos_p50",
        out_dir / "hist_pair_cos_p50",
        title="Distribution of median pairwise cosine similarity",
        xlabel="pair_cos_p50",
        bins=28,
    )
    _plot_hist(
        df, "eff_rank",
        out_dir / "hist_eff_rank",
        title="Distribution of effective rank across deployments",
        xlabel="eff_rank",
        bins=28,
    )

    if "mean_centroid_cosine_to_others" in df.columns:
        _plot_hist(
            df, "mean_centroid_cosine_to_others",
            out_dir / "hist_mean_centroid_cosine_to_others",
            title="Distribution of mean centroid cosine to others",
            xlabel="mean_centroid_cosine_to_others",
            bins=28,
        )

    plot_all_timeseries_together(
        df,
        out_dir / "ts_all_metrics",
        rolling=rolling,
    )


if __name__ == "__main__":
    visualize_geometry_metrics(
        metrics_csv=r"C:\alr4\analysis\geometry_all\geometry_metrics.csv",
        out_dir=r"C:\alr4\analysis\geometry_all",
        run_id=None,
        rolling=7,
        centroid_cosine_matrix_csv=r"C:\alr4\analysis\geometry_all\centroid_cosine_matrix.csv",
    )