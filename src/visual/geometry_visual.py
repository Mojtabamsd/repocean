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
    # annotate top k and bottom k
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

    # size proportional to sampled (but tame it)
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
        # annotate extremes on y
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


def visualize_geometry_metrics(
    metrics_csv: str | Path,
    out_dir: str | Path,
    run_id: str | None = None,
    rolling: int = 7,
):
    """
    rolling: window for an optional rolling mean overlay on time-series plots.
    """
    metrics_csv = Path(metrics_csv)
    out_dir = Path(out_dir) / "geometry_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv, sep=None, engine="python")
    if run_id is not None:
        df = df[df["run_id"] == run_id].copy()

    # keep as-is (no reordering), but optionally add dt column for later use
    df = _maybe_add_dt(df)
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


if __name__ == "__main__":
    visualize_geometry_metrics(
        metrics_csv=r"C:\alr4\analysis\geometry\geometry_metrics.csv",
        out_dir=r"C:\alr4\analysis\geometry",
        run_id=None,   # or "prediction_parti20260119121141"
        rolling=7,     # set 0 to disable rolling mean overlay
    )