from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

_gid_re = re.compile(
    r"^(?P<cruise>[^_]+)_(?P<date>\d{8})_(?P<hhmm>\d{4})_(?P<seg>\d{4})_d(?P<dep>\d+)$"
)


def _shorten_sample_id(raw: str) -> str:
    """
    Shorten long sample_id strings by keeping only the 3rd chunk.

    Example:
      'alr004_20251001_0012_0001_d0001' -> '0012'
    """
    if raw is None:
        return ""
    s = str(raw)
    parts = s.split("_")
    if len(parts) >= 3:
        val = parts[2]
        try:
            return f"{int(val):04d}"
        except ValueError:
            return val
    return s


def _with_group_id_short(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "group_id" in out.columns and "group_id_short" not in out.columns:
        out["group_id_short"] = out["group_id"].map(_shorten_sample_id)
    return out


def _get_label_col(df: pd.DataFrame, use_group_id_short: bool = False) -> str:
    if use_group_id_short and "group_id_short" in df.columns:
        return "group_id_short"
    return "group_id"


def _maybe_add_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" in df.columns:
        return df
    dts = []
    for gid in df["group_id"].astype(str).tolist():
        m = _gid_re.match(gid)
        if m:
            dt = pd.to_datetime(
                m.group("date") + m.group("hhmm"),
                format="%Y%m%d%H%M",
                errors="coerce",
            )
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
    good = np.isfinite(y)
    if good.sum() < (2 * k + 1):
        return
    idx = np.where(good)[0]
    order = idx[np.argsort(y[good])]
    picks = np.concatenate([order[:k], order[-k:]])
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
    use_group_id_short: bool = False,
):
    if col not in df.columns:
        return

    x = np.arange(len(df), dtype=int)
    y = pd.to_numeric(df[col], errors="coerce").values
    label_col = _get_label_col(df, use_group_id_short=use_group_id_short)
    labels = df[label_col].astype(str).tolist()

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
    id_col: str = "group_id",
    use_group_id_short: bool = False,
):
    if xcol not in df.columns or ycol not in df.columns:
        return

    x = pd.to_numeric(df[xcol], errors="coerce").values
    y = pd.to_numeric(df[ycol], errors="coerce").values

    label_col = id_col
    if id_col == "group_id" and use_group_id_short and "group_id_short" in df.columns:
        label_col = "group_id_short"

    labels = df[label_col].astype(str).tolist() if label_col in df.columns else [str(i) for i in range(len(df))]

    if "sampled" in df.columns:
        s = pd.to_numeric(df["sampled"], errors="coerce").fillna(0.0).values
        s_pos = s[s > 0]
        if len(s_pos) > 0:
            clip_hi = np.nanpercentile(s_pos, 95)
            s = np.clip(s, 0, clip_hi)
            sizes = 18 + 70 * (s / (s.max() if s.max() > 0 else 1.0))
        else:
            sizes = np.full(len(df), 30.0)
    else:
        sizes = np.full(len(df), 30.0)

    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=sizes, alpha=0.7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    _set_pretty_axes(ax)

    if annotate and len(df) >= 8:
        good = np.isfinite(x) & np.isfinite(y)
        if good.any():
            idx = np.where(good)[0]
            score = pd.Series(
                np.abs((x[good] - np.nanmean(x[good])) / (np.nanstd(x[good]) + 1e-12)) +
                np.abs((y[good] - np.nanmean(y[good])) / (np.nanstd(y[good]) + 1e-12))
            ).values
            pick = idx[np.argsort(score)[-6:]]
            for i in pick:
                ax.annotate(labels[i], (x[i], y[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    _save(fig, out_path)


def _plot_nmds_colored(
    df: pd.DataFrame,
    out_path: Path,
    color_col: str = "centroid_norm",
    title: str = "NMDS of deployments coloured by centroid norm",
    annotate: bool = True,
    use_group_id_short: bool = False,
):
    if "nmds1" not in df.columns or "nmds2" not in df.columns or color_col not in df.columns:
        return

    x = pd.to_numeric(df["nmds1"], errors="coerce").values
    y = pd.to_numeric(df["nmds2"], errors="coerce").values
    c = pd.to_numeric(df[color_col], errors="coerce").values

    label_col = _get_label_col(df, use_group_id_short=use_group_id_short)
    labels = df[label_col].astype(str).tolist()

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if good.sum() == 0:
        return

    if "sampled" in df.columns:
        s = pd.to_numeric(df["sampled"], errors="coerce").fillna(0.0).values
        s_pos = s[s > 0]
        if len(s_pos) > 0:
            clip_hi = np.nanpercentile(s_pos, 95)
            s = np.clip(s, 0, clip_hi)
            sizes = 24 + 80 * (s / (s.max() if s.max() > 0 else 1.0))
        else:
            sizes = np.full(len(df), 36.0)
    else:
        sizes = np.full(len(df), 36.0)

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        x[good],
        y[good],
        c=c[good],
        s=sizes[good],
        alpha=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_col, fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("NMDS1", fontsize=10)
    ax.set_ylabel("NMDS2", fontsize=10)
    _set_pretty_axes(ax)

    # nice reference lines
    ax.axhline(0, linewidth=0.8, alpha=0.25)
    ax.axvline(0, linewidth=0.8, alpha=0.25)

    if annotate and good.sum() >= 8:
        idx = np.where(good)[0]
        score = (
            np.abs((x[good] - np.nanmean(x[good])) / (np.nanstd(x[good]) + 1e-12)) +
            np.abs((y[good] - np.nanmean(y[good])) / (np.nanstd(y[good]) + 1e-12))
        )
        pick = idx[np.argsort(score)[-8:]]
        for i in pick:
            ax.annotate(
                labels[i],
                (x[i], y[i]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    _save(fig, out_path)

def _plot_hist(df: pd.DataFrame, col: str, out_path: Path, title: str, xlabel: str, bins: int = 30):
    if col not in df.columns:
        return
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values

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
    """
    IMPORTANT:
    Keep full group_id here for safe merging with centroid matrix.
    Do NOT use group_id_short here, because it may not be unique.
    """
    def _format_group_id(val):
        return str(val)

    out = df.copy()
    out["group_key"] = (
        out["run_id"].astype(str)
        + "::"
        + out["group_id"].map(_format_group_id)
    )
    mean_sim = _compute_mean_centroid_cosine_to_others(M)
    return out.merge(
        mean_sim.rename("mean_centroid_cosine_to_others"),
        left_on="group_key",
        right_index=True,
        how="left",
        validate="one_to_one",
    )


def _cluster_order_from_similarity(M: pd.DataFrame, method: str = "average") -> np.ndarray:
    A = M.astype(float).values.copy()
    A = np.clip(A, -1.0, 1.0)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    d_condensed = squareform(D, checks=False)
    Z = linkage(d_condensed, method=method)
    return leaves_list(Z)


def _plot_clustered_centroid_heatmap(
    M: pd.DataFrame,
    out_path: Path,
    title: str = "Clustered centroid cosine similarity between deployments",
    method: str = "average",
    use_group_id_short: bool = False,
):
    A = M.astype(float).values.copy()
    n = A.shape[0]
    if n == 0:
        return

    # order or not
    order_by_similarity = True
    if order_by_similarity:
        if n > 1:
            order = _cluster_order_from_similarity(M, method=method)
            M = M.iloc[order, order]

    A_ord = M.astype(float).values

    raw_labels = [l.split("::")[-1] for l in M.index.astype(str).tolist()]
    if use_group_id_short:
        labels = [_shorten_sample_id(l) for l in raw_labels]
    else:
        labels = raw_labels

    fig_w = max(8, min(18, 0.28 * n + 5))
    fig_h = max(7, min(16, 0.28 * n + 4))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    im = ax.imshow(A_ord, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("centroid cosine", fontsize=10)
    ax.set_title(title, fontsize=12)

    if n <= 200:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Deployments (clustered order)", fontsize=10)
        ax.set_ylabel("Deployments (clustered order)", fontsize=10)

    _save(fig, out_path)


def _plot_centroid_dendrogram(
    M: pd.DataFrame,
    out_path: Path,
    title: str = "Hierarchical clustering of deployments by centroid direction",
    method: str = "average",
    use_group_id_short: bool = False,
):
    A = M.astype(float).values.copy()
    n = A.shape[0]
    if n <= 1:
        return

    A = np.clip(A, -1.0, 1.0)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    d_condensed = squareform(D, checks=False)
    Z = linkage(d_condensed, method=method)

    fig_w = max(10, min(22, 0.18 * n + 8))
    fig_h = 5.8 if n < 120 else 7.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    raw_labels = [l.split("::")[-1] for l in M.index.astype(str).tolist()]
    labels = [_shorten_sample_id(l) for l in raw_labels] if use_group_id_short else raw_labels

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6 if n > 80 else 7,
        ax=ax,
        color_threshold=None,
    )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Distance = 1 - centroid cosine", fontsize=10)
    ax.set_xlabel("Deployments", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6 if n > 80 else 7)
    ax.tick_params(axis="y", labelsize=9)
    _save(fig, out_path)


def _plot_offdiag_hist(M: pd.DataFrame, out_path: Path, bins: int = 30):
    A = M.astype(float).values
    vals = A[~np.eye(A.shape[0], dtype=bool)] if A.shape[0] >= 2 else np.array([], dtype=float)
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=bins, alpha=0.9)
    ax.set_title("Distribution of centroid cosine between deployments", fontsize=12)
    ax.set_xlabel("centroid cosine (off-diagonal)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    _set_pretty_axes(ax)
    _save(fig, out_path)


def _make_exp_shannon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "exp_shannon" not in out.columns and "shannon" in out.columns:
        out["exp_shannon"] = np.exp(pd.to_numeric(out["shannon"], errors="coerce"))
    return out


def _plot_all_timeseries_together(
    df: pd.DataFrame,
    out_path: Path,
    rolling: int = 7,
    use_group_id_short: bool = False,
    xtick_every: int = 5,
):
    df = _make_exp_shannon(df)
    metrics = [
        ("centroid_norm", "Centroid norm\n(homogeneity ↑)"),
        ("eff_rank", "Effective rank\n(complexity ↑)"),
        # ("exp_shannon", "exp(Shannon)\n(effective #classes ↑)"),
        ("cos_p10", "Cosine p10\n(diversity tail ↓)"),
        ("mean_centroid_cosine_to_others", "Mean centroid cosine\nto others ↑"),
        # ("pair_cos_p50", "Pairwise cosine p50\n(repetitiveness ↑)"),
        # ("shannon", "Shannon entropy\n(alpha diversity ↑)"),
        # ("pca_dim_90", "PCA dim 90%\n(complexity ↑)"),
    ]
    metrics = [(c, y) for c, y in metrics if c in df.columns]
    if not metrics:
        return

    x = np.arange(len(df), dtype=int)
    label_col = _get_label_col(df, use_group_id_short=use_group_id_short)
    xlabels = df[label_col].astype(str).tolist()

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1, figsize=(12, 1.55 * len(metrics) + 2.0),
        sharex=True, constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, ylabel) in zip(axes, metrics):
        y = pd.to_numeric(df[col], errors="coerce").values
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.1, alpha=0.9)
        if rolling and rolling >= 3 and len(y) >= rolling:
            roll = pd.Series(y).rolling(rolling, center=True, min_periods=max(3, rolling // 3)).mean().values
            ax.plot(x, roll, linewidth=2.0, alpha=0.9)
        _set_pretty_axes(ax)
        ax.set_ylabel(ylabel, fontsize=9)

    step = max(1, xtick_every)
    tick_idx = list(range(0, len(x), step))
    axes[-1].set_xticks(tick_idx)
    axes[-1].set_xticklabels([xlabels[i] for i in tick_idx], rotation=90)
    axes[-1].set_xlabel("Deployment / sample", fontsize=10)
    fig.suptitle("Representation geometry", fontsize=13)
    _save(fig, out_path)


def _plot_all_timeseries_with_dual_shannon(
    df: pd.DataFrame,
    out_path: Path,
    rolling: int = 7,
    use_group_id_short: bool = False,
    xtick_every: int = 5,
):
    """
    Similar to ts_all_metrics_combined, but if tax_shannon exists and is not all NaN,
    add one extra bottom subplot that overlays:
        - shannon           (prediction)
        - tax_shannon       (taxonomist)
    """
    df = _make_exp_shannon(df)

    metrics = [
        ("centroid_norm", "Centroid norm\n(homogeneity ↑)"),
        ("eff_rank", "Effective rank\n(complexity ↑)"),
        ("cos_p10", "Cosine p10\n(diversity tail ↓)"),
        ("mean_centroid_cosine_to_others", "Mean centroid cosine\nto others ↑"),
    ]
    metrics = [(c, y) for c, y in metrics if c in df.columns]

    has_pred_shannon = "shannon" in df.columns
    has_tax_shannon = "tax_shannon" in df.columns and pd.to_numeric(
        df["tax_shannon"], errors="coerce"
    ).notna().any()

    if not metrics:
        return
    if not (has_pred_shannon or has_tax_shannon):
        return

    x = np.arange(len(df), dtype=int)
    label_col = _get_label_col(df, use_group_id_short=use_group_id_short)
    xlabels = df[label_col].astype(str).tolist()

    nrows = len(metrics) + 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12, 1.55 * nrows + 2.0),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    # top geometry panels
    for ax, (col, ylabel) in zip(axes[:-1], metrics):
        y = pd.to_numeric(df[col], errors="coerce").values
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.1, alpha=0.9, label=col)

        if rolling and rolling >= 3 and len(y) >= rolling:
            roll = pd.Series(y).rolling(
                rolling,
                center=True,
                min_periods=max(3, rolling // 3)
            ).mean().values
            ax.plot(x, roll, linewidth=2.0, alpha=0.9)

        _set_pretty_axes(ax)
        ax.set_ylabel(ylabel, fontsize=9)

    # bottom dual-shannon panel
    ax = axes[-1]

    if has_pred_shannon:
        y_pred = pd.to_numeric(df["shannon"], errors="coerce").values
        ax.plot(
            x, y_pred,
            marker="o",
            markersize=2.4,
            linewidth=1.2,
            alpha=0.9,
            label="Prediction Shannon",
        )
        if rolling and rolling >= 3 and len(y_pred) >= rolling:
            roll_pred = pd.Series(y_pred).rolling(
                rolling,
                center=True,
                min_periods=max(3, rolling // 3)
            ).mean().values
            ax.plot(x, roll_pred, linewidth=2.0, alpha=0.9)

    if has_tax_shannon:
        y_tax = pd.to_numeric(df["tax_shannon"], errors="coerce").values
        ax.plot(
            x, y_tax,
            marker="s",
            markersize=2.6,
            linewidth=1.2,
            alpha=0.9,
            linestyle="--",
            label="Taxonomist Shannon",
        )
        if rolling and rolling >= 3 and len(y_tax) >= rolling:
            roll_tax = pd.Series(y_tax).rolling(
                rolling,
                center=True,
                min_periods=max(3, rolling // 3)
            ).mean().values
            ax.plot(x, roll_tax, linewidth=2.0, alpha=0.9, linestyle="--")

    _set_pretty_axes(ax)
    ax.set_ylabel("Shannon\n(alpha diversity ↑)", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="best")

    step = max(1, xtick_every)
    tick_idx = list(range(0, len(x), step))
    axes[-1].set_xticks(tick_idx)
    axes[-1].set_xticklabels([xlabels[i] for i in tick_idx], rotation=90)
    axes[-1].set_xlabel("Deployment / sample", fontsize=10)

    fig.suptitle("Representation geometry + Shannon comparison", fontsize=13)
    _save(fig, out_path)


def _plot_zscore_heatmap(df: pd.DataFrame, cols: list[str], out_path: Path, title: str,
                         use_group_id_short: bool = False,
                         xtick_every: int = 5):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    Z = pd.DataFrame(index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        Z[c] = (v - v.mean()) / (v.std(ddof=0) + 1e-12)

    data = Z.T.values
    fig, ax = plt.subplots(figsize=(12, max(2.6, 0.55 * len(cols) + 0.8)))
    im = ax.imshow(data, aspect="auto")
    ax.set_yticks(range(len(cols)))
    import textwrap

    wrapped_cols = [
        "\n".join(textwrap.wrap(str(c), width=14))
        for c in cols
    ]
    ax.set_yticklabels(wrapped_cols, fontsize=9)


    x = np.arange(len(df), dtype=int)
    label_col = _get_label_col(df, use_group_id_short=use_group_id_short)
    xlabels = df[label_col].astype(str).tolist()

    step = max(1, xtick_every)
    tick_idx = list(range(0, len(x), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([xlabels[i] for i in tick_idx], rotation=90)
    ax.set_xlabel("Deployment / sample", fontsize=10)

    # tick_idx = np.unique(np.linspace(0, max(0, Z.shape[0] - 1), min(10, max(1, Z.shape[0]))).astype(int))
    # ax.set_xticks(tick_idx)
    # ax.set_xticklabels(tick_idx, fontsize=9)
    # ax.set_title(title)
    ax.set_xlabel("Deployment / sample", fontsize=10)
    # ax.set_ylabel("Metric (z-score)", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score")
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_rank_overlay(df: pd.DataFrame, cols: list[str], out_path: Path, title: str):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    x = np.arange(len(df), dtype=int)
    ranks = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").rank(method="average") for c in cols})

    fig, ax = plt.subplots(figsize=(12, 4.2))
    for c in cols:
        ax.plot(x, ranks[c].values, linewidth=1.4, label=c)
    ax.set_title(title)
    ax.set_xlabel("Deployment index (ordered)")
    ax.set_ylabel("Rank (higher = larger value)")
    _set_pretty_axes(ax)
    ax.legend(frameon=False, ncol=min(len(cols), 3), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def visualize_geometry_metrics_merged(
    metrics_csv: str | Path,
    out_dir: str | Path,
    run_id: str | None = None,
    rolling: int = 7,
    centroid_cosine_matrix_csv: str | Path | None = None,
    keep_individual_timeseries: bool = True,
    use_group_id_short: bool = False,
):
    metrics_csv = Path(metrics_csv)
    out_dir = Path(out_dir) / "geometry_viz_merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv, sep=None, engine="python")
    if run_id is not None:
        df = df[df["run_id"] == run_id].copy()

    df = _maybe_add_dt(df)
    df = _make_exp_shannon(df)
    df = _with_group_id_short(df)

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

            _plot_clustered_centroid_heatmap(
                M,
                out_dir / "heatmap_centroid_cosine_matrix_clustered",
                title="Clustered centroid cosine similarity between deployments",
                method="average",
                use_group_id_short=use_group_id_short,
            )
            _plot_centroid_dendrogram(
                M,
                out_dir / "dendrogram_centroid_cosine",
                title="Hierarchical clustering of deployments by centroid direction",
                method="average",
                use_group_id_short=use_group_id_short,
            )
            _plot_offdiag_hist(M, out_dir / "hist_centroid_cosine_offdiag", bins=32)

    df.to_csv(out_dir / "geometry_metrics_used.csv", index=False)

    _plot_all_timeseries_together(
        df,
        out_dir / "ts_all_metrics_combined",
        rolling=rolling,
        use_group_id_short=use_group_id_short,
    )

    if "tax_shannon" in df.columns and pd.to_numeric(df["tax_shannon"], errors="coerce").notna().any():
        _plot_all_timeseries_with_dual_shannon(
            df,
            out_dir / "ts_all_metrics_combined_with_taxonomist_shannon",
            rolling=rolling,
            use_group_id_short=use_group_id_short,
        )

    if keep_individual_timeseries:
        core_ts = [
            ("centroid_norm", "Centroid norm across deployments", "centroid_norm"),
            ("shannon", "Shannon entropy across deployments", "shannon"),
            ("exp_shannon", "exp(Shannon) across deployments", "exp_shannon"),
            ("cos_p10", "Cosine-to-centroid p10 across deployments", "cos_p10"),
            ("pair_cos_p50", "Median pairwise cosine similarity across deployments", "pair_cos_p50"),
            ("eff_rank", "Effective rank across deployments", "eff_rank"),
            ("mean_centroid_cosine_to_others", "Mean centroid cosine to other deployments", "mean_centroid_cosine_to_others"),
        ]
        for col, title, ylabel in core_ts:
            if col in df.columns:
                _plot_series(
                    df, col, out_dir / f"ts_{col}",
                    title=title,
                    ylabel=ylabel,
                    add_rolling=rolling,
                    annotate=True,
                    use_group_id_short=use_group_id_short,
                )

    _plot_rank_overlay(
        df,
        cols=["shannon", "centroid_norm", "eff_rank"],
        out_path=out_dir / "rank_overlay",
        title="Rank overlay: Shannon vs centroid norm vs effective rank",
    )

    _plot_zscore_heatmap(
        df,
        cols=["centroid_norm", "eff_rank", "cos_p10", "mean_centroid_cosine_to_others"],
        out_path=out_dir / "heatmap_metric_zscores",
        title="Metric comparison heatmap across deployments",
        use_group_id_short=use_group_id_short
    )

    scatters = [
        ("centroid_norm", "shannon", "sc_centroid_vs_shannon", "Concentration vs Shannon diversity"),
        ("centroid_norm", "exp_shannon", "sc_centroid_vs_exp_shannon", "Concentration vs effective diversity"),
        ("centroid_norm", "eff_rank", "sc_centroid_vs_effrank", "Concentration vs complexity"),
        ("centroid_norm", "cos_p10", "sc_centroid_vs_cos_p10", "Concentration vs diversity tail"),
        ("pair_cos_p50", "eff_rank", "sc_pair_p50_vs_effrank", "Repetitiveness vs complexity"),
    ]
    if "mean_centroid_cosine_to_others" in df.columns:
        scatters.append(
            ("mean_centroid_cosine_to_others", "eff_rank", "sc_mean_centroid_cosine_vs_effrank", "Directional typicality vs complexity")
        )

    for xcol, ycol, stem, title in scatters:
        _plot_scatter(
            df, xcol, ycol, out_dir / stem,
            title=title,
            xlabel=xcol,
            ylabel=ycol,
            annotate=True,
            use_group_id_short=use_group_id_short,
        )

    _plot_nmds_colored(
        df,
        out_dir / "sc_nmds_colored_by_centroid_norm",
        color_col="centroid_norm",
        title="NMDS of deployments coloured by centroid norm",
        annotate=True,
        use_group_id_short=use_group_id_short,
    )

    for col in ["centroid_norm", "shannon", "exp_shannon", "pair_cos_p50", "eff_rank"]:
        if col in df.columns:
            _plot_hist(
                df, col, out_dir / f"hist_{col}",
                title=f"Distribution of {col} across deployments",
                xlabel=col,
                bins=28,
            )

    print(f"Saved merged plots to: {out_dir}")


if __name__ == "__main__":
    # path = r'C:\alr4\analysis\partitrics\uvp6net\geometry'
    # path = r'C:\alr4\analysis\geometry'
    path = r'C:\alr4\analysis\geometry_all'
    # path = r'C:\alr4\analysis\geometry\ALL'
    visualize_geometry_metrics_merged(
        metrics_csv=path + r"\geometry_metrics.csv",
        out_dir=path,
        run_id=None,
        rolling=4,
        centroid_cosine_matrix_csv=path + r"\centroid_cosine_matrix.csv",
        keep_individual_timeseries=True,
        use_group_id_short=True,   # <- set False to use full group_id labels
    )