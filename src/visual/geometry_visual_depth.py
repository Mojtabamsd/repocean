from __future__ import annotations

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

_DEPTH_BIN_RE = re.compile(r"(?P<lo>\d+)\s*-\s*(?P<hi>\d+)\s*m", flags=re.IGNORECASE)


def _save(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_pretty_axes(ax: plt.Axes):
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))


def _shorten_sample_id(raw: str) -> str:
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


def _parse_depth_bin_mid(depth_bin: str) -> float:
    if depth_bin is None or pd.isna(depth_bin):
        return np.nan
    s = str(depth_bin)
    m = _DEPTH_BIN_RE.search(s)
    if not m:
        return np.nan
    lo = float(m.group("lo"))
    hi = float(m.group("hi"))
    return 0.5 * (lo + hi)


def _parse_profile_and_depth_from_group_id(group_id: str) -> tuple[str | float, str | float]:
    """
    Expect something like:
        alr004_20251001_0012_0001_d0001__000-010m
    """
    if group_id is None or pd.isna(group_id):
        return np.nan, np.nan

    s = str(group_id)
    if "__" in s:
        left, right = s.rsplit("__", 1)
        return left, right

    return np.nan, np.nan


def _ensure_depth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "profile_id" not in out.columns or out["profile_id"].isna().all():
        profs = []
        bins = []
        for gid in out["group_id"].astype(str).tolist():
            p, b = _parse_profile_and_depth_from_group_id(gid)
            profs.append(p)
            bins.append(b)

        if "profile_id" not in out.columns:
            out["profile_id"] = profs
        else:
            out["profile_id"] = out["profile_id"].where(out["profile_id"].notna(), pd.Series(profs, index=out.index))

        if "depth_bin" not in out.columns:
            out["depth_bin"] = bins
        else:
            out["depth_bin"] = out["depth_bin"].where(out["depth_bin"].notna(), pd.Series(bins, index=out.index))

    if "depth_mid" not in out.columns:
        out["depth_mid"] = out["depth_bin"].apply(_parse_depth_bin_mid)
    else:
        miss = pd.to_numeric(out["depth_mid"], errors="coerce").isna()
        out.loc[miss, "depth_mid"] = out.loc[miss, "depth_bin"].apply(_parse_depth_bin_mid)

    if "profile_id_short" not in out.columns and "profile_id" in out.columns:
        out["profile_id_short"] = out["profile_id"].apply(_shorten_sample_id)

    return out


def _choose_profile_label_col(df: pd.DataFrame, use_profile_id_short: bool = True) -> str:
    if use_profile_id_short and "profile_id_short" in df.columns:
        return "profile_id_short"
    return "profile_id"


# ---------------------------------------------------------------------
# plotting functions
# ---------------------------------------------------------------------

def _plot_depth_profiles_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    use_profile_id_short: bool = True,
    max_profiles_legend: int = 25,
):
    if metric_col not in df.columns:
        return

    prof_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    sub = df.copy()
    sub["depth_mid"] = pd.to_numeric(sub["depth_mid"], errors="coerce")
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub["depth_mid"].notna() & sub[metric_col].notna() & sub[prof_col].notna()].copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(9.0, 5.8))
    ax = fig.add_subplot(111)

    profiles = sorted(sub[prof_col].astype(str).unique().tolist())
    for p in profiles:
        s = sub[sub[prof_col].astype(str) == p].sort_values("depth_mid")
        if len(s) == 0:
            continue
        ax.plot(
            s["depth_mid"].values,
            s[metric_col].values,
            marker="o",
            linewidth=1.1,
            markersize=3.0,
            alpha=0.85,
            label=p,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Depth midpoint (m)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    _set_pretty_axes(ax)
    ax.invert_xaxis() if False else None  # keep normal increasing depth left->right by default

    if len(profiles) <= max_profiles_legend:
        ax.legend(frameon=False, fontsize=8, ncol=min(4, max(1, math.ceil(len(profiles) / 8))))

    _save(fig, out_path)


def _plot_depth_profile_panels_per_profile(
    df: pd.DataFrame,
    out_dir: Path,
    use_profile_id_short: bool = True,
):
    prof_label_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    metrics = [
        ("centroid_norm", "Centroid norm\n(homogeneity ↑)"),
        ("eff_rank", "Effective rank\n(complexity ↑)"),
        ("cos_p10", "Cosine p10\n(diversity tail ↓)"),
    ]

    has_pred_shannon = "shannon" in df.columns and pd.to_numeric(df["shannon"], errors="coerce").notna().any()
    has_tax_shannon = "tax_shannon" in df.columns and pd.to_numeric(df["tax_shannon"], errors="coerce").notna().any()

    profiles = df["profile_id"].dropna().astype(str).unique().tolist()
    profiles = sorted(profiles)

    for prof in profiles:
        sub = df[df["profile_id"].astype(str) == prof].copy()
        if sub.empty:
            continue

        sub["depth_mid"] = pd.to_numeric(sub["depth_mid"], errors="coerce")
        sub = sub[sub["depth_mid"].notna()].sort_values("depth_mid")
        if sub.empty:
            continue

        nrows = len(metrics) + (1 if (has_pred_shannon or has_tax_shannon) else 0)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(8.2, 1.8 * nrows + 1.8),
            sharex=True,
            constrained_layout=True,
        )
        if nrows == 1:
            axes = [axes]

        # top metric panels
        for ax, (col, ylabel) in zip(axes[:len(metrics)], metrics):
            if col not in sub.columns:
                ax.axis("off")
                continue

            y = pd.to_numeric(sub[col], errors="coerce").values
            ax.plot(sub["depth_mid"].values, y, marker="o", linewidth=1.2, markersize=3.0, alpha=0.9)
            _set_pretty_axes(ax)
            ax.set_ylabel(ylabel, fontsize=9)

        # bottom shannon panel
        if has_pred_shannon or has_tax_shannon:
            ax = axes[-1]
            if has_pred_shannon and "shannon" in sub.columns:
                y = pd.to_numeric(sub["shannon"], errors="coerce").values
                ax.plot(
                    sub["depth_mid"].values,
                    y,
                    marker="o",
                    linewidth=1.2,
                    markersize=3.0,
                    alpha=0.9,
                    label="Prediction Shannon",
                )

            if has_tax_shannon and "tax_shannon" in sub.columns:
                y = pd.to_numeric(sub["tax_shannon"], errors="coerce").values
                ax.plot(
                    sub["depth_mid"].values,
                    y,
                    marker="s",
                    linestyle="--",
                    linewidth=1.2,
                    markersize=3.2,
                    alpha=0.9,
                    label="Taxonomist Shannon",
                )

            _set_pretty_axes(ax)
            ax.set_ylabel("Shannon\n(alpha diversity ↑)", fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="best")

        axes[-1].set_xlabel("Depth midpoint (m)", fontsize=10)

        prof_label = sub[prof_label_col].iloc[0] if prof_label_col in sub.columns else prof
        fig.suptitle(f"Depth profile: {prof_label}", fontsize=12)

        stem = f"profile_{str(prof_label).replace('/', '_').replace(' ', '_')}_depth_panels"
        _save(fig, out_dir / stem)


def _plot_metric_heatmap_profile_depth(
    df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    use_profile_id_short: bool = True,
    row_zscore: bool = False,
):
    if metric_col not in df.columns:
        return

    prof_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    sub = df.copy()
    sub["depth_mid"] = pd.to_numeric(sub["depth_mid"], errors="coerce")
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub["depth_mid"].notna() & sub[metric_col].notna() & sub[prof_col].notna()].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index=prof_col,
        columns="depth_mid",
        values=metric_col,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    if pivot.empty:
        return

    data = pivot.values.astype(float)

    if row_zscore:
        mu = np.nanmean(data, axis=1, keepdims=True)
        sd = np.nanstd(data, axis=1, keepdims=True) + 1e-12
        data = (data - mu) / sd

    fig_h = max(4.0, min(14.0, 0.35 * pivot.shape[0] + 2.4))
    fig_w = max(7.0, min(16.0, 0.45 * pivot.shape[1] + 4.0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=12)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str).tolist(), fontsize=8)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns], rotation=90, fontsize=8)

    ax.set_xlabel("Depth midpoint (m)", fontsize=10)
    ax.set_ylabel("Profile", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"{metric_col}" + (" (row z-score)" if row_zscore else ""), fontsize=9)

    _save(fig, out_path)


def _plot_nmds_depth_colored(
    df: pd.DataFrame,
    out_path: Path,
    color_col: str = "depth_mid",
    title: str = "NMDS of profile-depth groups coloured by depth",
    annotate: bool = False,
    use_profile_id_short: bool = True,
):
    if "nmds1" not in df.columns or "nmds2" not in df.columns or color_col not in df.columns:
        return

    prof_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    sub = df.copy()
    sub["nmds1"] = pd.to_numeric(sub["nmds1"], errors="coerce")
    sub["nmds2"] = pd.to_numeric(sub["nmds2"], errors="coerce")
    sub[color_col] = pd.to_numeric(sub[color_col], errors="coerce")
    sub = sub[sub["nmds1"].notna() & sub["nmds2"].notna() & sub[color_col].notna()].copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        sub["nmds1"].values,
        sub["nmds2"].values,
        c=sub[color_col].values,
        s=38,
        alpha=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_col, fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("NMDS1", fontsize=10)
    ax.set_ylabel("NMDS2", fontsize=10)
    _set_pretty_axes(ax)
    ax.axhline(0, linewidth=0.8, alpha=0.25)
    ax.axvline(0, linewidth=0.8, alpha=0.25)

    if annotate:
        for _, r in sub.iterrows():
            lab = str(r.get(prof_col, r.get("profile_id", "")))
            dmid = r.get("depth_mid", np.nan)
            ax.annotate(f"{lab}:{dmid:.0f}", (r["nmds1"], r["nmds2"]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    _save(fig, out_path)


def _plot_nmds_profile_paths(
    df: pd.DataFrame,
    out_path: Path,
    use_profile_id_short: bool = True,
    annotate_endpoints: bool = True,
):
    if "nmds1" not in df.columns or "nmds2" not in df.columns:
        return

    prof_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    sub = df.copy()
    sub["nmds1"] = pd.to_numeric(sub["nmds1"], errors="coerce")
    sub["nmds2"] = pd.to_numeric(sub["nmds2"], errors="coerce")
    sub["depth_mid"] = pd.to_numeric(sub["depth_mid"], errors="coerce")
    sub = sub[sub["nmds1"].notna() & sub["nmds2"].notna() & sub["depth_mid"].notna() & sub["profile_id"].notna()].copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111)

    profiles = sorted(sub["profile_id"].astype(str).unique().tolist())
    for prof in profiles:
        s = sub[sub["profile_id"].astype(str) == prof].sort_values("depth_mid")
        if len(s) == 0:
            continue

        ax.plot(
            s["nmds1"].values,
            s["nmds2"].values,
            marker="o",
            linewidth=1.1,
            markersize=3.0,
            alpha=0.85,
        )

        if annotate_endpoints:
            lab = s[prof_col].iloc[0] if prof_col in s.columns else prof
            first = s.iloc[0]
            last = s.iloc[-1]
            ax.annotate(f"{lab} start", (first["nmds1"], first["nmds2"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.annotate(f"{lab} end", (last["nmds1"], last["nmds2"]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_title("NMDS profile trajectories across depth", fontsize=12)
    ax.set_xlabel("NMDS1", fontsize=10)
    ax.set_ylabel("NMDS2", fontsize=10)
    _set_pretty_axes(ax)
    ax.axhline(0, linewidth=0.8, alpha=0.25)
    ax.axvline(0, linewidth=0.8, alpha=0.25)

    _save(fig, out_path)


def _plot_vertical_turnover(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str = "centroid_norm",
    use_profile_id_short: bool = True,
):
    if metric_col not in df.columns:
        return

    prof_col = _choose_profile_label_col(df, use_profile_id_short=use_profile_id_short)

    sub = df.copy()
    sub["depth_mid"] = pd.to_numeric(sub["depth_mid"], errors="coerce")
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub["depth_mid"].notna() & sub[metric_col].notna() & sub["profile_id"].notna()].copy()
    if sub.empty:
        return

    rows = []
    for prof, s in sub.groupby("profile_id"):
        s = s.sort_values("depth_mid").reset_index(drop=True)
        if len(s) < 2:
            continue
        for i in range(len(s) - 1):
            a = s.iloc[i]
            b = s.iloc[i + 1]
            rows.append({
                "profile_id": prof,
                "profile_label": a.get(prof_col, prof),
                "depth_mid_from": a["depth_mid"],
                "depth_mid_to": b["depth_mid"],
                "transition_mid": 0.5 * (a["depth_mid"] + b["depth_mid"]),
                "delta_metric": float(b[metric_col] - a[metric_col]),
                "abs_delta_metric": float(abs(b[metric_col] - a[metric_col])),
            })

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        return

    fig = plt.figure(figsize=(8.0, 5.5))
    ax = fig.add_subplot(111)

    for prof, s in tdf.groupby("profile_label"):
        s = s.sort_values("transition_mid")
        ax.plot(
            s["transition_mid"].values,
            s["abs_delta_metric"].values,
            marker="o",
            linewidth=1.1,
            markersize=3.0,
            alpha=0.85,
            label=str(prof),
        )

    ax.set_title(f"Vertical turnover: |Δ {metric_col}| between adjacent depth bins", fontsize=12)
    ax.set_xlabel("Transition midpoint depth (m)", fontsize=10)
    ax.set_ylabel(f"|Δ {metric_col}|", fontsize=10)
    _set_pretty_axes(ax)

    if tdf["profile_label"].nunique() <= 20:
        ax.legend(frameon=False, fontsize=8, ncol=min(4, max(1, math.ceil(tdf['profile_label'].nunique() / 8))))

    _save(fig, out_path)


# ---------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------

def visualize_depth_metrics(
    metrics_csv: str | Path,
    out_dir: str | Path,
    run_id: str | None = None,
    use_profile_id_short: bool = True,
    min_images_per_group: int = 10,
):
    metrics_csv = Path(metrics_csv)
    out_dir = Path(out_dir) / "depth_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv, sep=None, engine="python")
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id].copy()

    df = _ensure_depth_columns(df)

    # keep only rows with usable depth/profile
    if "profile_id" not in df.columns or "depth_mid" not in df.columns:
        raise ValueError(
            "Could not find or derive 'profile_id' and 'depth_mid'. "
            "Need either explicit columns or group_id like profile__000-010m."
        )

    # -------------------------------------------------
    # Filter profiles by total number of images
    # -------------------------------------------------
    if "num_rows" not in df.columns:
        raise ValueError("Expected 'num_rows' column in geometry_metrics.csv")

    group_counts = (
        df.groupby("group_id")["num_rows"]
            .sum()
            .rename("total_images")
    )

    keep_groups = group_counts[group_counts >= min_images_per_group].index

    df = df[df["group_id"].isin(keep_groups)].copy()

    if df.empty:
        raise ValueError(
            f"No groups left after filtering with min_images_per_group={min_images_per_group}"
        )

    dropped = set(group_counts.index) - set(keep_groups)
    print(
        f"[depth_viz] kept {len(keep_groups)} groups, dropped {len(dropped)} (threshold={min_images_per_group})")

    df["depth_mid"] = pd.to_numeric(df["depth_mid"], errors="coerce")
    df = df[df["profile_id"].notna() & df["depth_mid"].notna()].copy()
    if df.empty:
        raise ValueError("No valid depth/profile rows found after parsing.")

    # save cleaned table used for plotting
    df = df.sort_values(["profile_id", "depth_mid"]).reset_index(drop=True)
    df.to_csv(out_dir / "depth_metrics_used.csv", index=False)

    # -------------------------------------------------
    # 1) overall line plots across profiles
    # -------------------------------------------------
    metric_specs = [
        ("centroid_norm", "Depth profile: centroid norm", "centroid_norm"),
        ("eff_rank", "Depth profile: effective rank", "eff_rank"),
        ("cos_p10", "Depth profile: cosine p10", "cos_p10"),
        ("shannon", "Depth profile: prediction Shannon", "shannon"),
        ("tax_shannon", "Depth profile: taxonomist Shannon", "tax_shannon"),
    ]

    for col, title, ylabel in metric_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            _plot_depth_profiles_for_metric(
                df,
                metric_col=col,
                out_path=out_dir / f"depth_profiles_{col}",
                title=title,
                ylabel=ylabel,
                use_profile_id_short=use_profile_id_short,
            )

    # # -------------------------------------------------
    # # 2) per-profile stacked panels
    # # -------------------------------------------------
    # per_profile_dir = out_dir / "per_profile_panels"
    # per_profile_dir.mkdir(parents=True, exist_ok=True)
    # _plot_depth_profile_panels_per_profile(
    #     df,
    #     out_dir=per_profile_dir,
    #     use_profile_id_short=use_profile_id_short,
    # )

    # -------------------------------------------------
    # 3) profile x depth heatmaps
    # -------------------------------------------------
    heatmap_specs = [
        ("centroid_norm", "Profile × depth heatmap: centroid norm"),
        ("eff_rank", "Profile × depth heatmap: effective rank"),
        ("cos_p10", "Profile × depth heatmap: cosine p10"),
        ("shannon", "Profile × depth heatmap: prediction Shannon"),
        ("tax_shannon", "Profile × depth heatmap: taxonomist Shannon"),
    ]

    for col, title in heatmap_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            _plot_metric_heatmap_profile_depth(
                df,
                metric_col=col,
                out_path=out_dir / f"heatmap_profile_depth_{col}",
                title=title,
                use_profile_id_short=use_profile_id_short,
                row_zscore=False,
            )
            _plot_metric_heatmap_profile_depth(
                df,
                metric_col=col,
                out_path=out_dir / f"heatmap_profile_depth_{col}_rowz",
                title=title + " (row z-score)",
                use_profile_id_short=use_profile_id_short,
                row_zscore=True,
            )

    # -------------------------------------------------
    # 4) NMDS views
    # -------------------------------------------------
    if "nmds1" in df.columns and "nmds2" in df.columns:
        _plot_nmds_depth_colored(
            df,
            out_path=out_dir / "nmds_colored_by_depth",
            color_col="depth_mid",
            title="NMDS of profile-depth groups coloured by depth",
            annotate=False,
            use_profile_id_short=use_profile_id_short,
        )

        if "centroid_norm" in df.columns:
            # simple reuse: color by centroid norm
            _plot_nmds_depth_colored(
                df,
                out_path=out_dir / "nmds_colored_by_centroid_norm",
                color_col="centroid_norm",
                title="NMDS of profile-depth groups coloured by centroid norm",
                annotate=False,
                use_profile_id_short=use_profile_id_short,
            )

        _plot_nmds_profile_paths(
            df,
            out_path=out_dir / "nmds_profile_paths",
            use_profile_id_short=use_profile_id_short,
            annotate_endpoints=True,
        )

    # -------------------------------------------------
    # 5) turnover plots
    # -------------------------------------------------
    for col in ["centroid_norm", "eff_rank", "cos_p10", "shannon", "tax_shannon"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            _plot_vertical_turnover(
                df,
                out_path=out_dir / f"vertical_turnover_{col}",
                metric_col=col,
                use_profile_id_short=use_profile_id_short,
            )

    print(f"Saved depth plots to: {out_dir}")


if __name__ == "__main__":
    # Example:
    path = r"C:\alr4\analysis\geometry"

    visualize_depth_metrics(
        metrics_csv=Path(path) / "geometry_metrics.csv",
        out_dir=Path(path),
        run_id=None,
        use_profile_id_short=False,
        min_images_per_group=0,
    )