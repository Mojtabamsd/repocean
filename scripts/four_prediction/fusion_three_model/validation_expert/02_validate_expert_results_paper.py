"""
03_plot_paper_figures.py

Publication-quality figures for the expert-validation of the
M1 / M2 / M3 / Fusion system.

This script is standalone: it does NOT re-run the validation. It only reads
the CSV files that 02_validate_expert_results.py already wrote to
`expert_validation_results/`, so you can iterate on the figures in seconds.

Usage
-----
    python 03_plot_paper_figures.py
    python 03_plot_paper_figures.py --results-dir D:\\some\\other\\results
    python 03_plot_paper_figures.py --out-dir figures_paper

Figures (each saved as vector PDF + 600-dpi PNG)
------------------------------------------------
Fig 1  Overall agreement with the expert (fine label and superclass),
       unweighted validation estimate + population-weighted estimate,
       and macro-F1.
Fig 2  Accuracy by Fusion decision pathway (where does Fusion work?).
Fig 3  Accuracy by expert superclass: fine-label and superclass heatmaps.
Fig 4  Fusion rescue vs. regression relative to each single model,
       with exact McNemar p-values.
Fig 5  Do Fusion confidence classes track real expert agreement?
Fig 6  Row-normalised superclass confusion matrix for Fusion.

Input files used (all optional except overall_metrics_unweighted.csv;
a figure is skipped with a message if its input is missing):
    overall_metrics_unweighted.csv
    overall_metrics_population_weighted.csv
    pathway_metrics.csv
    superclass_metrics.csv
    confidence_metrics.csv
    fusion_benefit.csv
    fusion_unique_outcomes.csv
    mcnemar_tests.csv
    validation_confusion_superclass.csv
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


# ======================================================================
# 1. SETTINGS
# ======================================================================

DEFAULT_RESULTS_DIR = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
    r"\expert_validation_results"
)

METHODS = ["M1", "M2", "M3", "Fusion"]

# Okabe-Ito colour-blind-safe palette; Fusion is the visual focus.
COLORS = {
    "M1": "#56B4E9",
    "M2": "#E69F00",
    "M3": "#009E73",
    "Fusion": "#D55E00",
}

C_RESCUE = "#2A9D8F"
C_REGRESS = "#C0392B"
C_BOTH_OK = "#B8C4CE"
C_BOTH_BAD = "#4A4A4A"

# Figure widths follow typical journal columns (inches).
COL1 = 3.5
COL15 = 5.4
COL2 = 7.2

# Superclasses with fewer expert-labelled objects than this are flagged.
MIN_N_FLAG = 5

PATH_LABELS = {
    "A_all_agree": "A\nall agree",
    "B_majority": "B\nmajority",
    "C_superclass": "C\nsuperclass",
    "D_score": "D\nscore",
    "D_score_close": "D'\nclose score",
    "abstention_m3_only": "Abst.\nM3 only",
}

CONF_ORDER = ["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # editable text in Illustrator / Inkscape
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )


# ======================================================================
# 2. HELPERS
# ======================================================================

def read_csv(results_dir: Path, name: str, **kwargs) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        print(f"  [skip] missing input: {name}")
        return None
    return pd.read_csv(path, **kwargs)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.13, 1.04, letter,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="right",
    )


def pretty(label: str, width: int = 26) -> str:
    return textwrap.shorten(str(label).replace("_", " "), width=width, placeholder="…")


def ci_err(acc, lo, hi):
    acc, lo, hi = map(np.asarray, (acc, lo, hi))
    return np.vstack([np.clip(acc - lo, 0, None), np.clip(hi - acc, 0, None)])


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def method_rows(df: pd.DataFrame, col: str = "method") -> pd.DataFrame:
    df = df.set_index(col)
    return df.loc[[m for m in METHODS if m in df.index]]


# ======================================================================
# 3. FIGURE 1 - OVERALL PERFORMANCE
# ======================================================================

def fig1_overall(results_dir: Path, out_dir: Path) -> None:
    overall = read_csv(results_dir, "overall_metrics_unweighted.csv")
    if overall is None:
        return
    weighted = read_csv(results_dir, "overall_metrics_population_weighted.csv")
    ov = method_rows(overall)
    wt = method_rows(weighted) if weighted is not None else None
    x = np.arange(len(ov))

    fig, axes = plt.subplots(
        1, 3, figsize=(COL2, 2.7),
        gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.38},
    )

    # ---- (a, b) accuracy with Wilson CI + population-weighted marker ----
    for ax, level, letter, title in [
        (axes[0], "fine", "a", "Fine label"),
        (axes[1], "superclass", "b", "Superclass"),
    ]:
        acc = ov[f"{level}_accuracy_valid"].values * 100
        lo = ov[f"{level}_accuracy_ci_low"].values * 100
        hi = ov[f"{level}_accuracy_ci_high"].values * 100
        ax.bar(
            x, acc, width=0.68,
            color=[COLORS[m] for m in ov.index],
            edgecolor="black", linewidth=0.5, zorder=2,
        )
        ax.errorbar(
            x, acc, yerr=ci_err(acc, lo, hi),
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=3,
        )
        for xi, a in zip(x, acc):
            ax.text(xi, 2.5, f"{a:.1f}", ha="center", va="bottom", fontsize=7,
                    color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.75))
        if wt is not None:
            w = wt[f"{level}_accuracy_population_weighted"].values * 100
            ax.scatter(
                x, w, marker="D", s=22, facecolor="white",
                edgecolor="black", linewidth=0.9, zorder=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(ov.index)
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Agreement with expert (%)" if letter == "a" else "")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        panel_letter(ax, letter)

    handles = [
        Patch(facecolor="#DDDDDD", edgecolor="black", linewidth=0.5,
              label="Validation set (bar, 95% Wilson CI)"),
    ]
    if wt is not None:
        handles.append(
            Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor="black", markersize=4.5,
                   label="Population-weighted estimate")
        )
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -0.36),
                   ncol=2, fontsize=7, handlelength=1.2, columnspacing=1.5)

    # ---- (c) macro-F1, fine vs superclass ----
    ax = axes[2]
    width = 0.36
    series = [
        ("fine_macro_f1", "", -0.5),
        ("superclass_macro_f1", "//", 0.5),
    ]
    for col, hatch, pos in series:
        if col not in ov.columns:
            continue
        vals = ov[col].values * 100
        for xi, v, m in zip(x, vals, ov.index):
            ax.bar(
                xi + pos * width, v, width=width * 0.94, color=COLORS[m],
                hatch=hatch, edgecolor="black", linewidth=0.5, zorder=2,
            )
            ax.text(xi + pos * width, v + 1.5, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ov.index)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Macro-F1", loc="left", fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    panel_letter(ax, "c")
    ax.legend(
        handles=[
            Patch(facecolor="#CCCCCC", edgecolor="black", linewidth=0.5,
                  label="Fine label"),
            Patch(facecolor="#CCCCCC", edgecolor="black", linewidth=0.5,
                  hatch="//", label="Superclass"),
        ],
        loc="lower left", bbox_to_anchor=(0, -0.36), ncol=2, fontsize=7,
        handlelength=1.2, columnspacing=1.5,
    )

    n = int(ov["n_validation"].iloc[0]) if "n_validation" in ov.columns else None
    if n:
        fig.text(
            0.01, -0.02,
            f"n = {n:,} expert-labelled objects.",
            fontsize=6.5, color="0.35", ha="left", va="top",
        )
    save(fig, out_dir, "fig1_overall_performance")


# ======================================================================
# 4. FIGURE 2 - PATHWAY
# ======================================================================

def fig2_pathway(results_dir: Path, out_dir: Path) -> None:
    df = read_csv(results_dir, "pathway_metrics.csv")
    if df is None:
        return
    df = df[df["n_valid"] > 0]
    paths = [p for p in PATH_LABELS if p in set(df["path"])]
    if not paths:
        return

    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    x = np.arange(len(paths))
    offsets = np.linspace(-0.27, 0.27, len(METHODS))

    for off, m in zip(offsets, METHODS):
        sub = df[df["method"] == m].set_index("path")
        xs, ys, los, his = [], [], [], []
        for xi, p in zip(x, paths):
            if p in sub.index and sub.loc[p, "n_valid"] > 0:
                xs.append(xi + off)
                ys.append(sub.loc[p, "accuracy"] * 100)
                los.append(sub.loc[p, "ci_low"] * 100)
                his.append(sub.loc[p, "ci_high"] * 100)
        is_f = m == "Fusion"
        ax.errorbar(
            xs, ys, yerr=ci_err(ys, los, his), fmt="o" if not is_f else "s",
            color=COLORS[m], markersize=5.5 if is_f else 4.2,
            markeredgecolor="black", markeredgewidth=0.5,
            elinewidth=1.0, capsize=2, label=m, zorder=4 if is_f else 3,
        )

    # sample sizes under each pathway label
    n_fusion = (
        df[df["method"] == "Fusion"].set_index("path")["n_valid"].to_dict()
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{PATH_LABELS[p]}\n(n = {int(n_fusion.get(p, 0)):,})" for p in paths]
    )
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="0.85", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.6, len(paths) - 0.4)
    ax.set_ylim(0, 103)
    ax.set_ylabel("Fine-label agreement with expert (%)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5,
              handletextpad=0.2, columnspacing=0.9)
    ax.text(
        1.0, 1.02,
        "Pathway A: all three models agree, so predictions are identical by design",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
        color="0.35",
    )
    save(fig, out_dir, "fig2_pathway_performance")


# ======================================================================
# 5. FIGURE 3 - EXPERT SUPERCLASS HEATMAPS
# ======================================================================

def _heatmap(ax, mat, nmat, show_ylabels, ylabels, cmap, title):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(METHODS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(ylabels if show_ylabels else [])
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#EEEEEE",
                                       edgecolor="white", linewidth=1.2))
                ax.text(j, i, "–", ha="center", va="center", color="0.5")
                continue
            low_n = nmat[i, j] < MIN_N_FLAG
            ax.text(
                j, i, f"{v * 100:.0f}",
                ha="center", va="center", fontsize=6.8,
                color="white" if v > 0.55 else "black",
                style="italic" if low_n else "normal",
            )
            if low_n:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       hatch="////", edgecolor="0.55",
                                       linewidth=0, alpha=0.6))
    # white grid lines
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    # outline Fusion column
    jf = METHODS.index("Fusion")
    ax.add_patch(Rectangle((jf - .5, -.5), 1, mat.shape[0], fill=False,
                           edgecolor=COLORS["Fusion"], linewidth=1.6,
                           zorder=10, clip_on=False))
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    return im


def fig3_superclass(results_dir: Path, out_dir: Path) -> None:
    df = read_csv(results_dir, "superclass_metrics.csv")
    if df is None:
        return
    fine = df.pivot(index="expert_superclass", columns="method", values="fine_accuracy")
    sc = df.pivot(index="expert_superclass", columns="method", values="superclass_accuracy")
    nmat_df = df.pivot(index="expert_superclass", columns="method", values="n")
    for t in (fine, sc, nmat_df):
        for m in METHODS:
            if m not in t.columns:
                t[m] = np.nan
    order = (
        nmat_df["Fusion"].fillna(nmat_df.max(axis=1))
        .sort_values(ascending=False).index
    )
    fine, sc, nmat_df = (t.loc[order, METHODS] for t in (fine, sc, nmat_df))
    n_ref = nmat_df.max(axis=1).astype(int)
    ylabels = [f"{pretty(i)}  ({n_ref[i]})" for i in order]

    rows = len(order)
    fig, axes = plt.subplots(
        1, 2, figsize=(COL2 * 0.95, 0.9 + 0.235 * rows),
        gridspec_kw={"wspace": 0.06},
    )
    cmap = plt.get_cmap("YlGnBu")
    nm = nmat_df.values
    im = _heatmap(axes[0], fine.values, nm, True, ylabels, cmap, "a  Fine-label agreement (%)")
    _heatmap(axes[1], sc.values, nm, False, ylabels, cmap, "b  Superclass agreement (%)")

    cb = fig.colorbar(im, ax=list(axes), orientation="horizontal",
                      fraction=0.03, pad=0.035, shrink=0.35, aspect=30)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.set_label("Agreement with expert (%)", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6.5)
    fig.text(0.5, 0.0,
             f"Rows: expert reference superclass (n objects), sorted by n. "
             f"Hatched italic cells: n < {MIN_N_FLAG}. Orange box: Fusion.",
             ha="center", va="top", fontsize=6.5, color="0.35")
    save(fig, out_dir, "fig3_superclass_performance")


# ======================================================================
# 6. FIGURE 4 - FUSION RESCUE / REGRESSION
# ======================================================================

def fig4_benefit(results_dir: Path, out_dir: Path) -> None:
    ben = read_csv(results_dir, "fusion_benefit.csv")
    if ben is None:
        return
    mc = read_csv(results_dir, "mcnemar_tests.csv")
    uniq = read_csv(results_dir, "fusion_unique_outcomes.csv")
    ben = ben.set_index("model").loc[[m for m in ["M1", "M2", "M3"]
                                      if m in set(ben["model"])]]
    models = list(ben.index)
    y = np.arange(len(models))[::-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(COL2, 2.3),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.6},
    )

    # ---- (a) diverging bars: regressions left, rescues right ----
    rescue = ben["fusion_rescue"].values
    regress = ben["fusion_regression"].values
    ax1.barh(y, rescue, color=C_RESCUE, height=0.55, edgecolor="black",
             linewidth=0.5, zorder=2)
    ax1.barh(y, -regress, color=C_REGRESS, height=0.55, edgecolor="black",
             linewidth=0.5, zorder=2)
    xmax = max(rescue.max(), regress.max(), 1)
    pad = xmax * 0.03 + 1
    for yi, r, g in zip(y, rescue, regress):
        ax1.text(r + pad, yi, f"{r:,}", va="center", ha="left", fontsize=7)
        ax1.text(-g - pad, yi, f"{g:,}", va="center", ha="right", fontsize=7)
    ax1.axvline(0, color="black", linewidth=0.8, zorder=3)
    ax1.set_xlim(-xmax * 1.35, xmax * 1.35)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"Fusion vs {m}" for m in models])
    ax1.set_xlabel("Number of objects")
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(int(v)):,}")
    )
    ax1.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax1.set_title("Fusion vs single model", loc="left",
                  fontweight="bold")
    ax1.legend(
        handles=[Patch(facecolor=C_REGRESS, edgecolor="black", linewidth=0.5,
                       label="Regression\n(model right, Fusion wrong)"),
                 Patch(facecolor=C_RESCUE, edgecolor="black", linewidth=0.5,
                       label="Rescue\n(model wrong, Fusion right)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=6.8,
        handlelength=1.1, columnspacing=1.2,
    )
    panel_letter(ax1, "a")

    # ---- (b) 100% stacked bars of all four outcomes + McNemar ----
    cats = [
        ("both_correct", C_BOTH_OK, "Both correct"),
        ("fusion_rescue", C_RESCUE, "Rescue"),
        ("fusion_regression", C_REGRESS, "Regression"),
        ("both_wrong", C_BOTH_BAD, "Both wrong"),
    ]
    paired = ben["paired_n"].values.astype(float)
    left = np.zeros(len(models))
    for col, color, lab in cats:
        frac = ben[col].values / paired * 100
        ax2.barh(y, frac, left=left, color=color, height=0.55,
                 edgecolor="black", linewidth=0.5, label=lab, zorder=2)
        for yi, f, l in zip(y, frac, left):
            if f >= 6:
                ax2.text(l + f / 2, yi, f"{f:.0f}", ha="center", va="center",
                         fontsize=6.8,
                         color="white" if color in (C_BOTH_BAD, C_REGRESS,
                                                    C_RESCUE) else "black")
        left += frac
    ax2.set_xlim(0, 100)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Share of paired objects (%)")
    ax2.set_title("Outcome composition, paired test", loc="left",
                  fontweight="bold")
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4,
               fontsize=6.8, handlelength=1.0, columnspacing=0.9)
    if mc is not None:
        mc = mc.set_index("comparison")
        for yi, m in zip(y, models):
            key = f"Fusion_vs_{m}"
            if key in mc.index:
                row = mc.loc[key]
                gain = int(row["net_fusion_gain"])
                ax2.text(102, yi, f"net {gain:+,}\n{fmt_p(row['mcnemar_exact_p'])}",
                         va="center", ha="left", fontsize=6.8, clip_on=False)
    panel_letter(ax2, "b")

    if uniq is not None and len(uniq) == 2:
        u = uniq.set_index("quantity")["n"]
        a = int(u.get("fusion_correct_all_three_models_wrong", 0))
        b = int(u.get("fusion_wrong_all_three_models_correct", 0))
        fig.text(
            0.01, -0.34,
            f"Fusion correct when all three models are wrong: {a:,} objects.   "
            f"Fusion wrong when all three models are correct: {b:,} objects.",
            fontsize=6.8, color="0.3", ha="left", va="top",
        )
    save(fig, out_dir, "fig4_fusion_benefit")


# ======================================================================
# 7. FIGURE 5 - CONFIDENCE
# ======================================================================

def fig5_confidence(results_dir: Path, out_dir: Path) -> None:
    df = read_csv(results_dir, "confidence_metrics.csv")
    if df is None:
        return
    df = df[df["n"] > 0].set_index("confidence")
    levels = [c for c in CONF_ORDER if c in df.index]
    if not levels:
        return
    overall = read_csv(results_dir, "overall_metrics_unweighted.csv")

    fig, ax = plt.subplots(figsize=(COL1 * 1.15, 2.7))
    x = np.arange(len(levels))
    acc = df.loc[levels, "accuracy"].values * 100
    lo = df.loc[levels, "ci_low"].values * 100
    hi = df.loc[levels, "ci_high"].values * 100
    shades = plt.get_cmap("Oranges")(np.linspace(0.85, 0.3, len(levels)))
    ax.bar(x, acc, width=0.65, color=shades, edgecolor="black", linewidth=0.5,
           zorder=2)
    ax.errorbar(x, acc, yerr=ci_err(acc, lo, hi), fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2.5, zorder=3)
    for xi, a, h in zip(x, acc, hi):
        ax.text(xi, h + 1.5, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
    if overall is not None:
        f = overall.set_index("method").loc["Fusion", "fine_accuracy_valid"] * 100
        ax.axhline(f, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(len(levels) - 0.45, f + 1.2, f"all objects: {f:.1f}",
                ha="right", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c.capitalize()}\n(n = {int(df.loc[c, 'n']):,})" for c in levels]
    )
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Fusion fine-label agreement\nwith expert (%)")
    ax.set_xlabel("Fusion confidence class")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    save(fig, out_dir, "fig5_confidence_validation")


# ======================================================================
# 8. FIGURE 6 - SUPERCLASS CONFUSION (FUSION)
# ======================================================================

def fig6_confusion(results_dir: Path, out_dir: Path) -> None:
    cm = read_csv(results_dir, "validation_confusion_superclass.csv", index_col=0)
    if cm is None:
        return
    sm = read_csv(results_dir, "superclass_metrics.csv")

    # Rows: only superclasses that have expert-labelled objects
    # (a class with no expert samples gives an all-zero, uninformative row).
    row_has_samples = cm.sum(axis=1) > 0
    row_labels = list(cm.index[row_has_samples])
    if sm is not None:
        n = (sm[sm["method"] == "Fusion"].set_index("expert_superclass")["n"]
             .reindex(row_labels).fillna(0))
        row_labels = sorted(row_labels, key=lambda l: -n.get(l, 0))
    else:
        n = pd.Series(0, index=row_labels)

    # Columns: the same classes, plus any other class Fusion actually predicted
    # (kept so that wrong predictions into it remain visible and rows sum to
    # 100%). Classes that are neither in the expert set nor ever predicted are
    # dropped.
    extra = [c for c in cm.columns
             if c not in row_labels and cm[c].sum() > 0]
    extra = sorted(extra, key=lambda c: -cm[c].sum())
    col_labels = row_labels + extra

    n_dropped = len(cm.index) - len(row_labels) - len(extra)
    print(f"  fig6: {len(row_labels)} expert classes shown, "
          f"{len(extra)} predicted-only column(s) kept, "
          f"{max(n_dropped, 0)} empty class(es) removed")

    mat = cm.loc[row_labels, col_labels].values
    nr, nc = len(row_labels), len(col_labels)

    width = min(COL2 * 0.95, 1.9 + 0.30 * nc)
    height = min(COL2 * 0.95, 1.5 + 0.30 * nr)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(mat, cmap="Blues", norm=PowerNorm(gamma=0.55, vmin=0, vmax=1),
                   aspect="equal")
    row_names = [pretty(l, 22) for l in row_labels]
    col_names = [pretty(l, 22) for l in col_labels]
    ax.set_xticks(range(nc))
    ax.set_xticklabels(col_names, rotation=60, ha="right",
                       rotation_mode="anchor", fontsize=6.8)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(
        [f"{nm} ({int(n.get(l, 0))})" for nm, l in zip(row_names, row_labels)],
        fontsize=6.8,
    )
    ax.set_xlabel("Fusion prediction (superclass)")
    ax.set_ylabel("Expert annotation (superclass, n)")
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white" if v > 0.5 else "black")
        j = i  # row i and column i are the same class (row_labels come first)
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor=COLORS["Fusion"], linewidth=1.2,
                               zorder=10, clip_on=False))
    if extra:
        ax.axvline(len(row_labels) - 0.5, color="0.4", linewidth=0.8,
                   linestyle="--", zorder=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Row-normalised share (%)", fontsize=7)
    cb.set_ticks([0, 0.1, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0", "10", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    note = ("Cells < 5% not annotated; orange outline = diagonal "
            "(correct superclass).")
    if extra:
        note += ("\nColumns right of the dashed line: classes with no expert "
                 "samples that Fusion nevertheless predicted.")
    ax.text(0.5, -0.27, note, transform=ax.transAxes, ha="center",
            va="top", fontsize=6.5, color="0.35")
    save(fig, out_dir, "fig6_superclass_confusion")


# ======================================================================
# 9. MAIN
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <results-dir>/paper_figures")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_dir / "paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    print(f"Reading results from: {args.results_dir}")
    print(f"Writing figures to:   {out_dir}\n")

    for fn in (fig1_overall, fig2_pathway, fig3_superclass,
               fig4_benefit, fig5_confidence, fig6_confusion):
        try:
            fn(args.results_dir, out_dir)
        except Exception as exc:  # keep going if one figure fails
            print(f"  [error] {fn.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()