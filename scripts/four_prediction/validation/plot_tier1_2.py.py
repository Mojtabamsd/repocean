import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

GEOMETRY_FILES = {
    0: Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\0\geometry\geometry_metrics.csv"),
    200: Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\200\geometry\geometry_metrics.csv"),
    1000: Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\1000\geometry\geometry_metrics.csv"),
    10000: Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\10000\geometry\geometry_metrics.csv"),
}

OUT_DIR = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


METRICS = [
    "shannon",
    "centroid_norm",
    "eff_rank",
    "cos_p10",
    "cos_p50",
    "pair_cos_p10",
    "pair_cos_p50",
]


SPEARMAN_MIN = 0.4
EFFECT_OVER_NOISE_MIN = 1.0


# ============================================================
# VISUAL DESIGN
# ============================================================

# Keep the same base color assignment across figures.
metric_colors = {
    metric: f"C{i}"
    for i, metric in enumerate(METRICS)
}


# Each metric also has its own marker shape.
METRIC_MARKERS = {
    "shannon": "o",
    "centroid_norm": "s",
    "eff_rank": "^",
    "cos_p10": "D",
    "cos_p50": "v",
    "pair_cos_p10": "P",
    "pair_cos_p50": "X",
}


# These three are the main metrics to visually emphasize.
HIGHLIGHT = [
    "pair_cos_p10",
    "cos_p50",
    "eff_rank",
]


DET_ORDER = [
    0,
    200,
    1000,
    10000,
]


# Equal categorical spacing.
DET_X = {
    detritus: i
    for i, detritus in enumerate(DET_ORDER)
}


# Slight offsets reduce point overlap.
offset_values = np.linspace(
    -0.10,
    0.10,
    len(METRICS),
)

METRIC_X_OFFSET = {
    metric: offset
    for metric, offset
    in zip(
        METRICS,
        offset_values,
    )
}


# ============================================================
# GROUP-ID PARSING
# ============================================================

BLOOM_RE = re.compile(
    r"abundance_bloom__bloom_(.+)_x(\d+)_rep(\d+)$"
)

NOVEL_RE = re.compile(
    r"novel_category__novel_(.+)_n(\d+)_rep(\d+)$"
)

COMPOSITION_RE = re.compile(
    r"composition_swap__composition_([AB])_rep(\d+)$"
)

NULL_RE = re.compile(
    r"null_control__null_control_(\d+)$"
)


def parse_group_id(group_id):

    gid = str(group_id)

    m = BLOOM_RE.match(gid)

    if m:
        return {
            "scenario": "bloom",
            "target": m.group(1),
            "level": int(m.group(2)),
            "replicate": int(m.group(3)),
        }

    m = NOVEL_RE.match(gid)

    if m:
        return {
            "scenario": "novel",
            "target": m.group(1),
            "level": int(m.group(2)),
            "replicate": int(m.group(3)),
        }

    m = COMPOSITION_RE.match(gid)

    if m:
        return {
            "scenario": "composition",
            "target": "A_vs_B",
            "level": m.group(1),
            "replicate": int(m.group(2)),
        }

    m = NULL_RE.match(gid)

    if m:
        return {
            "scenario": "null",
            "target": "null",
            "level": int(m.group(1)),
            "replicate": int(m.group(1)),
        }

    return {
        "scenario": "unknown",
        "target": "unknown",
        "level": np.nan,
        "replicate": np.nan,
    }


# ============================================================
# LOAD ALL FOUR RUNS
# ============================================================

frames = []


for detritus_n, path in GEOMETRY_FILES.items():

    print(
        f"Loading detritus={detritus_n}: {path}"
    )

    df = pd.read_csv(path)

    missing = [
        metric
        for metric in METRICS
        if metric not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing metrics: {missing}"
        )

    if "group_id" not in df.columns:
        raise ValueError(
            f"{path} does not contain group_id"
        )

    parsed = pd.DataFrame(
        df["group_id"]
        .apply(parse_group_id)
        .tolist(),
        index=df.index,
    )

    df = pd.concat(
        [
            df,
            parsed,
        ],
        axis=1,
    )

    df["detritus_n"] = detritus_n

    frames.append(df)


all_df = pd.concat(
    frames,
    ignore_index=True,
)


print(
    "\nParsed scenario counts:"
)

print(
    pd.crosstab(
        all_df["detritus_n"],
        all_df["scenario"],
    )
)


# ============================================================
# NULL NOISE
# ============================================================

noise_rows = []


for detritus_n in DET_ORDER:

    run_df = all_df[
        all_df["detritus_n"]
        == detritus_n
    ]

    null_df = run_df[
        run_df["scenario"]
        == "null"
    ]

    for metric in METRICS:

        values = (
            null_df[metric]
            .dropna()
            .to_numpy(float)
        )

        if len(values) < 2:

            noise = np.nan

        else:

            pairwise_diffs = [
                abs(
                    values[i]
                    - values[j]
                )
                for i in range(len(values))
                for j in range(
                    i + 1,
                    len(values),
                )
            ]

            noise = float(
                np.median(
                    pairwise_diffs
                )
            )

        noise_rows.append(
            {
                "detritus_n":
                    detritus_n,

                "metric":
                    metric,

                "noise_floor":
                    noise,
            }
        )


noise_df = pd.DataFrame(
    noise_rows
)


def get_noise(
    detritus_n,
    metric,
):

    row = noise_df[
        (
            noise_df["detritus_n"]
            == detritus_n
        )
        &
        (
            noise_df["metric"]
            == metric
        )
    ]

    if row.empty:
        return np.nan

    return float(
        row["noise_floor"]
        .iloc[0]
    )


# ============================================================
# SIGNED LOG EFFECT
# ============================================================

def signed_log_effect(x):

    return (
        np.sign(x)
        * np.log10(
            1 + np.abs(x)
        )
    )


TRANSFORMED_THRESHOLD = float(
    signed_log_effect(
        1.0
    )
)


# ============================================================
# BLOOM SCORING
# ============================================================

bloom_rows = []


for detritus_n in DET_ORDER:

    run_df = all_df[
        all_df["detritus_n"]
        == detritus_n
    ]

    bloom = run_df[
        run_df["scenario"]
        == "bloom"
    ]

    for metric in METRICS:

        g = bloom[
            [
                "level",
                metric,
            ]
        ].dropna()

        if (
            len(g) == 0
            or g["level"].nunique() < 3
        ):
            continue

        means = (
            g.groupby("level")[metric]
            .mean()
        )

        r, p = spearmanr(
            g["level"],
            g[metric],
        )

        noise = get_noise(
            detritus_n,
            metric,
        )

        swing = float(
            means.max()
            - means.min()
        )

        if (
            pd.isna(noise)
            or noise <= 0
        ):

            effect_over_noise = np.nan

        else:

            effect_over_noise = (
                abs(swing)
                / noise
            )

        working = (
            pd.notna(r)
            and abs(r) >= SPEARMAN_MIN
            and pd.notna(
                effect_over_noise
            )
            and effect_over_noise
            >= EFFECT_OVER_NOISE_MIN
        )

        signed_delta = float(
            means.loc[10]
            - means.loc[1]
        )

        if (
            pd.isna(noise)
            or noise <= 0
        ):

            signed_effect_noise = np.nan

        else:

            signed_effect_noise = (
                signed_delta
                / noise
            )

        bloom_rows.append(
            {
                "detritus_n":
                    detritus_n,

                "metric":
                    metric,

                "signed_effect_noise":
                    signed_effect_noise,

                "effect_over_noise":
                    effect_over_noise,

                "spearman_r":
                    r,

                "working":
                    working,
            }
        )


bloom_summary = pd.DataFrame(
    bloom_rows
)


# ============================================================
# NOVELTY SCORING
# ============================================================

novel_rows = []


for detritus_n in DET_ORDER:

    run_df = all_df[
        all_df["detritus_n"]
        == detritus_n
    ]

    novel = run_df[
        run_df["scenario"]
        == "novel"
    ]

    for target, target_df in novel.groupby(
        "target"
    ):

        for metric in METRICS:

            baseline = target_df[
                target_df["level"]
                == 0
            ][metric].dropna()

            if len(baseline) == 0:
                continue

            baseline_mean = float(
                baseline.mean()
            )

            noise = get_noise(
                detritus_n,
                metric,
            )

            for level in [
                5,
                20,
                60,
            ]:

                present = target_df[
                    target_df["level"]
                    == level
                ][metric].dropna()

                if len(present) == 0:
                    continue

                delta = float(
                    present.mean()
                    - baseline_mean
                )

                if (
                    pd.isna(noise)
                    or noise <= 0
                ):

                    signed_effect_noise = np.nan

                else:

                    signed_effect_noise = (
                        delta
                        / noise
                    )

                working = (
                    pd.notna(
                        signed_effect_noise
                    )
                    and abs(
                        signed_effect_noise
                    )
                    >= EFFECT_OVER_NOISE_MIN
                )

                novel_rows.append(
                    {
                        "detritus_n":
                            detritus_n,

                        "target":
                            target,

                        "metric":
                            metric,

                        "level":
                            level,

                        "signed_effect_noise":
                            signed_effect_noise,

                        "working":
                            working,
                    }
                )


novel_detail = pd.DataFrame(
    novel_rows
)


novel_summary = (
    novel_detail
    .groupby(
        [
            "detritus_n",
            "metric",
        ],
        as_index=False,
    )
    .agg(
        n_passed=(
            "working",
            "sum",
        ),

        n_tests=(
            "working",
            "size",
        ),
    )
)


novel_summary[
    "pass_fraction"
] = (
    novel_summary["n_passed"]
    / novel_summary["n_tests"]
)


# ============================================================
# COMPOSITION SCORING
# ============================================================

composition_rows = []


for detritus_n in DET_ORDER:

    run_df = all_df[
        all_df["detritus_n"]
        == detritus_n
    ]

    composition = run_df[
        run_df["scenario"]
        == "composition"
    ]

    for metric in METRICS:

        a = composition[
            composition["level"]
            == "A"
        ][metric].dropna()

        b = composition[
            composition["level"]
            == "B"
        ][metric].dropna()

        if (
            len(a) == 0
            or len(b) == 0
        ):
            continue

        delta = float(
            b.mean()
            - a.mean()
        )

        noise = get_noise(
            detritus_n,
            metric,
        )

        if (
            pd.isna(noise)
            or noise <= 0
        ):

            signed_effect_noise = np.nan

        else:

            signed_effect_noise = (
                delta
                / noise
            )

        working = (
            pd.notna(
                signed_effect_noise
            )
            and abs(
                signed_effect_noise
            )
            >= EFFECT_OVER_NOISE_MIN
        )

        composition_rows.append(
            {
                "detritus_n":
                    detritus_n,

                "metric":
                    metric,

                "signed_effect_noise":
                    signed_effect_noise,

                "working":
                    working,
            }
        )


composition_summary = pd.DataFrame(
    composition_rows
)


# ============================================================
# OVERALL PASS FRACTION
# ============================================================

overall_rows = []


for detritus_n in DET_ORDER:

    for metric in METRICS:

        bloom_row = bloom_summary[
            (
                bloom_summary[
                    "detritus_n"
                ]
                == detritus_n
            )
            &
            (
                bloom_summary[
                    "metric"
                ]
                == metric
            )
        ]

        bloom_pass = (
            int(
                bloom_row[
                    "working"
                ].iloc[0]
            )
            if len(
                bloom_row
            )
            else 0
        )

        novelty_rows = novel_detail[
            (
                novel_detail[
                    "detritus_n"
                ]
                == detritus_n
            )
            &
            (
                novel_detail[
                    "metric"
                ]
                == metric
            )
        ]

        novelty_pass = int(
            novelty_rows[
                "working"
            ].sum()
        )

        composition_row = (
            composition_summary[
                (
                    composition_summary[
                        "detritus_n"
                    ]
                    == detritus_n
                )
                &
                (
                    composition_summary[
                        "metric"
                    ]
                    == metric
                )
            ]
        )

        composition_pass = (
            int(
                composition_row[
                    "working"
                ].iloc[0]
            )
            if len(
                composition_row
            )
            else 0
        )

        total_passed = (
            bloom_pass
            + novelty_pass
            + composition_pass
        )

        overall_rows.append(
            {
                "detritus_n":
                    detritus_n,

                "metric":
                    metric,

                "n_passed":
                    total_passed,

                "n_tests":
                    8,

                "pass_fraction":
                    total_passed
                    / 8,
            }
        )


overall_summary = pd.DataFrame(
    overall_rows
)


# ============================================================
# VISUAL HELPER
# ============================================================

def visual_settings(metric):

    if metric in HIGHLIGHT:

        return {
            "color":
                metric_colors[
                    metric
                ],

            "alpha":
                1.0,

            "linewidth":
                2.2,

            "markersize":
                85,

            "zorder":
                4,
        }

    else:

        return {
            "color":
                "0.68",

            "alpha":
                0.75,

            "linewidth":
                1.0,

            "markersize":
                55,

            "zorder":
                2,
        }


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9),
)


ax_bloom = axes[0, 0]
ax_novel = axes[0, 1]
ax_comp = axes[1, 0]
ax_overall = axes[1, 1]


# ============================================================
# PANEL A — BLOOM
# ============================================================

for metric in METRICS:

    d = bloom_summary[
        bloom_summary["metric"]
        == metric
    ].sort_values(
        "detritus_n"
    )

    style = visual_settings(
        metric
    )

    xs = [
        DET_X[x]
        + METRIC_X_OFFSET[
            metric
        ]
        for x in d[
            "detritus_n"
        ]
    ]

    ys = [
        signed_log_effect(x)
        for x in d[
            "signed_effect_noise"
        ]
    ]

    ax_bloom.plot(
        xs,
        ys,

        color=style[
            "color"
        ],

        linewidth=style[
            "linewidth"
        ],

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ],
    )

    for (
        x,
        y,
        working,
    ) in zip(
        xs,
        ys,
        d["working"],
    ):

        if working:

            ax_bloom.scatter(
                x,
                y,

                s=style[
                    "markersize"
                ],

                marker=METRIC_MARKERS[
                    metric
                ],

                color=style[
                    "color"
                ],

                alpha=style[
                    "alpha"
                ],

                edgecolor="white",

                linewidth=0.7,

                zorder=style[
                    "zorder"
                ]
                + 1,
            )

        else:

            ax_bloom.scatter(
                x,
                y,

                s=style[
                    "markersize"
                ],

                marker=METRIC_MARKERS[
                    metric
                ],

                facecolors="white",

                edgecolors=style[
                    "color"
                ],

                linewidths=1.6,

                alpha=style[
                    "alpha"
                ],

                zorder=style[
                    "zorder"
                ]
                + 1,
            )


ax_bloom.axhline(
    0,
    linewidth=0.8,
    alpha=0.5,
)

ax_bloom.axhline(
    TRANSFORMED_THRESHOLD,
    linestyle=":",
    linewidth=1.2,
)

ax_bloom.axhline(
    -TRANSFORMED_THRESHOLD,
    linestyle=":",
    linewidth=1.2,
)

ax_bloom.set_title(
    "A  Abundance bloom"
)

ax_bloom.set_ylabel(
    "Signed effect / null noise\n(compressed scale)"
)


# ============================================================
# PANEL B — NOVELTY
# ============================================================

for metric in METRICS:

    d = novel_summary[
        novel_summary[
            "metric"
        ]
        == metric
    ].sort_values(
        "detritus_n"
    )

    style = visual_settings(
        metric
    )

    xs = [
        DET_X[x]
        + METRIC_X_OFFSET[
            metric
        ]
        for x in d[
            "detritus_n"
        ]
    ]

    ys = d[
        "pass_fraction"
    ].to_numpy()

    ax_novel.plot(
        xs,
        ys,

        color=style[
            "color"
        ],

        linewidth=style[
            "linewidth"
        ],

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ],
    )

    ax_novel.scatter(
        xs,
        ys,

        s=style[
            "markersize"
        ],

        marker=METRIC_MARKERS[
            metric
        ],

        color=style[
            "color"
        ],

        edgecolor="white",

        linewidth=0.7,

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ]
        + 1,
    )


ax_novel.set_ylim(
    -0.05,
    1.05,
)

ax_novel.set_yticks(
    [
        0,
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
        5 / 6,
        1,
    ]
)

ax_novel.set_yticklabels(
    [
        "0/6",
        "1/6",
        "2/6",
        "3/6",
        "4/6",
        "5/6",
        "6/6",
    ]
)

ax_novel.set_title(
    "B  Novel-category detection"
)

ax_novel.set_ylabel(
    "Novel comparisons passed"
)


# ============================================================
# PANEL C — COMPOSITION
# ============================================================

for metric in METRICS:

    d = composition_summary[
        composition_summary[
            "metric"
        ]
        == metric
    ].sort_values(
        "detritus_n"
    )

    style = visual_settings(
        metric
    )

    xs = [
        DET_X[x]
        + METRIC_X_OFFSET[
            metric
        ]
        for x in d[
            "detritus_n"
        ]
    ]

    ys = [
        signed_log_effect(x)
        for x in d[
            "signed_effect_noise"
        ]
    ]

    ax_comp.plot(
        xs,
        ys,

        color=style[
            "color"
        ],

        linewidth=style[
            "linewidth"
        ],

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ],
    )

    for (
        x,
        y,
        working,
    ) in zip(
        xs,
        ys,
        d["working"],
    ):

        if working:

            ax_comp.scatter(
                x,
                y,

                s=style[
                    "markersize"
                ],

                marker=METRIC_MARKERS[
                    metric
                ],

                color=style[
                    "color"
                ],

                edgecolor="white",

                linewidth=0.7,

                alpha=style[
                    "alpha"
                ],

                zorder=style[
                    "zorder"
                ]
                + 1,
            )

        else:

            ax_comp.scatter(
                x,
                y,

                s=style[
                    "markersize"
                ],

                marker=METRIC_MARKERS[
                    metric
                ],

                facecolors="white",

                edgecolors=style[
                    "color"
                ],

                linewidths=1.6,

                alpha=style[
                    "alpha"
                ],

                zorder=style[
                    "zorder"
                ]
                + 1,
            )


ax_comp.axhline(
    0,
    linewidth=0.8,
    alpha=0.5,
)

ax_comp.axhline(
    TRANSFORMED_THRESHOLD,
    linestyle=":",
    linewidth=1.2,
)

ax_comp.axhline(
    -TRANSFORMED_THRESHOLD,
    linestyle=":",
    linewidth=1.2,
)

ax_comp.set_title(
    "C  Community composition"
)

ax_comp.set_ylabel(
    "Signed effect / null noise\n(compressed scale)"
)


# ============================================================
# PANEL D — OVERALL
# ============================================================

for metric in METRICS:

    d = overall_summary[
        overall_summary[
            "metric"
        ]
        == metric
    ].sort_values(
        "detritus_n"
    )

    style = visual_settings(
        metric
    )

    xs = [
        DET_X[x]
        + METRIC_X_OFFSET[
            metric
        ]
        for x in d[
            "detritus_n"
        ]
    ]

    ys = d[
        "pass_fraction"
    ].to_numpy()

    ax_overall.plot(
        xs,
        ys,

        color=style[
            "color"
        ],

        linewidth=style[
            "linewidth"
        ],

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ],
    )

    ax_overall.scatter(
        xs,
        ys,

        s=style[
            "markersize"
        ],

        marker=METRIC_MARKERS[
            metric
        ],

        color=style[
            "color"
        ],

        edgecolor="white",

        linewidth=0.7,

        alpha=style[
            "alpha"
        ],

        zorder=style[
            "zorder"
        ]
        + 1,
    )


    # --------------------------------------------------------
    # Direct labels only for highlighted metrics
    # --------------------------------------------------------

    if metric in HIGHLIGHT:

        ax_overall.annotate(
            metric,

            xy=(
                xs[-1],
                ys[-1],
            ),

            xytext=(
                8,
                0,
            ),

            textcoords="offset points",

            color=metric_colors[
                metric
            ],

            fontsize=9,

            fontweight="bold",

            va="center",
        )


ax_overall.set_ylim(
    -0.05,
    1.05,
)

ax_overall.set_xlim(
    -0.25,
    len(DET_ORDER)
    - 1
    + 0.65,
)

ax_overall.set_title(
    "D  Overall Tier-1 performance"
)

ax_overall.set_ylabel(
    "Fraction of tests passed"
)


# ============================================================
# COMMON X AXIS
# ============================================================

for ax in axes.flat:

    ax.set_xticks(
        range(
            len(
                DET_ORDER
            )
        )
    )

    ax.set_xticklabels(
        [
            str(x)
            for x in DET_ORDER
        ]
    )

    ax.set_xlabel(
        "Number of detritus objects added"
    )

    ax.grid(
        alpha=0.12,
    )


# ============================================================
# LEGEND
# ============================================================

from matplotlib.lines import Line2D


legend_handles = []


for metric in METRICS:

    if metric in HIGHLIGHT:

        color = metric_colors[
            metric
        ]

        alpha = 1.0

    else:

        color = "0.55"

        alpha = 0.85


    legend_handles.append(
        Line2D(
            [0],
            [0],

            marker=METRIC_MARKERS[
                metric
            ],

            linestyle="-",

            color=color,

            markerfacecolor=color,

            markeredgecolor="white",

            linewidth=1.5,

            alpha=alpha,

            label=metric,
        )
    )


fig.legend(
    handles=legend_handles,

    loc="lower center",

    ncol=4,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        -0.01,
    ),
)


# ============================================================
# TITLE + NOTE
# ============================================================

fig.suptitle(
    "Effect of increasing detrital background on "
    "Tier-1 geometry-metric sensitivity",
    fontsize=15,
)


fig.text(
    0.5,
    0.045,

    "Panels A and C: filled markers pass the diagnostic criterion; "
    "open markers fail. Lines are visual guides. "
    "Highlighted metrics are pair_cos_p10, cos_p50 and eff_rank. "
    "Panel B summarizes six novelty comparisons; "
    "Panel D summarizes all eight Tier-1 tests.",

    ha="center",

    fontsize=9,
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0,
        0.085,
        1,
        0.95,
    ]
)


# ============================================================
# SAVE
# ============================================================

png_file = (
    OUT_DIR
    / "figure3_detritus_robustness_improved.png"
)

pdf_file = (
    OUT_DIR
    / "figure3_detritus_robustness_improved.pdf"
)


fig.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
)


# ============================================================
# SAVE TABLES
# ============================================================

bloom_summary.to_csv(
    OUT_DIR
    / "figure3_bloom_summary.csv",
    index=False,
)

novel_detail.to_csv(
    OUT_DIR
    / "figure3_novel_detail.csv",
    index=False,
)

novel_summary.to_csv(
    OUT_DIR
    / "figure3_novel_summary.csv",
    index=False,
)

composition_summary.to_csv(
    OUT_DIR
    / "figure3_composition_summary.csv",
    index=False,
)

overall_summary.to_csv(
    OUT_DIR
    / "figure3_overall_summary.csv",
    index=False,
)

noise_df.to_csv(
    OUT_DIR
    / "figure3_null_noise.csv",
    index=False,
)


plt.show()


print(
    "\nSaved:"
)

print(
    png_file
)

print(
    pdf_file
)