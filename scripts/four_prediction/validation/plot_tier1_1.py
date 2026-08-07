import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================


GEOMETRY_FILE = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\0\geometry\geometry_metrics.csv")

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
# ONE CONSISTENT COLOUR PER METRIC
# ============================================================

metric_colors = {
    metric: f"C{i}"
    for i, metric in enumerate(METRICS)
}


# ============================================================
# PARSE GROUP IDS
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


    # --------------------------------------------------------
    # BLOOM
    # --------------------------------------------------------

    m = BLOOM_RE.match(gid)

    if m:

        return {
            "scenario": "bloom",
            "target": m.group(1),
            "level": int(m.group(2)),
            "replicate": int(m.group(3)),
        }


    # --------------------------------------------------------
    # NOVEL
    # --------------------------------------------------------

    m = NOVEL_RE.match(gid)

    if m:

        return {
            "scenario": "novel",
            "target": m.group(1),
            "level": int(m.group(2)),
            "replicate": int(m.group(3)),
        }


    # --------------------------------------------------------
    # COMPOSITION
    # --------------------------------------------------------

    m = COMPOSITION_RE.match(gid)

    if m:

        return {
            "scenario": "composition",
            "target": "A_vs_B",
            "level": m.group(1),
            "replicate": int(m.group(2)),
        }


    # --------------------------------------------------------
    # NULL
    # --------------------------------------------------------

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
# LOAD DATA
# ============================================================

df = pd.read_csv(
    GEOMETRY_FILE
)


# Check requested metrics exist

missing_metrics = [
    metric
    for metric in METRICS
    if metric not in df.columns
]

if missing_metrics:

    raise ValueError(
        f"Missing metrics: {missing_metrics}"
    )


if "group_id" not in df.columns:

    raise ValueError(
        "geometry_metrics.csv must contain a 'group_id' column."
    )


# Parse group IDs

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


# ============================================================
# CHECK PARSING
# ============================================================

print(
    "\nParsed scenarios:"
)

print(
    df["scenario"]
    .value_counts()
)


unknown_rows = df[
    df["scenario"] == "unknown"
]

if len(unknown_rows) > 0:

    print(
        "\nWARNING: some group IDs were not parsed:"
    )

    print(
        unknown_rows["group_id"]
        .tolist()
    )


# ============================================================
# NULL NOISE
# ============================================================

null_df = df[
    df["scenario"] == "null"
]


if len(null_df) < 2:

    raise ValueError(
        "Need at least two null-control profiles."
    )


noise_floor = {}


for metric in METRICS:

    values = (
        null_df[metric]
        .dropna()
        .to_numpy(dtype=float)
    )


    if len(values) < 2:

        noise_floor[metric] = np.nan

        continue


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


    noise_floor[metric] = float(
        np.median(
            pairwise_diffs
        )
    )


print(
    "\nNull-control noise:"
)

for metric in METRICS:

    print(
        f"{metric:20s} "
        f"{noise_floor[metric]:.6g}"
    )


# ============================================================
# SIGNED LOG TRANSFORM
# ============================================================

def signed_log_effect(x):

    """
    Compress large effect/noise values while preserving:

    - sign
    - ordering
    - zero

    effect/noise = 1 becomes log10(2) ~ 0.301.
    """

    x = np.asarray(x)

    return (
        np.sign(x)
        * np.log10(
            1 + np.abs(x)
        )
    )


PASS_THRESHOLD_TRANSFORMED = float(
    signed_log_effect(
        1.0
    )
)


# ============================================================
# BLOOM SCORING
# ============================================================

bloom = df[
    df["scenario"] == "bloom"
]


bloom_rows = []


for metric in METRICS:

    g = bloom[
        [
            "level",
            "replicate",
            metric,
        ]
    ].dropna()


    if len(g) == 0:

        continue


    # --------------------------------------------------------
    # Means at x1, x3, x10
    # --------------------------------------------------------

    means = (
        g.groupby(
            "level"
        )[metric]
        .mean()
        .sort_index()
    )


    sds = (
        g.groupby(
            "level"
        )[metric]
        .std()
        .sort_index()
    )


    if (
        1 not in means.index
        or 3 not in means.index
        or 10 not in means.index
    ):

        continue


    # --------------------------------------------------------
    # Spearman uses ALL replicates across x1, x3, x10
    # --------------------------------------------------------

    r, p = spearmanr(
        g["level"],
        g[metric],
    )


    # --------------------------------------------------------
    # Diagnostic swing
    #
    # Same concept as report:
    # maximum level mean - minimum level mean
    # --------------------------------------------------------

    swing = float(
        means.max()
        - means.min()
    )


    noise = noise_floor[metric]


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
        and abs(r)
        >= SPEARMAN_MIN
        and pd.notna(
            effect_over_noise
        )
        and effect_over_noise
        >= EFFECT_OVER_NOISE_MIN
    )


    # --------------------------------------------------------
    # Signed endpoint effect for plotting
    #
    # The PASS test uses ALL levels above.
    # The plotted position shows overall direction x1 -> x10.
    # --------------------------------------------------------

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
            "metric": metric,

            "mean_x1": means.loc[1],
            "mean_x3": means.loc[3],
            "mean_x10": means.loc[10],

            "sd_x1": sds.loc[1],
            "sd_x3": sds.loc[3],
            "sd_x10": sds.loc[10],

            "spearman_r": r,
            "spearman_p": p,

            "effect_over_noise": effect_over_noise,

            "signed_effect_noise": signed_effect_noise,

            "working": working,
        }
    )


bloom_summary = pd.DataFrame(
    bloom_rows
)


# ============================================================
# NOVEL SCORING
# ============================================================

novel = df[
    df["scenario"] == "novel"
]


novel_targets = sorted(
    novel["target"]
    .dropna()
    .unique()
)


novel_rows = []


for target in novel_targets:

    target_df = novel[
        novel["target"] == target
    ]


    for metric in METRICS:

        baseline_values = target_df[
            target_df["level"] == 0
        ][metric].dropna()


        if len(
            baseline_values
        ) == 0:

            continue


        baseline_mean = float(
            baseline_values.mean()
        )

        baseline_sd = float(
            baseline_values.std()
        )


        noise = noise_floor[
            metric
        ]


        for level in [
            5,
            20,
            60,
        ]:

            present_values = target_df[
                target_df["level"]
                == level
            ][metric].dropna()


            if len(
                present_values
            ) == 0:

                continue


            present_mean = float(
                present_values.mean()
            )

            present_sd = float(
                present_values.std()
            )


            delta = (
                present_mean
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
                    "target": target,

                    "metric": metric,

                    "level": level,

                    "baseline_mean": baseline_mean,

                    "present_mean": present_mean,

                    "baseline_sd": baseline_sd,

                    "present_sd": present_sd,

                    "signed_effect_noise": signed_effect_noise,

                    "working": working,
                }
            )


novel_summary = pd.DataFrame(
    novel_rows
)


# ============================================================
# COMPOSITION SCORING
# ============================================================

composition = df[
    df["scenario"]
    == "composition"
]


composition_rows = []


for metric in METRICS:

    values_a = composition[
        composition["level"]
        == "A"
    ][metric].dropna()


    values_b = composition[
        composition["level"]
        == "B"
    ][metric].dropna()


    if (
        len(values_a) == 0
        or len(values_b) == 0
    ):

        continue


    mean_a = float(
        values_a.mean()
    )

    mean_b = float(
        values_b.mean()
    )


    sd_a = float(
        values_a.std()
    )

    sd_b = float(
        values_b.std()
    )


    delta = (
        mean_b
        - mean_a
    )


    noise = noise_floor[
        metric
    ]


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
            "metric": metric,

            "mean_A": mean_a,

            "mean_B": mean_b,

            "sd_A": sd_a,

            "sd_B": sd_b,

            "signed_effect_noise": signed_effect_noise,

            "working": working,
        }
    )


composition_summary = pd.DataFrame(
    composition_rows
)


# ============================================================
# Y POSITIONS
# ============================================================

metric_y = {
    metric: (
        len(METRICS)
        - 1
        - i
    )
    for i, metric
    in enumerate(METRICS)
}


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        13,
        9,
    ),
    sharey=True,
)


ax_bloom = axes[
    0,
    0,
]

ax_novel_1 = axes[
    0,
    1,
]

ax_novel_2 = axes[
    1,
    0,
]

ax_composition = axes[
    1,
    1,
]


# ============================================================
# COMMON AXIS STYLE
# ============================================================

def prepare_axis(
    ax
):

    # zero
    ax.axvline(
        0,
        linewidth=0.8,
        alpha=0.5,
    )


    # positive threshold
    ax.axvline(
        PASS_THRESHOLD_TRANSFORMED,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )


    # negative threshold
    ax.axvline(
        -PASS_THRESHOLD_TRANSFORMED,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )


    ax.grid(
        axis="x",
        alpha=0.15,
    )


    ax.set_yticks(
        [
            metric_y[m]
            for m in METRICS
        ]
    )


    ax.set_yticklabels(
        METRICS
    )


for ax in axes.flat:

    prepare_axis(
        ax
    )


# ============================================================
# PANEL A — BLOOM
# ============================================================

for _, row in bloom_summary.iterrows():

    metric = row[
        "metric"
    ]


    x = float(
        signed_log_effect(
            row[
                "signed_effect_noise"
            ]
        )
    )


    y = metric_y[
        metric
    ]


    if row[
        "working"
    ]:

        ax_bloom.scatter(
            x,
            y,
            s=90,
            color=metric_colors[
                metric
            ],
            zorder=3,
        )

    else:

        ax_bloom.scatter(
            x,
            y,
            s=90,
            facecolors="white",
            edgecolors=metric_colors[
                metric
            ],
            linewidths=1.8,
            zorder=3,
        )


ax_bloom.set_title(
    "A  Abundance bloom"
)


ax_bloom.set_xlabel(
    "Signed bloom response"
)


ax_bloom.set_ylabel(
    "Geometry metric"
)


# ============================================================
# NOVEL PANEL FUNCTION
# ============================================================

novel_markers = {
    5: "o",
    20: "s",
    60: "^",
}


novel_offsets = {
    5: -0.18,
    20: 0.0,
    60: 0.18,
}


def plot_novel_panel(
    ax,
    target,
    panel_letter,
):

    target_data = novel_summary[
        novel_summary[
            "target"
        ]
        == target
    ]


    for _, row in target_data.iterrows():

        metric = row[
            "metric"
        ]

        level = int(
            row[
                "level"
            ]
        )


        x = float(
            signed_log_effect(
                row[
                    "signed_effect_noise"
                ]
            )
        )


        y = (
            metric_y[
                metric
            ]
            + novel_offsets[
                level
            ]
        )


        marker = novel_markers[
            level
        ]


        if row[
            "working"
        ]:

            ax.scatter(
                x,
                y,
                s=75,
                marker=marker,
                color=metric_colors[
                    metric
                ],
                zorder=3,
            )

        else:

            ax.scatter(
                x,
                y,
                s=75,
                marker=marker,
                facecolors="white",
                edgecolors=metric_colors[
                    metric
                ],
                linewidths=1.6,
                zorder=3,
            )


    ax.set_title(
        f"{panel_letter}  Novel taxon: {target}"
    )


    ax.set_xlabel(
        "Signed novelty response"
    )


# ============================================================
# PANELS B + C
# ============================================================

if len(
    novel_targets
) >= 1:

    plot_novel_panel(
        ax_novel_1,
        novel_targets[
            0
        ],
        "B",
    )


if len(
    novel_targets
) >= 2:

    plot_novel_panel(
        ax_novel_2,
        novel_targets[
            1
        ],
        "C",
    )


ax_novel_2.set_ylabel(
    "Geometry metric"
)


# ============================================================
# PANEL D — COMPOSITION
# ============================================================

for _, row in composition_summary.iterrows():

    metric = row[
        "metric"
    ]


    x = float(
        signed_log_effect(
            row[
                "signed_effect_noise"
            ]
        )
    )


    y = metric_y[
        metric
    ]


    if row[
        "working"
    ]:

        ax_composition.scatter(
            x,
            y,
            s=90,
            color=metric_colors[
                metric
            ],
            zorder=3,
        )

    else:

        ax_composition.scatter(
            x,
            y,
            s=90,
            facecolors="white",
            edgecolors=metric_colors[
                metric
            ],
            linewidths=1.8,
            zorder=3,
        )


ax_composition.set_title(
    "D  Community composition"
)


ax_composition.set_xlabel(
    "Signed composition response"
)


# ============================================================
# BETTER X TICK LABELS
#
# Instead of showing signed-log values directly,
# show their corresponding original effect/noise values.
# ============================================================

original_ticks = [
    -20,
    -10,
    -5,
    -2,
    -1,
    0,
    1,
    2,
    5,
    10,
    20,
]


transformed_ticks = [
    float(
        signed_log_effect(
            value
        )
    )
    for value
    in original_ticks
]


for ax in axes.flat:

    ax.set_xticks(
        transformed_ticks
    )

    ax.set_xticklabels(
        [
            str(value)
            for value
            in original_ticks
        ]
    )


# ============================================================
# LEGEND — METRICS
# ============================================================

metric_handles = []


for metric in METRICS:

    metric_handles.append(
        Line2D(
            [0],
            [0],

            marker="o",

            linestyle="none",

            markerfacecolor=metric_colors[
                metric
            ],

            markeredgecolor=metric_colors[
                metric
            ],

            markersize=7,

            label=metric,
        )
    )


# ============================================================
# LEGEND — NOVEL LEVEL
# ============================================================

novel_level_handles = [
    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="none",

        color="black",

        markerfacecolor="black",

        markersize=7,

        label="n5",
    ),

    Line2D(
        [0],
        [0],

        marker="s",

        linestyle="none",

        color="black",

        markerfacecolor="black",

        markersize=7,

        label="n20",
    ),

    Line2D(
        [0],
        [0],

        marker="^",

        linestyle="none",

        color="black",

        markerfacecolor="black",

        markersize=7,

        label="n60",
    ),
]


# pass / fail explanation

pass_fail_handles = [
    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="none",

        color="black",

        markerfacecolor="black",

        markersize=7,

        label="Working",
    ),

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="none",

        color="black",

        markerfacecolor="white",

        markersize=7,

        label="Not working",
    ),
]


# ============================================================
# LEGENDS
# ============================================================

fig.legend(
    handles=metric_handles,

    loc="lower center",

    ncol=4,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        -0.005,
    ),
)


ax_novel_1.legend(
    handles=novel_level_handles,

    title="Novel level",

    frameon=False,

    loc="best",
)


ax_composition.legend(
    handles=pass_fail_handles,

    frameon=False,

    loc="best",
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Tier-1 response of representation geometry "
    "to controlled ecological perturbations",
    fontsize=15,
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.045,

    "x-axis values show signed effect / null noise "
    "on a compressed logarithmic scale. "
    "Dotted lines mark |effect / null noise| = 1. "
    "Filled markers pass the diagnostic criterion; "
    "open markers fail.",

    ha="center",

    fontsize=9,
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0,
        0.09,
        1,
        0.95,
    ]
)


# ============================================================
# SAVE FIGURE
# ============================================================

png_file = (
    OUT_DIR
    / "figure2_tier1_clean_compact.png"
)

pdf_file = (
    OUT_DIR
    / "figure2_tier1_clean_compact.pdf"
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
# SAVE NUMERICAL TABLES
# ============================================================

bloom_summary.to_csv(
    OUT_DIR
    / "figure2_bloom_summary.csv",
    index=False,
)


novel_summary.to_csv(
    OUT_DIR
    / "figure2_novel_summary.csv",
    index=False,
)


composition_summary.to_csv(
    OUT_DIR
    / "figure2_composition_summary.csv",
    index=False,
)


# ============================================================
# SHOW
# ============================================================

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