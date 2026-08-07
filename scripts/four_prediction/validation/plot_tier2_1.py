import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

GEOMETRY_FILE = Path(
    r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\geometry\geometry_metrics.csv"
)

OUT_DIR = Path(
    r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\plots"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
# VISUAL SETTINGS
# ============================================================

METRIC_COLORS = {
    metric: f"C{i}"
    for i, metric in enumerate(METRICS)
}


SCENARIOS = [
    "abundance_bloom_additive",
    "abundance_bloom_fixedn",
    "mixture_gradient",
    "novel_category_additive",
    "novel_category_fixedn",
    "real_profile_contrast",
]


SCENARIO_TITLES = {
    "abundance_bloom_additive":
        "A  Bloom — additive",

    "abundance_bloom_fixedn":
        "B  Bloom — fixed-N",

    "mixture_gradient":
        "C  Mixture gradient",

    "novel_category_additive":
        "D  Novel category — additive",

    "novel_category_fixedn":
        "E  Novel category — fixed-N",

    "real_profile_contrast":
        "F  Real-profile contrast",
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    GEOMETRY_FILE
)


required = [
    "group_id",
    *METRICS,
]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DISCOVER REAL ANCHORS FROM NULL CONTROLS
# ============================================================

NULL_RE = re.compile(
    r"^null_control__null_control_(.+)_rep(\d+)$"
)


anchors = []


for gid in df["group_id"]:

    m = NULL_RE.match(
        str(gid)
    )

    if m:

        anchors.append(
            m.group(1)
        )


anchors = sorted(
    set(anchors),
    key=len,
    reverse=True,
)


if not anchors:

    raise ValueError(
        "Could not reconstruct anchors from null-control group IDs."
    )


print("\nDiscovered anchors:")

for anchor in anchors:
    print("  ", anchor)


# ============================================================
# PARSER HELPERS
# ============================================================

def find_anchor_prefix(text):

    for anchor in anchors:

        if text.startswith(
            anchor
        ):
            return anchor

    return None


def parse_anchor_pair(text):

    for anchor_1 in anchors:

        prefix = (
            anchor_1
            + "_vs_"
        )

        if text.startswith(
            prefix
        ):

            anchor_2 = text[
                len(prefix):
            ]

            if anchor_2 in anchors:

                return (
                    anchor_1,
                    anchor_2,
                )

    return (
        None,
        None,
    )


# ============================================================
# PARSE GROUP ID
# ============================================================

def parse_group_id(group_id):

    gid = str(
        group_id
    )


    # --------------------------------------------------------
    # NULL
    # --------------------------------------------------------

    m = NULL_RE.match(
        gid
    )

    if m:

        return {
            "scenario_type":
                "null_control",

            "anchor":
                m.group(1),

            "anchor_2":
                None,

            "target":
                "null",

            "level":
                int(
                    m.group(2)
                ),

            "replicate":
                int(
                    m.group(2)
                ),

            "unit":
                m.group(1),
        }


    # --------------------------------------------------------
    # BLOOM ADDITIVE
    # --------------------------------------------------------

    prefix = (
        "abundance_bloom_additive"
        "__bloomadd_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_x(\d+)_rep(\d+)$",
            rest,
        )

        if m:

            anchor_target = m.group(1)

            anchor = find_anchor_prefix(
                anchor_target
            )

            if anchor is not None:

                target = anchor_target[
                    len(anchor) + 1:
                ]

                return {
                    "scenario_type":
                        "abundance_bloom_additive",

                    "anchor":
                        anchor,

                    "anchor_2":
                        None,

                    "target":
                        target,

                    "level":
                        int(
                            m.group(2)
                        ),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        anchor,
                }


    # --------------------------------------------------------
    # BLOOM FIXED-N
    # --------------------------------------------------------

    prefix = (
        "abundance_bloom_fixedn"
        "__bloomfix_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_x(\d+)_rep(\d+)$",
            rest,
        )

        if m:

            anchor_target = m.group(1)

            anchor = find_anchor_prefix(
                anchor_target
            )

            if anchor is not None:

                target = anchor_target[
                    len(anchor) + 1:
                ]

                return {
                    "scenario_type":
                        "abundance_bloom_fixedn",

                    "anchor":
                        anchor,

                    "anchor_2":
                        None,

                    "target":
                        target,

                    "level":
                        int(
                            m.group(2)
                        ),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        anchor,
                }


    # --------------------------------------------------------
    # NOVEL ADDITIVE
    # --------------------------------------------------------

    prefix = (
        "novel_category_additive"
        "__noveladd_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_n(\d+)_rep(\d+)$",
            rest,
        )

        if m:

            anchor_target = m.group(1)

            anchor = find_anchor_prefix(
                anchor_target
            )

            if anchor is not None:

                target = anchor_target[
                    len(anchor) + 1:
                ]

                return {
                    "scenario_type":
                        "novel_category_additive",

                    "anchor":
                        anchor,

                    "anchor_2":
                        None,

                    "target":
                        target,

                    "level":
                        int(
                            m.group(2)
                        ),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        anchor,
                }


    # --------------------------------------------------------
    # NOVEL FIXED-N
    # --------------------------------------------------------

    prefix = (
        "novel_category_fixedn"
        "__novelfix_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_n(\d+)_rep(\d+)$",
            rest,
        )

        if m:

            anchor_target = m.group(1)

            anchor = find_anchor_prefix(
                anchor_target
            )

            if anchor is not None:

                target = anchor_target[
                    len(anchor) + 1:
                ]

                return {
                    "scenario_type":
                        "novel_category_fixedn",

                    "anchor":
                        anchor,

                    "anchor_2":
                        None,

                    "target":
                        target,

                    "level":
                        int(
                            m.group(2)
                        ),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        anchor,
                }


    # --------------------------------------------------------
    # MIXTURE
    # --------------------------------------------------------

    prefix = (
        "mixture_gradient"
        "__mixture_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_p(\d+)_rep(\d+)$",
            rest,
        )

        if m:

            pair_text = m.group(1)

            anchor_1, anchor_2 = (
                parse_anchor_pair(
                    pair_text
                )
            )

            if (
                anchor_1 is not None
                and anchor_2 is not None
            ):

                unit = (
                    anchor_1
                    + "_vs_"
                    + anchor_2
                )

                return {
                    "scenario_type":
                        "mixture_gradient",

                    "anchor":
                        anchor_1,

                    "anchor_2":
                        anchor_2,

                    "target":
                        unit,

                    "level":
                        int(
                            m.group(2)
                        ),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        unit,
                }


    # --------------------------------------------------------
    # REAL PROFILE CONTRAST
    # --------------------------------------------------------

    prefix = (
        "real_profile_contrast"
        "__contrast_"
    )

    if gid.startswith(
        prefix
    ):

        rest = gid[
            len(prefix):
        ]

        m = re.match(
            r"(.+)_([AB])_rep(\d+)$",
            rest,
        )

        if m:

            pair_text = m.group(1)

            anchor_1, anchor_2 = (
                parse_anchor_pair(
                    pair_text
                )
            )

            if (
                anchor_1 is not None
                and anchor_2 is not None
            ):

                unit = (
                    anchor_1
                    + "_vs_"
                    + anchor_2
                )

                return {
                    "scenario_type":
                        "real_profile_contrast",

                    "anchor":
                        anchor_1,

                    "anchor_2":
                        anchor_2,

                    "target":
                        unit,

                    "level":
                        m.group(2),

                    "replicate":
                        int(
                            m.group(3)
                        ),

                    "unit":
                        unit,
                }


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        "scenario_type":
            "unknown",

        "anchor":
            None,

        "anchor_2":
            None,

        "target":
            None,

        "level":
            np.nan,

        "replicate":
            np.nan,

        "unit":
            None,
    }


# ============================================================
# APPLY PARSER
# ============================================================

parsed = pd.DataFrame(
    df["group_id"]
    .apply(
        parse_group_id
    )
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


print(
    "\nParsed scenarios:"
)

print(
    df[
        "scenario_type"
    ].value_counts()
)


unknown_n = int(
    (
        df[
            "scenario_type"
        ]
        == "unknown"
    ).sum()
)


if unknown_n > 0:

    print(
        f"\nWARNING: {unknown_n} groups could not be parsed."
    )


# ============================================================
# ANCHOR-SPECIFIC NULL NOISE
# ============================================================

noise_rows = []


null = df[
    df["scenario_type"]
    == "null_control"
]


for anchor, g in null.groupby(
    "anchor"
):

    for metric in METRICS:

        values = (
            g[metric]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(values) < 2:

            noise = np.nan

        else:

            pairwise = [
                abs(
                    values[i]
                    - values[j]
                )

                for i, j
                in combinations(
                    range(
                        len(values)
                    ),
                    2,
                )
            ]

            noise = float(
                np.median(
                    pairwise
                )
            )

        noise_rows.append(
            {
                "anchor":
                    anchor,

                "metric":
                    metric,

                "noise":
                    noise,
            }
        )


noise_df = pd.DataFrame(
    noise_rows
)


def anchor_noise(
    anchor,
    metric,
):

    row = noise_df[
        (
            noise_df["anchor"]
            == anchor
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
        row["noise"]
        .iloc[0]
    )


def pair_noise(
    anchor_1,
    anchor_2,
    metric,
):

    values = [
        anchor_noise(
            anchor_1,
            metric,
        ),

        anchor_noise(
            anchor_2,
            metric,
        ),
    ]

    values = [
        value
        for value in values
        if pd.notna(
            value
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# SCORE TABLE
# ============================================================

score_rows = []


# ============================================================
# BLOOM + MIXTURE
# ============================================================

ORDINAL_SCENARIOS = [
    "abundance_bloom_additive",
    "abundance_bloom_fixedn",
    "mixture_gradient",
]


for scenario_type in ORDINAL_SCENARIOS:

    sub = df[
        df["scenario_type"]
        == scenario_type
    ]

    for unit, g in sub.groupby(
        "unit"
    ):

        anchor = g[
            "anchor"
        ].iloc[0]

        for metric in METRICS:

            values = g[
                [
                    "level",
                    metric,
                ]
            ].dropna()

            if (
                len(values) < 4
                or values["level"].nunique()
                < 2
            ):
                continue


            r, p = spearmanr(
                values["level"],
                values[metric],
            )


            means = (
                g.groupby(
                    "level"
                )[metric]
                .mean()
            )


            effect = float(
                means.max()
                - means.min()
            )


            noise = anchor_noise(
                anchor,
                metric,
            )


            effect_noise = (
                abs(effect)
                / noise

                if (
                    pd.notna(noise)
                    and noise > 0
                )

                else np.nan
            )


            working = (
                pd.notna(r)
                and abs(r)
                >= SPEARMAN_MIN
                and pd.notna(
                    effect_noise
                )
                and effect_noise
                >= EFFECT_OVER_NOISE_MIN
            )


            score_rows.append(
                {
                    "scenario_type":
                        scenario_type,

                    "unit":
                        unit,

                    "metric":
                        metric,

                    "level":
                        np.nan,

                    "effect_noise":
                        effect_noise,

                    "working":
                        working,

                    "spearman_r":
                        r,
                }
            )


# ============================================================
# NOVELTY
# ============================================================

NOVEL_SCENARIOS = [
    "novel_category_additive",
    "novel_category_fixedn",
]


for scenario_type in NOVEL_SCENARIOS:

    sub = df[
        df["scenario_type"]
        == scenario_type
    ]


    for (
        unit,
        target,
    ), g in sub.groupby(
        [
            "unit",
            "target",
        ]
    ):

        anchor = g[
            "anchor"
        ].iloc[0]


        baseline = g[
            g["level"]
            == 0
        ]


        for level in [
            5,
            20,
            60,
        ]:

            present = g[
                g["level"]
                == level
            ]


            for metric in METRICS:

                baseline_values = (
                    baseline[metric]
                    .dropna()
                )

                present_values = (
                    present[metric]
                    .dropna()
                )


                if (
                    len(baseline_values) < 2
                    or len(present_values) < 2
                ):
                    continue


                effect = float(
                    present_values.mean()
                    - baseline_values.mean()
                )


                noise = anchor_noise(
                    anchor,
                    metric,
                )


                effect_noise = (
                    abs(effect)
                    / noise

                    if (
                        pd.notna(noise)
                        and noise > 0
                    )

                    else np.nan
                )


                working = (
                    pd.notna(
                        effect_noise
                    )
                    and effect_noise
                    >= EFFECT_OVER_NOISE_MIN
                )


                score_rows.append(
                    {
                        "scenario_type":
                            scenario_type,

                        "unit":
                            unit,

                        "metric":
                            metric,

                        "level":
                            level,

                        "effect_noise":
                            effect_noise,

                        "working":
                            working,

                        "spearman_r":
                            np.nan,
                    }
                )


# ============================================================
# REAL PROFILE CONTRAST
# ============================================================

sub = df[
    df["scenario_type"]
    == "real_profile_contrast"
]


for unit, g in sub.groupby(
    "unit"
):

    anchor_1 = g[
        "anchor"
    ].iloc[0]

    anchor_2 = g[
        "anchor_2"
    ].iloc[0]


    side_a = g[
        g["level"]
        == "A"
    ]

    side_b = g[
        g["level"]
        == "B"
    ]


    for metric in METRICS:

        a = (
            side_a[metric]
            .dropna()
        )

        b = (
            side_b[metric]
            .dropna()
        )


        if (
            len(a) < 2
            or len(b) < 2
        ):
            continue


        effect = float(
            b.mean()
            - a.mean()
        )


        noise = pair_noise(
            anchor_1,
            anchor_2,
            metric,
        )


        effect_noise = (
            abs(effect)
            / noise

            if (
                pd.notna(noise)
                and noise > 0
            )

            else np.nan
        )


        working = (
            pd.notna(
                effect_noise
            )
            and effect_noise
            >= EFFECT_OVER_NOISE_MIN
        )


        score_rows.append(
            {
                "scenario_type":
                    "real_profile_contrast",

                "unit":
                    unit,

                "metric":
                    metric,

                "level":
                    np.nan,

                "effect_noise":
                    effect_noise,

                "working":
                    working,

                "spearman_r":
                    np.nan,
            }
        )


score = pd.DataFrame(
    score_rows
)


# ============================================================
# COLLAPSE NOVELTY
#
# Take weakest n5/n20/n60 response for each anchor.
#
# This represents the most conservative novelty performance.
# ============================================================

novel_collapsed = (
    score[
        score["scenario_type"]
        .isin(
            NOVEL_SCENARIOS
        )
    ]
    .groupby(
        [
            "scenario_type",
            "unit",
            "metric",
        ],
        as_index=False,
    )
    .agg(
        effect_noise=(
            "effect_noise",
            "min",
        ),

        working=(
            "working",
            "all",
        ),
    )
)


# ============================================================
# OTHER SCENARIOS
# ============================================================

other_scores = score[
    ~score["scenario_type"]
    .isin(
        NOVEL_SCENARIOS
    )
][
    [
        "scenario_type",
        "unit",
        "metric",
        "effect_noise",
        "working",
    ]
].copy()


# ============================================================
# FINAL PLOT DATA
# ============================================================

plot_df = pd.concat(
    [
        other_scores,
        novel_collapsed,
    ],
    ignore_index=True,
)


plot_df.to_csv(
    OUT_DIR
    / "figure4_tier2_anchor_effects.csv",
    index=False,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = (
    plot_df
    .groupby(
        [
            "scenario_type",
            "metric",
        ]
    )
    .agg(
        median_effect_noise=(
            "effect_noise",
            "median",
        ),

        minimum_effect_noise=(
            "effect_noise",
            "min",
        ),

        maximum_effect_noise=(
            "effect_noise",
            "max",
        ),

        n_units=(
            "unit",
            "nunique",
        ),

        n_passed=(
            "working",
            "sum",
        ),
    )
    .reset_index()
)


summary[
    "pass_fraction"
] = (
    summary["n_passed"]
    / summary["n_units"]
)


summary.to_csv(
    OUT_DIR
    / "figure4_tier2_anchor_effect_summary.csv",
    index=False,
)


print(
    "\nTier-2 plotting summary:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# GLOBAL LOG X LIMITS
# ============================================================

finite_effects = (
    plot_df["effect_noise"]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .dropna()
)


finite_effects = finite_effects[
    finite_effects > 0
]


if finite_effects.empty:

    XMIN = 0.1
    XMAX = 100

else:

    XMIN = max(
        0.1,

        10 ** np.floor(
            np.log10(
                finite_effects.min()
            )
        ),
    )


    XMAX = (
        10 ** np.ceil(
            np.log10(
                finite_effects.max()
            )
        )
    )


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        15,
        8.7,
    ),
    sharex=True,
    sharey=True,
)


axes = axes.ravel()


# ============================================================
# METRIC Y POSITIONS
# ============================================================

metric_y = {
    metric:
        len(METRICS)
        - 1
        - i

    for i, metric
    in enumerate(
        METRICS
    )
}


# ============================================================
# DRAW PANELS
# ============================================================

for ax, scenario_type in zip(
    axes,
    SCENARIOS,
):

    scenario_df = plot_df[
        plot_df["scenario_type"]
        == scenario_type
    ]


    # --------------------------------------------------------
    # Null-noise threshold
    # --------------------------------------------------------

    ax.axvline(
        1,

        color="0.35",

        linestyle="--",

        linewidth=1.25,

        alpha=0.8,

        zorder=0,
    )


    # --------------------------------------------------------
    # Each metric
    # --------------------------------------------------------

    for metric in METRICS:

        metric_df = scenario_df[
            scenario_df["metric"]
            == metric
        ].copy()


        if metric_df.empty:
            continue


        metric_df = metric_df.sort_values(
            "unit"
        )


        n_points = len(
            metric_df
        )


        # Slight vertical jitter to reveal anchors
        if n_points == 1:

            offsets = np.array(
                [0.0]
            )

        else:

            offsets = np.linspace(
                -0.14,
                0.14,
                n_points,
            )


        y_base = metric_y[
            metric
        ]


        # ----------------------------------------------------
        # Individual anchors
        # ----------------------------------------------------

        for offset, (_, row) in zip(
            offsets,
            metric_df.iterrows(),
        ):

            x = row[
                "effect_noise"
            ]


            if (
                pd.isna(x)
                or x <= 0
            ):
                continue


            y = (
                y_base
                + offset
            )


            if row[
                "working"
            ]:

                ax.scatter(
                    x,
                    y,

                    s=40,

                    marker="o",

                    color=METRIC_COLORS[
                        metric
                    ],

                    alpha=0.5,

                    edgecolors="none",

                    zorder=2,
                )

            else:

                ax.scatter(
                    x,
                    y,

                    s=44,

                    marker="o",

                    facecolors="white",

                    edgecolors=METRIC_COLORS[
                        metric
                    ],

                    linewidths=1.3,

                    zorder=2,
                )


        # ----------------------------------------------------
        # Median
        # ----------------------------------------------------

        median_x = float(
            metric_df[
                "effect_noise"
            ].median()
        )


        if (
            pd.notna(
                median_x
            )
            and median_x > 0
        ):

            ax.scatter(
                median_x,
                y_base,

                s=115,

                marker="D",

                color=METRIC_COLORS[
                    metric
                ],

                edgecolor="black",

                linewidth=0.8,

                zorder=4,
            )


    # --------------------------------------------------------
    # PANEL STYLE
    # --------------------------------------------------------

    ax.set_xscale(
        "log"
    )


    ax.set_xlim(
        XMIN,
        XMAX,
    )


    ax.set_title(
        SCENARIO_TITLES[
            scenario_type
        ],
        fontsize=11,
        pad=7,
    )


    ax.grid(
        axis="x",
        which="both",
        alpha=0.10,
    )


    # subtle horizontal metric guides
    for y in metric_y.values():

        ax.axhline(
            y,
            color="0.93",
            linewidth=0.7,
            zorder=0,
        )


# ============================================================
# Y AXIS
# ============================================================

for ax in axes:

    ax.set_yticks(
        [
            metric_y[m]
            for m in METRICS
        ]
    )

    ax.set_yticklabels(
        METRICS
    )


axes[0].set_ylabel(
    "Geometry metric"
)

axes[3].set_ylabel(
    "Geometry metric"
)


# ============================================================
# X AXIS
# ============================================================

for ax in axes[3:]:

    ax.set_xlabel(
        "Ecological effect / anchor-specific null noise"
    )


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Tier-2 effect strength and generalisation "
    "across real community anchors",
    fontsize=16,
    y=0.965,
)


# ============================================================
# LEGEND
#
# Keep this compact.
# Do NOT put manuscript caption text inside the figure.
# ============================================================

legend_handles = [

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="none",

        markerfacecolor="0.55",

        markeredgecolor="none",

        markersize=6,

        label="Individual anchor / pair",
    ),


    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="none",

        markerfacecolor="white",

        markeredgecolor="0.35",

        markeredgewidth=1.3,

        markersize=6,

        label="Diagnostic fail",
    ),


    Line2D(
        [0],
        [0],

        marker="D",

        linestyle="none",

        markerfacecolor="0.55",

        markeredgecolor="black",

        markersize=7,

        label="Median across anchors",
    ),


    Line2D(
        [0],
        [0],

        linestyle="--",

        color="0.35",

        linewidth=1.2,

        label="Effect = null noise",
    ),
]


fig.legend(
    handles=legend_handles,

    loc="lower center",

    bbox_to_anchor=(
        0.5,
        0.018,
    ),

    ncol=4,

    frameon=False,

    fontsize=9,
)


# ============================================================
# LAYOUT
#
# Manual spacing avoids the previous overlap.
# ============================================================

fig.subplots_adjust(
    left=0.11,
    right=0.985,
    top=0.90,
    bottom=0.13,
    wspace=0.09,
    hspace=0.13,
)


# ============================================================
# SAVE
# ============================================================

png_file = (
    OUT_DIR
    / "figure4_tier2_effect_strength_by_anchor_final.png"
)


pdf_file = (
    OUT_DIR
    / "figure4_tier2_effect_strength_by_anchor_final.pdf"
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


plt.show()


print("\nSaved:")
print(png_file)
print(pdf_file)