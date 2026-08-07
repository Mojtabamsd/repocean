import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Only representative metrics for this explanatory figure
# ------------------------------------------------------------

METRICS = [
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


# ============================================================
# GROUP-ID PARSING
# ============================================================

BLOOM_RE = re.compile(
    r"abundance_bloom__bloom_(.+)_x(\d+)_rep(\d+)$"
)


def parse_bloom_group(group_id):

    gid = str(group_id)

    m = BLOOM_RE.match(gid)

    if m:

        return {
            "scenario": "bloom",
            "target": m.group(1),
            "level": int(m.group(2)),
            "replicate": int(m.group(3)),
        }

    return {
        "scenario": "other",
        "target": "",
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

    df = pd.read_csv(
        path
    )

    missing = [
        metric
        for metric in METRICS
        if metric not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{path} is missing metrics: {missing}"
        )


    parsed = pd.DataFrame(
        df["group_id"]
        .apply(parse_bloom_group)
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


    df["detritus_n"] = (
        detritus_n
    )


    # Keep bloom only
    df = df[
        df["scenario"]
        == "bloom"
    ].copy()


    frames.append(
        df
    )


all_df = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# CHECK
# ============================================================

print(
    "\nBloom rows by detritus:"
)

print(
    all_df.groupby(
        "detritus_n"
    ).size()
)


print(
    "\nBloom levels:"
)

print(
    pd.crosstab(
        all_df["detritus_n"],
        all_df["level"],
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for detritus_n in DET_ORDER:

    run_df = all_df[
        all_df["detritus_n"]
        == detritus_n
    ]


    for metric in METRICS:

        for level in [
            1,
            3,
            10,
        ]:

            values = run_df[
                run_df["level"]
                == level
            ][metric].dropna()


            if len(values) == 0:
                continue


            summary_rows.append(
                {
                    "detritus_n":
                        detritus_n,

                    "metric":
                        metric,

                    "level":
                        level,

                    "mean":
                        float(
                            values.mean()
                        ),

                    "sd":
                        float(
                            values.std()
                        ),

                    "n":
                        len(values),
                }
            )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT NUMBERS
# ============================================================

print(
    "\nSummary:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        14,
        4.8,
    ),
)


# ------------------------------------------------------------
# Give each detritus condition one consistent style.
#
# Colour now represents DETRITUS, not metric, because each
# panel already represents one metric.
# ------------------------------------------------------------

detritus_colors = {
    detritus: f"C{i}"
    for i, detritus
    in enumerate(
        DET_ORDER
    )
}


detritus_markers = {
    0: "o",
    200: "s",
    1000: "^",
    10000: "D",
}


# ------------------------------------------------------------
# Human-readable panel names
# ------------------------------------------------------------

titles = {
    "pair_cos_p10":
        "A  Pairwise cosine p10",

    "cos_p50":
        "B  Cosine p50",

    "eff_rank":
        "C  Effective rank",
}


# ============================================================
# EACH METRIC GETS ITS OWN PANEL
# ============================================================

for ax, metric in zip(
    axes,
    METRICS,
):

    metric_df = summary[
        summary["metric"]
        == metric
    ]


    for detritus_n in DET_ORDER:

        d = metric_df[
            metric_df[
                "detritus_n"
            ]
            == detritus_n
        ].sort_values(
            "level"
        )


        if d.empty:
            continue


        ax.errorbar(
            d["level"],
            d["mean"],

            yerr=d["sd"],

            marker=detritus_markers[
                detritus_n
            ],

            markersize=7,

            color=detritus_colors[
                detritus_n
            ],

            linewidth=1.7,

            capsize=3,

            label=(
                f"{detritus_n}"
            ),

            zorder=3,
        )


    ax.set_title(
        titles[
            metric
        ]
    )


    ax.set_xlabel(
        "Bloom multiplier"
    )


    ax.set_ylabel(
        metric
    )


    ax.set_xticks(
        [
            1,
            3,
            10,
        ]
    )


    ax.set_xticklabels(
        [
            "x1",
            "x3",
            "x10",
        ]
    )


    ax.grid(
        alpha=0.15,
    )


# ============================================================
# LEGEND
# ============================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,

    title=(
        "Detritus objects added"
    ),

    loc="lower center",

    ncol=4,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        -0.04,
    ),
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Detrital background alters the geometry response "
    "to the same abundance perturbation",
    fontsize=15,
)


fig.text(
    0.5,
    0.015,

    "Points show mean metric value across four replicate "
    "synthetic profiles; error bars show ±1 SD.",

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
        0.92,
    ]
)


# ============================================================
# SAVE
# ============================================================

png_file = (
    OUT_DIR
    / "figure4_bloom_geometry_mechanism.png"
)


pdf_file = (
    OUT_DIR
    / "figure4_bloom_geometry_mechanism.pdf"
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


summary.to_csv(
    OUT_DIR
    / "figure4_bloom_raw_summary.csv",
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