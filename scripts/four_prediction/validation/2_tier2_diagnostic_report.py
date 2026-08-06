"""
tier2_diagnostic_report_final.py

Goal
----
Turn Tier-2 geometry_metrics.csv into a readable diagnostic report showing:

1. Whether each geometry metric responds to realistic ecological perturbations.
2. Whether responses generalise across different real anchor profiles.
3. Whether replicate variability is small relative to the ecological effect.

Scenario logic
--------------
- Bloom additive / fixed-N:
    Spearman trend across bloom levels + effect-over-anchor-null-noise.

- Mixture gradient:
    Spearman trend across donor fractions + effect-over-anchor-null-noise.

- Novel category additive / fixed-N:
    Compare n=0 separately with every positive injected count.
    No monotonicity requirement is imposed.

- Real profile contrast / legacy composition swap:
    Compare A versus B mean shift against the relevant anchor null noise.

Outputs
-------
- diagnostic_scorecard.csv
- diagnostic_anchor_rollup.csv
- diagnostic_report.txt
- spotlight PNGs
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

IN_PATH = Path(
    r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\geometry\geometry_metrics.csv"
)
OUT_DIR = Path(
    r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\geometry\tier2_metric_analysis"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS = 5
SPEARMAN_MIN = 0.4
EFFECT_OVER_NOISE_MIN = 1.0

METRICS = [
    "pred_num_classes_present",
    "shannon",
    "exp_shannon",
    "centroid_norm",
    "cos_mean",
    "cos_p10",
    "cos_p50",
    "cos_p90",
    "pair_cos_mean",
    "pair_cos_p10",
    "pair_cos_p50",
    "pair_cos_p90",
    "eff_rank",
    "pca_dim_90",
    "pca_dim_95",
]


# ---------------------------------------------------------------------------
# GROUP-ID PARSING
# ---------------------------------------------------------------------------

SCENARIO_PATTERNS = [
    # Tier 2
    (
        "null_control",
        re.compile(r"_null_control_(.+)_rep(\d+)$"),
        "null",
    ),
    (
        "abundance_bloom_additive",
        re.compile(r"_bloomadd_(.+)_x(\d+)_rep(\d+)$"),
        "ordinal",
    ),
    (
        "abundance_bloom_fixedn",
        re.compile(r"_bloomfix_(.+)_x(\d+)_rep(\d+)$"),
        "ordinal",
    ),
    (
        "novel_category_additive",
        re.compile(r"_noveladd_(.+)_n(\d+)_rep(\d+)$"),
        "ordinal",
    ),
    (
        "novel_category_fixedn",
        re.compile(r"_novelfix_(.+)_n(\d+)_rep(\d+)$"),
        "ordinal",
    ),
    (
        "real_profile_contrast",
        re.compile(r"_contrast_(.+)_(A|B)_rep(\d+)$"),
        "pair",
    ),
    (
        "mixture_gradient",
        re.compile(r"_mixture_(.+)_p(\d+)_rep(\d+)$"),
        "ordinal",
    ),
    # Backward compatibility
    (
        "abundance_bloom_additive",
        re.compile(r"_bloom_(.+)_x(\d+)(?:_rep(\d+))?$"),
        "ordinal",
    ),
    (
        "novel_category_additive",
        re.compile(r"_novel_(.+)_n(\d+)(?:_rep(\d+))?$"),
        "ordinal",
    ),
    (
        "composition_swap",
        re.compile(r"_composition_(?:(.+)_)?(A|B)(?:_rep(\d+))?$"),
        "pair",
    ),
]


def discover_anchor_ids(group_ids: pd.Series) -> list[str]:
    """Discover Tier-2 anchor IDs from null-control group names.

    geometry_metrics.csv often contains only ``group_id`` and metric values,
    not the original ``anchor_sample_id`` column. Null-control IDs are the
    safest source because their structure is unambiguous:

        ..._null_control_<anchor_id>_rep1

    The discovered IDs are sorted longest-first so one anchor cannot be
    mistaken for a prefix of another.
    """
    anchors: set[str] = set()
    pattern = re.compile(r"_null_control_(.+)_rep\d+$")

    for group_id in group_ids.astype(str):
        match = pattern.search(group_id)
        if match:
            anchors.add(match.group(1))

    return sorted(anchors, key=lambda value: (-len(value), value))


def split_anchor_and_target(body: str, anchors: list[str]) -> tuple[str, str]:
    """Split ``<anchor>_<target>`` using known exact anchor IDs."""
    for anchor in anchors:
        prefix = f"{anchor}_"
        if body.startswith(prefix):
            return anchor, body[len(prefix) :]
        if body == anchor:
            return anchor, "unknown"
    return "global", body


def split_anchor_pair(body: str, anchors: list[str]) -> tuple[str, str]:
    """Split ``<anchor_x>_vs_<anchor_y>`` using known anchor IDs."""
    for anchor_x in anchors:
        prefix = f"{anchor_x}_vs_"
        if not body.startswith(prefix):
            continue
        remainder = body[len(prefix) :]
        for anchor_y in anchors:
            if remainder == anchor_y:
                return anchor_x, anchor_y
    return "global", "global"


def parse_group_id(
    group_id: str,
    anchors: list[str],
) -> tuple[str, str, float, float, str, str | float]:
    """Parse one Tier-2 group ID.

    Returns
    -------
    scenario_type, scenario_target, scenario_level, replicate,
    anchor_sample_id, donor_sample_id

    Anchor IDs are reconstructed from the group name when the geometry CSV
    does not preserve metadata columns.
    """
    text = str(group_id)

    match = re.search(r"_null_control_(.+)_rep(\d+)$", text)
    if match:
        anchor = match.group(1)
        return "null_control", "null_control", 0.0, float(match.group(2)), anchor, np.nan

    ordinal_specs = [
        ("abundance_bloom_additive", "bloomadd", "x"),
        ("abundance_bloom_fixedn", "bloomfix", "x"),
        ("novel_category_additive", "noveladd", "n"),
        ("novel_category_fixedn", "novelfix", "n"),
    ]

    for scenario_type, prefix, level_letter in ordinal_specs:
        pattern = rf"_{prefix}_(.+)_{level_letter}(\d+)_rep(\d+)$"
        match = re.search(pattern, text)
        if match:
            body = match.group(1)
            anchor, target = split_anchor_and_target(body, anchors)
            return (
                scenario_type,
                target,
                float(match.group(2)),
                float(match.group(3)),
                anchor,
                np.nan,
            )

    match = re.search(r"_mixture_(.+)_p(\d+)_rep(\d+)$", text)
    if match:
        body = match.group(1)
        anchor_x, anchor_y = split_anchor_pair(body, anchors)
        target = f"{anchor_x}_vs_{anchor_y}" if anchor_x != "global" else body
        return (
            "mixture_gradient",
            target,
            float(match.group(2)),
            float(match.group(3)),
            anchor_x,
            anchor_y if anchor_y != "global" else np.nan,
        )

    match = re.search(r"__contrast_(.+)_(A|B)_rep(\d+)$", text)
    if match:
        body = match.group(1)
        side = match.group(2)
        anchor_x, anchor_y = split_anchor_pair(body, anchors)
        target = f"{anchor_x}_vs_{anchor_y}" if anchor_x != "global" else body
        side_anchor = anchor_x if side == "A" else anchor_y
        other_anchor = anchor_y if side == "A" else anchor_x
        return (
            "real_profile_contrast",
            target,
            0.0 if side == "A" else 1.0,
            float(match.group(3)),
            side_anchor,
            other_anchor if other_anchor != "global" else np.nan,
        )

    # Backward-compatible Tier-1 names. These do not generally encode a
    # recoverable anchor, so they remain under the global pseudo-anchor.
    match = re.search(r"_bloom_(.+)_x(\d+)(?:_rep(\d+))?$", text)
    if match:
        replicate = match.group(3)
        return (
            "abundance_bloom_additive",
            match.group(1),
            float(match.group(2)),
            float(replicate) if replicate else np.nan,
            "global",
            np.nan,
        )

    match = re.search(r"_novel_(.+)_n(\d+)(?:_rep(\d+))?$", text)
    if match:
        replicate = match.group(3)
        return (
            "novel_category_additive",
            match.group(1),
            float(match.group(2)),
            float(replicate) if replicate else np.nan,
            "global",
            np.nan,
        )

    match = re.search(r"_composition_(?:(.+)_)?(A|B)(?:_rep(\d+))?$", text)
    if match:
        replicate = match.group(3)
        return (
            "composition_swap",
            match.group(1) or "composition_swap",
            0.0 if match.group(2) == "A" else 1.0,
            float(replicate) if replicate else np.nan,
            "global",
            np.nan,
        )

    return "unknown", "unknown", np.nan, np.nan, "global", np.nan


def load_and_parse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"group_id", "num_rows"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    missing_metrics = [metric for metric in METRICS if metric not in df.columns]
    if missing_metrics:
        print(f"[warn] Missing metrics will be skipped: {missing_metrics}")

    discovered_anchors = discover_anchor_ids(df["group_id"])
    if discovered_anchors:
        print(f"Discovered {len(discovered_anchors)} anchors from group_id:")
        for anchor in discovered_anchors:
            print(f"  - {anchor}")
    else:
        print("[warn] No anchor IDs could be discovered from null-control group IDs.")

    parsed = df["group_id"].apply(
        lambda group_id: parse_group_id(group_id, discovered_anchors)
    )
    df["scenario_type"] = parsed.apply(lambda value: value[0])
    df["scenario_target"] = parsed.apply(lambda value: value[1])
    df["scenario_level"] = parsed.apply(lambda value: value[2])
    df["replicate"] = parsed.apply(lambda value: value[3])
    parsed_anchor = parsed.apply(lambda value: value[4]).astype(str)
    parsed_donor = parsed.apply(lambda value: value[5])

    # Prefer a valid explicit metadata column when one exists, but recover
    # anchors from group_id when the geometry pipeline dropped that metadata.
    if "anchor_sample_id" in df.columns:
        explicit_anchor = df["anchor_sample_id"].astype("string")
        valid_explicit = (
            explicit_anchor.notna()
            & (explicit_anchor.str.len() > 0)
            & (explicit_anchor != "global")
        )
        df["anchor_sample_id"] = parsed_anchor
        df.loc[valid_explicit, "anchor_sample_id"] = explicit_anchor[valid_explicit]
    else:
        df["anchor_sample_id"] = parsed_anchor

    if "donor_sample_id" in df.columns:
        explicit_donor = df["donor_sample_id"].astype("string")
        valid_donor = explicit_donor.notna() & (explicit_donor.str.len() > 0)
        df["donor_sample_id"] = parsed_donor
        df.loc[valid_donor, "donor_sample_id"] = explicit_donor[valid_donor]
    else:
        df["donor_sample_id"] = parsed_donor

    # Prefer explicit scenario names when preserved by the geometry pipeline.
    if "scenario_name" in df.columns:
        explicit_map = {
            "null_control": "null_control",
            "abundance_bloom_additive": "abundance_bloom_additive",
            "abundance_bloom_fixedn": "abundance_bloom_fixedn",
            "novel_category_additive": "novel_category_additive",
            "novel_category_fixedn": "novel_category_fixedn",
            "real_profile_contrast": "real_profile_contrast",
            "mixture_gradient": "mixture_gradient",
            "composition_swap": "composition_swap",
        }
        mapped = df["scenario_name"].map(explicit_map)
        df.loc[mapped.notna(), "scenario_type"] = mapped[mapped.notna()]

    unknown = int((df["scenario_type"] == "unknown").sum())
    if unknown:
        print(f"[warn] {unknown} rows could not be assigned to a known scenario.")

    nonlegacy = df[~df["scenario_type"].isin(["unknown", "composition_swap"])]
    if not nonlegacy.empty and (nonlegacy["anchor_sample_id"] == "global").any():
        bad = int((nonlegacy["anchor_sample_id"] == "global").sum())
        print(
            f"[warn] {bad} Tier-2 rows still have anchor='global'. "
            "Inspect their group_id format before trusting anchor rollups."
        )

    # Save a compact parser audit so incorrect IDs are immediately visible.
    audit_columns = [
        "group_id",
        "scenario_type",
        "anchor_sample_id",
        "donor_sample_id",
        "scenario_target",
        "scenario_level",
        "replicate",
    ]
    audit_path = OUT_DIR / "group_id_parser_audit.csv"
    df[audit_columns].to_csv(audit_path, index=False)
    print(f"Saved parser audit -> {audit_path}")

    return df

def available_metrics(df: pd.DataFrame) -> list[str]:
    return [metric for metric in METRICS if metric in df.columns]


# ---------------------------------------------------------------------------
# COVERAGE AND NULL NOISE
# ---------------------------------------------------------------------------


def coverage_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    small = int((df["num_rows"] < MIN_ROWS).sum())
    pct_small = (small / total * 100) if total else 0.0
    by_scenario = (
        df.groupby("scenario_type")["num_rows"].median().sort_index().to_dict()
    )
    unknown = int((df["scenario_type"] == "unknown").sum())
    return {
        "total": total,
        "small": small,
        "pct_small": pct_small,
        "median_by_scenario": by_scenario,
        "unknown": unknown,
    }


def pairwise_abs_differences(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.array([], dtype=float)
    return np.array(
        [abs(left - right) for left, right in itertools.combinations(values, 2)],
        dtype=float,
    )


def null_noise_floor_by_anchor(df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.Series]:
    """Median pairwise null difference using all available null replicates."""
    null_df = df[
        (df["scenario_type"] == "null_control")
        & (df["num_rows"] >= MIN_ROWS)
    ]

    result: dict[str, pd.Series] = {}

    for anchor, group in null_df.groupby("anchor_sample_id"):
        noise = {}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            differences = pairwise_abs_differences(values)
            noise[metric] = (
                float(np.median(differences)) if len(differences) else np.nan
            )
        result[str(anchor)] = pd.Series(noise, dtype=float)

    return result


def global_noise(noise_by_anchor: dict[str, pd.Series], metrics: list[str]) -> pd.Series:
    if not noise_by_anchor:
        return pd.Series({metric: np.nan for metric in metrics}, dtype=float)
    return pd.concat(noise_by_anchor.values(), axis=1).median(axis=1)


def get_noise_for_anchor(
    noise_by_anchor: dict[str, pd.Series],
    anchor: str,
    metrics: list[str],
) -> pd.Series:
    if anchor in noise_by_anchor:
        return noise_by_anchor[anchor]
    return global_noise(noise_by_anchor, metrics)


def get_noise_for_pair(
    noise_by_anchor: dict[str, pd.Series],
    anchors: list[str],
    metrics: list[str],
) -> pd.Series:
    matched = [noise_by_anchor[a] for a in anchors if a in noise_by_anchor]
    if matched:
        return pd.concat(matched, axis=1).mean(axis=1)
    return global_noise(noise_by_anchor, metrics)


# ---------------------------------------------------------------------------
# TARGET AND PAIR HELPERS
# ---------------------------------------------------------------------------


def clean_target_from_anchor(target: str, anchor: str) -> str:
    """Remove anchor prefix from parsed target when the ID embeds both."""
    target = str(target)
    anchor = str(anchor)
    prefix = f"{anchor}_"
    if target.startswith(prefix):
        return target[len(prefix) :]
    return target


def pair_key(group: pd.DataFrame) -> str:
    anchors = sorted(group["anchor_sample_id"].astype(str).unique())
    if len(anchors) >= 2:
        return "_vs_".join(anchors)
    targets = group["scenario_target"].dropna().astype(str).unique()
    return targets[0] if len(targets) else "unknown_pair"


# ---------------------------------------------------------------------------
# SCORING FUNCTIONS
# ---------------------------------------------------------------------------


def score_ordinal_scenario(
    df: pd.DataFrame,
    scenario_type: str,
    noise_by_anchor: dict[str, pd.Series],
    metrics: list[str],
) -> pd.DataFrame:
    """Bloom and mixture-gradient trend scoring, within each anchor."""
    sub = df[
        (df["scenario_type"] == scenario_type)
        & (df["num_rows"] >= MIN_ROWS)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["scenario_target"] = sub.apply(
        lambda row: clean_target_from_anchor(
            row["scenario_target"], row["anchor_sample_id"]
        ),
        axis=1,
    )

    rows = []

    for (anchor, target), group in sub.groupby(
        ["anchor_sample_id", "scenario_target"], dropna=False
    ):
        if group["scenario_level"].nunique() < 2:
            continue

        noise = get_noise_for_anchor(noise_by_anchor, str(anchor), metrics)

        for metric in metrics:
            valid = group[["scenario_level", metric]].dropna()
            if len(valid) < 4 or valid["scenario_level"].nunique() < 2:
                continue

            r, p = spearmanr(valid["scenario_level"], valid[metric])
            level_stats = group.groupby("scenario_level")[metric].agg(
                mean="mean", sd="std", n="count"
            )
            swing = float(level_stats["mean"].max() - level_stats["mean"].min())
            noise_floor = float(noise.get(metric, np.nan))
            ratio = (
                abs(swing) / noise_floor
                if pd.notna(noise_floor) and noise_floor > 0
                else np.nan
            )
            working = (
                pd.notna(r)
                and abs(float(r)) >= SPEARMAN_MIN
                and pd.notna(ratio)
                and ratio >= EFFECT_OVER_NOISE_MIN
            )

            rows.append(
                {
                    "scenario_type": scenario_type,
                    "anchor": str(anchor),
                    "scenario_target": str(target),
                    "comparison": "ordinal_trend",
                    "metric": metric,
                    "spearman_r": float(r) if pd.notna(r) else np.nan,
                    "spearman_p": float(p) if pd.notna(p) else np.nan,
                    "swing": swing,
                    "noise_floor": noise_floor,
                    "effect_over_noise": ratio,
                    "mean_replicate_sd": float(level_stats["sd"].mean()),
                    "max_replicate_sd": float(level_stats["sd"].max()),
                    "level_summary": "; ".join(
                        f"{level:g}: {row['mean']:.6g} +/- {row['sd']:.6g} (n={int(row['n'])})"
                        for level, row in level_stats.iterrows()
                    ),
                    "n_groups": int(len(valid)),
                    "working": bool(working),
                }
            )

    return pd.DataFrame(rows)


def score_novel_presence(
    df: pd.DataFrame,
    scenario_type: str,
    noise_by_anchor: dict[str, pd.Series],
    metrics: list[str],
) -> pd.DataFrame:
    """Compare n=0 separately against every positive novelty level."""
    sub = df[
        (df["scenario_type"] == scenario_type)
        & (df["num_rows"] >= MIN_ROWS)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["scenario_target"] = sub.apply(
        lambda row: clean_target_from_anchor(
            row["scenario_target"], row["anchor_sample_id"]
        ),
        axis=1,
    )

    rows = []

    for (anchor, target), group in sub.groupby(
        ["anchor_sample_id", "scenario_target"], dropna=False
    ):
        baseline = group[group["scenario_level"] == 0]
        if baseline.empty:
            print(
                f"[warn] Missing n=0 baseline for {scenario_type}, "
                f"anchor={anchor}, target={target}"
            )
            continue

        noise = get_noise_for_anchor(noise_by_anchor, str(anchor), metrics)
        positive_levels = sorted(
            level
            for level in group["scenario_level"].dropna().unique()
            if level > 0
        )

        for level in positive_levels:
            present = group[group["scenario_level"] == level]

            for metric in metrics:
                baseline_values = baseline[metric].dropna()
                present_values = present[metric].dropna()

                if len(baseline_values) < 2 or len(present_values) < 2:
                    continue

                baseline_mean = float(baseline_values.mean())
                baseline_sd = float(baseline_values.std())
                present_mean = float(present_values.mean())
                present_sd = float(present_values.std())
                delta = present_mean - baseline_mean
                swing = abs(delta)
                noise_floor = float(noise.get(metric, np.nan))
                ratio = (
                    swing / noise_floor
                    if pd.notna(noise_floor) and noise_floor > 0
                    else np.nan
                )
                working = (
                    pd.notna(ratio) and ratio >= EFFECT_OVER_NOISE_MIN
                )

                rows.append(
                    {
                        "scenario_type": scenario_type,
                        "anchor": str(anchor),
                        "scenario_target": str(target),
                        "comparison": f"n0_vs_n{int(level)}",
                        "baseline_level": 0,
                        "present_level": float(level),
                        "metric": metric,
                        "baseline_mean": baseline_mean,
                        "baseline_sd": baseline_sd,
                        "present_mean": present_mean,
                        "present_sd": present_sd,
                        "delta": delta,
                        "spearman_r": np.nan,
                        "spearman_p": np.nan,
                        "swing": swing,
                        "noise_floor": noise_floor,
                        "effect_over_noise": ratio,
                        "mean_replicate_sd": float(
                            np.nanmean([baseline_sd, present_sd])
                        ),
                        "max_replicate_sd": float(
                            np.nanmax([baseline_sd, present_sd])
                        ),
                        "level_summary": (
                            f"n0: {baseline_mean:.6g} +/- {baseline_sd:.6g} "
                            f"(n={len(baseline_values)}); "
                            f"n{int(level)}: {present_mean:.6g} +/- {present_sd:.6g} "
                            f"(n={len(present_values)})"
                        ),
                        "n_baseline": int(len(baseline_values)),
                        "n_present": int(len(present_values)),
                        "n_groups": int(
                            len(baseline_values) + len(present_values)
                        ),
                        "working": bool(working),
                    }
                )

    return pd.DataFrame(rows)


def score_pair_scenario(
    df: pd.DataFrame,
    scenario_type: str,
    noise_by_anchor: dict[str, pd.Series],
    metrics: list[str],
) -> pd.DataFrame:
    """Compare A and B for real-profile contrast or legacy composition swap."""
    sub = df[
        (df["scenario_type"] == scenario_type)
        & (df["num_rows"] >= MIN_ROWS)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    rows = []

    # Parsed target normally identifies the anchor pair. If parsing is imperfect,
    # grouping by the sorted anchors still keeps each contrast separate.
    sub["pair_key"] = sub.groupby("scenario_target", dropna=False)[
        "anchor_sample_id"
    ].transform(lambda values: "_vs_".join(sorted(set(map(str, values)))))

    grouping_columns = ["scenario_target", "pair_key"]

    for (target, parsed_pair_key), group in sub.groupby(
        grouping_columns, dropna=False
    ):
        a = group[group["scenario_level"] == 0]
        b = group[group["scenario_level"] == 1]
        if a.empty or b.empty:
            continue

        anchors = sorted(group["anchor_sample_id"].astype(str).unique())
        actual_pair_key = "_vs_".join(anchors) if len(anchors) >= 2 else str(parsed_pair_key)
        noise = get_noise_for_pair(noise_by_anchor, anchors, metrics)

        for metric in metrics:
            a_values = a[metric].dropna()
            b_values = b[metric].dropna()

            if len(a_values) < 3 or len(b_values) < 3:
                continue

            a_mean = float(a_values.mean())
            a_sd = float(a_values.std())
            b_mean = float(b_values.mean())
            b_sd = float(b_values.std())
            swing = abs(a_mean - b_mean)
            noise_floor = float(noise.get(metric, np.nan))
            ratio = (
                swing / noise_floor
                if pd.notna(noise_floor) and noise_floor > 0
                else np.nan
            )
            working = pd.notna(ratio) and ratio >= EFFECT_OVER_NOISE_MIN

            rows.append(
                {
                    "scenario_type": scenario_type,
                    "anchor": actual_pair_key,
                    "scenario_target": str(target),
                    "comparison": "A_vs_B",
                    "metric": metric,
                    "a_mean": a_mean,
                    "a_sd": a_sd,
                    "b_mean": b_mean,
                    "b_sd": b_sd,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "swing": swing,
                    "noise_floor": noise_floor,
                    "effect_over_noise": ratio,
                    "mean_replicate_sd": float(np.nanmean([a_sd, b_sd])),
                    "max_replicate_sd": float(np.nanmax([a_sd, b_sd])),
                    "level_summary": (
                        f"A: {a_mean:.6g} +/- {a_sd:.6g} (n={len(a_values)}); "
                        f"B: {b_mean:.6g} +/- {b_sd:.6g} (n={len(b_values)})"
                    ),
                    "n_groups": int(len(a_values) + len(b_values)),
                    "working": bool(working),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ROLLUPS
# ---------------------------------------------------------------------------


def anchor_metric_detail(scorecard: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario, metric, and anchor with comparison-level pass rates."""
    if scorecard.empty:
        return pd.DataFrame()

    rows = []
    for (scenario, metric, anchor), group in scorecard.groupby(
        ["scenario_type", "metric", "anchor"], dropna=False
    ):
        comparisons = group["working"].astype(bool)
        rows.append(
            {
                "scenario_type": scenario,
                "metric": metric,
                "anchor": anchor,
                "n_comparisons_tested": int(len(group)),
                "n_comparisons_passed": int(comparisons.sum()),
                "pct_comparisons_passed": float(comparisons.mean() * 100),
                "passed_any": bool(comparisons.any()),
                "passed_all": bool(comparisons.all()),
                "median_abs_spearman": float(group["spearman_r"].abs().median()),
                "median_effect_over_noise": float(
                    group["effect_over_noise"].median()
                ),
                "median_replicate_sd": float(
                    group["mean_replicate_sd"].median()
                ),
            }
        )

    return pd.DataFrame(rows)


def anchor_rollup(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Summarise consistency of metric performance across anchors."""
    detail = anchor_metric_detail(scorecard)
    if detail.empty:
        return pd.DataFrame()

    rows = []
    for (scenario, metric), group in detail.groupby(
        ["scenario_type", "metric"], dropna=False
    ):
        rows.append(
            {
                "scenario_type": scenario,
                "metric": metric,
                "n_anchors_tested": int(group["anchor"].nunique()),
                "n_anchors_passed_any": int(group["passed_any"].sum()),
                "n_anchors_passed_all": int(group["passed_all"].sum()),
                "pct_anchors_passed_any": float(group["passed_any"].mean() * 100),
                "pct_anchors_passed_all": float(group["passed_all"].mean() * 100),
                "median_pct_comparisons_passed": float(
                    group["pct_comparisons_passed"].median()
                ),
                "median_abs_spearman": float(
                    group["median_abs_spearman"].median()
                ),
                "median_effect_over_noise": float(
                    group["median_effect_over_noise"].median()
                ),
                "median_replicate_sd": float(
                    group["median_replicate_sd"].median()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scenario_type", "pct_anchors_passed_any", "median_effect_over_noise"],
        ascending=[True, False, False],
    )


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


def safe_text(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def spotlight_plot(
    df: pd.DataFrame,
    scenario_type: str,
    anchor: str,
    target: str,
    metric: str,
    tag: str,
) -> Path | None:
    sub = df[
        (df["scenario_type"] == scenario_type)
        & (df["num_rows"] >= MIN_ROWS)
    ].copy()

    if scenario_type in {"real_profile_contrast", "composition_swap"}:
        anchors = str(anchor).split("_vs_")
        sub = sub[sub["anchor_sample_id"].astype(str).isin(anchors)]
        sub = sub[sub["scenario_target"].astype(str) == str(target)]
    else:
        sub["scenario_target"] = sub.apply(
            lambda row: clean_target_from_anchor(
                row["scenario_target"], row["anchor_sample_id"]
            ),
            axis=1,
        )
        sub = sub[
            (sub["anchor_sample_id"].astype(str) == str(anchor))
            & (sub["scenario_target"].astype(str) == str(target))
        ]

    if sub.empty or metric not in sub.columns:
        return None

    stats = sub.groupby("scenario_level")[metric].agg(mean="mean", sd="std")
    if stats.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        stats.index,
        stats["mean"],
        yerr=stats["sd"].fillna(0),
        marker="o",
        capsize=4,
    )
    ax.set_title(
        f"{tag}: {metric}\n{scenario_type} / {anchor} / {target}",
        fontsize=9,
    )
    ax.set_xlabel("manipulation level")
    ax.set_ylabel(metric)
    fig.tight_layout()

    path = OUT_DIR / (
        f"spotlight_{safe_text(tag)}_{safe_text(scenario_type)}_"
        f"{safe_text(anchor)}_{safe_text(target)}_{safe_text(metric)}.png"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------


def fmt_number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def write_report(
    coverage: dict,
    scorecard: pd.DataFrame,
    rollup: pd.DataFrame,
    spotlight_paths: dict[str, Path | None],
    metrics: list[str],
) -> None:
    lines: list[str] = []

    lines.append("TIER 2 METRIC DIAGNOSTIC REPORT")
    lines.append("=" * 42)
    lines.append("")

    lines.append("1) DATA COVERAGE CHECK")
    lines.append(f"   Total synthetic groups: {coverage['total']}")
    lines.append(
        f"   Groups with < {MIN_ROWS} rows: {coverage['small']} "
        f"({coverage['pct_small']:.1f}%) -- excluded from scoring."
    )
    lines.append(f"   Unparsed/unknown scenario rows: {coverage['unknown']}")
    for scenario, median_rows in coverage["median_by_scenario"].items():
        lines.append(f"     median rows in '{scenario}': {median_rows:.1f}")
    lines.append("")

    lines.append("2) HOW EACH SCENARIO IS SCORED")
    lines.append(
        f"   Bloom and mixture gradients: |Spearman r| >= {SPEARMAN_MIN} "
        f"AND effect/noise >= {EFFECT_OVER_NOISE_MIN}."
    )
    lines.append(
        "   Novel categories: n=0 is compared separately with every n>0 level; "
        "effect/noise must exceed the threshold. No monotonicity is required."
    )
    lines.append(
        "   Real-profile contrasts: A-versus-B mean shift must exceed the "
        "relevant anchor-specific null noise."
    )
    lines.append(
        "   Null noise is the median of all pairwise absolute differences among "
        "the anchor's null-control replicates."
    )
    lines.append(
        "   Replicate mean +/- SD is saved in the scorecard and shown in the "
        "level_summary field."
    )
    lines.append("")

    lines.append("3) PER-ANCHOR GENERALISATION -- PRIMARY TIER-2 RESULT")
    lines.append(
        "   This section asks whether a metric works across several real anchor "
        "profiles, rather than being driven by one favourable anchor."
    )
    lines.append("")

    if rollup.empty:
        lines.append("   No anchor roll-up was available.")
    else:
        for scenario in rollup["scenario_type"].unique():
            subset = rollup[rollup["scenario_type"] == scenario].head(8)
            lines.append(f"   {scenario}:")
            for _, row in subset.iterrows():
                lines.append(
                    f"     {row['metric']:<28s} "
                    f"any={int(row['n_anchors_passed_any'])}/"
                    f"{int(row['n_anchors_tested'])} anchors "
                    f"({row['pct_anchors_passed_any']:.0f}%); "
                    f"all-levels={int(row['n_anchors_passed_all'])}/"
                    f"{int(row['n_anchors_tested'])}; "
                    f"median comparisons passed="
                    f"{row['median_pct_comparisons_passed']:.0f}%; "
                    f"median|r|={fmt_number(row['median_abs_spearman'])}; "
                    f"median effect/noise="
                    f"{fmt_number(row['median_effect_over_noise'])}"
                )
            lines.append("")

    lines.append("4) RAW COMPARISON-LEVEL RANKING -- SECONDARY RESULT")
    if scorecard.empty:
        lines.append("   No scenarios had enough usable data to score.")
    else:
        ranking = scorecard.groupby("metric")["working"].agg(["sum", "count"])
        ranking["pct"] = ranking["sum"] / ranking["count"] * 100
        ranking = ranking.sort_values(["pct", "sum"], ascending=False)
        for metric, row in ranking.iterrows():
            lines.append(
                f"     {metric:<28s} {int(row['sum'])}/{int(row['count'])} "
                f"comparisons ({row['pct']:.0f}%)"
            )
    lines.append("")

    lines.append("5) PER-SCENARIO DETAIL")
    if not scorecard.empty:
        for scenario in scorecard["scenario_type"].unique():
            subset = scorecard[scorecard["scenario_type"] == scenario]
            lines.append(
                f"   {scenario}: {int(subset['working'].sum())}/{len(subset)} "
                "comparison rows passed."
            )
            top = subset.sort_values(
                "effect_over_noise", ascending=False, na_position="last"
            ).head(5)
            for _, row in top.iterrows():
                lines.append(
                    f"     {row['metric']}: anchor={row['anchor']}; "
                    f"target={row['scenario_target']}; "
                    f"comparison={row.get('comparison', 'NA')}; "
                    f"effect/noise={fmt_number(row['effect_over_noise'])}; "
                    f"mean replicate SD={fmt_number(row.get('mean_replicate_sd'))}; "
                    f"working={bool(row['working'])}"
                )
                if pd.notna(row.get("level_summary")):
                    lines.append(f"       {row['level_summary']}")
            lines.append("")

    lines.append("6) HOW TO INTERPRET A METRIC")
    lines.append("   Strong candidate site descriptor:")
    lines.append("     - passes on most anchors, not only one anchor;")
    lines.append("     - ecological shift exceeds anchor-specific null noise;")
    lines.append("     - replicate SD is small relative to the shift;")
    lines.append("     - bloom/mixture additionally shows a monotonic trend.")
    lines.append("   Stable but insensitive:")
    lines.append("     - low replicate SD, but effect/noise below threshold.")
    lines.append("   Responsive but context-dependent:")
    lines.append("     - strong effects on some anchors but failures on others.")
    lines.append("   Responsive but noisy:")
    lines.append("     - mean changes, but replicate SD is large relative to the shift.")
    lines.append("")

    lines.append("7) SPOTLIGHT PLOTS")
    for label, path in spotlight_paths.items():
        if path:
            lines.append(f"   {label}: {path.name}")
    lines.append(
        "   Error bars are replicate SD. Look for ecological separation that is "
        "large relative to those error bars."
    )
    lines.append("")

    lines.append("8) SUGGESTED NEXT STEP")
    if scorecard.empty or coverage["pct_small"] > 40:
        lines.append(
            "   Resolve coverage/parsing problems before interpreting metric performance."
        )
    else:
        working_metrics = int(scorecard.groupby("metric")["working"].any().sum())
        lines.append(
            f"   {working_metrics}/{len(metrics)} available metrics passed at least one "
            "Tier-2 comparison."
        )
        lines.append(
            "   Prioritise metrics that pass across most anchors for the ecological "
            "property of interest; do not expect one metric to encode every property."
        )

    report_path = OUT_DIR / "diagnostic_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved -> {report_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    df = load_and_parse(IN_PATH)
    metrics = available_metrics(df)

    if not metrics:
        raise ValueError("None of the configured metrics were found in the input CSV.")

    coverage = coverage_summary(df)
    noise_by_anchor = null_noise_floor_by_anchor(df, metrics)

    ordinal_types = [
        "abundance_bloom_additive",
        "abundance_bloom_fixedn",
        "mixture_gradient",
    ]
    novel_types = [
        "novel_category_additive",
        "novel_category_fixedn",
    ]
    pair_types = [
        "composition_swap",
        "real_profile_contrast",
    ]

    scorecards = [
        score_ordinal_scenario(df, scenario, noise_by_anchor, metrics)
        for scenario in ordinal_types
    ]
    scorecards.extend(
        score_novel_presence(df, scenario, noise_by_anchor, metrics)
        for scenario in novel_types
    )
    scorecards.extend(
        score_pair_scenario(df, scenario, noise_by_anchor, metrics)
        for scenario in pair_types
    )

    scorecards = [score for score in scorecards if not score.empty]
    scorecard = (
        pd.concat(scorecards, ignore_index=True, sort=False)
        if scorecards
        else pd.DataFrame()
    )

    scorecard_path = OUT_DIR / "diagnostic_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    print(f"Saved -> {scorecard_path}")

    detail = anchor_metric_detail(scorecard)
    detail_path = OUT_DIR / "diagnostic_anchor_detail.csv"
    detail.to_csv(detail_path, index=False)
    print(f"Saved -> {detail_path}")

    rollup = anchor_rollup(scorecard)
    rollup_path = OUT_DIR / "diagnostic_anchor_rollup.csv"
    rollup.to_csv(rollup_path, index=False)
    print(f"Saved -> {rollup_path}")

    spotlight_paths: dict[str, Path | None] = {}

    if not scorecard.empty:
        ranking = scorecard.groupby("metric")["working"].mean().sort_values()
        worst_metric = ranking.index[0]
        best_metric = ranking.index[-1]

        best_rows = scorecard[scorecard["metric"] == best_metric]
        best_row = best_rows.loc[
            best_rows["effect_over_noise"].fillna(-1).idxmax()
        ]

        worst_rows = scorecard[scorecard["metric"] == worst_metric]
        worst_row = worst_rows.loc[
            worst_rows["effect_over_noise"].fillna(-1).idxmax()
        ]

        spotlight_paths["best_metric"] = spotlight_plot(
            df,
            str(best_row["scenario_type"]),
            str(best_row["anchor"]),
            str(best_row["scenario_target"]),
            best_metric,
            "BEST",
        )
        spotlight_paths["worst_metric"] = spotlight_plot(
            df,
            str(worst_row["scenario_type"]),
            str(worst_row["anchor"]),
            str(worst_row["scenario_target"]),
            worst_metric,
            "WORST",
        )

    write_report(coverage, scorecard, rollup, spotlight_paths, metrics)


if __name__ == "__main__":
    main()
