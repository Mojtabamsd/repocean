"""
tier1_diagnostic_report.py

Goal: turn a big geometry_metrics.csv (hundreds of groups x 15 metrics)
into ONE readable verdict: which metrics are actually responding to your
Tier-1 manipulations, which aren't, and what to change next.

Two numbers decide the verdict for every (metric, scenario) pair:

  1. SPEARMAN CORRELATION between the manipulation strength (bloom
     multiplier, novel-category count) and the metric value. This checks
     DIRECTION/MONOTONICITY: does the metric move consistently as you
     dial the manipulation up, or does it just bounce around?

  2. EFFECT-OVER-NOISE RATIO: (biggest mean swing across manipulation
     levels) / (null-control noise floor, i.e. how much two IDENTICAL
     profiles differ from bootstrap sampling alone). This checks
     MAGNITUDE: does the swing rise above the noise you'd see even if
     nothing changed? Ratio < 1 means "the metric's response is smaller
     than what sampling noise alone produces" -> not trustworthy yet.

A metric only counts as WORKING for a scenario if it clears both bars.
That's the core idea: direction alone can be a coincidence in a small
sample, and magnitude alone can be a big number that still doesn't track
your manipulation. You need both.

USAGE
  1. Point IN_PATH at your geometry_metrics.csv
  2. Run this script
  3. Read diagnostic_report.txt first (plain-English summary + what to
     change next), then diagnostic_scorecard.csv for the full numbers,
     then look at the 2-3 spotlight PNGs it names for you.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

IN_PATH = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\geometry\geometry_metrics.csv")
OUT_DIR = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier1_synthetic\geometry\tier1_metric_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Below this many rows, geometry metrics collapse to degenerate values
# (cos_p10=1.0, eff_rank~0, etc) and mostly reflect sample-size, not
# composition. Excluded from the scorecard, but reported as a coverage
# problem since it's very likely to be limiting your results.
MIN_ROWS = 5

# Bar for calling a metric's response "working"
SPEARMAN_MIN = 0.4          # |r| below this = direction isn't consistent
EFFECT_OVER_NOISE_MIN = 1.0  # ratio below this = swing is within noise floor

METRICS = [
    "pred_num_classes_present", "shannon", "exp_shannon",
    "centroid_norm", "cos_mean", "cos_p10", "cos_p50", "cos_p90",
    "pair_cos_mean", "pair_cos_p10", "pair_cos_p50", "pair_cos_p90",
    "eff_rank", "pca_dim_90", "pca_dim_95",
]

SCENARIO_PATTERNS = [
    ("null_control", re.compile(r"_null_control_(\d+)$")),
    ("bloom", re.compile(r"_bloom_(.+)_x(\d+)(?:_rep\d+)?$")),
    ("composition_swap", re.compile(r"_composition_([AB])(?:_rep\d+)?$")),
    ("novel_category", re.compile(r"_novel_(.+)_n(\d+)(?:_rep\d+)?$")),
]


def parse_group_id(group_id: str):
    for scenario_type, pat in SCENARIO_PATTERNS:
        m = pat.search(group_id)
        if m:
            original = group_id[: m.start()]
            if scenario_type == "null_control":
                return original, scenario_type, "null_control", int(m.group(1))
            if scenario_type == "bloom":
                return original, scenario_type, m.group(1), int(m.group(2))
            if scenario_type == "composition_swap":
                return original, scenario_type, "composition_swap", (0 if m.group(1) == "A" else 1)
            if scenario_type == "novel_category":
                return original, scenario_type, m.group(1), int(m.group(2))
    return group_id, "unknown", "unknown", np.nan


def load_and_parse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["group_id"].apply(parse_group_id)
    df["original_sample_id"] = parsed.apply(lambda t: t[0])
    df["scenario_type"] = parsed.apply(lambda t: t[1])
    df["scenario_target"] = parsed.apply(lambda t: t[2])   # e.g. bloom category or novel species
    df["scenario_level"] = parsed.apply(lambda t: t[3])    # ordinal: multiplier / injected count
    return df


def coverage_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    small = int((df["num_rows"] < MIN_ROWS).sum())
    pct_small = small / total * 100
    by_scenario = df.groupby("scenario_type")["num_rows"].median().to_dict()
    return {"total": total, "small": small, "pct_small": pct_small, "median_by_scenario": by_scenario}


def null_noise_floor(df: pd.DataFrame) -> pd.Series:
    nc = df[(df["scenario_type"] == "null_control") & (df["num_rows"] >= MIN_ROWS)]
    if nc.empty:
        return pd.Series({m: np.nan for m in METRICS})
    piv = nc.pivot_table(index="original_sample_id", columns="scenario_level", values=METRICS)
    diffs = {}
    levels = sorted(nc["scenario_level"].dropna().unique())
    if len(levels) < 2:
        return pd.Series({m: np.nan for m in METRICS})
    lo, hi = levels[0], levels[1]
    for m in METRICS:
        try:
            diffs[m] = (piv[m][lo] - piv[m][hi]).abs().mean()
        except KeyError:
            diffs[m] = np.nan
    return pd.Series(diffs)


def score_ordinal_scenario(df: pd.DataFrame, scenario_type: str, noise: pd.Series) -> pd.DataFrame:
    """For bloom / novel_category: correlate metric value with scenario_level."""
    sub = df[(df["scenario_type"] == scenario_type) & (df["num_rows"] >= MIN_ROWS)]
    rows = []
    for target, g in sub.groupby("scenario_target"):
        if g["scenario_level"].nunique() < 2:
            continue
        for m in METRICS:
            vals = g[m].dropna()
            levels = g.loc[vals.index, "scenario_level"]
            if len(vals) < 4:
                continue
            r, p = spearmanr(levels, vals)
            level_means = g.groupby("scenario_level")[m].mean()
            swing = level_means.max() - level_means.min()
            n = noise.get(m, np.nan)
            ratio = abs(swing) / n if (n and n > 0) else np.nan
            works = (abs(r) >= SPEARMAN_MIN) and (ratio is not np.nan) and (ratio >= EFFECT_OVER_NOISE_MIN)
            rows.append({
                "scenario_type": scenario_type, "scenario_target": target, "metric": m,
                "spearman_r": r, "spearman_p": p, "swing": swing,
                "noise_floor": n, "effect_over_noise": ratio,
                "n_groups": len(vals), "working": works,
            })
    return pd.DataFrame(rows)


def score_composition_swap(df: pd.DataFrame, noise: pd.Series) -> pd.DataFrame:
    sub = df[(df["scenario_type"] == "composition_swap") & (df["num_rows"] >= MIN_ROWS)]
    rows = []
    if sub.empty:
        return pd.DataFrame(rows)
    for m in METRICS:
        a = sub[sub["scenario_level"] == 0][m].dropna()
        b = sub[sub["scenario_level"] == 1][m].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        swing = abs(a.mean() - b.mean())
        n = noise.get(m, np.nan)
        ratio = swing / n if (n and n > 0) else np.nan
        works = (ratio is not np.nan) and (ratio >= EFFECT_OVER_NOISE_MIN)
        rows.append({
            "scenario_type": "composition_swap", "scenario_target": "A_vs_B", "metric": m,
            "spearman_r": np.nan, "spearman_p": np.nan, "swing": swing,
            "noise_floor": n, "effect_over_noise": ratio,
            "n_groups": len(a) + len(b), "working": works,
        })
    return pd.DataFrame(rows)


def spotlight_plot(df: pd.DataFrame, scenario_type: str, target: str, metric: str, tag: str):
    sub = df[(df["scenario_type"] == scenario_type) & (df["scenario_target"] == target)
             & (df["num_rows"] >= MIN_ROWS)]
    if sub.empty:
        return None
    means = sub.groupby("scenario_level")[metric].mean()
    stds = sub.groupby("scenario_level")[metric].std()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(means.index, means.values, yerr=stds.values, marker="o")
    ax.set_title(f"{tag}: {metric}\n({scenario_type} / {target})", fontsize=10)
    ax.set_xlabel("manipulation level")
    ax.set_ylabel(metric)
    fig.tight_layout()
    out_path = OUT_DIR / f"spotlight_{tag}_{scenario_type}_{target}_{metric}.png".replace("<", "").replace(">", "")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def write_report(coverage: dict, scorecard: pd.DataFrame, spotlight_paths: dict):
    lines = []
    lines.append("TIER 1 METRIC DIAGNOSTIC REPORT")
    lines.append("=" * 40)
    lines.append("")

    # ---- 1. Coverage check (usually the first thing worth fixing) ----
    lines.append("1) DATA COVERAGE CHECK")
    lines.append(f"   Total synthetic groups: {coverage['total']}")
    lines.append(f"   Groups with < {MIN_ROWS} rows: {coverage['small']} ({coverage['pct_small']:.1f}%) "
                 f"-- excluded from scoring below.")
    for scen, med in coverage["median_by_scenario"].items():
        lines.append(f"     median rows in '{scen}': {med:.1f}")
    if coverage["pct_small"] > 40:
        lines.append("   >> WHAT TO CHANGE: over 40% of your groups are too small to trust.")
        lines.append("      Go back to the Tier-1 generator and either (a) raise BASE_N_PER_CAT / ")
        lines.append("      detritus_total so every synthetic profile has more rows, or (b) stop ")
        lines.append("      anchoring sampling to a single original profile's own pool and sample ")
        lines.append("      from the full dataset instead -- small original profiles are currently ")
        lines.append("      capping how many rows a synthetic profile can get.")
    lines.append("")

    # ---- 2. Scorecard summary ----
    lines.append("2) WHICH METRICS ARE WORKING")
    lines.append(f"   'Working' = |Spearman r| >= {SPEARMAN_MIN} (moves consistently with the ")
    lines.append(f"   manipulation) AND swing/noise-floor >= {EFFECT_OVER_NOISE_MIN} (the swing is ")
    lines.append("   bigger than what two IDENTICAL profiles differ by from sampling alone).")
    lines.append("")
    if scorecard.empty:
        lines.append("   No scenario had enough same-sized groups to score. See coverage issue above.")
    else:
        per_metric = scorecard.groupby("metric")["working"].agg(["sum", "count"])
        per_metric["pct_working"] = (per_metric["sum"] / per_metric["count"] * 100).round(0)
        per_metric = per_metric.sort_values("pct_working", ascending=False)
        lines.append("   Metric ranking (fraction of scenarios where it passed both bars):")
        for metric, row in per_metric.iterrows():
            lines.append(f"     {metric:<28s} {int(row['sum'])}/{int(row['count'])} scenarios "
                         f"({row['pct_working']:.0f}%)")
        lines.append("")
        best = per_metric.index[0]
        worst = per_metric.index[-1]
        lines.append(f"   >> LOOK FIRST AT: '{best}' -- most consistently responsive metric so far.")
        lines.append(f"   >> LEAST USEFUL SO FAR: '{worst}' -- either drop it or investigate why ")
        lines.append("      it doesn't track composition changes (could be a floor/ceiling effect, ")
        lines.append("      e.g. always near 1.0 or 0 when groups are small).")
    lines.append("")

    # ---- 3. Per-scenario detail ----
    lines.append("3) PER-SCENARIO DETAIL (full numbers in diagnostic_scorecard.csv)")
    if not scorecard.empty:
        for scen in scorecard["scenario_type"].unique():
            s = scorecard[scorecard["scenario_type"] == scen]
            n_working = s["working"].sum()
            lines.append(f"   {scen}: {n_working}/{len(s)} (metric,target) pairs passed both bars.")
    lines.append("")

    # ---- 4. Where to look ----
    lines.append("4) PLOTS TO LOOK AT (saved as PNGs)")
    for tag, path in spotlight_paths.items():
        if path:
            lines.append(f"   {tag}: {path.name}")
    lines.append("")
    lines.append("   For each spotlight plot: look for a MONOTONIC line (mean point goes up or down")
    lines.append("   consistently as manipulation level increases) with error bars (std across")
    lines.append("   original profiles) that don't swamp the trend. A flat or zig-zag line, or ")
    lines.append("   error bars bigger than the gap between points, means that metric isn't yet ")
    lines.append("   distinguishing this manipulation given your current sample sizes.")
    lines.append("")

    # ---- 5. What to change next ----
    lines.append("5) SUGGESTED NEXT STEP")
    if scorecard.empty or coverage["pct_small"] > 40:
        lines.append("   Fix sample-size coverage first (see section 1) and rerun this report --")
        lines.append("   right now most groups are too small for the scorecard to be conclusive.")
    else:
        n_any_working = scorecard.groupby("metric")["working"].any().sum()
        if n_any_working == 0:
            lines.append("   None of the metrics cleared both bars on any scenario. Before concluding")
            lines.append("   the metrics don't work, try: (a) more extreme manipulation levels (bigger")
            lines.append("   bloom multipliers, larger novel-category counts), since your current")
            lines.append("   swings may just be too subtle; (b) larger MIN_ROWS filter / bigger base N")
            lines.append("   per profile, since noise floor scales down as group size goes up.")
        else:
            lines.append(f"   {n_any_working}/{len(METRICS)} metrics cleared the bar on at least one")
            lines.append("   scenario. Focus validation effort on those metrics; consider dropping or")
            lines.append("   deprioritizing metrics that never cleared the bar in any scenario.")

    report_path = OUT_DIR / "diagnostic_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    df = load_and_parse(IN_PATH)
    coverage = coverage_summary(df)
    noise = null_noise_floor(df)

    scorecards = []
    for scen in ["bloom", "novel_category"]:
        scorecards.append(score_ordinal_scenario(df, scen, noise))
    scorecards.append(score_composition_swap(df, noise))
    scorecard = pd.concat(scorecards, ignore_index=True) if scorecards else pd.DataFrame()

    scorecard_path = OUT_DIR / "diagnostic_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    print(f"Saved -> {scorecard_path}")

    # spotlight plots: best metric overall, worst metric overall, and
    # the metric with the strongest single (scenario,target) result
    spotlight_paths = {}
    if not scorecard.empty:
        per_metric = scorecard.groupby("metric")["working"].agg(["sum", "count"])
        per_metric["pct"] = per_metric["sum"] / per_metric["count"]
        best_metric = per_metric["pct"].idxmax()
        worst_metric = per_metric["pct"].idxmin()

        best_row = scorecard.loc[scorecard["metric"] == best_metric].iloc[
            scorecard.loc[scorecard["metric"] == best_metric, "effect_over_noise"].fillna(-1).values.argmax()
        ]
        worst_row = scorecard.loc[scorecard["metric"] == worst_metric].iloc[0]

        spotlight_paths["best_metric"] = spotlight_plot(
            df, best_row["scenario_type"], best_row["scenario_target"], best_metric, "BEST")
        spotlight_paths["worst_metric"] = spotlight_plot(
            df, worst_row["scenario_type"], worst_row["scenario_target"], worst_metric, "WORST")

    write_report(coverage, scorecard, spotlight_paths)