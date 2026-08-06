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

IN_PATH = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\geometry\geometry_metrics.csv")
OUT_DIR = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106\tier2_synthetic\geometry\tier2_metric_analysis")
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
    ("null_control", re.compile(r"_null_control_.*?(\d+)$"), "null"),
    # bloom now comes in additive/fixedn variants with distinct prefixes
    ("abundance_bloom_additive", re.compile(r"_bloomadd_(.+)_x(\d+)(?:_rep\d+)?$"), "ordinal"),
    ("abundance_bloom_fixedn", re.compile(r"_bloomfix_(.+)_x(\d+)(?:_rep\d+)?$"), "ordinal"),
    ("novel_category_additive", re.compile(r"_noveladd_(.+)_n(\d+)(?:_rep\d+)?$"), "ordinal"),
    ("novel_category_fixedn", re.compile(r"_novelfix_(.+)_n(\d+)(?:_rep\d+)?$"), "ordinal"),
    # legacy Tier-1/older-Tier-2 names, kept for backward compatibility
    ("abundance_bloom_additive", re.compile(r"_bloom_(.+)_x(\d+)(?:_rep\d+)?$"), "ordinal"),
    ("novel_category_additive", re.compile(r"_novel_(.+)_n(\d+)(?:_rep\d+)?$"), "ordinal"),
    ("composition_swap", re.compile(r"_composition_(?:(.+)_)?(A|B)(?:_rep\d+)?$"), "pair"),
    # renamed, fixed-size version of composition_swap
    ("real_profile_contrast", re.compile(r"_contrast_(.+)_(A|B)(?:_rep\d+)?$"), "pair"),
    ("mixture_gradient", re.compile(r"_mixture_(.+)_p(\d+)(?:_rep\d+)?$"), "ordinal"),
]


def parse_group_id(group_id: str):
    """Returns (scenario_type, scenario_target, scenario_level).
    scenario_target is the category/species/pair name parsed from the
    string -- used as a FALLBACK label only. Anchor identity (which real
    profile a scenario is tied to) comes from the anchor_sample_id
    column when present, not from string parsing, since Tier 2 anchor
    IDs can contain arbitrary characters that make robust parsing
    fragile."""
    for scenario_type, pat, shape in SCENARIO_PATTERNS:
        m = pat.search(group_id)
        if not m:
            continue
        if shape == "null":
            return scenario_type, "null_control", int(m.group(1))
        if shape == "ordinal":
            return scenario_type, m.group(1), int(m.group(2))
        if shape == "pair":
            pair = m.group(1) if m.group(1) else scenario_type
            return scenario_type, pair, (0 if m.group(2) == "A" else 1)
    return "unknown", "unknown", np.nan


def load_and_parse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["group_id"].apply(parse_group_id)
    df["scenario_type"] = parsed.apply(lambda t: t[0])
    df["scenario_target"] = parsed.apply(lambda t: t[1])   # category / novel species / anchor-pair
    df["scenario_level"] = parsed.apply(lambda t: t[2])    # ordinal: multiplier / injected count / rep

    # anchor identity: Tier 2 carries an explicit anchor_sample_id column.
    # Tier 1 has no such column (one global scenario) -> use a constant
    # so all downstream grouping-by-anchor logic still works unchanged.
    if "anchor_sample_id" not in df.columns:
        df["anchor_sample_id"] = "global"
    df["anchor_sample_id"] = df["anchor_sample_id"].fillna("global")
    return df


def coverage_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    small = int((df["num_rows"] < MIN_ROWS).sum())
    pct_small = small / total * 100
    by_scenario = df.groupby("scenario_type")["num_rows"].median().to_dict()
    return {"total": total, "small": small, "pct_small": pct_small, "median_by_scenario": by_scenario}


def null_noise_floor_by_anchor(df: pd.DataFrame) -> dict:
    """Returns {anchor_sample_id: pd.Series(metric -> noise floor)}.

    Tier 1 has one anchor ('global') -> effectively one noise floor, same
    as before. Tier 2 has several real anchors with very different real
    sample sizes, so each gets its OWN noise floor: comparing a bloom
    scenario on a 52-row anchor against a noise floor measured on a
    33,000-row anchor would be comparing apples to oranges."""
    nc = df[(df["scenario_type"] == "null_control") & (df["num_rows"] >= MIN_ROWS)]
    result = {}
    if nc.empty:
        return result
    for anchor, g in nc.groupby("anchor_sample_id"):
        levels = sorted(g["scenario_level"].dropna().unique())
        if len(levels) < 2:
            continue
        lo, hi = levels[0], levels[1]
        diffs = {}
        for m in METRICS:
            try:
                v_lo = g[g["scenario_level"] == lo][m].mean()
                v_hi = g[g["scenario_level"] == hi][m].mean()
                diffs[m] = abs(v_lo - v_hi)
            except KeyError:
                diffs[m] = np.nan
        result[anchor] = pd.Series(diffs)
    return result


def get_noise_for_anchor(noise_by_anchor: dict, anchor: str) -> pd.Series:
    """Look up an anchor's noise floor; fall back to the mean across all
    known anchors if this specific one has no null-control data (e.g. it
    got filtered out by MIN_ROWS)."""
    if anchor in noise_by_anchor:
        return noise_by_anchor[anchor]
    if noise_by_anchor:
        return pd.concat(noise_by_anchor.values(), axis=1).mean(axis=1)
    return pd.Series({m: np.nan for m in METRICS})


def get_noise_for_pair(noise_by_anchor: dict, pair_target: str) -> pd.Series:
    """composition_swap targets are 'anchor_x_vs_anchor_y' pairs -- use
    the average of both anchors' individual noise floors when we can
    identify them, else fall back to the overall mean."""
    if pair_target in noise_by_anchor:  # tier1 constant "composition_swap"
        return noise_by_anchor[pair_target]
    matches = [noise for a, noise in noise_by_anchor.items() if a in pair_target]
    if len(matches) >= 1:
        return pd.concat(matches, axis=1).mean(axis=1)
    return get_noise_for_anchor(noise_by_anchor, pair_target)


def score_ordinal_scenario(df: pd.DataFrame, scenario_type: str, noise_by_anchor: dict) -> pd.DataFrame:
    """For bloom / novel_category: correlate metric value with scenario_level.
    Grouped by (anchor, target) so two different anchors that happen to
    pick the same category/species name never get merged together."""
    sub = df[(df["scenario_type"] == scenario_type) & (df["num_rows"] >= MIN_ROWS)]
    rows = []
    for (anchor, target), g in sub.groupby(["anchor_sample_id", "scenario_target"]):
        if g["scenario_level"].nunique() < 2:
            continue
        noise = get_noise_for_anchor(noise_by_anchor, anchor)
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
            works = (abs(r) >= SPEARMAN_MIN) and (not pd.isna(ratio)) and (ratio >= EFFECT_OVER_NOISE_MIN)
            rows.append({
                "scenario_type": scenario_type, "anchor": anchor, "scenario_target": target, "metric": m,
                "spearman_r": r, "spearman_p": p, "swing": swing,
                "noise_floor": n, "effect_over_noise": ratio,
                "n_groups": len(vals), "working": works,
            })
    return pd.DataFrame(rows)


def score_pair_scenario(df: pd.DataFrame, scenario_type: str, noise_by_anchor: dict) -> pd.DataFrame:
    """For composition_swap / real_profile_contrast: compare the two
    sides (level 0 vs level 1) of each anchor-pair target."""
    sub = df[(df["scenario_type"] == scenario_type) & (df["num_rows"] >= MIN_ROWS)]
    rows = []
    if sub.empty:
        return pd.DataFrame(rows)
    for target, g in sub.groupby("scenario_target"):
        noise = get_noise_for_pair(noise_by_anchor, target)
        a = g[g["scenario_level"] == 0]
        b = g[g["scenario_level"] == 1]
        for m in METRICS:
            av, bv = a[m].dropna(), b[m].dropna()
            if len(av) < 3 or len(bv) < 3:
                continue
            swing = abs(av.mean() - bv.mean())
            n = noise.get(m, np.nan)
            ratio = swing / n if (n and n > 0) else np.nan
            works = (not pd.isna(ratio)) and (ratio >= EFFECT_OVER_NOISE_MIN)
            rows.append({
                "scenario_type": scenario_type, "anchor": target, "scenario_target": target, "metric": m,
                "spearman_r": np.nan, "spearman_p": np.nan, "swing": swing,
                "noise_floor": n, "effect_over_noise": ratio,
                "n_groups": len(av) + len(bv), "working": works,
            })
    return pd.DataFrame(rows)


def spotlight_plot(df: pd.DataFrame, scenario_type: str, anchor: str, target: str, metric: str, tag: str):
    sub = df[(df["scenario_type"] == scenario_type) & (df["anchor_sample_id"] == anchor)
             & (df["scenario_target"] == target) & (df["num_rows"] >= MIN_ROWS)]
    if sub.empty:
        return None
    means = sub.groupby("scenario_level")[metric].mean()
    stds = sub.groupby("scenario_level")[metric].std()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(means.index, means.values, yerr=stds.values, marker="o")
    ax.set_title(f"{tag}: {metric}\n({scenario_type} / {anchor} / {target})", fontsize=9)
    ax.set_xlabel("manipulation level")
    ax.set_ylabel(metric)
    fig.tight_layout()
    safe_anchor = str(anchor).replace("<", "").replace(">", "")
    safe_target = str(target).replace("<", "").replace(">", "")
    out_path = OUT_DIR / f"spotlight_{tag}_{scenario_type}_{safe_anchor}_{safe_target}_{metric}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def anchor_rollup(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario_type, metric): how many DISTINCT anchors was this
    tested on, how many passed, and what's the median Spearman /
    median effect-over-noise ACROSS anchors. This is the fix for
    pooling all (anchor,target) pairs into one tally -- a metric that
    works great on one anchor and fails on four others should not look
    the same as one that works consistently across all five."""
    if scorecard.empty:
        return pd.DataFrame()
    rows = []
    for (scen, metric), g in scorecard.groupby(["scenario_type", "metric"]):
        rows.append({
            "scenario_type": scen, "metric": metric,
            "n_anchors_tested": g["anchor"].nunique(),
            "n_anchors_passed": g.loc[g["working"], "anchor"].nunique(),
            "median_abs_spearman": g["spearman_r"].abs().median(),
            "median_effect_over_noise": g["effect_over_noise"].median(),
        })
    out = pd.DataFrame(rows)
    out["pct_anchors_passed"] = (out["n_anchors_passed"] / out["n_anchors_tested"] * 100).round(0)
    return out.sort_values(["scenario_type", "pct_anchors_passed"], ascending=[True, False])


def write_report(coverage: dict, scorecard: pd.DataFrame, rollup: pd.DataFrame, spotlight_paths: dict):
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

    # ---- 2b. Per-anchor rollup: does the metric generalize across anchors? ----
    lines.append("2b) PER-ANCHOR GENERALIZATION (does it work on ONE anchor or MOST anchors?)")
    lines.append("   A metric passing overall could still be driven by one anchor and fail on")
    lines.append("   the rest. This breaks that out: how many distinct anchors was each")
    lines.append("   (scenario, metric) tested on, and what fraction of THOSE anchors passed.")
    lines.append("")
    if rollup.empty:
        lines.append("   No anchor rollup available (see coverage issue above).")
    else:
        for scen in rollup["scenario_type"].unique():
            r = rollup[rollup["scenario_type"] == scen].head(5)
            lines.append(f"   {scen}:")
            for _, row in r.iterrows():
                lines.append(f"     {row['metric']:<28s} {int(row['n_anchors_passed'])}/"
                             f"{int(row['n_anchors_tested'])} anchors passed "
                             f"({row['pct_anchors_passed']:.0f}%), median|r|="
                             f"{row['median_abs_spearman']:.2f}, median effect/noise="
                             f"{row['median_effect_over_noise']:.2f}")
        lines.append("")
        lines.append("   >> A metric near 100% here is reliable across different real profiles.")
        lines.append("      A metric that's high overall (section 2) but low here is only working")
        lines.append("      on specific anchors -- worth checking what's different about those.")
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
    noise_by_anchor = null_noise_floor_by_anchor(df)

    ordinal_types = ["abundance_bloom_additive", "abundance_bloom_fixedn",
                      "novel_category_additive", "novel_category_fixedn",
                      "mixture_gradient"]
    pair_types = ["composition_swap", "real_profile_contrast"]

    scorecards = [score_ordinal_scenario(df, t, noise_by_anchor) for t in ordinal_types]
    scorecards += [score_pair_scenario(df, t, noise_by_anchor) for t in pair_types]
    scorecards = [s for s in scorecards if not s.empty]
    scorecard = pd.concat(scorecards, ignore_index=True) if scorecards else pd.DataFrame()

    scorecard_path = OUT_DIR / "diagnostic_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    print(f"Saved -> {scorecard_path}")

    rollup = anchor_rollup(scorecard)
    rollup_path = OUT_DIR / "diagnostic_anchor_rollup.csv"
    rollup.to_csv(rollup_path, index=False)
    print(f"Saved -> {rollup_path}")

    # spotlight plots: best metric overall, worst metric overall
    spotlight_paths = {}
    if not scorecard.empty:
        per_metric = scorecard.groupby("metric")["working"].agg(["sum", "count"])
        per_metric["pct"] = per_metric["sum"] / per_metric["count"]
        best_metric = per_metric["pct"].idxmax()
        worst_metric = per_metric["pct"].idxmin()

        best_rows = scorecard.loc[scorecard["metric"] == best_metric]
        best_row = best_rows.loc[best_rows["effect_over_noise"].fillna(-1).idxmax()]
        worst_row = scorecard.loc[scorecard["metric"] == worst_metric].iloc[0]

        spotlight_paths["best_metric"] = spotlight_plot(
            df, best_row["scenario_type"], best_row["anchor"], best_row["scenario_target"], best_metric, "BEST")
        spotlight_paths["worst_metric"] = spotlight_plot(
            df, worst_row["scenario_type"], worst_row["anchor"], worst_row["scenario_target"], worst_metric, "WORST")

    write_report(coverage, scorecard, rollup, spotlight_paths)