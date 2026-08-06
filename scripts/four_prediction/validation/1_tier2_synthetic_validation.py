"""
tier2_synthetic_validation.py

TIER 2 -- realistic synthetic profile validation.

Tier 1 asked: "can the metric detect a hand-designed, decontaminated
difference at all?" Every profile there was built from an arbitrary
category dict, pooled globally.

Tier 2 asks: "does the metric detect the kind of difference that
actually shows up between real ship profiles?" The key structural change
from Tier 1:

  - Every synthetic profile is ANCHORED to one (or two) REAL sample_id's
    actual composition -- real category counts, real detritus ratio,
    real co-occurrence structure -- instead of a hand-picked dict.
  - The manipulation (bloom / novel category / composition difference)
    is a PERTURBATION layered on top of that real baseline, sourced
    preferentially from the anchor profile's own rows, falling back to
    the wider dataset only when the anchor doesn't have enough.
  - Composition "swap" is no longer two arbitrary category lists -- it's
    two REAL profiles' real compositions, compared directly.

Everything else (replicate profiles per level, output schema, sample_id
as the grouping key, source_row_uid for traceability) matches Tier 1 so
the same diagnostic_report.py works unmodified on this output.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import load_run_config
from src.metadata import load_run_metadata
from src.index import build_run_index

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PARENT_DIR = Path(r"D:\mojmas\files\Projects\Partitrics\result_validate_Repo\prediction_parti20260325122106")
PRED_CSV = PARENT_DIR / "predictions_with_top3_scores.csv"
OUT_DIR = PARENT_DIR / "tier2_synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

JUNK_KEYWORDS = ["detritus", "reflection", "artefact", "artifact", "fiber", "bubble"]
# QC-flag / placeholder / ambiguous-morphotype labels -- not real taxonomic
# signal, so they must never be auto-picked as a "bloom" or "novel" category.
AMBIGUOUS_KEYWORDS = ["othertocheck", "zoom-in", "like<", "temporary<", "crystal",
                       "solitaryblack", "solitaryglobule", "dark<sphere"]
EXCLUDE_KEYWORDS = JUNK_KEYWORDS + AMBIGUOUS_KEYWORDS
T0XX_PATTERN = None  # set below, needs re


def is_excluded_category(cat: str) -> bool:
    import re
    c = cat.lower()
    if any(k in c for k in EXCLUDE_KEYWORDS):
        return True
    if re.fullmatch(r"t\d{3}", c):  # placeholder codes like t001, t002, ...
        return True
    return False
CAT_COL = "object_annotation_category"

N_ANCHORS = 5          # how many real profiles to anchor scenarios to
TARGET_N = 250          # target rows per synthetic profile (bootstrap up to this if anchor is smaller)
N_REPLICATES = 4        # independent draws per scenario level, same reasoning as Tier 1


# --------------------------------------------------------------------------
# STEP 1: build full_df (same as Tier 1)
# --------------------------------------------------------------------------
def build_full_df() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV, sep=",")
    runs = build_run_index(str(PARENT_DIR))

    merged_frames = []
    for _, r in runs.iterrows():
        run_cfg_path = r["run_cfg"]
        cfg = load_run_config(run_cfg_path)
        input_path = cfg.get("input_path")

        need_cols = ["sample_id", "object_lat", "object_lon", "object_depth_min"]
        need_cols = list(dict.fromkeys(need_cols))
        numeric_cols = ["object_lat", "object_lon"]

        meta = load_run_metadata(input_path, cols=need_cols, numeric_cols=numeric_cols)
        meta = meta.reset_index(drop=True)
        meta = meta.rename(columns={"image_name": "Image Name"})

        m = pd.merge(df, meta, on="Image Name", how="inner")
        merged_frames.append(m)

    full = pd.concat(merged_frames, ignore_index=True)
    full = full.drop_duplicates(subset=["Image Name"]).reset_index(drop=True)
    return full


# --------------------------------------------------------------------------
# STEP 2: pick anchor profiles -- a spread of real, typical profiles,
# not the smallest or largest outliers
# --------------------------------------------------------------------------
def select_anchor_profiles(full_df: pd.DataFrame, n_anchors: int = N_ANCHORS) -> pd.DataFrame:
    stats = full_df.groupby("sample_id").agg(
        n_rows=(CAT_COL, "size"),
        n_categories=(CAT_COL, "nunique"),
    ).reset_index()

    # drop degenerate profiles (too few rows to be a meaningful baseline)
    stats = stats[stats["n_rows"] >= 30].sort_values("n_rows").reset_index(drop=True)
    if len(stats) < n_anchors:
        raise ValueError(f"Only {len(stats)} profiles have >=30 rows; lower N_ANCHORS or the threshold.")

    # A realistic spread across the size distribution -- deliberately
    # avoiding the literal min/max, since your profile sizes are heavily
    # long-tailed (52 to 30k+ rows) and the extremes are outliers, not
    # "typical" profiles. 10th/30th/50th/70th/90th percentile by rank
    # gives a representative spread instead.
    if n_anchors == 1:
        percentiles = [50]
    else:
        percentiles = np.linspace(10, 90, n_anchors)
    idx = np.percentile(np.arange(len(stats)), percentiles).round().astype(int)
    idx = np.clip(idx, 0, len(stats) - 1)
    idx = sorted(set(idx))  # de-dup in case of small n or coarse rounding
    anchors = stats.iloc[idx].reset_index(drop=True)

    print("=== Selected anchor profiles ===")
    print(anchors.to_string(index=False))
    anchors.to_csv(OUT_DIR / "anchor_profiles.csv", index=False)
    return anchors


def get_anchor_pool_and_composition(full_df: pd.DataFrame, anchor_id: str):
    pool = full_df[full_df["sample_id"] == anchor_id]
    composition = pool[CAT_COL].value_counts().to_dict()
    return pool, composition


# --------------------------------------------------------------------------
# STEP 3: preferential sampling -- anchor's own rows first, top up from
# the global pool only if the anchor doesn't have enough of a category
# --------------------------------------------------------------------------
def sample_rows_preferential(full_df: pd.DataFrame, anchor_pool: pd.DataFrame,
                              category: str, n: int) -> pd.DataFrame:
    """Anchor's own rows first, topped up from the global pool if needed.

    FIX (was bug): previously always included ALL local rows whenever
    local < n, so every replicate shared the exact same local rows for
    scarce categories -- replicate variability was artificially low for
    exactly the categories where variability matters most. Now local
    rows are themselves sampled (with replacement if n exceeds what's
    locally available), so each replicate draws independently."""
    local = anchor_pool[anchor_pool[CAT_COL] == category]
    local_n = min(n, len(local))

    parts = []
    if local_n > 0:
        local_idx = rng.choice(local.index.to_numpy(), size=local_n,
                                replace=len(local) < local_n)
        parts.append(full_df.loc[local_idx])

    remaining = n - local_n
    if remaining > 0:
        global_pool = full_df[(full_df[CAT_COL] == category) & (~full_df.index.isin(local.index))]
        if len(global_pool) == 0:
            # nothing outside the anchor either -> fall back to local with replacement
            global_pool = local
        if len(global_pool) == 0:
            raise ValueError(f"No rows anywhere for category '{category}'")
        replace = len(global_pool) < remaining
        if replace:
            print(f"  [warn] category '{category}' has only {len(local) + len(global_pool)} rows "
                  f"dataset-wide, sampling {n} WITH replacement")
        idx = rng.choice(global_pool.index.to_numpy(), size=remaining, replace=replace)
        parts.append(full_df.loc[idx])

    return pd.concat(parts, ignore_index=False)


def assemble_profile(full_df, anchor_pool, category_counts: dict, profile_id: str,
                      scenario_name: str, anchor_id: str) -> pd.DataFrame:
    parts = []
    for cat, n in category_counts.items():
        if n <= 0:
            continue
        parts.append(sample_rows_preferential(full_df, anchor_pool, cat, n))
    prof = pd.concat(parts, ignore_index=True)

    prof["original_sample_id"] = prof["sample_id"]      # real profile each row came from
    prof["anchor_sample_id"] = anchor_id                  # which real profile this scenario is anchored to
    prof["synthetic_profile_id"] = profile_id
    prof["scenario_name"] = scenario_name
    prof["sample_id"] = scenario_name + "__" + prof["synthetic_profile_id"].astype(str)
    prof["source_row_uid"] = prof["original_sample_id"].astype(str) + "_" + prof["sample_id"].astype(str)
    return prof


def bootstrap_baseline_rows(anchor_pool: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """One independent bootstrap draw of target_n rows from the anchor's
    own real rows. Category proportions come out matching the real
    profile automatically. This is the SAME regime null control, bloom,
    and novelty all now share -- previously only null control used
    target_n; bloom/novelty started from the anchor's full (differently
    sized) composition, so the noise floor and the manipulated scenario
    were measured under different sample-size regimes."""
    replace = len(anchor_pool) < target_n
    idx = rng.choice(anchor_pool.index.to_numpy(), size=target_n, replace=replace)
    return anchor_pool.loc[idx]


def rows_to_counts(rows: pd.DataFrame) -> dict:
    return rows[CAT_COL].value_counts().to_dict()


def bump_category_additive(counts: dict, category: str, base_n: int, mult: int) -> dict:
    """Additive bloom/novelty: category count set to base_n * mult, on
    top of the unchanged baseline -- total profile size grows with mult.
    Tests: does the metric detect more objects of this category,
    INCLUDING the resulting shift in total N and biological:detritus
    ratio (a real bloom does both at once, so this is a valid scenario
    in its own right -- just needs to be interpreted as such)."""
    cats = dict(counts)
    cats[category] = base_n * mult
    return cats


def bump_category_fixed_n(counts: dict, category: str, base_n: int, mult: int, rng_local) -> dict:
    """Fixed-size bloom/novelty: add rows for `category`, remove the same
    number of OTHER non-junk biological rows so total N stays constant.
    Tests: does the metric detect the compositional shift ALONE, isolated
    from any change in total profile size."""
    cats = dict(counts)
    current = cats.get(category, 0)
    target = base_n * mult
    added = max(target - current, 0)
    cats[category] = current + added
    if added == 0:
        return cats

    # remove `added` rows from other non-junk categories, proportionally
    donors = {c: n for c, n in cats.items() if c != category and not is_excluded_category(c)}
    donor_total = sum(donors.values())
    if donor_total == 0:
        return cats  # nothing to remove from; falls back to additive behavior
    to_remove = added
    for c, n in sorted(donors.items(), key=lambda kv: -kv[1]):
        if to_remove <= 0:
            break
        share = min(n - 1, round(n / donor_total * added))  # leave at least 1
        share = min(share, to_remove)
        cats[c] = max(cats[c] - share, 0)
        to_remove -= share
    return cats


# --------------------------------------------------------------------------
# STEP 4: scenarios
# --------------------------------------------------------------------------
def scenario_null_control(full_df, anchor_pool, anchor_id, target_n=TARGET_N, n_replicates=N_REPLICATES):
    """Multiple independent target_n bootstrap draws from the SAME real
    profile. Defines the noise floor for THIS anchor, at the SAME sample
    size every other scenario for this anchor now uses."""
    frames = []
    for rep in range(1, n_replicates + 1):
        baseline_rows = bootstrap_baseline_rows(anchor_pool, target_n)
        cats = rows_to_counts(baseline_rows)
        pid = f"null_control_{anchor_id}_rep{rep}"
        frames.append(assemble_profile(full_df, anchor_pool, cats, pid, "null_control", anchor_id))
    return pd.concat(frames, ignore_index=True)


def scenario_abundance_bloom(full_df, anchor_pool, anchor_id, composition: dict,
                              bloom_category: str = None, multipliers=(1, 3, 10),
                              target_n=TARGET_N, n_replicates=N_REPLICATES):
    """Bloom scenario, built on a fresh target_n baseline PER REPLICATE
    (same regime as null control -- fixes the baseline-size mismatch).
    Generates both variants:
      abundance_bloom_additive : total N grows with the bloom
      abundance_bloom_fixedn   : total N held constant, other
                                  categories proportionally reduced
    so a metric's response can be attributed to "more objects overall"
    vs "compositional shift alone"."""
    if bloom_category is None:
        non_junk = {c: n for c, n in composition.items() if not is_excluded_category(c)}
        if not non_junk:
            raise ValueError(f"Anchor {anchor_id} has no non-junk categories to bloom.")
        bloom_category = max(non_junk, key=non_junk.get)

    frames = []
    for mult in multipliers:
        for rep in range(1, n_replicates + 1):
            baseline_rows = bootstrap_baseline_rows(anchor_pool, target_n)
            baseline_cats = rows_to_counts(baseline_rows)
            base_n = baseline_cats.get(bloom_category, max(1, composition.get(bloom_category, 5)))

            cats_add = bump_category_additive(baseline_cats, bloom_category, base_n, mult)
            pid_add = f"bloomadd_{anchor_id}_{bloom_category.replace(' ', '_')}_x{mult}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats_add, pid_add,
                                            "abundance_bloom_additive", anchor_id))

            cats_fixed = bump_category_fixed_n(baseline_cats, bloom_category, base_n, mult, rng)
            pid_fixed = f"bloomfix_{anchor_id}_{bloom_category.replace(' ', '_')}_x{mult}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats_fixed, pid_fixed,
                                            "abundance_bloom_fixedn", anchor_id))
    return pd.concat(frames, ignore_index=True), bloom_category


def scenario_novel_category(full_df, anchor_pool, anchor_id, composition: dict,
                             novel_category: str, novel_counts, target_n=TARGET_N,
                             n_replicates=N_REPLICATES):
    """Novel-category injection, built on a fresh target_n baseline PER
    REPLICATE (same fix as bloom). Both additive and fixed-n variants."""
    frames = []
    for n in novel_counts:
        for rep in range(1, n_replicates + 1):
            baseline_rows = bootstrap_baseline_rows(anchor_pool, target_n)
            baseline_cats = rows_to_counts(baseline_rows)
            baseline_cats.pop(novel_category, None)  # anchor shouldn't have it; drop if bootstrap got unlucky

            cats_add = dict(baseline_cats)
            if n > 0:
                cats_add[novel_category] = n
            pid_add = f"noveladd_{anchor_id}_{novel_category.replace(' ', '_')}_n{n}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats_add, pid_add,
                                            "novel_category_additive", anchor_id))

            cats_fixed = bump_category_fixed_n(baseline_cats, novel_category, 0, 1, rng) if n == 0 \
                else bump_category_fixed_n(baseline_cats, novel_category, n, 1, rng)
            pid_fixed = f"novelfix_{anchor_id}_{novel_category.replace(' ', '_')}_n{n}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats_fixed, pid_fixed,
                                            "novel_category_fixedn", anchor_id))
    return pd.concat(frames, ignore_index=True)


def scenario_real_profile_contrast(full_df, anchor_pool_x, anchor_id_x,
                                    anchor_pool_y, anchor_id_y,
                                    target_n=TARGET_N, n_replicates=N_REPLICATES):
    """Two REAL profiles' compositions, each independently bootstrapped
    to the SAME target_n (previously used the full, differently-sized
    anchor compositions -- inconsistent with every other scenario).

    Renamed from 'composition_swap': this is NOT a controlled
    perturbation. X and Y can differ in taxonomy, detritus ratio,
    richness, depth, location, batch effects, all at once. A metric
    responding here only tells you it can tell X and Y apart -- not
    WHICH difference it's responding to. Use scenario_mixture_gradient
    for an isolated, controlled version of this question."""
    frames = []
    for rep in range(1, n_replicates + 1):
        rows_x = bootstrap_baseline_rows(anchor_pool_x, target_n)
        rows_y = bootstrap_baseline_rows(anchor_pool_y, target_n)
        cats_x = rows_to_counts(rows_x)
        cats_y = rows_to_counts(rows_y)
        pid_x = f"contrast_{anchor_id_x}_vs_{anchor_id_y}_A_rep{rep}"
        pid_y = f"contrast_{anchor_id_x}_vs_{anchor_id_y}_B_rep{rep}"
        frames.append(assemble_profile(full_df, anchor_pool_x, cats_x, pid_x,
                                        "real_profile_contrast", anchor_id_x))
        frames.append(assemble_profile(full_df, anchor_pool_y, cats_y, pid_y,
                                        "real_profile_contrast", anchor_id_y))
    return pd.concat(frames, ignore_index=True)


def scenario_mixture_gradient(full_df, anchor_pool_x, anchor_id_x,
                               anchor_pool_y, anchor_id_y,
                               fractions=(0, 10, 25, 50), target_n=TARGET_N,
                               n_replicates=N_REPLICATES):
    """Controlled, FIXED-SIZE gradient: X progressively contaminated with
    a known percentage of Y, total N held constant throughout. This is
    what 'real_profile_contrast' can't give you -- an isolated answer to
    'does the metric respond monotonically as profile Y's influence
    grows', with everything else (total N) held fixed."""
    frames = []
    for pct in fractions:
        for rep in range(1, n_replicates + 1):
            n_y = round(target_n * pct / 100)
            n_x = target_n - n_y
            rows_x = bootstrap_baseline_rows(anchor_pool_x, n_x) if n_x > 0 else anchor_pool_x.iloc[0:0]
            rows_y = bootstrap_baseline_rows(anchor_pool_y, n_y) if n_y > 0 else anchor_pool_y.iloc[0:0]
            mix_rows = pd.concat([rows_x, rows_y], ignore_index=False)
            cats = rows_to_counts(mix_rows)
            pid = f"mixture_{anchor_id_x}_vs_{anchor_id_y}_p{pct}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool_x, cats, pid,
                                            "mixture_gradient", anchor_id_x))
    return pd.concat(frames, ignore_index=True)


def pick_novel_category_for_anchor(full_df, composition: dict, min_rows_elsewhere=80):
    """A real category absent from this anchor but well-represented
    dataset-wide -- a plausible 'newcomer'. min_rows_elsewhere is set
    above the largest novel_counts level (60) with headroom, so the
    injection doesn't fall back to heavy with-replacement sampling."""
    all_counts = full_df[CAT_COL].value_counts()
    candidates = [c for c in all_counts.index
                  if c not in composition
                  and not is_excluded_category(c)
                  and all_counts[c] >= min_rows_elsewhere]
    if not candidates:
        return None
    # prefer a moderately-rare real category, not the single rarest or
    # the single most common, for a realistic "newcomer" scenario
    candidates_sorted = sorted(candidates, key=lambda c: all_counts[c])
    return candidates_sorted[len(candidates_sorted) // 3]


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    full_df = build_full_df()
    anchors = select_anchor_profiles(full_df, N_ANCHORS)
    anchor_ids = anchors["sample_id"].tolist()

    all_scenarios = []
    anchor_data = {}
    for aid in anchor_ids:
        pool, comp = get_anchor_pool_and_composition(full_df, aid)
        anchor_data[aid] = (pool, comp)

    # null control + bloom + novel category, per anchor
    for aid in anchor_ids:
        pool, comp = anchor_data[aid]

        all_scenarios.append(scenario_null_control(full_df, pool, aid))

        bloom_df, bloom_cat = scenario_abundance_bloom(full_df, pool, aid, comp)
        all_scenarios.append(bloom_df)

        novel_cat = pick_novel_category_for_anchor(full_df, comp)
        if novel_cat:
            all_scenarios.append(
                scenario_novel_category(full_df, pool, aid, comp, novel_cat,
                                         novel_counts=(0, 5, 20, 60))
            )
        else:
            print(f"  [info] no suitable novel category found for anchor {aid}, skipping")

    # real profile contrast + mixture gradient: pair up consecutive
    # anchors (by size rank) so each pair is a realistic "these two real
    # profiles differ" test
    for i in range(0, len(anchor_ids) - 1, 2):
        aid_x, aid_y = anchor_ids[i], anchor_ids[i + 1]
        pool_x, _ = anchor_data[aid_x]
        pool_y, _ = anchor_data[aid_y]
        all_scenarios.append(
            scenario_real_profile_contrast(full_df, pool_x, aid_x, pool_y, aid_y)
        )
        all_scenarios.append(
            scenario_mixture_gradient(full_df, pool_x, aid_x, pool_y, aid_y)
        )

    final = pd.concat(all_scenarios, ignore_index=True)
    out_path = OUT_DIR / "tier2_synthetic_profiles.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved Tier-2 synthetic profiles -> {out_path}")
    print(final.groupby(["scenario_name", "anchor_sample_id"])["sample_id"].nunique())