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
    local = anchor_pool[anchor_pool[CAT_COL] == category]
    if len(local) >= n:
        idx = rng.choice(local.index.values, size=n, replace=False)
        return full_df.loc[idx]

    parts = [local]
    remaining = n - len(local)
    global_pool = full_df[(full_df[CAT_COL] == category) & (~full_df.index.isin(local.index))]
    if len(global_pool) == 0:
        raise ValueError(f"No rows anywhere for category '{category}'")
    replace = len(global_pool) < remaining
    if replace:
        print(f"  [warn] category '{category}' has only {len(local) + len(global_pool)} rows "
              f"dataset-wide, sampling {n} WITH replacement")
    idx = rng.choice(global_pool.index.values, size=remaining, replace=replace)
    parts.append(full_df.loc[idx])
    return pd.concat(parts)


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


def bootstrap_to_target(anchor_pool: pd.DataFrame, target_n: int) -> dict:
    """Resample uniformly from the anchor's own rows (with replacement if
    needed) up to target_n. Since sampling is uniform over the anchor's
    real rows, category proportions come out matching the real profile
    automatically -- no manual category dict needed for a pure baseline."""
    replace = len(anchor_pool) < target_n
    idx = rng.choice(anchor_pool.index.values, size=target_n, replace=replace)
    sampled = anchor_pool.loc[idx]
    return sampled[CAT_COL].value_counts().to_dict()


# --------------------------------------------------------------------------
# STEP 4: scenarios
# --------------------------------------------------------------------------
def scenario_null_control(full_df, anchor_pool, anchor_id, target_n=TARGET_N, n_replicates=N_REPLICATES):
    """Multiple bootstrap draws from the SAME real profile's own
    composition. Defines the noise floor for THIS anchor specifically."""
    frames = []
    for rep in range(1, n_replicates + 1):
        cats = bootstrap_to_target(anchor_pool, target_n)
        pid = f"null_control_{anchor_id}_rep{rep}"
        frames.append(assemble_profile(full_df, anchor_pool, cats, pid, "null_control", anchor_id))
    return pd.concat(frames, ignore_index=True)


def scenario_abundance_bloom(full_df, anchor_pool, anchor_id, composition: dict,
                              bloom_category: str = None, multipliers=(1, 3, 10),
                              n_replicates=N_REPLICATES):
    """Real baseline composition, one real category multiplied. If no
    category given, auto-picks the anchor's most abundant non-junk
    category (a realistic bloom target)."""
    if bloom_category is None:
        non_junk = {c: n for c, n in composition.items() if not is_excluded_category(c)}
        if not non_junk:
            raise ValueError(f"Anchor {anchor_id} has no non-junk categories to bloom.")
        bloom_category = max(non_junk, key=non_junk.get)

    base_n = composition.get(bloom_category, 5)
    frames = []
    for mult in multipliers:
        cats = dict(composition)
        cats[bloom_category] = base_n * mult
        for rep in range(1, n_replicates + 1):
            pid = f"bloom_{anchor_id}_{bloom_category.replace(' ', '_')}_x{mult}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats, pid, "abundance_bloom", anchor_id))
    return pd.concat(frames, ignore_index=True), bloom_category


def scenario_composition_swap(full_df, anchor_pool_x, anchor_id_x, composition_x,
                               anchor_pool_y, anchor_id_y, composition_y,
                               n_replicates=N_REPLICATES):
    """Two REAL profiles' real compositions, resampled up to comparable
    size. This is the realism-anchored version of Tier 1's hand-picked
    category-list swap: the difference here is whatever two real
    profiles actually differ by."""
    frames = []
    for rep in range(1, n_replicates + 1):
        pid_x = f"composition_{anchor_id_x}_vs_{anchor_id_y}_A_rep{rep}"
        pid_y = f"composition_{anchor_id_x}_vs_{anchor_id_y}_B_rep{rep}"
        frames.append(assemble_profile(full_df, anchor_pool_x, composition_x, pid_x,
                                        "composition_swap", anchor_id_x))
        frames.append(assemble_profile(full_df, anchor_pool_y, composition_y, pid_y,
                                        "composition_swap", anchor_id_y))
    return pd.concat(frames, ignore_index=True)


def scenario_novel_category(full_df, anchor_pool, anchor_id, composition: dict,
                             novel_category: str, novel_counts, n_replicates=N_REPLICATES):
    """Real baseline (which genuinely lacks novel_category), real
    novel_category rows injected at increasing counts, sourced from the
    global pool since the anchor has none by definition."""
    baseline = {k: v for k, v in composition.items() if k != novel_category}
    frames = []
    for n in novel_counts:
        cats = dict(baseline)
        if n > 0:
            cats[novel_category] = n
        for rep in range(1, n_replicates + 1):
            pid = f"novel_{anchor_id}_{novel_category.replace(' ', '_')}_n{n}_rep{rep}"
            frames.append(assemble_profile(full_df, anchor_pool, cats, pid, "novel_category", anchor_id))
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

    # composition swap: pair up consecutive anchors (by size rank) so
    # each pair is a realistic "these two real profiles differ" test
    for i in range(0, len(anchor_ids) - 1, 2):
        aid_x, aid_y = anchor_ids[i], anchor_ids[i + 1]
        pool_x, comp_x = anchor_data[aid_x]
        pool_y, comp_y = anchor_data[aid_y]
        all_scenarios.append(
            scenario_composition_swap(full_df, pool_x, aid_x, comp_x, pool_y, aid_y, comp_y)
        )

    final = pd.concat(all_scenarios, ignore_index=True)
    out_path = OUT_DIR / "tier2_synthetic_profiles.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved Tier-2 synthetic profiles -> {out_path}")
    print(final.groupby(["scenario_name", "anchor_sample_id"])["sample_id"].nunique())