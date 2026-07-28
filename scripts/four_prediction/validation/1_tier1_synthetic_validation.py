"""
tier1_synthetic_validation.py

TIER 1 — idealized synthetic profile validation.

Idea: we do NOT fabricate any score/embedding values. Every row in every
synthetic profile is a REAL row taken from your existing
`predictions_with_top3_scores.csv` (merged with metadata for sample_id /
category / depth). We only control WHICH real rows go into WHICH synthetic
profile, and HOW MANY of each category. This keeps latent vectors / scores
authentic while giving you full control over composition, abundance, and
novelty — the three axes you described.

Workflow:
  STEP 1: build `full_df` (predictions + metadata merged across runs)
  STEP 2: explore categories -> decide which N are "obvious" enough to use
  STEP 3: generate Tier-1 scenarios (bloom / composition-swap / novel-class
          / null-control) by resampling from `full_df`
  STEP 4: write one CSV, same columns as predictions_with_top3_scores.csv
          + a few bookkeeping columns, ready for your metric pipeline
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
OUT_DIR = PARENT_DIR / "tier1_synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Rows whose category matches (case-insensitive substring) one of these
# are treated as "junk/background" classes rather than taxonomic signal.
# Adjust to your label vocabulary.
JUNK_KEYWORDS = ["detritus", "reflection", "artefact", "artifact", "fiber", "bubble"]


# --------------------------------------------------------------------------
# STEP 1: build full_df (predictions + metadata, all runs)
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
# STEP 2: category exploration
# --------------------------------------------------------------------------
def explore_categories(full_df: pd.DataFrame) -> pd.DataFrame:
    cat_col = "object_annotation_category"

    total_counts = full_df[cat_col].value_counts()
    n_profiles_per_cat = full_df.groupby(cat_col)["sample_id"].nunique()
    n_profiles_total = full_df["sample_id"].nunique()

    stats = pd.DataFrame({
        "total_count": total_counts,
        "n_profiles_present_in": n_profiles_per_cat,
    }).fillna(0)
    stats["pct_of_all_rows"] = (stats["total_count"] / len(full_df) * 100).round(2)
    stats["pct_profiles_present_in"] = (stats["n_profiles_present_in"] / n_profiles_total * 100).round(1)
    stats["is_junk_like"] = stats.index.to_series().str.lower().apply(
        lambda s: any(k in s for k in JUNK_KEYWORDS)
    )
    stats = stats.sort_values("total_count", ascending=False)

    print(f"\nTotal rows: {len(full_df)}  |  Total profiles (sample_id): {n_profiles_total}")
    print(f"Total distinct categories: {full_df[cat_col].nunique()}\n")
    print(stats.to_string())

    stats.to_csv(OUT_DIR / "category_exploration.csv")
    print(f"\nSaved -> {OUT_DIR / 'category_exploration.csv'}")

    # A practical starting filter: categories that are NOT junk-like,
    # appear in a reasonable number of profiles (so you can resample them
    # realistically), and have enough total rows to sample from.
    candidates = stats[(~stats["is_junk_like"]) & (stats["total_count"] >= 30)]
    print(f"\nCandidate 'obvious' categories (non-junk, count>=30): {len(candidates)}")
    print(candidates.head(20).to_string())

    return stats


# --------------------------------------------------------------------------
# STEP 3 helpers: sampling a real-row pool per category
# --------------------------------------------------------------------------
def sample_rows(full_df: pd.DataFrame, category: str, n: int) -> pd.DataFrame:
    """Sample n real rows of a given category. Sample with replacement
    only if the pool is smaller than n (and warn)."""
    pool = full_df[full_df["object_annotation_category"] == category]
    if len(pool) == 0:
        raise ValueError(f"No rows found for category '{category}'")
    replace = len(pool) < n
    if replace:
        print(f"  [warn] category '{category}' has only {len(pool)} rows, "
              f"sampling {n} WITH replacement")
    idx = rng.choice(pool.index.values, size=n, replace=replace)
    return full_df.loc[idx]


def assemble_profile(full_df, category_counts: dict, profile_id: str, scenario_name: str) -> pd.DataFrame:
    """category_counts: {category_name: n_rows_to_sample}"""
    parts = []
    for cat, n in category_counts.items():
        if n <= 0:
            continue
        parts.append(sample_rows(full_df, cat, n))
    prof = pd.concat(parts, ignore_index=True)
    prof["synthetic_profile_id"] = profile_id
    prof["scenario_name"] = scenario_name
    prof["original_sample_id"] = prof["sample_id"]
    # `sample_id` is overwritten here to be the grouping key you'll actually
    # use downstream: real source profile + synthetic scenario/profile.
    # `original_sample_id` preserves the untouched real sample_id for
    # traceability back to the source profile.
    prof["sample_id"] = prof["original_sample_id"].astype(str) + "_" + prof["synthetic_profile_id"].astype(str)
    return prof


# --------------------------------------------------------------------------
# STEP 3: scenario definitions
# --------------------------------------------------------------------------
def scenario_null_control(full_df, base_categories: dict, n_replicates=2):
    """Two (or more) profiles bootstrapped from the SAME composition.
    Metrics computed on these should show ~no difference -> defines your
    noise floor / null distribution."""
    frames = []
    for i in range(n_replicates):
        pid = f"null_control_{i + 1}"
        frames.append(assemble_profile(full_df, base_categories, pid, "null_control"))
    return pd.concat(frames, ignore_index=True)


def scenario_abundance_bloom(full_df, base_categories: dict, bloom_category: str,
                             multipliers=(1, 3, 10)):
    """Same category set every time; one category's count is multiplied
    to simulate a bloom, others held constant."""
    frames = []
    base_n = base_categories.get(bloom_category, 20)
    for mult in multipliers:
        cats = dict(base_categories)
        cats[bloom_category] = base_n * mult
        pid = f"bloom_{bloom_category.replace(' ', '_')}_x{mult}"
        frames.append(assemble_profile(full_df, cats, pid, "abundance_bloom"))
    return pd.concat(frames, ignore_index=True)


def scenario_composition_swap(full_df, categories_a: dict, categories_b: dict):
    """Two profiles, same total N, different category membership
    (partial or full overlap allowed)."""
    frames = [
        assemble_profile(full_df, categories_a, "composition_A", "composition_swap"),
        assemble_profile(full_df, categories_b, "composition_B", "composition_swap"),
    ]
    return pd.concat(frames, ignore_index=True)


def scenario_novel_category(full_df, base_categories: dict, novel_category: str,
                            novel_counts=(0, 5, 20, 60)):
    """Baseline composition held fixed; a category absent from the
    baseline (novel_counts[0] == 0) is introduced at increasing
    prevalence in subsequent profiles."""
    frames = []
    for n in novel_counts:
        cats = dict(base_categories)
        pid = f"novel_{novel_category.replace(' ', '_')}_n{n}"
        if n > 0:
            cats[novel_category] = n
        frames.append(assemble_profile(full_df, cats, pid, "novel_category"))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    full_df = build_full_df()
    stats = explore_categories(full_df)

    # ---- Chosen after inspecting category_exploration.csv ----
    # Real taxa only (QC flags / "t0xx" placeholder codes / "like<"/"temporary<"
    # labels excluded), with decent row counts and profile coverage.
    # Chaetognatha and Thecosomata dropped (noisy labels per manual review).
    CHOSEN_CATEGORIES = [
        "Copepoda<Maxillopoda",  # 11233 rows, 97.7% profiles - dominant real taxon
        "Eumalacostraca",  # 438 rows, 81.8% profiles
        "Limacinidae",  # 307 rows, 77.3% profiles
        "Hydrozoa",  # 170 rows, 75.0% profiles
        "Ostracoda",  # 166 rows, 70.5% profiles
        "Siphonophorae",  # 121 rows, 68.2% profiles
        "Annelida",  # 146 rows, 68.2% profiles
        "Acantharia",  # 152 rows, 56.8% profiles
        "Cydippida",  # 75 rows, 63.6% profiles
    ]

    # Detritus is structurally different from a "rare class" - it's a
    # dominant, heterogeneous background. Pool its sub-labels together
    # rather than treating them as separate taxonomic categories.
    DETRITUS_LABELS = ["detritus", "filament<detritus", "fiber<detritus"]

    # Two novel-category candidates at different rarity levels, since
    # "moderately rare newcomer" and "barely-there newcomer" test
    # different sensitivity thresholds.
    NOVEL_LABEL_MODERATE = "Coelographis"  # 69 rows, 65.9% profiles
    NOVEL_LABEL_RARE = "Foraminifera"  # 18 rows, 27.3% profiles

    BASE_N_PER_CAT = 20
    base_categories = {c: BASE_N_PER_CAT for c in CHOSEN_CATEGORIES}

    detritus_total = 200  # detritus dominates, as in reality
    per_label = detritus_total // len(DETRITUS_LABELS)
    for lbl in DETRITUS_LABELS:
        base_categories[lbl] = per_label

    all_scenarios = []

    all_scenarios.append(scenario_null_control(full_df, base_categories))

    if len(CHOSEN_CATEGORIES) >= 1:
        all_scenarios.append(
            scenario_abundance_bloom(full_df, base_categories, CHOSEN_CATEGORIES[0])
        )

    if len(CHOSEN_CATEGORIES) >= 2:
        cats_a = {c: BASE_N_PER_CAT for c in CHOSEN_CATEGORIES[: len(CHOSEN_CATEGORIES) // 2]}
        cats_b = {c: BASE_N_PER_CAT for c in CHOSEN_CATEGORIES[len(CHOSEN_CATEGORIES) // 2:]}
        for lbl in DETRITUS_LABELS:
            cats_a[lbl] = per_label
            cats_b[lbl] = per_label
        all_scenarios.append(scenario_composition_swap(full_df, cats_a, cats_b))

    # moderate-rarity novel category
    baseline_without_novel = {k: v for k, v in base_categories.items() if k != NOVEL_LABEL_MODERATE}
    all_scenarios.append(
        scenario_novel_category(full_df, baseline_without_novel, NOVEL_LABEL_MODERATE,
                                novel_counts=(0, 5, 20, 60))
    )

    # rare novel category (fewer available rows -> smaller max count)
    baseline_without_rare = {k: v for k, v in base_categories.items() if k != NOVEL_LABEL_RARE}
    all_scenarios.append(
        scenario_novel_category(full_df, baseline_without_rare, NOVEL_LABEL_RARE,
                                novel_counts=(0, 3, 8, 15))
    )

    final = pd.concat(all_scenarios, ignore_index=True)

    # `synthetic_profile_id` alone isn't guaranteed unique across scenarios
    # (e.g. every scenario could in principle reuse names). Build an
    # explicit, globally unique profile key to group/aggregate metrics on.
    final["profile_uid"] = final["scenario_name"] + "__" + final["synthetic_profile_id"]
    uid_map = {uid: i for i, uid in enumerate(final["profile_uid"].unique(), start=1)}
    final["profile_int_id"] = final["profile_uid"].map(uid_map)

    # keep the same columns as predictions_with_top3_scores.csv, plus
    # the bookkeeping columns we added
    out_path = OUT_DIR / "tier1_synthetic_profiles.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved Tier-1 synthetic profiles -> {out_path}")
    print(final.groupby(["scenario_name", "synthetic_profile_id"]).size())