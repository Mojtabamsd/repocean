"""
validate_expert_results.py

Expert validation analysis for the three-model fusion framework.

======================================================================
INPUTS
======================================================================

1. EcoTaxa expert export
   ----------------------
   Contains expert annotations and original metadata.

   Important fields:
       object_id
       object_annotation_status
       object_annotation_category
       object_annotation_hierarchy
       object_major

2. fused_predictions.csv
   ----------------------
   Contains:
       m1_label
       m2_label
       m3_label
       final_label
       path
       confidence
       scores
       etc.

3. 01_sampling_records.csv
   ------------------------
   Private model-specific sampling provenance.

4. 03_sampling_report.csv
   -----------------------
   Available/sample counts for each model-superclass-bin stratum.

5. label_to_int.csv
   -----------------
   Fine label -> ecotaxa_20 superclass mapping.

======================================================================
MAIN ANALYSES
======================================================================

A. Overall performance
   M1 vs expert
   M2 vs expert
   M3 vs expert
   Fusion vs expert

   Metrics:
       fine-label accuracy
       fine-label macro F1
       fine-label balanced accuracy
       superclass accuracy
       superclass macro F1
       superclass balanced accuracy
       prediction coverage

B. Performance by fusion pathway
   A_all_agree
   B_majority
   C_superclass
   D_score
   D_score_close

C. Performance by expert reference superclass

D. Fusion rescue / regression
   - fusion correct, model wrong
   - fusion wrong, model correct
   - all models wrong but fusion correct
   - all models correct but fusion wrong

E. Confidence validation

F. Paired McNemar tests:
   Fusion vs M1
   Fusion vs M2
   Fusion vs M3

G. Sampling-aware population estimates
   using inverse inclusion probabilities.

======================================================================
OUTPUTS
======================================================================

results/
    validation_merged.csv
    expert_label_mapping_report.csv
    overall_metrics_unweighted.csv
    overall_metrics_population_weighted.csv
    pathway_metrics.csv
    superclass_metrics.csv
    confidence_metrics.csv
    fusion_benefit.csv
    mcnemar_tests.csv
    validation_confusion_superclass.csv
    population_inclusion_probabilities.csv
    validation_report.txt

Figures:
    fig1_overall_performance.png
    fig2_pathway_performance.png
    fig3_superclass_performance.png
    fig4_fusion_benefit.png
    fig5_confidence_validation.png
    fig6_superclass_confusion.png
    fig7_weighted_vs_unweighted.png

======================================================================
"""

from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ----------------------------------------------------------------------
# Optional/scientific packages
# ----------------------------------------------------------------------

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        balanced_accuracy_score,
        confusion_matrix,
    )
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install with:\n"
        "pip install scikit-learn"
    ) from exc


try:
    from scipy.stats import binomtest
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install with:\n"
        "pip install scipy"
    ) from exc


# ======================================================================
# 1. USER SETTINGS
# ======================================================================

# ----------------------------------------------------------------------
# Expert EcoTaxa export
# ----------------------------------------------------------------------

EXPERT_EXPORT = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
    r"\ecotaxa_export_EXPERT.tsv"
)


# ----------------------------------------------------------------------
# Full fusion results
# ----------------------------------------------------------------------

FUSED_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\fused_predictions.csv"
)


# ----------------------------------------------------------------------
# Private sampling provenance
# ----------------------------------------------------------------------

SAMPLING_RECORDS_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
    r"\01_sampling_records.csv"
)


SAMPLING_REPORT_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
    r"\03_sampling_report.csv"
)


# ----------------------------------------------------------------------
# Label -> superclass mapping
# ----------------------------------------------------------------------

LABEL_MAP_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\label_to_int.csv"
)


METADATA_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\merge_three_prediction_all.csv"
)

# ----------------------------------------------------------------------
# Output directory
# ----------------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
    r"\expert_validation_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 2. COLUMN SETTINGS
# ======================================================================

OBJECT_ID = "object_id"

EXPERT_LABEL_COL = (
    "object_annotation_category"
)

EXPERT_HIERARCHY_COL = (
    "object_annotation_hierarchy"
)

EXPERT_STATUS_COL = (
    "object_annotation_status"
)

BIN_VARIABLE = "object_major"


MODEL_LABELS = {
    "M1": "m1_label",
    "M2": "m2_label",
    "M3": "m3_label",
}

FUSION_LABEL = "final_label"

FUSION_PATH = "path"

FUSION_CONFIDENCE = "confidence"


# ======================================================================
# 3. SAMPLING DESIGN
# ======================================================================

N_BINS = 10
K_PER_BIN = 10

USE_LOG_BINS = True

EPS = 1e-6


# ----------------------------------------------------------------------
# Whether to calculate population-weighted estimates.
#
# Recommended: True
# ----------------------------------------------------------------------

CALCULATE_POPULATION_WEIGHTED = True


# ======================================================================
# 4. LABEL EQUIVALENCES USED BY THE FUSION ENGINE
# ======================================================================

VOTE_EQUIV = {
    "like<feces": "detritus",
    "badfocus<artefact": "artefact",
}


# ======================================================================
# 5. PLOT SETTINGS
# ======================================================================

RANDOM_SEED = 42

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ======================================================================
# 6. GENERAL HELPERS
# ======================================================================

def normalized_text(value):
    """
    Normalize text conservatively.

    Intended for:
        - labels
        - taxonomy names
        - label-map lookup
    """

    if pd.isna(value):
        return None

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.strip().lower()

    # Normalize Unicode dash variants
    text = (
        text
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    # Normalize whitespace around "<"
    text = re.sub(
        r"\s*<\s*",
        "<",
        text,
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def loose_label_key(value):
    """
    A second-level label key.

    Treats:
        'zoom in'
        'zoom-in'
        'zoom_in'

    as equivalent.

    This is useful because EcoTaxa may export a label differently
    from the flat model label vocabulary.
    """

    text = normalized_text(value)

    if text is None:
        return None

    return re.sub(
        r"[\s_-]+",
        "",
        text,
    )


def apply_vote_equivalence(value):
    """
    Apply the equivalences explicitly used by the fusion engine.
    """

    text = normalized_text(value)

    if text is None:
        return None

    return normalized_text(
        VOTE_EQUIV.get(
            text,
            text,
        )
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ======================================================================
# 7. WILSON CONFIDENCE INTERVAL
# ======================================================================

def wilson_interval(
    correct,
    n,
    z=1.96,
):
    """
    Wilson 95% confidence interval for a binomial proportion.
    """

    if n == 0:
        return np.nan, np.nan

    p = correct / n

    denominator = (
        1 + z**2 / n
    )

    center = (
        p
        + z**2 / (2*n)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                p*(1-p)/n
            )
            + z**2/(4*n**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


# ======================================================================
# 8. LOAD LABEL MAP
# ======================================================================

def load_superclass_map(path):
    """
    Load:

        label
        superclass_ecotaxa_20
    """

    mapping_df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "label",
        "superclass_ecotaxa_20",
    }

    missing = (
        required
        - set(mapping_df.columns)
    )

    if missing:
        raise KeyError(
            "Missing columns in label map:\n"
            f"{sorted(missing)}"
        )

    mapping = {}

    for _, row in mapping_df.iterrows():

        label = normalized_text(
            row["label"]
        )

        superclass = (
            str(
                row["superclass_ecotaxa_20"]
            ).strip()
            if not pd.isna(
                row["superclass_ecotaxa_20"]
            )
            else None
        )

        if label is None or superclass is None:
            continue

        mapping[
            loose_label_key(label)
        ] = superclass

    return mapping


# ======================================================================
# 9. LOAD MODEL LABEL VOCABULARY
# ======================================================================

def build_model_vocabulary(fused):
    """
    Build a loose-key -> canonical model label map from M1/M2/M3.
    """

    vocabulary = {}

    collision_rows = []

    for model, column in MODEL_LABELS.items():

        values = (
            fused[column]
            .dropna()
            .astype(str)
            .unique()
        )

        for value in values:

            canonical = normalized_text(
                value
            )

            key = loose_label_key(
                canonical
            )

            if key is None:
                continue

            if key in vocabulary:

                if vocabulary[key] != canonical:

                    collision_rows.append({
                        "loose_key": key,
                        "existing": vocabulary[key],
                        "new": canonical,
                    })

            else:

                vocabulary[key] = canonical


    if collision_rows:

        collision_df = pd.DataFrame(
            collision_rows
        )

        collision_df.to_csv(
            OUTPUT_DIR
            / "label_vocabulary_collisions.csv",
            index=False,
        )

        print(
            "\nWARNING:"
            f" {len(collision_df)} label-vocabulary "
            "collisions detected."
        )

    return vocabulary


# ======================================================================
# 10. CANONICALIZE EXPERT LABEL
# ======================================================================

def expert_label_candidates(
    category,
    hierarchy,
):
    """
    Build reasonable representations of an EcoTaxa expert label.

    Example:

        detritus<not-living
        ->
        detritus<not-living
        detritus
        detritus

    Example:

        Copepoda<Multicrustacea
        ->
        Copepoda<Multicrustacea
        Copepoda
        Copepoda
    """

    category = normalized_text(
        category
    )

    hierarchy = normalized_text(
        hierarchy
    )

    candidates = []

    def add(value):

        if value is None:
            return

        if value not in candidates:
            candidates.append(value)


    # Complete EcoTaxa category
    add(category)


    # Main label before '<'
    if category and "<" in category:

        add(
            category.split(
                "<",
                1,
            )[0].strip()
        )


    # Final hierarchy node
    if hierarchy:

        hierarchy_parts = [
            x.strip()
            for x in hierarchy.split(
                ">"
            )
            if x.strip()
        ]

        if hierarchy_parts:

            # Prefer leaf
            add(
                hierarchy_parts[-1]
            )

            # Also make parent nodes available
            for part in reversed(
                hierarchy_parts[:-1]
            ):
                add(part)


    return candidates


def canonicalize_expert_label(
    category,
    hierarchy,
    model_vocab,
):
    """
    Select the expert representation that best matches the model
    label vocabulary.

    Returns:
        expert_label_for_evaluation
        matched_to_model_vocabulary
        match_source
    """

    candidates = expert_label_candidates(
        category,
        hierarchy,
    )


    # ---------------------------------------------------------------
    # Prefer exact/vocabulary-compatible representations
    # ---------------------------------------------------------------

    for i, candidate in enumerate(
        candidates
    ):

        key = loose_label_key(
            candidate
        )

        if (
            key is not None
            and key in model_vocab
        ):

            return (
                model_vocab[key],
                True,
                (
                    "candidate_"
                    + str(i + 1)
                ),
            )


    # ---------------------------------------------------------------
    # Otherwise use the first available representation
    # ---------------------------------------------------------------

    if candidates:

        return (
            candidates[0],
            False,
            "unmatched_candidate",
        )


    return (
        None,
        False,
        "missing",
    )


# ======================================================================
# 11. MAP EXPERT SUPERCLASS
# ======================================================================

def map_expert_superclass(
    category,
    hierarchy,
    superclass_map,
):
    """
    Map expert annotation to ecotaxa_20.

    The mapping is done by testing:
        category
        category main label
        hierarchy leaf
        hierarchy parents

    against label_to_int.csv.
    """

    candidates = expert_label_candidates(
        category,
        hierarchy,
    )


    for i, candidate in enumerate(
        candidates
    ):

        key = loose_label_key(
            candidate
        )

        if (
            key is not None
            and key in superclass_map
        ):

            return (
                superclass_map[key],
                f"candidate_{i+1}",
            )


    return (
        None,
        "unmapped",
    )


# ======================================================================
# 12. READ AND PREPARE DATA
# ======================================================================

def load_data():

    print("=" * 78)
    print("LOADING EXPERT VALIDATION DATA")
    print("=" * 78)


    # ============================================================
    # EXPERT EXPORT
    # ============================================================

    expert = pd.read_csv(
        EXPERT_EXPORT,
        sep="\t",
        low_memory=False,
        dtype={
            OBJECT_ID: str
        },
    )

    print(
        f"\nExpert export:"
        f"\n  rows    : {len(expert):,}"
        f"\n  columns : {len(expert.columns):,}"
    )


    required_expert = {
        OBJECT_ID,
        EXPERT_LABEL_COL,
        EXPERT_HIERARCHY_COL,
        EXPERT_STATUS_COL,
        BIN_VARIABLE,
    }

    missing = (
        required_expert
        - set(expert.columns)
    )

    if missing:

        raise KeyError(
            "\nMissing required expert columns:\n"
            f"{sorted(missing)}"
        )


    status_counts = (
        expert[
            EXPERT_STATUS_COL
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nExpert annotation status:"
    )

    print(
        status_counts.to_string()
    )


    # ============================================================
    # FUSED PREDICTIONS
    # ============================================================

    fused = pd.read_csv(
        FUSED_PATH,
        low_memory=False,
        dtype={
            OBJECT_ID: str
        },
    )

    print(
        f"\nFused predictions:"
        f"\n  rows    : {len(fused):,}"
        f"\n  columns : {len(fused.columns):,}"
    )


    required_fused = {
        OBJECT_ID,

        "m1_label",
        "m2_label",
        "m3_label",

        "final_label",
        "path",
        "confidence",

        "score_p_r",
        "score_p_f",
        "score_e_f",
    }


    missing = (
        required_fused
        - set(fused.columns)
    )

    if missing:

        raise KeyError(
            "\nMissing required fused columns:\n"
            f"{sorted(missing)}"
        )


    # ============================================================
    # SAMPLING RECORDS
    # ============================================================

    sampling = None

    if SAMPLING_RECORDS_PATH.exists():

        sampling = pd.read_csv(
            SAMPLING_RECORDS_PATH,
            low_memory=False,
            dtype={
                OBJECT_ID: str
            },
        )

        print(
            f"\nSampling records:"
            f"\n  rows: {len(sampling):,}"
            f"\n  unique objects: "
            f"{sampling[OBJECT_ID].nunique():,}"
        )


    # ============================================================
    # SAMPLING REPORT
    # ============================================================

    sampling_report = None

    if SAMPLING_REPORT_PATH.exists():

        sampling_report = pd.read_csv(
            SAMPLING_REPORT_PATH,
            low_memory=False,
        )


    # ============================================================
    # LOAD ORIGINAL METADATA FOR POPULATION SAMPLING PROBABILITY
    # ============================================================

    metadata = pd.read_csv(
        METADATA_PATH,
        usecols=[
            OBJECT_ID,
            BIN_VARIABLE,
        ],
        low_memory=False,
        dtype={
            OBJECT_ID: str
        },
    )

    print(
        f"\nPopulation metadata:"
        f"\n  rows: {len(metadata):,}"
    )


    # ============================================================
    # Check metadata uniqueness
    # ============================================================

    metadata_duplicates = (
        metadata[
            OBJECT_ID
        ]
        .duplicated()
        .sum()
    )

    if metadata_duplicates:

        raise ValueError(
            "Population metadata contains duplicate "
            f"{OBJECT_ID} values: "
            f"{metadata_duplicates:,}"
        )


    # ============================================================
    # Build minimal population table
    #
    # This is ONLY for reconstructing the sampling design.
    # ============================================================

    population_labels = fused[
        [
            OBJECT_ID,
            "m1_label",
            "m2_label",
            "m3_label",
        ]
    ].copy()


    population_df = metadata.merge(
        population_labels,
        on=OBJECT_ID,
        how="inner",
        validate="one_to_one",
    )


    print(
        f"\nPopulation table for sampling weights:"
        f"\n  rows: {len(population_df):,}"
    )


    if len(population_df) != len(fused):

        raise ValueError(
            "\nPopulation metadata + fused prediction merge "
            "did not recover every fused object.\n"
            f"Expected {len(fused):,}, "
            f"got {len(population_df):,}."
        )


    return (
        expert,
        fused,
        sampling,
        sampling_report,
        population_df,
    )


# ======================================================================
# 13. BUILD VALIDATION DATASET
# ======================================================================

def build_validation_dataset(
    expert,
    fused,
    sampling,
    superclass_map,
):
    """
    One row per unique expert-labelled object.
    """

    print()
    print("=" * 78)
    print("BUILDING ONE-ROW-PER-OBJECT VALIDATION DATASET")
    print("=" * 78)


    model_vocab = (
        build_model_vocabulary(
            fused
        )
    )


    print(
        f"\nModel label vocabulary: "
        f"{len(model_vocab):,} unique normalized labels"
    )


    # ------------------------------------------------------------------
    # Merge expert + model/fusion results
    # ------------------------------------------------------------------

    fused_keep = [
        OBJECT_ID,

        "m1_label",
        "m2_label",
        "m3_label",

        "final_label",

        "score_p_r",
        "score_p_f",
        "score_e_f",

        "final_score",

        "superclass",
        "sc_ecotaxa_20",
        "sc_lineage",

        "confidence",
        "path",

        "n_models_active",
    ]


    fused_keep = [
        c
        for c in fused_keep
        if c in fused.columns
    ]


    validation = expert.merge(
        fused[
            fused_keep
        ],
        on=OBJECT_ID,
        how="inner",
        validate="one_to_one",
    )


    print(
        f"\nExpert rows: "
        f"{len(expert):,}"
    )

    print(
        f"Successfully matched to fusion: "
        f"{len(validation):,}"
    )


    if len(validation) != len(expert):

        expert_ids = set(
            expert[OBJECT_ID]
        )

        matched_ids = set(
            validation[OBJECT_ID]
        )

        missing_ids = sorted(
            expert_ids
            - matched_ids
        )

        pd.DataFrame({
            OBJECT_ID:
                missing_ids
        }).to_csv(
            OUTPUT_DIR
            / "expert_objects_missing_from_fused.csv",
            index=False,
        )

        print(
            "\nWARNING:"
            f" {len(missing_ids)} expert objects "
            "were not found in fused_predictions.csv."
        )


    # ------------------------------------------------------------------
    # Expert label mapping
    # ------------------------------------------------------------------

    expert_map_results = []

    for _, row in validation.iterrows():

        (
            canonical,
            shared,
            source,
        ) = canonicalize_expert_label(
            row[EXPERT_LABEL_COL],
            row[EXPERT_HIERARCHY_COL],
            model_vocab,
        )


        (
            expert_sc,
            sc_source,
        ) = map_expert_superclass(
            row[EXPERT_LABEL_COL],
            row[EXPERT_HIERARCHY_COL],
            superclass_map,
        )


        expert_map_results.append({

            OBJECT_ID:
                row[OBJECT_ID],

            "expert_label_raw":
                row[EXPERT_LABEL_COL],

            "expert_hierarchy":
                row[EXPERT_HIERARCHY_COL],

            "expert_label":
                canonical,

            "expert_label_shared_with_models":
                shared,

            "expert_label_mapping_source":
                source,

            "expert_superclass":
                expert_sc,

            "expert_superclass_mapping_source":
                sc_source,
        })


    expert_map_df = pd.DataFrame(
        expert_map_results
    )


    validation = validation.merge(
        expert_map_df,
        on=OBJECT_ID,
        how="left",
        validate="one_to_one",
    )


    # ------------------------------------------------------------------
    # Model-space normalized labels
    # ------------------------------------------------------------------

    for model, column in MODEL_LABELS.items():

        validation[
            f"{model}_fine"
        ] = validation[
            column
        ].map(
            normalized_text
        )


        validation[
            f"{model}_fine_eval"
        ] = validation[
            column
        ].map(
            apply_vote_equivalence
        )


        # model superclass from label map
        validation[
            f"{model}_superclass"
        ] = (
            validation[
                column
            ]
            .map(
                loose_label_key
            )
            .map(
                superclass_map
            )
        )


    # Fusion
    validation[
        "Fusion_fine"
    ] = validation[
        FUSION_LABEL
    ].map(
        normalized_text
    )


    validation[
        "Fusion_fine_eval"
    ] = validation[
        FUSION_LABEL
    ].map(
        apply_vote_equivalence
    )


    validation[
        "Fusion_superclass"
    ] = (
        validation[
            FUSION_LABEL
        ]
        .map(
            loose_label_key
        )
        .map(
            superclass_map
        )
    )


    # Expert evaluation representation
    validation[
        "expert_fine_eval"
    ] = validation[
        "expert_label"
    ].map(
        apply_vote_equivalence
    )


    # ------------------------------------------------------------------
    # Correctness indicators
    # ------------------------------------------------------------------

    for model in [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]:

        validation[
            f"{model}_fine_correct"
        ] = (
            validation[
                f"{model}_fine_eval"
            ]
            ==
            validation[
                "expert_fine_eval"
            ]
        )


        validation[
            f"{model}_superclass_correct"
        ] = (
            validation[
                f"{model}_superclass"
            ]
            ==
            validation[
                "expert_superclass"
            ]
        )


    # ------------------------------------------------------------------
    # Sampling provenance
    # ------------------------------------------------------------------

    if sampling is not None:

        provenance = (
            sampling
            .groupby(
                OBJECT_ID
            )
            .agg(
                selected_by=(
                    "sampled_for_model",
                    lambda x:
                        "|".join(
                            sorted(
                                set(
                                    x.astype(str)
                                )
                            )
                        ),
                ),

                n_sampling_records=(
                    "sampled_for_model",
                    "size",
                ),
            )
            .reset_index()
        )


        validation = validation.merge(
            provenance,
            on=OBJECT_ID,
            how="left",
            validate="one_to_one",
        )


        for model in [
            "M1",
            "M2",
            "M3",
        ]:

            validation[
                f"selected_for_{model}"
            ] = (
                validation[
                    "selected_by"
                ]
                .fillna("")
                .str.contains(
                    model
                )
            )


    # ------------------------------------------------------------------
    # Save mapping diagnostics
    # ------------------------------------------------------------------

    mapping_report = (
        validation[
            [
                EXPERT_LABEL_COL,
                EXPERT_HIERARCHY_COL,
                "expert_label",
                "expert_label_shared_with_models",
                "expert_label_mapping_source",
                "expert_superclass",
                "expert_superclass_mapping_source",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "expert_label_shared_with_models",
                EXPERT_LABEL_COL,
            ]
        )
    )


    mapping_report.to_csv(
        OUTPUT_DIR
        / "expert_label_mapping_report.csv",
        index=False,
    )


    return (
        validation,
        model_vocab,
    )


# ======================================================================
# 14. BASIC PERFORMANCE METRICS
# ======================================================================

def classification_metrics(
    y_true,
    y_pred,
    weights=None,
):
    """
    Calculate:

        accuracy
        macro precision
        macro recall
        macro F1
        balanced accuracy

    For unweighted metrics:
        each validation object has equal weight.

    For weighted metrics:
        inverse-sampling weights can be supplied.

    Balanced accuracy is calculated as the mean recall across
    classes present in y_true. This avoids sklearn warnings when
    y_pred contains a class that is absent from y_true.
    """

    # ------------------------------------------------------------
    # Valid paired observations
    # ------------------------------------------------------------

    valid = (
        y_true.notna()
        &
        y_pred.notna()
    )


    y_true = y_true.loc[
        valid
    ].copy()

    y_pred = y_pred.loc[
        valid
    ].copy()


    if weights is not None:

        weights = weights.loc[
            valid
        ].copy()


    n = len(
        y_true
    )


    if n == 0:

        return {
            "n_valid": 0,
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
        }


    # ------------------------------------------------------------
    # Label universe
    # ------------------------------------------------------------

    labels = sorted(
        set(y_true)
        |
        set(y_pred),
        key=str,
    )


    # ------------------------------------------------------------
    # UNWEIGHTED
    # ------------------------------------------------------------

    if weights is None:

        yt = y_true.to_numpy()
        yp = y_pred.to_numpy()


        # Accuracy
        accuracy = float(
            np.mean(
                yt == yp
            )
        )


        # Macro precision / recall / F1
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                yt,
                yp,
                labels=labels,
                zero_division=0,
            )
        )


        macro_precision = float(
            np.mean(precision)
        )

        macro_recall = float(
            np.mean(recall)
        )

        macro_f1 = float(
            np.mean(f1)
        )


        # --------------------------------------------------------
        # Balanced accuracy
        #
        # Only classes actually present in expert truth are used.
        # --------------------------------------------------------

        true_labels = sorted(
            set(yt),
            key=str,
        )


        class_recalls = []


        for label in true_labels:

            mask = (
                yt == label
            )

            if mask.sum() == 0:
                continue

            recall_value = float(
                np.mean(
                    yp[mask] == label
                )
            )

            class_recalls.append(
                recall_value
            )


        balanced_accuracy = float(
            np.mean(
                class_recalls
            )
        )


    # ------------------------------------------------------------
    # WEIGHTED
    # ------------------------------------------------------------

    else:

        yt = y_true.to_numpy()
        yp = y_pred.to_numpy()

        w = weights.to_numpy(
            dtype=float
        )


        # Remove invalid weights
        weight_valid = (
            np.isfinite(w)
            & (w > 0)
        )


        yt = yt[
            weight_valid
        ]

        yp = yp[
            weight_valid
        ]

        w = w[
            weight_valid
        ]


        if len(w) == 0:

            return {
                "n_valid": 0,
                "accuracy": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "macro_f1": np.nan,
                "balanced_accuracy": np.nan,
            }


        # --------------------------------------------------------
        # Weighted accuracy
        # --------------------------------------------------------

        accuracy = float(
            np.sum(
                w * (yt == yp)
            )
            /
            np.sum(w)
        )


        # --------------------------------------------------------
        # Weighted per-class metrics
        # --------------------------------------------------------

        precisions = []
        recalls = []
        f1s = []


        for label in labels:

            true_mask = (
                yt == label
            )

            pred_mask = (
                yp == label
            )


            tp = np.sum(
                w[
                    true_mask
                    & pred_mask
                ]
            )

            fp = np.sum(
                w[
                    ~true_mask
                    & pred_mask
                ]
            )

            fn = np.sum(
                w[
                    true_mask
                    & ~pred_mask
                ]
            )


            if tp + fp > 0:

                precision_value = (
                    tp
                    / (tp + fp)
                )

            else:

                precision_value = 0.0


            if tp + fn > 0:

                recall_value = (
                    tp
                    / (tp + fn)
                )

            else:

                recall_value = 0.0


            if (
                precision_value
                + recall_value
                > 0
            ):

                f1_value = (
                    2
                    * precision_value
                    * recall_value
                    /
                    (
                        precision_value
                        + recall_value
                    )
                )

            else:

                f1_value = 0.0


            precisions.append(
                precision_value
            )

            recalls.append(
                recall_value
            )

            f1s.append(
                f1_value
            )


        macro_precision = float(
            np.mean(
                precisions
            )
        )

        macro_recall = float(
            np.mean(
                recalls
            )
        )

        macro_f1 = float(
            np.mean(
                f1s
            )
        )


        # --------------------------------------------------------
        # Weighted balanced accuracy
        #
        # Mean class recall, using weighted observations.
        # --------------------------------------------------------

        true_labels = sorted(
            set(yt),
            key=str,
        )


        class_recalls = []


        for label in true_labels:

            mask = (
                yt == label
            )


            denominator = np.sum(
                w[mask]
            )


            if denominator == 0:
                continue


            numerator = np.sum(
                w[
                    mask
                    & (yp == label)
                ]
            )


            class_recalls.append(
                numerator
                / denominator
            )


        balanced_accuracy = float(
            np.mean(
                class_recalls
            )
        )


    return {
        "n_valid": int(n),

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "balanced_accuracy":
            balanced_accuracy,
    }


# ======================================================================
# 15. OVERALL METRICS
# ======================================================================

def calculate_overall_metrics(
    validation,
):
    """
    Calculate overall expert-validation performance.

    Reports both:

        1. Fine-label performance
        2. Superclass performance

    for:

        M1
        M2
        M3
        Fusion
    """

    rows = []


    total_n = len(
        validation
    )


    # Expert labels that can actually be represented in the
    # shared model vocabulary.
    shared_mask = (
        validation[
            "expert_label_shared_with_models"
        ]
        .fillna(False)
        .astype(bool)
    )


    for method in [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]:

        # --------------------------------------------------------
        # Fine labels
        # --------------------------------------------------------

        fine_pred = validation[
            f"{method}_fine_eval"
        ]

        fine_true = validation[
            "expert_fine_eval"
        ]


        fine_valid = (
            fine_true.notna()
            &
            fine_pred.notna()
        )


        fine_correct = (
            fine_pred
            ==
            fine_true
        )


        # Invalid/missing predictions count as incorrect in
        # "accuracy_all".
        fine_correct_all = (
            fine_correct
            .where(
                fine_valid,
                False,
            )
        )


        fine_accuracy_all = float(
            fine_correct_all.mean()
        )


        # Metrics using only valid paired labels.
        fine = classification_metrics(
            fine_true,
            fine_pred,
        )


        # Shared model/expert vocabulary only.
        fine_shared = classification_metrics(
            fine_true.loc[
                shared_mask
            ],
            fine_pred.loc[
                shared_mask
            ],
        )


        # --------------------------------------------------------
        # Superclass
        # --------------------------------------------------------

        superclass_true = validation[
            "expert_superclass"
        ]

        superclass_pred = validation[
            f"{method}_superclass"
        ]


        superclass_valid = (
            superclass_true.notna()
            &
            superclass_pred.notna()
        )


        superclass_correct = (
            superclass_pred
            ==
            superclass_true
        )


        superclass_correct_all = (
            superclass_correct
            .where(
                superclass_valid,
                False,
            )
        )


        superclass_accuracy_all = float(
            superclass_correct_all.mean()
        )


        superclass = classification_metrics(
            superclass_true,
            superclass_pred,
        )


        # --------------------------------------------------------
        # Wilson interval for fine-label accuracy
        # --------------------------------------------------------

        n_fine_valid = int(
            fine_valid.sum()
        )


        n_fine_correct = int(
            fine_correct_all.loc[
                fine_valid
            ].sum()
        )


        fine_ci_low, fine_ci_high = (
            wilson_interval(
                n_fine_correct,
                n_fine_valid,
            )
        )


        # --------------------------------------------------------
        # Wilson interval for superclass accuracy
        # --------------------------------------------------------

        n_sc_valid = int(
            superclass_valid.sum()
        )


        n_sc_correct = int(
            superclass_correct_all.loc[
                superclass_valid
            ].sum()
        )


        sc_ci_low, sc_ci_high = (
            wilson_interval(
                n_sc_correct,
                n_sc_valid,
            )
        )


        # --------------------------------------------------------
        # Output row
        # --------------------------------------------------------

        row = {

            "method":
                method,

            "n_validation":
                total_n,


            # Fine label
            "fine_prediction_coverage":
                float(
                    fine_valid.mean()
                ),

            "fine_accuracy_all":
                fine_accuracy_all,

            "fine_accuracy_valid":
                fine[
                    "accuracy"
                ],

            "fine_macro_precision":
                fine[
                    "macro_precision"
                ],

            "fine_macro_recall":
                fine[
                    "macro_recall"
                ],

            "fine_macro_f1":
                fine[
                    "macro_f1"
                ],

            "fine_balanced_accuracy":
                fine[
                    "balanced_accuracy"
                ],


            # Shared vocabulary
            "fine_macro_f1_shared_label_space":
                fine_shared[
                    "macro_f1"
                ],

            "fine_balanced_accuracy_shared_label_space":
                fine_shared[
                    "balanced_accuracy"
                ],


            # Superclass
            "superclass_prediction_coverage":
                float(
                    superclass_valid.mean()
                ),

            "superclass_accuracy_all":
                superclass_accuracy_all,

            "superclass_accuracy_valid":
                superclass[
                    "accuracy"
                ],

            "superclass_macro_precision":
                superclass[
                    "macro_precision"
                ],

            "superclass_macro_recall":
                superclass[
                    "macro_recall"
                ],

            "superclass_macro_f1":
                superclass[
                    "macro_f1"
                ],

            "superclass_balanced_accuracy":
                superclass[
                    "balanced_accuracy"
                ],


            # Expert/model label compatibility
            "n_expert_labels_outside_model_vocabulary":
                int(
                    (
                        ~shared_mask
                    ).sum()
                ),


            "n_expert_labels_shared_with_models":
                int(
                    shared_mask.sum()
                ),


            # Confidence intervals
            "fine_accuracy_ci_low":
                fine_ci_low,

            "fine_accuracy_ci_high":
                fine_ci_high,

            "superclass_accuracy_ci_low":
                sc_ci_low,

            "superclass_accuracy_ci_high":
                sc_ci_high,
        }


        rows.append(
            row
        )


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "overall_metrics_unweighted.csv",
        index=False,
    )


    return result


# ======================================================================
# 16. PATHWAY METRICS
# ======================================================================

def calculate_pathway_metrics(
    validation,
):
    """
    Evaluate model/fusion performance separately for each fusion path.
    """

    paths = [
        "A_all_agree",
        "B_majority",
        "C_superclass",
        "D_score",
        "D_score_close",
        "abstention_m3_only",
    ]


    rows = []


    for path in paths:

        subset = validation[
            validation[
                FUSION_PATH
            ]
            == path
        ]


        if subset.empty:
            continue


        for method in [
            "M1",
            "M2",
            "M3",
            "Fusion",
        ]:

            pred = subset[
                f"{method}_fine_eval"
            ]

            true = subset[
                "expert_fine_eval"
            ]


            valid = (
                true.notna()
                &
                pred.notna()
            )


            correct = (
                subset.loc[
                    valid,
                    f"{method}_fine_correct"
                ]
                .sum()
            )


            n = int(
                valid.sum()
            )


            lo, hi = wilson_interval(
                int(correct),
                n,
            )


            rows.append({

                "path":
                    path,

                "method":
                    method,

                "n_total":
                    len(subset),

                "n_valid":
                    n,

                "coverage":
                    valid.mean(),

                "accuracy":
                    (
                        correct / n
                        if n
                        else np.nan
                    ),

                "ci_low":
                    lo,

                "ci_high":
                    hi,
            })


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "pathway_metrics.csv",
        index=False,
    )


    return result


# ======================================================================
# 17. SUPERCLASS METRICS
# ======================================================================

def calculate_superclass_metrics(
    validation,
):
    """
    Calculate exact fine-label agreement and superclass agreement
    as a function of the expert reference superclass.
    """

    rows = []


    superclasses = sorted(
        validation[
            "expert_superclass"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    for superclass in superclasses:

        subset = validation[
            validation[
                "expert_superclass"
            ].astype(str)
            == superclass
        ]


        for method in [
            "M1",
            "M2",
            "M3",
            "Fusion",
        ]:

            fine_valid = (
                subset[
                    f"{method}_fine_eval"
                ].notna()
                &
                subset[
                    "expert_fine_eval"
                ].notna()
            )


            sc_valid = (
                subset[
                    f"{method}_superclass"
                ].notna()
                &
                subset[
                    "expert_superclass"
                ].notna()
            )


            fine_n = int(
                fine_valid.sum()
            )

            fine_correct = int(
                subset.loc[
                    fine_valid,
                    f"{method}_fine_correct"
                ].sum()
            )


            sc_n = int(
                sc_valid.sum()
            )

            sc_correct = int(
                subset.loc[
                    sc_valid,
                    f"{method}_superclass_correct"
                ].sum()
            )


            lo_f, hi_f = (
                wilson_interval(
                    fine_correct,
                    fine_n,
                )
            )


            lo_s, hi_s = (
                wilson_interval(
                    sc_correct,
                    sc_n,
                )
            )


            rows.append({

                "expert_superclass":
                    superclass,

                "method":
                    method,

                "n":
                    len(subset),

                "fine_accuracy":
                    (
                        fine_correct
                        / fine_n
                        if fine_n
                        else np.nan
                    ),

                "fine_ci_low":
                    lo_f,

                "fine_ci_high":
                    hi_f,

                "superclass_accuracy":
                    (
                        sc_correct
                        / sc_n
                        if sc_n
                        else np.nan
                    ),

                "superclass_ci_low":
                    lo_s,

                "superclass_ci_high":
                    hi_s,
            })


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "superclass_metrics.csv",
        index=False,
    )


    return result


# ======================================================================
# 18. CONFIDENCE VALIDATION
# ======================================================================

def calculate_confidence_metrics(
    validation,
):
    """
    Test whether the fusion confidence levels are empirically
    associated with correctness.
    """

    order = [
        "HIGH",
        "MEDIUM",
        "LOW",
        "UNCERTAIN",
    ]


    rows = []


    for confidence in order:

        subset = validation[
            validation[
                FUSION_CONFIDENCE
            ]
            == confidence
        ]


        valid = (
            subset[
                "Fusion_fine_eval"
            ].notna()
            &
            subset[
                "expert_fine_eval"
            ].notna()
        )


        n = int(
            valid.sum()
        )


        correct = int(
            subset.loc[
                valid,
                "Fusion_fine_correct"
            ].sum()
        )


        lo, hi = wilson_interval(
            correct,
            n,
        )


        rows.append({

            "confidence":
                confidence,

            "n":
                n,

            "correct":
                correct,

            "accuracy":
                (
                    correct / n
                    if n
                    else np.nan
                ),

            "ci_low":
                lo,

            "ci_high":
                hi,
        })


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "confidence_metrics.csv",
        index=False,
    )


    return result


# ======================================================================
# 19. FUSION BENEFIT / REGRESSION
# ======================================================================

def calculate_fusion_benefit(
    validation,
):
    """
    Determine when fusion helps or hurts relative to each model.
    """

    rows = []


    for model in [
        "M1",
        "M2",
        "M3",
    ]:

        fusion_correct = (
            validation[
                "Fusion_fine_correct"
            ]
        )

        model_correct = (
            validation[
                f"{model}_fine_correct"
            ]
        )


        pair_valid = (
            validation[
                f"{model}_fine_eval"
            ].notna()
            &
            validation[
                "Fusion_fine_eval"
            ].notna()
        )


        both_correct = (
            pair_valid
            &
            fusion_correct
            &
            model_correct
        ).sum()


        both_wrong = (
            pair_valid
            &
            ~fusion_correct
            &
            ~model_correct
        ).sum()


        fusion_rescue = (
            pair_valid
            &
            fusion_correct
            &
            ~model_correct
        ).sum()


        fusion_regression = (
            pair_valid
            &
            ~fusion_correct
            &
            model_correct
        ).sum()


        discordant = (
            fusion_rescue
            + fusion_regression
        )


        p_rescue = (
            fusion_rescue
            / discordant
            if discordant
            else np.nan
        )


        rows.append({

            "model":
                model,

            "paired_n":
                int(pair_valid.sum()),

            "both_correct":
                int(both_correct),

            "both_wrong":
                int(both_wrong),

            "fusion_rescue":
                int(fusion_rescue),

            "fusion_regression":
                int(fusion_regression),

            "discordant":
                int(discordant),

            "fraction_of_discordant_favouring_fusion":
                p_rescue,

            "net_fusion_gain":
                int(
                    fusion_rescue
                    - fusion_regression
                ),
        })


    # ---------------------------------------------------------------
    # Fusion uniquely correct/wrong relative to all three
    # ---------------------------------------------------------------

    all_valid = (
        validation[
            "M1_fine_eval"
        ].notna()
        &
        validation[
            "M2_fine_eval"
        ].notna()
        &
        validation[
            "M3_fine_eval"
        ].notna()
        &
        validation[
            "Fusion_fine_eval"
        ].notna()
    )


    all_models_wrong_fusion_correct = (
        all_valid
        &
        ~validation[
            "M1_fine_correct"
        ]
        &
        ~validation[
            "M2_fine_correct"
        ]
        &
        ~validation[
            "M3_fine_correct"
        ]
        &
        validation[
            "Fusion_fine_correct"
        ]
    ).sum()


    all_models_correct_fusion_wrong = (
        all_valid
        &
        validation[
            "M1_fine_correct"
        ]
        &
        validation[
            "M2_fine_correct"
        ]
        &
        validation[
            "M3_fine_correct"
        ]
        &
        ~validation[
            "Fusion_fine_correct"
        ]
    ).sum()


    unique_summary = pd.DataFrame({

        "quantity": [
            "fusion_correct_all_three_models_wrong",
            "fusion_wrong_all_three_models_correct",
        ],

        "n": [
            int(
                all_models_wrong_fusion_correct
            ),
            int(
                all_models_correct_fusion_wrong
            ),
        ],
    })


    unique_summary.to_csv(
        OUTPUT_DIR
        / "fusion_unique_outcomes.csv",
        index=False,
    )


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "fusion_benefit.csv",
        index=False,
    )


    return (
        result,
        unique_summary,
    )


# ======================================================================
# 20. MCNEMAR TESTS
# ======================================================================

def exact_mcnemar(
    fusion_correct,
    model_correct,
):
    """
    Exact two-sided McNemar test.

    b = model correct, fusion wrong
    c = model wrong, fusion correct
    """

    valid = (
        fusion_correct.notna()
        &
        model_correct.notna()
    )


    f = (
        fusion_correct.loc[
            valid
        ].astype(bool)
    )

    m = (
        model_correct.loc[
            valid
        ].astype(bool)
    )


    b = int(
        (
            m
            & ~f
        ).sum()
    )


    c = int(
        (
            ~m
            & f
        ).sum()
    )


    n_discordant = (
        b + c
    )


    if n_discordant == 0:

        p_value = 1.0

    else:

        p_value = (
            binomtest(
                min(b, c),
                n=n_discordant,
                p=0.5,
                alternative="two-sided",
            )
            .pvalue
        )


    return (
        int(valid.sum()),
        b,
        c,
        p_value,
    )


def calculate_mcnemar(
    validation,
):
    rows = []


    for model in [
        "M1",
        "M2",
        "M3",
    ]:

        (
            n,
            model_correct_fusion_wrong,
            model_wrong_fusion_correct,
            p,
        ) = exact_mcnemar(
            validation[
                "Fusion_fine_correct"
            ],
            validation[
                f"{model}_fine_correct"
            ],
        )


        rows.append({

            "comparison":
                f"Fusion_vs_{model}",

            "paired_n":
                n,

            "model_correct_fusion_wrong":
                model_correct_fusion_wrong,

            "model_wrong_fusion_correct":
                model_wrong_fusion_correct,

            "net_fusion_gain":
                (
                    model_wrong_fusion_correct
                    -
                    model_correct_fusion_wrong
                ),

            "mcnemar_exact_p":
                p,
        })


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "mcnemar_tests.csv",
        index=False,
    )


    return result


# ======================================================================
# 21. BIN ASSIGNMENT
# ======================================================================

def assign_bins_to_group(
    values,
):
    """
    Reproduce the binning logic used by the sampling script.

    Equal-width bins in log10(object_major).
    """

    x = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid = np.isfinite(
        x
    )


    if USE_LOG_BINS:

        valid &= (
            x > 0
        )


    transformed = np.full(
        len(x),
        np.nan,
    )


    if USE_LOG_BINS:

        transformed[
            valid
        ] = np.log10(
            np.clip(
                x[valid],
                EPS,
                None,
            )
        )

    else:

        transformed[
            valid
        ] = x[
            valid
        ]


    usable = transformed[
        np.isfinite(
            transformed
        )
    ]


    if len(usable) == 0:

        return (
            np.full(
                len(x),
                -1,
                dtype=int,
            ),
            None,
        )


    minimum = float(
        usable.min()
    )

    maximum = float(
        usable.max()
    )


    if minimum == maximum:

        maximum = (
            minimum
            + 1e-9
        )


    edges = np.linspace(
        minimum,
        maximum,
        N_BINS + 1,
    )


    bin_ids = np.full(
        len(x),
        -1,
        dtype=int,
    )


    idx = np.searchsorted(
        edges,
        transformed[valid],
        side="right",
    ) - 1


    idx = np.clip(
        idx,
        0,
        N_BINS - 1,
    )


    bin_ids[
        valid
    ] = idx


    return (
        bin_ids,
        edges,
    )


# ======================================================================
# 22. POPULATION INCLUSION PROBABILITIES
# ======================================================================

def compute_model_inclusion_probability(
    population_df,
    model,
    label_column,
    superclass_map,
):
    """
    Reconstruct the inclusion probability for each object under
    one model's stratified sampling procedure.

    Sampling design:

        model predicted superclass
            x 10 bins of object_major
            x 10 samples per bin

    For a stratum containing N objects:

        n_sample = min(10, N)

        inclusion probability = n_sample / N
    """

    columns = [
        OBJECT_ID,
        label_column,
        BIN_VARIABLE,
    ]


    work = population_df[
        columns
    ].copy()


    work[
        "sample_superclass"
    ] = (
        work[
            label_column
        ]
        .map(
            loose_label_key
        )
        .map(
            superclass_map
        )
    )


    work[
        BIN_VARIABLE
    ] = pd.to_numeric(
        work[
            BIN_VARIABLE
        ],
        errors="coerce",
    )


    valid = (
        work[
            "sample_superclass"
        ].notna()
        &
        work[
            BIN_VARIABLE
        ].notna()
    )


    if USE_LOG_BINS:

        valid &= (
            work[
                BIN_VARIABLE
            ] > 0
        )


    work[
        "bin_id"
    ] = -1


    work[
        f"{model}_p_inclusion"
    ] = 0.0


    # ============================================================
    # IMPORTANT:
    # Reproduce the EXACT binning procedure used during sampling.
    # ============================================================

    for superclass in (
        work.loc[
            valid,
            "sample_superclass"
        ]
        .dropna()
        .unique()
    ):

        mask = (
            valid
            &
            (
                work[
                    "sample_superclass"
                ]
                == superclass
            )
        )


        group = work.loc[
            mask
        ].copy()


        (
            bins,
            edges,
        ) = assign_bins_to_group(
            group[
                BIN_VARIABLE
            ]
        )


        if edges is None:
            continue


        group[
            "bin_id"
        ] = bins


        # --------------------------------------------------------
        # Stratum size N
        # --------------------------------------------------------

        counts = (
            group[
                "bin_id"
            ]
            .value_counts()
            .to_dict()
        )


        for bin_id, N in counts.items():

            if bin_id < 0:
                continue


            # Same sampling rule as the extraction script
            n_sample = min(
                K_PER_BIN,
                N,
            )


            inclusion_probability = (
                n_sample / N
            )


            target_index = (
                group.index[
                    group[
                        "bin_id"
                    ]
                    == bin_id
                ]
            )


            work.loc[
                target_index,
                "bin_id"
            ] = int(
                bin_id
            )


            work.loc[
                target_index,
                f"{model}_p_inclusion"
            ] = (
                inclusion_probability
            )


    return work[
        [
            OBJECT_ID,
            "sample_superclass",
            "bin_id",
            f"{model}_p_inclusion",
        ]
    ].rename(
        columns={
            "sample_superclass":
                f"{model}_sampling_superclass",

            "bin_id":
                f"{model}_sampling_bin",
        }
    )


def calculate_population_weights(
    validation,
    population_df,
    superclass_map,
    sampling,
):
    """
    Calculate union inclusion probability across M1/M2/M3.

    Assuming independent model-specific sampling:

        pi_union =
            1 - (1-p1)(1-p2)(1-p3)

    Then:

        weight = 1 / pi_union
    """

    print()
    print("=" * 78)
    print("CALCULATING SAMPLING INCLUSION PROBABILITIES")
    print("=" * 78)


    result = validation[
        [
            OBJECT_ID
        ]
    ].copy()


    for model, label_col in MODEL_LABELS.items():

        model_probs = (
            compute_model_inclusion_probability(
                population_df,
                model,
                label_col,
                superclass_map,
            )
        )


        result = result.merge(
            model_probs,
            on=OBJECT_ID,
            how="left",
            validate="one_to_one",
        )


        p_sum = (
            model_probs[
                f"{model}_p_inclusion"
            ].sum()
        )


        print(
            f"\n{model}:"
            f"\n  Sum of inclusion probabilities: "
            f"{p_sum:,.2f}"
        )


        if sampling is not None:

            actual_n = int(
                (
                    sampling[
                        "sampled_for_model"
                    ]
                    == model
                ).sum()
            )

            print(
                f"  Actual sampled records: "
                f"{actual_n:,}"
            )


    p1 = result[
        "M1_p_inclusion"
    ].fillna(0)

    p2 = result[
        "M2_p_inclusion"
    ].fillna(0)

    p3 = result[
        "M3_p_inclusion"
    ].fillna(0)


    result[
        "union_inclusion_probability"
    ] = (
        1
        - (1-p1)
        * (1-p2)
        * (1-p3)
    )


    result[
        "population_weight"
    ] = np.where(
        result[
            "union_inclusion_probability"
        ] > 0,

        1
        / result[
            "union_inclusion_probability"
        ],

        np.nan,
    )


    # Effective sample size
    w = result[
        "population_weight"
    ].dropna().to_numpy(
        dtype=float
    )


    effective_n = (
        w.sum()**2
        / np.sum(
            w**2
        )
        if len(w)
        else np.nan
    )


    print(
        f"\nValidation objects with "
        f"non-zero union inclusion probability: "
        f"{result['union_inclusion_probability'].gt(0).sum():,}"
    )


    print(
        f"Approximate weighted effective sample size: "
        f"{effective_n:.1f}"
    )


    result.to_csv(
        OUTPUT_DIR
        / "population_inclusion_probabilities.csv",
        index=False,
    )


    validation = validation.merge(
        result,
        on=OBJECT_ID,
        how="left",
        validate="one_to_one",
    )


    return validation


# ======================================================================
# 23. POPULATION-WEIGHTED METRICS
# ======================================================================

def calculate_weighted_metrics(
    validation,
):
    """
    Calculate model performance estimates weighted by the inverse
    probability that an object entered the union validation sample.
    """

    weights = validation[
        "population_weight"
    ]


    rows = []


    for method in [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]:

        fine = classification_metrics(
            validation[
                "expert_fine_eval"
            ],
            validation[
                f"{method}_fine_eval"
            ],
            weights=weights,
        )


        superclass = classification_metrics(
            validation[
                "expert_superclass"
            ],
            validation[
                f"{method}_superclass"
            ],
            weights=weights,
        )


        rows.append({

            "method":
                method,

            "fine_accuracy_population_weighted":
                fine[
                    "accuracy"
                ],

            "fine_macro_f1_population_weighted":
                fine[
                    "macro_f1"
                ],

            "fine_balanced_accuracy_population_weighted":
                fine[
                    "balanced_accuracy"
                ],

            "superclass_accuracy_population_weighted":
                superclass[
                    "accuracy"
                ],

            "superclass_macro_f1_population_weighted":
                superclass[
                    "macro_f1"
                ],

            "superclass_balanced_accuracy_population_weighted":
                superclass[
                    "balanced_accuracy"
                ],
        })


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_DIR
        / "overall_metrics_population_weighted.csv",
        index=False,
    )


    return result


# ======================================================================
# 24. SUPERCLASS CONFUSION MATRIX
# ======================================================================

def calculate_superclass_confusion(
    validation,
):
    """
    Row-normalized confusion matrix:
        rows    = expert superclass
        columns = predicted superclass

    Main useful output is Fusion.
    """

    true = validation[
        "expert_superclass"
    ]


    pred = validation[
        "Fusion_superclass"
    ]


    valid = (
        true.notna()
        &
        pred.notna()
    )


    labels = sorted(
        set(
            true.loc[
                valid
            ]
        )
        |
        set(
            pred.loc[
                valid
            ]
        )
    )


    cm = confusion_matrix(
        true.loc[
            valid
        ],
        pred.loc[
            valid
        ],
        labels=labels,
    )


    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )


    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float,
        ),
        where=row_sums != 0,
    )


    result = pd.DataFrame(
        cm_norm,
        index=labels,
        columns=labels,
    )


    result.to_csv(
        OUTPUT_DIR
        / "validation_confusion_superclass.csv"
    )


    return result


# ======================================================================
# 25. PLOT 1 — OVERALL PERFORMANCE
# ======================================================================

def plot_overall_performance(
    overall,
):

    methods = [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]


    pretty = {
        "M1": "M1",
        "M2": "M2",
        "M3": "M3",
        "Fusion": "Fusion",
    }


    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )


    x = np.arange(
        len(methods)
    )


    accuracy = []
    lower = []
    upper = []


    for method in methods:

        row = overall[
            overall[
                "method"
            ]
            == method
        ].iloc[0]


        accuracy.append(
            row[
                "fine_accuracy_valid"
            ] * 100
        )


        lower.append(
            row[
                "fine_accuracy_ci_low"
            ] * 100
        )


        upper.append(
            row[
                "fine_accuracy_ci_high"
            ] * 100
        )


    accuracy = np.array(
        accuracy
    )

    lower = np.array(
        lower
    )

    upper = np.array(
        upper
    )


    error = np.vstack([
        accuracy - lower,
        upper - accuracy,
    ])


    bars = ax.bar(
        x,
        accuracy,
        yerr=error,
        capsize=4,
        width=0.65,
        color="#4C78A8",
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            pretty[m]
            for m in methods
        ]
    )


    ax.set_ylabel(
        "Fine-label agreement with expert (%)"
    )


    ax.set_ylim(
        0,
        min(
            100,
            max(100, upper.max() + 8)
        )
    )


    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )


    ax.set_axisbelow(
        True
    )


    for bar, value, row_method in zip(
        bars,
        accuracy,
        methods,
    ):

        row = overall[
            overall[
                "method"
            ]
            == row_method
        ].iloc[0]


        ax.text(
            bar.get_x()
            + bar.get_width()/2,
            value + 2,
            f"{value:.1f}%\n"
            f"F1={row['fine_macro_f1']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


    ax.set_title(
        "Expert-validated fine-label performance",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig1_overall_performance.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig1_overall_performance.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 26. PLOT 2 — PATHWAY PERFORMANCE
# ======================================================================

def plot_pathway_performance(
    pathway,
):

    path_order = [
        "A_all_agree",
        "B_majority",
        "C_superclass",
        "D_score",
        "D_score_close",
    ]


    path_names = {
        "A_all_agree":
            "A  Unanimous",

        "B_majority":
            "B  Majority",

        "C_superclass":
            "C  Superclass",

        "D_score":
            "D  Score",

        "D_score_close":
            "D  Score close",
    }


    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )


    for method, color, marker in [
        ("M1", "#4C78A8", "o"),
        ("M2", "#59A14F", "s"),
        ("M3", "#E17C05", "^"),
        ("Fusion", "#000000", "D"),
    ]:

        subset = pathway[
            pathway[
                "method"
            ]
            == method
        ].copy()


        subset[
            "order"
        ] = subset[
            "path"
        ].map(
            {
                p: i
                for i, p in enumerate(
                    path_order
                )
            }
        )


        subset = subset.sort_values(
            "order"
        )


        x = np.arange(
            len(subset)
        )


        y = (
            subset[
                "accuracy"
            ]
            * 100
        )


        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=2
            if method == "Fusion"
            else 1.3,
            markersize=7,
            color=color,
            label=method,
        )


        for xi, yi, n in zip(
            x,
            y,
            subset[
                "n_valid"
            ],
        ):

            if n >= 10:

                ax.text(
                    xi,
                    yi + 3,
                    f"n={n}",
                    ha="center",
                    fontsize=7,
                    color=color,
                )


    ax.set_xticks(
        np.arange(
            len(path_order)
        )
    )


    ax.set_xticklabels([
        path_names[p]
        for p in path_order
    ])


    ax.set_ylabel(
        "Fine-label agreement with expert (%)"
    )


    ax.set_ylim(
        0,
        100,
    )


    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )


    ax.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )


    ax.set_title(
        "Expert-validated performance across fusion pathways",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig2_pathway_performance.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig2_pathway_performance.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 27. PLOT 3 — SUPERCLASS PERFORMANCE
# ======================================================================

def plot_superclass_performance(
    superclass_metrics,
):

    methods = [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]


    superclasses = sorted(
        superclass_metrics[
            "expert_superclass"
        ]
        .unique()
    )


    matrix = np.full(
        (
            len(superclasses),
            len(methods),
        ),
        np.nan,
    )


    ns = np.zeros(
        len(superclasses),
        dtype=int,
    )


    for i, superclass in enumerate(
        superclasses
    ):

        subset = superclass_metrics[
            superclass_metrics[
                "expert_superclass"
            ]
            == superclass
        ]


        if len(subset):

            ns[i] = int(
                subset[
                    "n"
                ].iloc[0]
            )


        for j, method in enumerate(
            methods
        ):

            row = subset[
                subset[
                    "method"
                ]
                == method
            ]


            if len(row):

                matrix[
                    i,
                    j
                ] = (
                    row[
                        "fine_accuracy"
                    ].iloc[0]
                    * 100
                )


    fig, ax = plt.subplots(
        figsize=(8.5, 10)
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
    )


    ax.set_xticks(
        np.arange(
            len(methods)
        )
    )

    ax.set_xticklabels(
        methods
    )


    y_labels = [
        f"{sc}  (n={n:,})"
        for sc, n in zip(
            superclasses,
            ns,
        )
    ]


    ax.set_yticks(
        np.arange(
            len(superclasses)
        )
    )

    ax.set_yticklabels(
        y_labels
    )


    for i in range(
        len(superclasses)
    ):

        for j in range(
            len(methods)
        ):

            value = matrix[
                i,
                j
            ]


            if not np.isnan(
                value
            ):

                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white"
                    if value < 60
                    else "black",
                    fontsize=8,
                    fontweight="bold",
                )


    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )


    cbar.set_label(
        "Exact fine-label agreement (%)"
    )


    ax.set_xlabel(
        "Prediction source"
    )


    ax.set_ylabel(
        "Expert reference superclass"
    )


    ax.set_title(
        "Expert-validated fine-label performance by superclass",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig3_superclass_performance.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig3_superclass_performance.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 28. PLOT 4 — FUSION RESCUE / REGRESSION
# ======================================================================

def plot_fusion_benefit(
    benefit,
):

    models = [
        "M1",
        "M2",
        "M3",
    ]


    rescue = []
    regression = []


    for model in models:

        row = benefit[
            benefit[
                "model"
            ]
            == model
        ].iloc[0]


        rescue.append(
            row[
                "fusion_rescue"
            ]
        )


        regression.append(
            -row[
                "fusion_regression"
            ]
        )


    y = np.arange(
        len(models)
    )


    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )


    ax.barh(
        y,
        rescue,
        color="#4C78A8",
        label="Fusion correct; model wrong",
    )


    ax.barh(
        y,
        regression,
        color="#E15759",
        label="Fusion wrong; model correct",
    )


    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        models
    )


    ax.set_xlabel(
        "Number of paired validation cases"
    )


    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )


    ax.set_title(
        "Fusion rescues and regressions relative to individual models",
        loc="left",
        fontweight="bold",
    )


    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4,
    )


    ax.set_axisbelow(
        True
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig4_fusion_benefit.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig4_fusion_benefit.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 29. PLOT 5 — CONFIDENCE VALIDATION
# ======================================================================

def plot_confidence(
    confidence,
):

    order = [
        "HIGH",
        "MEDIUM",
        "LOW",
        "UNCERTAIN",
    ]


    confidence = (
        confidence
        .set_index(
            "confidence"
        )
        .reindex(
            order
        )
        .reset_index()
    )


    x = np.arange(
        len(order)
    )


    y = (
        confidence[
            "accuracy"
        ]
        * 100
    )


    lo = (
        confidence[
            "ci_low"
        ]
        * 100
    )


    hi = (
        confidence[
            "ci_high"
        ]
        * 100
    )


    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )


    ax.errorbar(
        x,
        y,
        yerr=np.vstack([
            y - lo,
            hi - y,
        ]),
        fmt="o-",
        color="#4C78A8",
        linewidth=2,
        markersize=7,
        capsize=4,
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        order
    )


    ax.set_ylabel(
        "Fusion fine-label agreement with expert (%)"
    )


    ax.set_ylim(
        0,
        100,
    )


    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )


    for xi, yi, n in zip(
        x,
        y,
        confidence[
            "n"
        ],
    ):

        ax.text(
            xi,
            yi + 5,
            f"n={int(n):,}",
            ha="center",
            fontsize=8,
        )


    ax.set_title(
        "Empirical validation of fusion confidence levels",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig5_confidence_validation.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig5_confidence_validation.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 30. PLOT 6 — SUPERCLASS CONFUSION MATRIX
# ======================================================================

def plot_superclass_confusion(
    confusion,
):

    fig, ax = plt.subplots(
        figsize=(9.5, 8.5)
    )


    image = ax.imshow(
        confusion.to_numpy()
        * 100,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="equal",
    )


    labels = (
        confusion.index
        .astype(str)
        .tolist()
    )


    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )


    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=8,
    )


    ax.set_yticklabels(
        labels,
        fontsize=8,
    )


    matrix = (
        confusion.to_numpy()
        * 100
    )


    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = matrix[
                i,
                j
            ]


            if value >= 5:

                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white"
                    if value > 55
                    else "black",
                )


    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )


    cbar.set_label(
        "Proportion of expert class (%)"
    )


    ax.set_xlabel(
        "Fusion-predicted superclass"
    )

    ax.set_ylabel(
        "Expert reference superclass"
    )


    ax.set_title(
        "Fusion superclass confusion matrix",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig6_superclass_confusion.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig6_superclass_confusion.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 31. PLOT 7 — WEIGHTED VS UNWEIGHTED
# ======================================================================

def plot_weighted_vs_unweighted(
    overall,
    weighted,
):

    methods = [
        "M1",
        "M2",
        "M3",
        "Fusion",
    ]


    unweighted = []
    population_weighted = []


    for method in methods:

        r1 = overall[
            overall[
                "method"
            ]
            == method
        ].iloc[0]


        r2 = weighted[
            weighted[
                "method"
            ]
            == method
        ].iloc[0]


        unweighted.append(
            r1[
                "fine_accuracy_valid"
            ] * 100
        )


        population_weighted.append(
            r2[
                "fine_accuracy_population_weighted"
            ] * 100
        )


    x = np.arange(
        len(methods)
    )


    width = 0.36


    fig, ax = plt.subplots(
        figsize=(8.5, 5)
    )


    ax.bar(
        x - width/2,
        unweighted,
        width,
        label="Unweighted validation sample",
        color="#4C78A8",
    )


    ax.bar(
        x + width/2,
        population_weighted,
        width,
        label="Population-weighted estimate",
        color="#59A14F",
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        methods
    )


    ax.set_ylabel(
        "Fine-label agreement (%)"
    )


    ax.set_ylim(
        0,
        100,
    )


    ax.legend(
        frameon=False,
    )


    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )


    ax.set_axisbelow(
        True
    )


    ax.set_title(
        "Effect of validation-sampling design on performance estimates",
        loc="left",
        fontweight="bold",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "fig7_weighted_vs_unweighted.png",
        dpi=600,
        bbox_inches="tight",
    )


    fig.savefig(
        OUTPUT_DIR
        / "fig7_weighted_vs_unweighted.pdf",
        bbox_inches="tight",
    )


    plt.show()


# ======================================================================
# 32. WRITE TEXT REPORT
# ======================================================================

def write_text_report(
    validation,
    overall,
    pathway,
    superclass,
    confidence,
    benefit,
    mcnemar,
    weighted=None,
):
    """
    Save a compact human-readable report.
    """

    path = (
        OUTPUT_DIR
        / "validation_report.txt"
    )


    lines = []


    lines.append(
        "THREE-MODEL EXPERT VALIDATION REPORT"
    )

    lines.append(
        "=" * 72
    )


    lines.append(
        f"Expert-labelled objects: "
        f"{len(validation):,}"
    )


    if "selected_by" in validation.columns:

        unique_sampled = (
            validation[
                "selected_by"
            ]
            .notna()
            .sum()
        )

        lines.append(
            f"Objects linked to sampling provenance: "
            f"{unique_sampled:,}"
        )


    lines.append("")
    lines.append(
        "OVERALL PERFORMANCE"
    )
    lines.append(
        "-" * 72
    )


    for _, row in overall.iterrows():

        lines.append(
            f"{row['method']:>7} | "
            f"Fine accuracy={row['fine_accuracy_valid']*100:.2f}% | "
            f"Macro F1={row['fine_macro_f1']:.3f} | "
            f"Balanced acc={row['fine_balanced_accuracy']:.3f} | "
            f"Superclass acc={row['superclass_accuracy_valid']*100:.2f}%"
        )


    lines.append("")
    lines.append(
        "CONFIDENCE"
    )
    lines.append(
        "-" * 72
    )


    for _, row in confidence.iterrows():

        lines.append(
            f"{row['confidence']:>10} | "
            f"n={int(row['n']):>5,} | "
            f"accuracy={row['accuracy']*100:.2f}%"
        )


    lines.append("")
    lines.append(
        "FUSION RESCUE / REGRESSION"
    )
    lines.append(
        "-" * 72
    )


    for _, row in benefit.iterrows():

        lines.append(
            f"{row['model']}: "
            f"rescue={int(row['fusion_rescue']):,}, "
            f"regression={int(row['fusion_regression']):,}, "
            f"net={int(row['net_fusion_gain']):+,}"
        )


    lines.append("")
    lines.append(
        "MCNEMAR TESTS"
    )
    lines.append(
        "-" * 72
    )


    for _, row in mcnemar.iterrows():

        lines.append(
            f"{row['comparison']}: "
            f"p={row['mcnemar_exact_p']:.6g}"
        )


    if weighted is not None:

        lines.append("")
        lines.append(
            "POPULATION-WEIGHTED ESTIMATES"
        )

        lines.append(
            "-" * 72
        )


        for _, row in weighted.iterrows():

            lines.append(
                f"{row['method']:>7} | "
                f"Fine accuracy="
                f"{row['fine_accuracy_population_weighted']*100:.2f}% | "
                f"Macro F1="
                f"{row['fine_macro_f1_population_weighted']:.3f} | "
                f"Balanced acc="
                f"{row['fine_balanced_accuracy_population_weighted']:.3f}"
            )


    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ======================================================================
# 33. MAIN
# ======================================================================

def main():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "THREE-MODEL EXPERT VALIDATION"
    )

    print(
        "=" * 78
    )


    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    (
        expert,
        fused,
        sampling,
        sampling_report,
        population_df,
    ) = load_data()

    # ------------------------------------------------------------------
    # Load superclass map
    # ------------------------------------------------------------------

    superclass_map = load_superclass_map(
        LABEL_MAP_PATH
    )


    print(
        f"\nSuperclass mappings loaded: "
        f"{len(superclass_map):,}"
    )


    print(
        "Unique mapped superclasses:"
    )

    print(
        sorted(
            set(
                superclass_map.values()
            )
        )
    )


    # ------------------------------------------------------------------
    # Build validation dataset
    # ------------------------------------------------------------------

    validation, model_vocab = (
        build_validation_dataset(
            expert,
            fused,
            sampling,
            superclass_map,
        )
    )


    # ------------------------------------------------------------------
    # Save validation table
    # ------------------------------------------------------------------

    validation.to_csv(
        OUTPUT_DIR
        / "validation_merged.csv",
        index=False,
    )


    print(
        f"\nValidation table saved:"
        f"\n  {OUTPUT_DIR / 'validation_merged.csv'}"
    )


    # ------------------------------------------------------------------
    # Expert mapping summary
    # ------------------------------------------------------------------

    print(
        "\nExpert labels:"
    )

    print(
        validation[
            "expert_label"
        ]
        .value_counts()
        .to_string()
    )


    n_shared = int(
        validation[
            "expert_label_shared_with_models"
        ]
        .fillna(False)
        .sum()
    )


    n_unshared = (
        len(validation)
        - n_shared
    )


    print(
        f"\nExpert labels representable by "
        f"model vocabulary: "
        f"{n_shared:,}/{len(validation):,}"
        f" ({100*n_shared/len(validation):.1f}%)"
    )


    print(
        f"Expert labels outside model vocabulary: "
        f"{n_unshared:,}"
    )


    # ------------------------------------------------------------------
    # Basic overall metrics
    # ------------------------------------------------------------------

    overall = (
        calculate_overall_metrics(
            validation
        )
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "OVERALL UNWEIGHTED PERFORMANCE"
    )

    print(
        "=" * 78
    )


    display_cols = [
        "method",
        "fine_prediction_coverage",
        "fine_accuracy_valid",
        "fine_macro_f1",
        "fine_balanced_accuracy",
        "superclass_accuracy_valid",
        "superclass_macro_f1",
        "superclass_balanced_accuracy",
    ]


    print(
        overall[
            display_cols
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    # ------------------------------------------------------------------
    # Pathway
    # ------------------------------------------------------------------

    pathway = (
        calculate_pathway_metrics(
            validation
        )
    )


    # ------------------------------------------------------------------
    # Superclass
    # ------------------------------------------------------------------

    superclass = (
        calculate_superclass_metrics(
            validation
        )
    )


    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    confidence = (
        calculate_confidence_metrics(
            validation
        )
    )


    # ------------------------------------------------------------------
    # Fusion benefit
    # ------------------------------------------------------------------

    (
        benefit,
        unique_fusion_outcomes,
    ) = calculate_fusion_benefit(
        validation
    )


    # ------------------------------------------------------------------
    # McNemar
    # ------------------------------------------------------------------

    mcnemar = (
        calculate_mcnemar(
            validation
        )
    )


    # ------------------------------------------------------------------
    # Population weighting
    # ------------------------------------------------------------------

    weighted = None

    if CALCULATE_POPULATION_WEIGHTED:

        validation = (
            calculate_population_weights(
                validation,
                population_df,
                superclass_map,
                sampling,
            )
        )


        validation.to_csv(
            OUTPUT_DIR
            / "validation_merged.csv",
            index=False,
        )


        weighted = (
            calculate_weighted_metrics(
                validation
            )
        )


        print(
            "\n"
            + "=" * 78
        )

        print(
            "POPULATION-WEIGHTED PERFORMANCE"
        )

        print(
            "=" * 78
        )


        print(
            weighted.round(4)
            .to_string(
                index=False
            )
        )


    # ------------------------------------------------------------------
    # Superclass confusion matrix
    # ------------------------------------------------------------------

    confusion = (
        calculate_superclass_confusion(
            validation
        )
    )


    # ------------------------------------------------------------------
    # Print important fusion findings
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FUSION BENEFIT / REGRESSION"
    )

    print(
        "=" * 78
    )


    print(
        benefit.to_string(
            index=False
        )
    )


    print(
        "\nUnique fusion outcomes:"
    )

    print(
        unique_fusion_outcomes.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "MCNEMAR TESTS"
    )

    print(
        "=" * 78
    )


    print(
        mcnemar.to_string(
            index=False
        )
    )


    # ------------------------------------------------------------------
    # Save final validation table
    # ------------------------------------------------------------------

    validation.to_csv(
        OUTPUT_DIR
        / "validation_merged.csv",
        index=False,
    )


    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "GENERATING FIGURES"
    )

    print(
        "=" * 78
    )


    plot_overall_performance(
        overall
    )


    plot_pathway_performance(
        pathway
    )


    plot_superclass_performance(
        superclass
    )


    plot_fusion_benefit(
        benefit
    )


    plot_confidence(
        confidence
    )


    plot_superclass_confusion(
        confusion
    )


    if weighted is not None:

        plot_weighted_vs_unweighted(
            overall,
            weighted,
        )


    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    write_text_report(
        validation,
        overall,
        pathway,
        superclass,
        confidence,
        benefit,
        mcnemar,
        weighted=weighted,
    )


    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "VALIDATION ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )


    print(
        f"\nExpert-labelled objects: "
        f"{len(validation):,}"
    )


    print(
        "\nOutputs:"
    )


    for filename in [

        "validation_merged.csv",

        "expert_label_mapping_report.csv",

        "overall_metrics_unweighted.csv",

        "overall_metrics_population_weighted.csv",

        "pathway_metrics.csv",

        "superclass_metrics.csv",

        "confidence_metrics.csv",

        "fusion_benefit.csv",

        "fusion_unique_outcomes.csv",

        "mcnemar_tests.csv",

        "validation_confusion_superclass.csv",

        "population_inclusion_probabilities.csv",

        "validation_report.txt",

    ]:

        path = (
            OUTPUT_DIR
            / filename
        )

        if path.exists():

            print(
                f"  {path}"
            )


    print(
        "\nFigures:"
    )


    for filename in [

        "fig1_overall_performance.png",

        "fig2_pathway_performance.png",

        "fig3_superclass_performance.png",

        "fig4_fusion_benefit.png",

        "fig5_confidence_validation.png",

        "fig6_superclass_confusion.png",

        "fig7_weighted_vs_unweighted.png",

    ]:

        path = (
            OUTPUT_DIR
            / filename
        )

        if path.exists():

            print(
                f"  {path}"
            )


if __name__ == "__main__":
    main()