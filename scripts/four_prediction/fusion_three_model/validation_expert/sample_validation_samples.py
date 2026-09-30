"""
sample_validation_samples.py

Create a stratified expert-validation dataset and EcoTaxa upload package
from three classifier predictions.

=======================================================================
DESIGN
=======================================================================

Two source tables are used:

1. merge_three_prediction_all.csv
   --------------------------------
   Authoritative source for:
       - original EcoTaxa metadata
       - object morphology measurements
       - object_major
       - object_id
       - image_name
       - sample/process/acquisition metadata

2. fused_predictions.csv
   ----------------------
   Authoritative source for:
       - m1_label
       - m2_label
       - m3_label
       - canonical labels
       - model scores
       - final fused label
       - fusion decision path
       - superclass outputs
       - confidence

The tables are merged using:
    object_id

=======================================================================
SAMPLING
=======================================================================

Each model is sampled independently.

For every model:

    predicted superclass
        x 10 bins based on object_major
        x 10 random objects per bin

Nominal target:

    20 superclasses
    x 10 bins
    x 10 samples
    = 2,000 sampling records per model

For 3 models:
    = 6,000 sampling records

The SAME physical object is allowed to be independently selected
for M1, M2, and/or M3.

This duplication is preserved in the PRIVATE sampling table.

=======================================================================
ECOTAXA
=======================================================================

EcoTaxa requires unique object_id values.

Therefore, after model-specific sampling:

    ~6000 sampling records
              |
              v
       collapse by object_id
              |
              v
       unique physical images

Only one copy of each object is uploaded to EcoTaxa.

The private provenance table retains which models/strata selected it.

The EcoTaxa metadata are BLIND:
    - AI predictions removed
    - scores removed
    - fusion information removed
    - sampling provenance removed
    - previous annotation information cleared

Original object/sample/process/acquisition metadata are retained.

=======================================================================
AFTER EXPERT ANNOTATION
=======================================================================

After exporting annotated objects from EcoTaxa:

    expert_export.tsv

join it back to:

    01_sampling_records.csv

using:

    object_id

This restores expert label + model prediction information and allows:

    M1 vs expert
    M2 vs expert
    M3 vs expert
    Fusion vs expert

=======================================================================
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image


# =====================================================================
# 1. USER SETTINGS
# =====================================================================

# ---------------------------------------------------------------------
# Input tables
# ---------------------------------------------------------------------

METADATA_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\merge_three_prediction_all.csv"
)

FUSED_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\fused_predictions.csv"
)


# ---------------------------------------------------------------------
# Fine-label -> superclass mapping
#
# This file must contain:
#
#     label
#     superclass_ecotaxa_20
#
# IMPORTANT:
# It should cover labels predicted by M1, M2 and M3.
# ---------------------------------------------------------------------

LABEL_MAP_PATH = Path(
    r"C:\alr4\ai_predict\ai_predict_d\label_to_int.csv"
)


# ---------------------------------------------------------------------
# Original image directory
#
# Example:
#
# IMAGE_ROOT = Path(r"C:\alr4\images")
#
# If image_name in the metadata is:
#
#     img_000123.png
#
# source image becomes:
#
#     C:\alr4\images\img_000123.png
#
# If image_name already contains subfolders, that also works.
# ---------------------------------------------------------------------

IMAGE_ROOT = Path(
    r"C:\alr4\ecodata\d"
)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"C:\alr4\ai_predict\ai_predict_d\expert_validation"
)

PROJECT_NAME = "three_model_expert_validation"


# =====================================================================
# 2. COLUMN SETTINGS
# =====================================================================

OBJECT_ID_COL = "object_id"
IMAGE_NAME_COL = "image_name"

# Variable used to divide each model-superclass into bins
BIN_VARIABLE = "object_major"


# ------------------------------------------------------------ ---------
# Model labels
# ---------------------------------------------------------------------

MODEL_COLUMNS = {
    "M1": "m1_label",
    "M2": "m2_label",
    "M3": "m3_label",
}


# ---------------------------------------------------------------------
# Superclass mapping columns
# ---------------------------------------------------------------------

MAP_LABEL_COL = "label"
MAP_SUPERCLASS_COL = "superclass_ecotaxa_20"


# =====================================================================
# 3. SAMPLING SETTINGS
# =====================================================================

N_BINS = 10
K_PER_BIN = 10

RANDOM_SEED = 42


# Your previous sampling code used equally spaced bins in log10 space.
USE_LOG_BINS = True

EPS = 1e-6


# If True, abort if any model-predicted labels cannot be mapped
# to the selected superclass scheme.
#
# I recommend keeping this True for the final validation dataset.
REQUIRE_COMPLETE_SUPERCLASS_MAPPING = True


# =====================================================================
# 4. IMAGE SETTINGS
# =====================================================================

# Normally preserve original images.
CONVERT_IMAGES_TO_PNG = True


# Only used if CONVERT_IMAGES_TO_PNG=True
INVERT_IMAGES = True

ECOTAXA_MINIMAL_TEST = True


# =====================================================================
# 5. OUTPUT PATHS
# =====================================================================

ECOTAXA_DIR = (
    OUTPUT_ROOT /
    "ecotaxa_upload"
)

ECOTAXA_TSV = (
    ECOTAXA_DIR /
    f"ecotaxa_{PROJECT_NAME}.tsv"
)

ZIP_PATH = (
    OUTPUT_ROOT /
    f"ecotaxa_{PROJECT_NAME}.zip"
)

SAMPLING_RECORDS_PATH = (
    OUTPUT_ROOT /
    "01_sampling_records.csv"
)

PRIVATE_UNIQUE_MASTER_PATH = (
    OUTPUT_ROOT /
    "02_unique_objects_private_master.csv"
)

SAMPLING_REPORT_PATH = (
    OUTPUT_ROOT /
    "03_sampling_report.csv"
)

DUPLICATE_REPORT_PATH = (
    OUTPUT_ROOT /
    "04_cross_model_duplicates.csv"
)

MISSING_IMAGES_PATH = (
    OUTPUT_ROOT /
    "05_missing_images.csv"
)

MERGE_REPORT_PATH = (
    OUTPUT_ROOT /
    "06_merge_report.csv"
)

MAPPING_REPORT_PATH = (
    OUTPUT_ROOT /
    "07_superclass_mapping_report.csv"
)


# =====================================================================
# 6. COLUMNS THAT MUST NEVER BE SHOWN TO THE EXPERT
# =====================================================================

AI_COLUMNS_TO_DROP_FROM_ECOTAXA = {

    # -----------------------------------------------------------------
    # raw model predictions/scores
    # -----------------------------------------------------------------

    "class_p_r",
    "score_p_r",

    "class_p_f",
    "score_p_f",

    "class_e_f",
    "score_e_f",

    # -----------------------------------------------------------------
    # canonical predictions
    # -----------------------------------------------------------------

    "canon_m1",
    "canon_m2",
    "canon_m3",

    # -----------------------------------------------------------------
    # fusion model labels
    # -----------------------------------------------------------------

    "m1_label",
    "m2_label",
    "m3_label",

    # -----------------------------------------------------------------
    # fusion output
    # -----------------------------------------------------------------

    "final_label",
    "final_score",

    "confidence",
    "path",

    # -----------------------------------------------------------------
    # superclass output
    # -----------------------------------------------------------------

    "superclass",

    "sc_taxo_20",
    "sc_morpho_24",
    "sc_taxo_30",
    "sc_ecotaxa_20",
    "sc_lineage",

    # -----------------------------------------------------------------
    # diagnostic flags
    # -----------------------------------------------------------------

    "flag_m1_null",
    "flag_m2_null",
    "flag_m3_null",

    "flag_m1_m2_identical",

    "n_models_active",
}


# =====================================================================
# 7. GENERAL HELPERS
# =====================================================================

def normalize_label(value):
    """Normalize a label for dictionary matching."""

    if pd.isna(value):
        return None

    return str(value).strip().lower()


def safe_filename(value):
    """
    Convert an object_id to something safe for a Windows filename.

    The object_id stored in the EcoTaxa metadata is NOT modified.
    Only the physical image filename is sanitized.
    """

    text = str(value).strip()

    text = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        text,
    )

    return text


# =====================================================================
# 8. LOAD AND MERGE INPUT TABLES
# =====================================================================

def load_and_merge_inputs(
    metadata_path,
    fused_path,
):
    """
    Merge original metadata with model/fusion outputs using object_id.

    merge_three_prediction_all.csv:
        authoritative metadata source

    fused_predictions.csv:
        authoritative source for derived model/fusion columns
    """

    print()
    print("=" * 78)
    print("1. LOADING INPUT TABLES")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    print(f"\nMetadata file:\n{metadata_path}")

    metadata = pd.read_csv(
        metadata_path,
        low_memory=False,
    )

    print(
        f"  rows    : {len(metadata):,}\n"
        f"  columns : {len(metadata.columns):,}"
    )


    # -----------------------------------------------------------------
    # Fusion table
    # -----------------------------------------------------------------

    print(f"\nFusion file:\n{fused_path}")

    fused = pd.read_csv(
        fused_path,
        low_memory=False,
    )

    print(
        f"  rows    : {len(fused):,}\n"
        f"  columns : {len(fused.columns):,}"
    )


    # -----------------------------------------------------------------
    # Check key column
    # -----------------------------------------------------------------

    for name, table in [
        ("metadata", metadata),
        ("fused", fused),
    ]:

        if OBJECT_ID_COL not in table.columns:

            raise KeyError(
                f"\n'{OBJECT_ID_COL}' is missing "
                f"from the {name} table."
            )


    # -----------------------------------------------------------------
    # Object ID uniqueness
    # -----------------------------------------------------------------

    metadata_duplicate_count = (
        metadata[OBJECT_ID_COL]
        .duplicated()
        .sum()
    )

    fused_duplicate_count = (
        fused[OBJECT_ID_COL]
        .duplicated()
        .sum()
    )


    print("\nObject-ID uniqueness:")

    print(
        f"  metadata duplicate IDs : "
        f"{metadata_duplicate_count:,}"
    )

    print(
        f"  fused duplicate IDs    : "
        f"{fused_duplicate_count:,}"
    )


    if metadata_duplicate_count:

        raise ValueError(
            "\nmerge_three_prediction_all.csv contains "
            "duplicate object_id values.\n"
            "A one-to-one validation merge cannot safely continue."
        )


    if fused_duplicate_count:

        raise ValueError(
            "\nfused_predictions.csv contains duplicate object_id values.\n"
            "A one-to-one validation merge cannot safely continue."
        )


    # -----------------------------------------------------------------
    # Object ID correspondence
    # -----------------------------------------------------------------

    metadata_ids = set(
        metadata[OBJECT_ID_COL].astype(str)
    )

    fused_ids = set(
        fused[OBJECT_ID_COL].astype(str)
    )


    metadata_only = (
        metadata_ids - fused_ids
    )

    fused_only = (
        fused_ids - metadata_ids
    )


    print("\nObject-ID correspondence:")

    print(
        f"  only in metadata : "
        f"{len(metadata_only):,}"
    )

    print(
        f"  only in fused    : "
        f"{len(fused_only):,}"
    )


    merge_report = pd.DataFrame({
        "metric": [
            "metadata_rows",
            "fused_rows",
            "metadata_duplicate_object_ids",
            "fused_duplicate_object_ids",
            "ids_only_in_metadata",
            "ids_only_in_fused",
        ],
        "value": [
            len(metadata),
            len(fused),
            metadata_duplicate_count,
            fused_duplicate_count,
            len(metadata_only),
            len(fused_only),
        ],
    })


    # -----------------------------------------------------------------
    # Columns wanted from fused_predictions.csv
    # -----------------------------------------------------------------

    fused_columns_wanted = [

        OBJECT_ID_COL,

        # raw model outputs
        "class_p_r",
        "score_p_r",

        "class_p_f",
        "score_p_f",

        "class_e_f",
        "score_e_f",

        # canonical labels
        "canon_m1",
        "canon_m2",
        "canon_m3",

        # model labels
        "m1_label",
        "m2_label",
        "m3_label",

        # fusion
        "final_label",
        "final_score",
        "confidence",
        "path",

        # superclasses
        "superclass",
        "sc_taxo_20",
        "sc_morpho_24",
        "sc_taxo_30",
        "sc_ecotaxa_20",
        "sc_lineage",

        # flags
        "flag_m1_null",
        "flag_m2_null",
        "flag_m3_null",
        "flag_m1_m2_identical",

        "n_models_active",
    ]


    fused_columns = [
        col
        for col in fused_columns_wanted
        if col in fused.columns
    ]


    fused_subset = fused[
        fused_columns
    ].copy()


    # -----------------------------------------------------------------
    # Some raw score/prediction columns already exist in metadata.
    #
    # For those, preserve metadata version and do not create suffixes.
    #
    # Derived fusion fields should come from fused_predictions.
    # -----------------------------------------------------------------

    force_from_fused = {

        "canon_m1",
        "canon_m2",
        "canon_m3",

        "m1_label",
        "m2_label",
        "m3_label",

        "final_label",
        "final_score",

        "confidence",
        "path",

        "superclass",

        "sc_taxo_20",
        "sc_morpho_24",
        "sc_taxo_30",
        "sc_ecotaxa_20",
        "sc_lineage",

        "flag_m1_null",
        "flag_m2_null",
        "flag_m3_null",

        "flag_m1_m2_identical",

        "n_models_active",
    }


    overlapping = (
        set(metadata.columns)
        & set(fused_subset.columns)
    )

    overlapping.discard(
        OBJECT_ID_COL
    )


    remove_from_fused = [
        col
        for col in overlapping
        if col not in force_from_fused
    ]


    if remove_from_fused:

        fused_subset = fused_subset.drop(
            columns=remove_from_fused
        )


    # -----------------------------------------------------------------
    # If metadata happened to contain an old derived fusion column,
    # remove it so fused_predictions remains authoritative.
    # -----------------------------------------------------------------

    derived_overlap_in_metadata = [
        col
        for col in force_from_fused
        if (
            col in metadata.columns
            and col in fused_subset.columns
        )
    ]


    if derived_overlap_in_metadata:

        metadata = metadata.drop(
            columns=derived_overlap_in_metadata
        )


    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------

    combined = metadata.merge(
        fused_subset,
        how="inner",
        on=OBJECT_ID_COL,
        validate="one_to_one",
    )


    print("\nCombined working table:")

    print(
        f"  rows    : {len(combined):,}\n"
        f"  columns : {len(combined.columns):,}"
    )


    merge_report.loc[
        len(merge_report)
    ] = [
        "merged_rows",
        len(combined),
    ]


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    merge_report.to_csv(
        MERGE_REPORT_PATH,
        index=False,
    )


    # -----------------------------------------------------------------
    # Required fields
    # -----------------------------------------------------------------

    required = {

        OBJECT_ID_COL,
        IMAGE_NAME_COL,
        BIN_VARIABLE,

        "m1_label",
        "m2_label",
        "m3_label",
    }


    missing = (
        required
        - set(combined.columns)
    )


    if missing:

        raise KeyError(
            "\nRequired fields are still missing after "
            "combining the two files:\n"
            f"{sorted(missing)}\n"
        )


    print("\nRequired validation fields:")

    for col in sorted(required):
        print(f"  OK  {col}")


    return combined


# =====================================================================
# 9. SUPERCLASS MAPPING
# =====================================================================

def load_superclass_mapping(path):
    """
    Build dictionary:

        fine label -> ecotaxa_20 superclass
    """

    print()
    print("=" * 78)
    print("2. LOADING SUPERCLASS MAPPING")
    print("=" * 78)

    print(f"\nMapping file:\n{path}")


    mapping_df = pd.read_csv(
        path,
        low_memory=False,
    )


    required = {
        MAP_LABEL_COL,
        MAP_SUPERCLASS_COL,
    }


    missing = (
        required
        - set(mapping_df.columns)
    )


    if missing:

        raise KeyError(
            "\nSuperclass mapping file is missing:\n"
            f"{sorted(missing)}\n\n"
            "Available columns:\n"
            f"{mapping_df.columns.tolist()}"
        )


    mapping_df = mapping_df[
        [
            MAP_LABEL_COL,
            MAP_SUPERCLASS_COL,
        ]
    ].copy()


    mapping_df["_label_key"] = (
        mapping_df[
            MAP_LABEL_COL
        ]
        .map(normalize_label)
    )


    # Remove empty mappings
    mapping_df = mapping_df[
        mapping_df[
            MAP_SUPERCLASS_COL
        ].notna()
    ].copy()


    mapping = dict(
        zip(
            mapping_df["_label_key"],
            mapping_df[
                MAP_SUPERCLASS_COL
            ],
        )
    )


    unique_superclasses = sorted(
        mapping_df[
            MAP_SUPERCLASS_COL
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    print(
        f"\nFine labels mapped : "
        f"{len(mapping):,}"
    )

    print(
        f"Unique superclasses: "
        f"{len(unique_superclasses):,}"
    )


    print("\nSuperclass values:")

    for superclass in unique_superclasses:
        print(f"  {superclass}")


    return mapping


# =====================================================================
# 10. CHECK MODEL-SPECIFIC MAPPING COVERAGE
# =====================================================================

def check_mapping_coverage(
    df,
    superclass_map,
):
    """
    Ensure M1/M2/M3 labels can be mapped to ecotaxa_20 independently.
    """

    print()
    print("=" * 78)
    print("3. CHECKING MODEL-SPECIFIC SUPERCLASS COVERAGE")
    print("=" * 78)


    rows = []

    any_unmapped = False


    for model, column in MODEL_COLUMNS.items():

        labels = (
            df[column]
            .dropna()
            .map(normalize_label)
        )


        mapped_mask = labels.isin(
            superclass_map.keys()
        )


        n_total = len(labels)

        n_mapped = int(
            mapped_mask.sum()
        )

        n_unmapped = (
            n_total - n_mapped
        )


        percentage = (
            100 * n_mapped / n_total
            if n_total
            else np.nan
        )


        print(
            f"\n{model}:"
            f"\n  predicted labels : {n_total:,}"
            f"\n  mapped           : {n_mapped:,}"
            f"\n  unmapped         : {n_unmapped:,}"
            f"\n  coverage         : {percentage:.2f}%"
        )


        unmapped_labels = sorted(
            set(
                labels[
                    ~mapped_mask
                ]
            )
        )


        if unmapped_labels:

            any_unmapped = True

            print(
                f"  unique unmapped labels: "
                f"{len(unmapped_labels)}"
            )

            for label in unmapped_labels:
                print(f"    {label}")


        rows.append({
            "model":
                model,

            "n_predictions":
                n_total,

            "n_mapped":
                n_mapped,

            "n_unmapped":
                n_unmapped,

            "coverage_percent":
                percentage,

            "unmapped_labels":
                "|".join(
                    unmapped_labels
                ),
        })


    report = pd.DataFrame(
        rows
    )


    report.to_csv(
        MAPPING_REPORT_PATH,
        index=False,
    )


    if (
        any_unmapped
        and REQUIRE_COMPLETE_SUPERCLASS_MAPPING
    ):

        raise ValueError(
            "\nNot all M1/M2/M3 labels can be mapped "
            f"to '{MAP_SUPERCLASS_COL}'.\n\n"
            "For a balanced superclass validation design, "
            "I recommend fixing the superclass mapping first.\n\n"
            f"See:\n{MAPPING_REPORT_PATH}"
        )


# =====================================================================
# 11. BINNING
# =====================================================================

def assign_bins(values):
    """
    Assign equal-width bins to one model-superclass subset.

    With USE_LOG_BINS=True:

        object_major
              |
              v
        log10(object_major)
              |
              v
        10 equal-width bins

    This reproduces the approach used in the user's previous
    reservoir-sampling script.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid = np.isfinite(
        values
    )


    if USE_LOG_BINS:

        valid &= (
            values > 0
        )


    transformed = np.full(
        len(values),
        np.nan,
        dtype=float,
    )


    if USE_LOG_BINS:

        transformed[
            valid
        ] = np.log10(
            np.clip(
                values[valid],
                EPS,
                None,
            )
        )

    else:

        transformed[
            valid
        ] = values[
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
                len(values),
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
            minimum + 1e-9
        )


    edges = np.linspace(
        minimum,
        maximum,
        N_BINS + 1,
    )


    bin_ids = np.full(
        len(values),
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


# =====================================================================
# 12. SAMPLE ONE MODEL
# =====================================================================

def sample_one_model(
    df,
    model_name,
    prediction_column,
    superclass_map,
    rng,
):
    """
    Independently sample one model:

        model prediction
            -> predicted ecotaxa superclass
            -> 10 object_major bins
            -> 10 samples/bin
    """

    print()
    print("-" * 78)
    print(
        f"SAMPLING {model_name}"
    )
    print("-" * 78)


    work = df.copy()


    # -----------------------------------------------------------------
    # Record which model caused selection
    # -----------------------------------------------------------------

    work[
        "sampled_for_model"
    ] = model_name


    work[
        "sampled_model_prediction"
    ] = work[
        prediction_column
    ]


    # -----------------------------------------------------------------
    # Map this specific model's prediction to superclass
    # -----------------------------------------------------------------

    work[
        "sampled_model_superclass"
    ] = (
        work[
            "sampled_model_prediction"
        ]
        .map(
            normalize_label
        )
        .map(
            superclass_map
        )
    )


    # -----------------------------------------------------------------
    # Numeric bin variable
    # -----------------------------------------------------------------

    work[
        BIN_VARIABLE
    ] = pd.to_numeric(
        work[
            BIN_VARIABLE
        ],
        errors="coerce",
    )


    usable = (
        work[
            "sampled_model_prediction"
        ].notna()
        &
        work[
            "sampled_model_superclass"
        ].notna()
        &
        work[
            BIN_VARIABLE
        ].notna()
    )


    if USE_LOG_BINS:

        usable &= (
            work[
                BIN_VARIABLE
            ] > 0
        )


    work = work.loc[
        usable
    ].copy()


    superclasses = sorted(
        work[
            "sampled_model_superclass"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    print(
        f"Mapped superclasses present: "
        f"{len(superclasses)}"
    )


    selected_groups = []
    report_rows = []


    # -----------------------------------------------------------------
    # One superclass at a time
    # -----------------------------------------------------------------

    for superclass in superclasses:

        superclass_df = work[
            work[
                "sampled_model_superclass"
            ].astype(str)
            == superclass
        ].copy()


        bin_ids, edges = assign_bins(
            superclass_df[
                BIN_VARIABLE
            ]
        )


        superclass_df[
            "_sampling_bin"
        ] = bin_ids


        superclass_df = superclass_df[
            superclass_df[
                "_sampling_bin"
            ] >= 0
        ].copy()


        if edges is None:
            continue


        # -------------------------------------------------------------
        # One bin at a time
        # -------------------------------------------------------------

        for bin_number in range(
            N_BINS
        ):

            candidates = superclass_df[
                superclass_df[
                    "_sampling_bin"
                ]
                == bin_number
            ].copy()


            available = len(
                candidates
            )


            number_to_sample = min(
                K_PER_BIN,
                available,
            )


            if number_to_sample > 0:

                local_seed = int(
                    rng.integers(
                        0,
                        np.iinfo(
                            np.int32
                        ).max,
                    )
                )


                selected = candidates.sample(
                    n=number_to_sample,
                    replace=False,
                    random_state=local_seed,
                ).copy()


                selected[
                    "_sampling_bin_left"
                ] = edges[
                    bin_number
                ]


                selected[
                    "_sampling_bin_right"
                ] = edges[
                    bin_number + 1
                ]


                selected_groups.append(
                    selected
                )


            report_rows.append({

                "model":
                    model_name,

                "superclass":
                    superclass,

                "bin_id":
                    bin_number,

                "available":
                    available,

                "requested":
                    K_PER_BIN,

                "sampled":
                    number_to_sample,

                "complete":
                    (
                        number_to_sample
                        == K_PER_BIN
                    ),

                "bin_left_transformed":
                    edges[
                        bin_number
                    ],

                "bin_right_transformed":
                    edges[
                        bin_number + 1
                    ],
            })


    # -----------------------------------------------------------------
    # Combine sampled records
    # -----------------------------------------------------------------

    if selected_groups:

        sampled = pd.concat(
            selected_groups,
            ignore_index=True,
        )

    else:

        sampled = pd.DataFrame()


    report = pd.DataFrame(
        report_rows
    )


    print(
        f"\n{model_name} sampled records: "
        f"{len(sampled):,}"
    )


    return (
        sampled,
        report,
    )


# =====================================================================
# 13. COLLAPSE CROSS-MODEL DUPLICATES FOR ECOTAXA
# =====================================================================

def create_unique_private_master(
    sampling_records,
):
    """
    Collapse duplicated selections to one physical object for EcoTaxa.

    Example:

        object 123 selected for M1
        object 123 selected for M3

    becomes one unique-object row, but private fields preserve:

        validation_selected_by = M1|M3

        validation_M1_selected = True
        validation_M3_selected = True

        validation_M1_bin = ...
        validation_M3_bin = ...
    """

    print()
    print("=" * 78)
    print("5. COLLAPSING DUPLICATES FOR ECOTAXA")
    print("=" * 78)


    unique_rows = []


    for object_id, group in sampling_records.groupby(
        OBJECT_ID_COL,
        sort=False,
    ):

        base = group.iloc[
            0
        ].copy()


        selected_models = sorted(
            group[
                "sampled_for_model"
            ]
            .dropna()
            .astype(str)
            .unique()
        )


        base[
            "validation_selected_by"
        ] = "|".join(
            selected_models
        )


        base[
            "validation_n_sampling_records"
        ] = len(
            group
        )


        base[
            "validation_n_models_selected"
        ] = len(
            selected_models
        )


        # -------------------------------------------------------------
        # Per-model provenance
        # -------------------------------------------------------------

        for model in MODEL_COLUMNS:

            model_group = group[
                group[
                    "sampled_for_model"
                ]
                == model
            ]


            if len(
                model_group
            ):

                first = model_group.iloc[
                    0
                ]


                base[
                    f"validation_{model}_selected"
                ] = True


                base[
                    f"validation_{model}_sampled_label"
                ] = first[
                    "sampled_model_prediction"
                ]


                base[
                    f"validation_{model}_sampled_superclass"
                ] = first[
                    "sampled_model_superclass"
                ]


                base[
                    f"validation_{model}_bin"
                ] = int(
                    first[
                        "_sampling_bin"
                    ]
                )


                base[
                    f"validation_{model}_bin_left"
                ] = first[
                    "_sampling_bin_left"
                ]


                base[
                    f"validation_{model}_bin_right"
                ] = first[
                    "_sampling_bin_right"
                ]


            else:

                base[
                    f"validation_{model}_selected"
                ] = False


                base[
                    f"validation_{model}_sampled_label"
                ] = np.nan


                base[
                    f"validation_{model}_sampled_superclass"
                ] = np.nan


                base[
                    f"validation_{model}_bin"
                ] = np.nan


                base[
                    f"validation_{model}_bin_left"
                ] = np.nan


                base[
                    f"validation_{model}_bin_right"
                ] = np.nan


        unique_rows.append(
            base.to_dict()
        )


    unique_master = pd.DataFrame(
        unique_rows
    )


    print(
        f"\nSampling records : "
        f"{len(sampling_records):,}"
    )

    print(
        f"Unique objects   : "
        f"{len(unique_master):,}"
    )

    print(
        f"Collapsed records: "
        f"{len(sampling_records) - len(unique_master):,}"
    )


    return unique_master


# =====================================================================
# 14. BUILD DUPLICATE REPORT
# =====================================================================

def build_duplicate_report(
    sampling_records,
):
    """
    Show objects selected by more than one model.
    """

    report = (
        sampling_records
        .groupby(
            OBJECT_ID_COL
        )
        .agg(

            n_sampling_records=(
                "sampled_for_model",
                "size",
            ),

            selected_by=(
                "sampled_for_model",
                lambda values:
                    "|".join(
                        sorted(
                            set(
                                values.astype(
                                    str
                                )
                            )
                        )
                    ),
            ),

        )
        .reset_index()
    )


    report = report[
        report[
            "n_sampling_records"
        ] > 1
    ].copy()


    report = report.sort_values(
        [
            "n_sampling_records",
            OBJECT_ID_COL,
        ],
        ascending=[
            False,
            True,
        ],
    )


    return report


# =====================================================================
# 15. PREPARE BLIND ECOTAXA METADATA
# =====================================================================

def create_blind_ecotaxa_dataframe(
    unique_master,
):
    """
    Start from the merged original metadata.

    Remove all AI/fusion/private validation information.

    Preserve original object/sample/process/acquisition metadata.
    """

    print()
    print("=" * 78)
    print("6. CREATING BLIND ECOTAXA METADATA")
    print("=" * 78)


    ecotaxa = unique_master.copy()


    # -----------------------------------------------------------------
    # AI/fusion columns
    # -----------------------------------------------------------------

    columns_to_drop = set(
        AI_COLUMNS_TO_DROP_FROM_ECOTAXA
    )


    # -----------------------------------------------------------------
    # Private validation provenance
    # -----------------------------------------------------------------

    for column in ecotaxa.columns:

        if column.startswith(
            "validation_"
        ):

            columns_to_drop.add(
                column
            )


    columns_to_drop.update({

        "sampled_for_model",

        "sampled_model_prediction",

        "sampled_model_superclass",

        "_sampling_bin",

        "_sampling_bin_left",

        "_sampling_bin_right",
        IMAGE_NAME_COL,
    })


    actual_drop = [
        col
        for col in columns_to_drop
        if col in ecotaxa.columns
    ]


    ecotaxa = ecotaxa.drop(
        columns=actual_drop
    )


    # -----------------------------------------------------------------
    # Clear old annotation information
    #
    # We want the expert to annotate blindly.
    # -----------------------------------------------------------------

    annotation_columns_to_clear = [

        "object_annotation_category",

        "object_annotation_hierarchy",

        "object_annotation_date",

        "object_annotation_time",

        "object_annotation_person_name",

        "object_annotation_person_email",

        "object_annotation_status",
    ]


    for column in annotation_columns_to_clear:

        if column in ecotaxa.columns:

            ecotaxa[
                column
            ] = ""


    print(
        f"EcoTaxa metadata columns retained: "
        f"{len(ecotaxa.columns):,}"
    )

    if ECOTAXA_MINIMAL_TEST:
        minimal_columns = [
            "img_file_name",  # added later, so don't require here yet
            "object_id",
            "sample_id",
            "object_lat",
            "object_lon",
            "object_depth_min",
            "object_depth_max",
        ]

        # img_file_name doesn't exist yet at this stage
        minimal_columns = [
            c
            for c in minimal_columns
            if c in ecotaxa.columns
        ]

        ecotaxa = ecotaxa[
            minimal_columns
        ].copy()


    return ecotaxa


# =====================================================================
# 16. FIND ORIGINAL IMAGE
# =====================================================================

def resolve_image_path(
    image_name,
):
    """
    Reconstruct local image path from IMAGE_ROOT + image_name.
    """

    if pd.isna(
        image_name
    ):

        return None


    text = str(
        image_name
    ).strip()


    if not text:

        return None


    candidate = (
        IMAGE_ROOT /
        Path(text)
    )


    return candidate


# =====================================================================
# 17. SAVE IMAGE
# =====================================================================

def save_image(
    source,
    destination,
):
    """
    Either copy the original image or convert to grayscale PNG.
    """

    source = Path(
        source
    )


    if not source.exists():

        raise FileNotFoundError(
            str(source)
        )


    if not CONVERT_IMAGES_TO_PNG:

        shutil.copy2(
            source,
            destination,
        )

        return


    image = Image.open(
        source
    )


    image = image.convert(
        "L"
    )


    if INVERT_IMAGES:

        array = np.asarray(
            image
        )


        maximum = np.iinfo(
            array.dtype
        ).max


        inverted = (
            maximum - array
        )


        image = Image.fromarray(
            inverted
        )


    image.save(
        destination,
        format="PNG",
    )


# =====================================================================
# 18. COPY UNIQUE IMAGES TO ECOTAXA DIRECTORY
# =====================================================================

def prepare_ecotaxa_images(
    unique_master,
    ecotaxa_df,
):
    """
    Copy each unique physical object once.

    Adds / replaces:
        img_file_name
    """

    print()
    print("=" * 78)
    print("7. PREPARING ECOTAXA IMAGES")
    print("=" * 78)


    if ECOTAXA_DIR.exists():

        shutil.rmtree(
            ECOTAXA_DIR
        )


    ECOTAXA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output = ecotaxa_df.copy()


    filenames = []

    missing_rows = []


    total = len(
        unique_master
    )


    for number, (_, row) in enumerate(
        unique_master.iterrows(),
        start=1,
    ):

        object_id = row[
            OBJECT_ID_COL
        ]


        image_name = row[
            IMAGE_NAME_COL
        ]


        source = resolve_image_path(
            image_name
        )


        if (
            source is None
            or not source.exists()
        ):

            filenames.append(
                np.nan
            )


            missing_rows.append({

                OBJECT_ID_COL:
                    object_id,

                IMAGE_NAME_COL:
                    image_name,

                "expected_path":
                    (
                        str(source)
                        if source is not None
                        else ""
                    ),
            })


            continue


        # -------------------------------------------------------------
        # EcoTaxa image filename
        # -------------------------------------------------------------

        if CONVERT_IMAGES_TO_PNG:

            filename = (
                safe_filename(
                    object_id
                )
                + ".png"
            )

        else:

            extension = (
                source.suffix.lower()
            )


            if not extension:

                extension = ".png"


            filename = (
                safe_filename(
                    object_id
                )
                + extension
            )


        destination = (
            ECOTAXA_DIR /
            filename
        )


        # -------------------------------------------------------------
        # Safety check: different objects should never overwrite
        # another image accidentally.
        # -------------------------------------------------------------

        if destination.exists():

            raise FileExistsError(
                "\nTwo different records generated the same "
                f"EcoTaxa image filename:\n{destination}\n"
                "Check object_id uniqueness / filename sanitization."
            )


        save_image(
            source,
            destination,
        )


        filenames.append(
            filename
        )


        if (
            number % 500 == 0
            or number == total
        ):

            print(
                f"  {number:,} / "
                f"{total:,}"
            )


    output[
        "img_file_name"
    ] = filenames


    # -----------------------------------------------------------------
    # Missing image report
    # -----------------------------------------------------------------

    missing_df = pd.DataFrame(
        missing_rows
    )


    missing_df.to_csv(
        MISSING_IMAGES_PATH,
        index=False,
    )


    print(
        f"\nMissing images: "
        f"{len(missing_df):,}"
    )


    if len(
        missing_df
    ):

        print(
            f"Report:\n{MISSING_IMAGES_PATH}"
        )


    # -----------------------------------------------------------------
    # Don't upload metadata rows with no image
    # -----------------------------------------------------------------

    output = output[
        output[
            "img_file_name"
        ].notna()
    ].copy()


    return output


# =====================================================================
# 19. ECOTAXA FIELD-TYPE ROW
# =====================================================================

def create_ecotaxa_type_row(df):

    types = {}

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            types[column] = "[f]"
        else:
            types[column] = "[t]"

    # force text fields
    for column in [
        OBJECT_ID_COL,
        "img_file_name",
        "sample_id",
        "process_id",
        "acq_id",
    ]:
        if column in types:
            types[column] = "[t]"

    return pd.DataFrame([types])


# =====================================================================
# 20. SAVE ECOTAXA TSV
# =====================================================================

def save_ecotaxa_tsv(ecotaxa_df):

    print()
    print("=" * 78)
    print("8. SAVING ECOTAXA TSV")
    print("=" * 78)

    # ---------------------------------------------------------
    # Arrange important columns first
    # ---------------------------------------------------------

    preferred_first = [
        "img_file_name",
        OBJECT_ID_COL,
        "sample_id",
    ]

    existing_first = [
        col
        for col in preferred_first
        if col in ecotaxa_df.columns
    ]

    remaining = [
        col
        for col in ecotaxa_df.columns
        if col not in existing_first
    ]

    ecotaxa_df = ecotaxa_df[
        existing_first + remaining
    ].copy()

    # ---------------------------------------------------------
    # Validate headers
    # ---------------------------------------------------------

    validate_ecotaxa_headers(ecotaxa_df)

    # ---------------------------------------------------------
    # Create EcoTaxa type row
    # ---------------------------------------------------------

    type_row = create_ecotaxa_type_row(
        ecotaxa_df
    )

    # ---------------------------------------------------------
    # Add type row as first data row
    # ---------------------------------------------------------

    final_df = pd.concat(
        [
            type_row,
            ecotaxa_df,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save TSV
    # ---------------------------------------------------------

    final_df.to_csv(
        ECOTAXA_TSV,
        sep="\t",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    # ---------------------------------------------------------
    # Verify first two lines
    # ---------------------------------------------------------

    with open(
        ECOTAXA_TSV,
        "r",
        encoding="utf-8",
    ) as f:

        header = f.readline().rstrip("\r\n")
        type_line = f.readline().rstrip("\r\n")

    print("\nHeader:")
    print(header)

    print("\nType row:")
    print(type_line)

    print(
        f"\nSaved:\n{ECOTAXA_TSV}"
    )

    print(
        f"Objects in TSV: {len(ecotaxa_df):,}"
    )


def validate_ecotaxa_headers(df):
    """
    Validate EcoTaxa column names and print suspicious columns.

    Allowed:
        img_file_name

        object_*
        sample_*
        process_*
        acq_*

    Also checks:
        - blank column names
        - leading/trailing whitespace
        - duplicate headers
        - suspicious punctuation
    """

    allowed_exact = {
        "img_file_name",
    }

    allowed_prefixes = (
        "object_",
        "sample_",
        "process_",
        "acq_",
    )

    print("\n" + "=" * 78)
    print("ECOTAXA HEADER VALIDATION")
    print("=" * 78)

    problems = []

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    duplicate_columns = (
        pd.Index(df.columns)[
            pd.Index(df.columns).duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        print("\nDuplicate column names:")

        for c in duplicate_columns:
            print(f"  {repr(c)}")

        problems.extend(
            [
                f"duplicate:{c}"
                for c in duplicate_columns
            ]
        )


    # --------------------------------------------------------
    # Check every column
    # --------------------------------------------------------

    print(
        f"\nChecking {len(df.columns)} columns..."
    )

    for i, original_column in enumerate(
        df.columns
    ):

        column = str(
            original_column
        )

        stripped = column.strip()


        # ----------------------------------------------------
        # Empty header
        # ----------------------------------------------------

        if stripped == "":

            print(
                f"\nINVALID EMPTY HEADER "
                f"at column index {i}"
            )

            problems.append(
                f"empty_header_index_{i}"
            )

            continue


        # ----------------------------------------------------
        # Leading/trailing whitespace
        # ----------------------------------------------------

        if column != stripped:

            print(
                f"\nWHITESPACE IN HEADER:"
                f"\n  index : {i}"
                f"\n  raw   : {repr(column)}"
            )

            problems.append(
                f"whitespace:{column}"
            )


        # ----------------------------------------------------
        # Prefix
        # ----------------------------------------------------

        prefix_ok = (
            stripped in allowed_exact
            or stripped.startswith(
                allowed_prefixes
            )
        )


        if not prefix_ok:

            print(
                f"\nINVALID PREFIX:"
                f"\n  index : {i}"
                f"\n  column: {repr(stripped)}"
            )

            problems.append(
                f"prefix:{stripped}"
            )


        # ----------------------------------------------------
        # Suspicious punctuation
        # ----------------------------------------------------

        suspicious_chars = [
            "%",
            ".",
            " ",
            "\t",
            "\n",
            "\r",
            "/",
            "\\",
            ":",
            ";",
            "[",
            "]",
            "(",
            ")",
        ]


        found_chars = [
            char
            for char in suspicious_chars
            if char in stripped
        ]


        if found_chars:

            print(
                f"\nSUSPICIOUS HEADER:"
                f"\n  index : {i}"
                f"\n  column: {repr(stripped)}"
                f"\n  chars : {found_chars}"
            )


    # --------------------------------------------------------
    # Print all headers with indexes
    # --------------------------------------------------------

    print(
        "\nFull EcoTaxa header list:"
    )

    for i, column in enumerate(
        df.columns
    ):

        print(
            f"{i:>4}: {repr(column)}"
        )


    # --------------------------------------------------------
    # Stop on definite structural problems
    # --------------------------------------------------------

    if problems:

        raise ValueError(
            "\nEcoTaxa header validation failed.\n"
            "Review the diagnostic output above."
        )


    print(
        "\nEcoTaxa structural header validation: OK"
    )


# =====================================================================
# 21. CREATE ZIP
# =====================================================================

def create_zip():
    """
    Zip all files in ecotaxa_upload directly into ZIP root.

    Structure becomes:

        ecotaxa_three_model_expert_validation.tsv
        image1.png
        image2.png
        ...
    """

    print()
    print("=" * 78)
    print("9. CREATING ECOTAXA ZIP")
    print("=" * 78)


    if ZIP_PATH.exists():

        ZIP_PATH.unlink()


    files = [
        path
        for path in ECOTAXA_DIR.iterdir()
        if path.is_file()
    ]


    with ZipFile(
        ZIP_PATH,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as archive:

        for number, file in enumerate(
            files,
            start=1,
        ):

            archive.write(
                file,
                arcname=file.name,
            )


            if (
                number % 1000 == 0
                or number == len(files)
            ):

                print(
                    f"  zipped "
                    f"{number:,} / "
                    f"{len(files):,}"
                )


    print(
        f"\nZIP created:\n{ZIP_PATH}"
    )


# =====================================================================
# 22. PRINT FINAL SAMPLING SUMMARY
# =====================================================================

def print_summary(
    sampling_records,
    unique_master,
    sampling_report,
):
    """
    Print diagnostics before finishing.
    """

    print()
    print("=" * 78)
    print("FINAL VALIDATION-SAMPLING SUMMARY")
    print("=" * 78)


    # -----------------------------------------------------------------
    # Per model
    # -----------------------------------------------------------------

    print(
        "\nSampling records per model:"
    )


    per_model = (
        sampling_records[
            "sampled_for_model"
        ]
        .value_counts()
        .reindex(
            [
                "M1",
                "M2",
                "M3",
            ],
            fill_value=0,
        )
    )


    print(
        per_model.to_string()
    )


    # -----------------------------------------------------------------
    # Number superclasses represented/model
    # -----------------------------------------------------------------

    print(
        "\nSuperclasses represented:"
    )


    sc_counts = (
        sampling_records
        .groupby(
            "sampled_for_model"
        )[
            "sampled_model_superclass"
        ]
        .nunique()
    )


    print(
        sc_counts.to_string()
    )


    # -----------------------------------------------------------------
    # Expected target
    # -----------------------------------------------------------------

    expected_per_model = (
        N_BINS
        * K_PER_BIN
    )


    print(
        "\nSampling design:"
    )

    print(
        f"  bins/superclass : "
        f"{N_BINS}"
    )

    print(
        f"  samples/bin     : "
        f"{K_PER_BIN}"
    )

    print(
        "  target per superclass : "
        f"{expected_per_model}"
    )


    # -----------------------------------------------------------------
    # Incomplete bins
    # -----------------------------------------------------------------

    incomplete = sampling_report[
        sampling_report[
            "sampled"
        ]
        <
        sampling_report[
            "requested"
        ]
    ].copy()


    print(
        f"\nIncomplete model-superclass-bin strata: "
        f"{len(incomplete):,}"
    )


    if len(
        incomplete
    ):

        print(
            "\nFirst 30 incomplete strata:"
        )


        print(
            incomplete[
                [
                    "model",
                    "superclass",
                    "bin_id",
                    "available",
                    "sampled",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )


    # -----------------------------------------------------------------
    # Duplicate selections
    # -----------------------------------------------------------------

    print(
        "\nCross-model sampling overlap:"
    )


    distribution = (
        unique_master[
            "validation_n_models_selected"
        ]
        .value_counts()
        .sort_index()
    )


    for n_models, count in distribution.items():

        print(
            f"  selected by "
            f"{int(n_models)} model(s): "
            f"{count:,} unique objects"
        )


    print(
        f"\nTotal sampling records : "
        f"{len(sampling_records):,}"
    )

    print(
        f"Unique objects         : "
        f"{len(unique_master):,}"
    )

    print(
        f"Duplicate records removed "
        f"for EcoTaxa upload: "
        f"{len(sampling_records) - len(unique_master):,}"
    )


# =====================================================================
# 23. MAIN
# =====================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------------------
    # Check image root
    # -----------------------------------------------------------------

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            "\nIMAGE_ROOT does not exist:\n"
            f"{IMAGE_ROOT}\n\n"
            "Edit IMAGE_ROOT near the top of the script."
        )


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    # =================================================================
    # A. Merge metadata + fusion predictions
    # =================================================================

    combined = load_and_merge_inputs(
        METADATA_PATH,
        FUSED_PATH,
    )


    # =================================================================
    # B. Load superclass mapping
    # =================================================================

    superclass_map = load_superclass_mapping(
        LABEL_MAP_PATH
    )


    # =================================================================
    # C. Check model-specific superclass coverage
    # =================================================================

    check_mapping_coverage(
        combined,
        superclass_map,
    )


    # =================================================================
    # D. Sample each model INDEPENDENTLY
    #
    # Deliberately do NOT exclude objects selected for previous models.
    # =================================================================

    print()
    print("=" * 78)
    print("4. STRATIFIED MODEL-SPECIFIC SAMPLING")
    print("=" * 78)


    all_sampling_records = []
    all_sampling_reports = []


    for model_name, prediction_column in MODEL_COLUMNS.items():

        sampled, report = sample_one_model(

            df=combined,

            model_name=model_name,

            prediction_column=prediction_column,

            superclass_map=superclass_map,

            rng=rng,
        )


        all_sampling_records.append(
            sampled
        )


        all_sampling_reports.append(
            report
        )


    # -----------------------------------------------------------------
    # Combine three model samples
    # -----------------------------------------------------------------

    sampling_records = pd.concat(
        all_sampling_records,
        ignore_index=True,
    )


    sampling_report = pd.concat(
        all_sampling_reports,
        ignore_index=True,
    )


    # -----------------------------------------------------------------
    # Give every sampling event a unique ID.
    #
    # The same object_id may therefore have several validation_record_id
    # values if it was independently selected for multiple models.
    # -----------------------------------------------------------------

    sampling_records[
        "validation_record_id"
    ] = [

        f"VALR_{i:06d}"

        for i in range(
            1,
            len(sampling_records) + 1,
        )
    ]


    # -----------------------------------------------------------------
    # Save complete private sampling table
    # -----------------------------------------------------------------

    sampling_records.to_csv(
        SAMPLING_RECORDS_PATH,
        index=False,
    )


    sampling_report.to_csv(
        SAMPLING_REPORT_PATH,
        index=False,
    )


    print(
        f"\nPrivate sampling table saved:\n"
        f"{SAMPLING_RECORDS_PATH}"
    )


    # =================================================================
    # E. Collapse duplicate object IDs for EcoTaxa
    # =================================================================

    unique_master = create_unique_private_master(
        sampling_records
    )


    unique_master.to_csv(
        PRIVATE_UNIQUE_MASTER_PATH,
        index=False,
    )


    # -----------------------------------------------------------------
    # Duplicate/overlap report
    # -----------------------------------------------------------------

    duplicate_report = build_duplicate_report(
        sampling_records
    )


    duplicate_report.to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
    )


    # =================================================================
    # F. Make blind EcoTaxa metadata
    # =================================================================

    ecotaxa_df = create_blind_ecotaxa_dataframe(
        unique_master
    )


    # =================================================================
    # G. Copy images
    # =================================================================

    ecotaxa_df = prepare_ecotaxa_images(
        unique_master,
        ecotaxa_df,
    )


    # =================================================================
    # H. Save EcoTaxa TSV
    # =================================================================

    save_ecotaxa_tsv(
        ecotaxa_df
    )


    # =================================================================
    # I. ZIP TSV + images
    # =================================================================

    create_zip()


    # =================================================================
    # J. Summary
    # =================================================================

    print_summary(
        sampling_records,
        unique_master,
        sampling_report,
    )


    print()
    print("=" * 78)
    print("FILES CREATED")
    print("=" * 78)


    print(
        "\nPRIVATE — DO NOT UPLOAD TO ECOTAXA:"
    )

    print(
        f"  1. {SAMPLING_RECORDS_PATH}"
    )

    print(
        f"  2. {PRIVATE_UNIQUE_MASTER_PATH}"
    )

    print(
        f"  3. {SAMPLING_REPORT_PATH}"
    )

    print(
        f"  4. {DUPLICATE_REPORT_PATH}"
    )

    print(
        f"  5. {MAPPING_REPORT_PATH}"
    )

    print(
        f"  6. {MERGE_REPORT_PATH}"
    )


    print(
        "\nECOTAXA:"
    )

    print(
        f"  TSV : {ECOTAXA_TSV}"
    )

    print(
        f"  ZIP : {ZIP_PATH}"
    )


    print()
    print("=" * 78)
    print("IMPORTANT NEXT STEP")
    print("=" * 78)

    print(
        "\nAfter the taxonomist finishes annotation:"
        "\n"
        "\n1. Export the annotated data from EcoTaxa."
        "\n2. Keep object_id in the exported file."
        "\n3. Merge the expert export with:"
        f"\n\n   {SAMPLING_RECORDS_PATH}"
        "\n\nusing object_id."
        "\n"
        "\nThat will restore:"
        "\n  - expert label"
        "\n  - M1 prediction"
        "\n  - M2 prediction"
        "\n  - M3 prediction"
        "\n  - fusion prediction"
        "\n  - sampling superclass"
        "\n  - sampling bin"
        "\n  - model that caused the object to enter the sample."
    )


# =====================================================================
# 24. ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()