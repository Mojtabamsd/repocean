"""
fusion.py
=========
Merges predictions from M1, M2, and M3 into a single canonical label.

Pipeline per sample
-------------------
1. Normalise each model's raw label  →  canonical label
2. Normalise each model's raw score  →  [0, 1] comparable score
3. Group predictions by canonical label (lineage-aware: labels within
   distance <= MAX_MERGE_DISTANCE are treated as the same class,
   resolved to the deeper/more-specific one)
4. Compute a weighted confidence per canonical label
5. Apply per-model weights and M2 specialisation boost
6. Return the label with highest weighted confidence, plus diagnostics

Key design decisions
--------------------
- M2 was fine-tuned on 35 classes: when it fires on a class within its
  known set, its weight is boosted (M2_BOOST).
- When a model has no prediction for a sample (abstention), it
  contributes zero weight for that sample.
- Labels within lineage distance <= MAX_MERGE_DISTANCE are merged
  before voting; the deeper (more specific) label wins.
- A result whose top confidence is below MIN_CONFIDENCE is flagged
  as "uncertain" rather than being forced to a label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from label_mapping import (
    normalise_label,
    lineage_distance,
    deeper_label,
    build_canonical_table,
    build_normaliser,
    load_label_tables,
)
from score_normaliser import BaseNormaliser, RankNormaliser


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-model base weights (tune these based on known accuracy)
MODEL_WEIGHTS: dict[str, float] = {
    "m1": 1.0,   # 82 classes, reference images
    "m2": 1.0,   # 52 classes, fine-tuned — boosted separately for its classes
    "m3": 1.0,   # 54 classes, different architecture
}

# Extra weight multiplier for M2 when it predicts within its known 52 classes
M2_BOOST: float = 1.5

# Labels within this lineage distance are treated as the same class
# 0 = exact match only | 1 = include parent/child | 2 = include cousins
MAX_MERGE_DISTANCE: int = 1

# Predictions below this normalised score are treated as abstentions
MIN_SCORE_THRESHOLD: float = 0.0   # set to e.g. 0.3 to ignore low-confidence preds

# Final merged confidence below this → result flagged as "uncertain"
MIN_CONFIDENCE: float = 0.0        # set to e.g. 0.4 to flag uncertain cases

UNCERTAIN_LABEL: str = "uncertain"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """A single prediction from one model for one sample."""
    model_id: str          # "m1", "m2", or "m3"
    raw_label: str         # as it came out of the model
    raw_score: float       # raw confidence score


@dataclass
class FusionResult:
    """Output of the fusion step for one sample."""
    label: str                          # canonical final label (or UNCERTAIN_LABEL)
    confidence: float                   # weighted score in [0, 1]
    is_uncertain: bool
    votes: dict[str, float]             # canonical_label -> weighted score
    per_model: dict[str, str]           # model_id -> canonical label used
    raw_predictions: list[ModelPrediction]


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Stateful engine: set up once with the label tables and normaliser,
    then call .fuse() per sample or .fuse_dataframe() for batch.

    Parameters
    ----------
    canonical_table : dict[str, str]
        From label_mapping.build_canonical_table()
    norm_map : dict[str, str]
        From label_mapping.build_normaliser()
    score_normaliser : BaseNormaliser
        Fitted RankNormaliser or MinMaxNormaliser.
    m2_known_labels : set[str]
        Canonical labels that M2 was trained on (for the boost).
    """

    def __init__(
        self,
        canonical_table: dict[str, str],
        norm_map: dict[str, str],
        score_normaliser: BaseNormaliser,
        m2_known_labels: Optional[set[str]] = None,
        model_weights: dict[str, float] = MODEL_WEIGHTS,
        m2_boost: float = M2_BOOST,
        max_merge_distance: int = MAX_MERGE_DISTANCE,
        min_score_threshold: float = MIN_SCORE_THRESHOLD,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.canonical_table = canonical_table
        self.norm_map = norm_map
        self.score_normaliser = score_normaliser
        self.m2_known_labels = m2_known_labels or set()
        self.model_weights = model_weights
        self.m2_boost = m2_boost
        self.max_merge_distance = max_merge_distance
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def fuse(self, predictions: list[ModelPrediction]) -> FusionResult:
        """
        Fuse a list of predictions (one per model) for a single sample.
        Models may be absent (abstention) — just omit them from the list.
        """
        # Step 1: normalise labels and scores
        normed: list[tuple[str, str, float]] = []   # (model_id, canon_label, norm_score)
        per_model_label: dict[str, str] = {}

        for pred in predictions:
            canon = normalise_label(pred.raw_label, self.norm_map)
            if canon is None:
                per_model_label[pred.model_id] = f"[unknown: {pred.raw_label}]"
                continue

            try:
                norm_score = self.score_normaliser.transform(pred.model_id, pred.raw_score)
            except ValueError:
                norm_score = pred.raw_score  # fallback: use raw if not fitted

            if norm_score < self.min_score_threshold:
                per_model_label[pred.model_id] = f"[below threshold: {canon}]"
                continue

            per_model_label[pred.model_id] = canon
            normed.append((pred.model_id, canon, norm_score))

        if not normed:
            return FusionResult(
                label=UNCERTAIN_LABEL,
                confidence=0.0,
                is_uncertain=True,
                votes={},
                per_model=per_model_label,
                raw_predictions=predictions,
            )

        # Step 2: merge labels within lineage distance <= max_merge_distance
        # Build groups: each prediction slots into a canonical "bucket"
        buckets: dict[str, list[tuple[str, str, float]]] = {}
        # buckets key = the representative canonical label for the group

        for model_id, canon, score in normed:
            placed = False
            for bucket_label in list(buckets.keys()):
                dist, _ = lineage_distance(
                    canon, bucket_label,
                    self.canonical_table, self.norm_map
                )
                if 0 <= dist <= self.max_merge_distance:
                    # Merge: upgrade bucket key to the deeper label
                    deeper = deeper_label(
                        canon, bucket_label,
                        self.canonical_table, self.norm_map
                    )
                    if deeper != bucket_label:
                        buckets[deeper] = buckets.pop(bucket_label)
                    buckets[deeper].append((model_id, canon, score))
                    placed = True
                    break
            if not placed:
                buckets[canon] = [(model_id, canon, score)]

        # Step 3: compute weighted confidence per bucket
        votes: dict[str, float] = {}
        for bucket_label, preds in buckets.items():
            total = 0.0
            for model_id, _canon, score in preds:
                weight = self.model_weights.get(model_id, 1.0)
                # M2 boost: if M2 predicts within its known label set
                if model_id == "m2" and _canon in self.m2_known_labels:
                    weight *= self.m2_boost
                total += weight * score
            votes[bucket_label] = total

        # Step 4: pick winner
        best_label = max(votes, key=lambda k: votes[k])
        # Normalise confidence to [0, 1] by dividing by max possible weight
        max_weight = sum(self.model_weights.values()) * max(1.0, self.m2_boost)
        confidence = votes[best_label] / max_weight

        is_uncertain = confidence < self.min_confidence

        return FusionResult(
            label=UNCERTAIN_LABEL if is_uncertain else best_label,
            confidence=round(confidence, 4),
            is_uncertain=is_uncertain,
            votes={k: round(v / max_weight, 4) for k, v in votes.items()},
            per_model=per_model_label,
            raw_predictions=predictions,
        )

    # ------------------------------------------------------------------
    # Batch fusion from a DataFrame
    # ------------------------------------------------------------------

    def fuse_dataframe(
        self,
        df: pd.DataFrame,
        sample_col: str = "sample_id",
        model_col: str = "model",
        label_col: str = "label",
        score_col: str = "score",
    ) -> pd.DataFrame:
        """
        Fuse a DataFrame with one row per (sample, model) prediction.

        Expected columns: sample_id | model | label | score

        Returns a DataFrame with one row per sample:
            sample_id | label | confidence | is_uncertain | m1_label | m2_label | m3_label | votes
        """
        rows = []
        for sample_id, group in df.groupby(sample_col):
            preds = [
                ModelPrediction(
                    model_id=row[model_col],
                    raw_label=row[label_col],
                    raw_score=row[score_col],
                )
                for _, row in group.iterrows()
            ]
            result = self.fuse(preds)
            rows.append({
                sample_col:      sample_id,
                "label":         result.label,
                "confidence":    result.confidence,
                "is_uncertain":  result.is_uncertain,
                "m1_label":      result.per_model.get("m1", "—"),
                "m2_label":      result.per_model.get("m2", "—"),
                "m3_label":      result.per_model.get("m3", "—"),
                "votes":         result.votes,
            })

        return pd.DataFrame(rows)
