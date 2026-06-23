"""
score_normaliser.py
===================
Makes confidence scores comparable across models.

Each model has its own score range and distribution.
A 0.7 from M1 is not the same certainty as 0.7 from M3.

Two normalisation strategies are provided:
  - MinMaxNormaliser  : scales scores to [0, 1] using per-model min/max
  - RankNormaliser    : converts scores to percentile ranks (more robust
                        to outliers, recommended when score distributions
                        are skewed or unknown)

Usage
-----
    norm = RankNormaliser()
    norm.fit(model_id="m1", scores=[0.3, 0.5, 0.7, 0.9, ...])
    normalised = norm.transform(model_id="m1", score=0.7)
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseNormaliser:
    def fit(self, model_id: str, scores: list[float]) -> None:
        raise NotImplementedError

    def transform(self, model_id: str, score: float) -> float:
        raise NotImplementedError

    def fit_transform(self, model_id: str, scores: list[float]) -> list[float]:
        self.fit(model_id, scores)
        return [self.transform(model_id, s) for s in scores]


# ---------------------------------------------------------------------------
# Min-Max normaliser
# ---------------------------------------------------------------------------

@dataclass
class MinMaxNormaliser(BaseNormaliser):
    """
    Scales each model's scores linearly to [0, 1].
    Simple, but sensitive to the min/max values seen during fit.
    Recommended when you have a large representative sample per model.
    """
    _params: dict = field(default_factory=dict)

    def fit(self, model_id: str, scores: list[float]) -> None:
        arr = np.array(scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            hi = lo + 1e-9   # avoid division by zero for constant arrays
        self._params[model_id] = {"min": lo, "max": hi}

    def transform(self, model_id: str, score: float) -> float:
        if model_id not in self._params:
            raise ValueError(f"Model '{model_id}' not fitted. Call fit() first.")
        p = self._params[model_id]
        return float(np.clip((score - p["min"]) / (p["max"] - p["min"]), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rank (percentile) normaliser
# ---------------------------------------------------------------------------

@dataclass
class RankNormaliser(BaseNormaliser):
    """
    Converts a raw score to its percentile rank within the model's
    fitted distribution.  Robust to outliers and skew.

    Returns a value in [0, 1] representing the fraction of fitted
    scores that the given score exceeds.
    """
    _fitted: dict = field(default_factory=dict)

    def fit(self, model_id: str, scores: list[float]) -> None:
        self._fitted[model_id] = np.sort(np.array(scores, dtype=float))

    def transform(self, model_id: str, score: float) -> float:
        if model_id not in self._fitted:
            raise ValueError(f"Model '{model_id}' not fitted. Call fit() first.")
        arr = self._fitted[model_id]
        rank = float(np.searchsorted(arr, score, side="right")) / len(arr)
        return float(np.clip(rank, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fit from a predictions DataFrame
# ---------------------------------------------------------------------------

def fit_normaliser_from_df(
    normaliser: BaseNormaliser,
    df,                          # DataFrame with columns [model_id, score]
    model_col: str = "model",
    score_col: str = "score",
) -> None:
    """
    Convenience: fit a normaliser from a DataFrame that has one row
    per prediction, with a model identifier and raw score column.

    Example DataFrame columns:  model | label | score
    """
    for model_id, group in df.groupby(model_col):
        normaliser.fit(model_id, group[score_col].tolist())
