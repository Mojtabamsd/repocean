from __future__ import annotations
from typing import Iterable, Literal, Sequence, Tuple

from pathlib import Path
import numpy as np
import pandas as pd

from src.stream import (
    open_h5,
    get_h5_shapes,
    sample_indices_uniform,
    read_rows_by_indices,
    load_predictions_map,
)


def collect_group_samples(
    groups: pd.DataFrame,
    group_mode: Literal["run", "meta"] = "run",
    sample_per_group: int = 2000,
    rng: np.random.Generator | None = None,
    preds_cols: Sequence[str] = (
        "Image Name",
        "Top-1 Predicted Label",
        "Top-1 Confidence Score",
    ),
    attach_preds: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generic utility to sample features and metadata per group.

    Parameters
    ----------
    groups : DataFrame
        Output of build_group_index(...).
        Must have columns:
          - run_id, group_id, features, preds, indices (indices may be None for mode='run')
    group_mode : {"run", "meta"}
        "run"  -> each row == one run; all rows in H5 are eligible.
        "meta" -> each row == one (run_id, group_id) subset; use `indices` column.
    sample_per_group : int
        Max number of rows to sample per group.
    rng : np.random.Generator, optional
        If None, a default generator is created.
    preds_cols : list of str
        Column names to load from predictions CSV.
    attach_preds : bool
        If True, joins predictions metadata (pred1_label, pred1_conf).

    Returns
    -------
    X_all : np.ndarray
        Stacked feature matrix from all sampled groups.
    meta_all : pd.DataFrame
        Metadata for each sampled row, including:
          - run_id
          - group_id
          - image_name
          - (optional) pred1_label, pred1_conf if attach_preds=True
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_parts: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        group_id = g.get("group_id", run_id)
        features_path = g["features"]
        preds_path = g.get("preds", None)

        with open_h5(features_path) as h5f:
            n, _ = get_h5_shapes(h5f)
            if n == 0:
                continue

            # Decide which indices belong to this group
            if group_mode == "run":
                idx_all = np.arange(n, dtype=np.int64)
            else:
                # group_mode == "meta": subset of indices
                idx_group = g.get("indices", None)
                if idx_group is None:
                    continue
                idx_all = np.asarray(idx_group, dtype=np.int64)
                if idx_all.size == 0:
                    continue

            # Sample within this group
            k = min(sample_per_group, idx_all.size)
            if k <= 0:
                continue

            rel_idx = sample_indices_uniform(idx_all.size, k, rng)
            sel_idx = np.sort(idx_all[rel_idx])

            part = read_rows_by_indices(h5f, sel_idx)
            X = part["features"]
            names = part["image_names"]

        # Build metadata row for each sampled image
        meta_dict = {
            "run_id": run_id,
            "group_id": group_id if group_mode == "meta" else run_id,
            "image_name": names,
        }

        if attach_preds and preds_path is not None:
            preds = load_predictions_map(preds_path, cols=list(preds_cols))
            # Expecting specific column renames; adapt if you want other preds_cols
            re = preds.reindex(names)
            meta_dict["pred1_label"] = re["pred1_label"].values
            meta_dict["pred1_conf"] = re["pred1_conf"].values

        meta = pd.DataFrame(meta_dict)

        X_parts.append(X)
        metas.append(meta)

    if not X_parts:
        raise RuntimeError("No features collected (check grouping and sampling settings).")

    X_all = np.vstack(X_parts)
    meta_all = pd.concat(metas, ignore_index=True)
    return X_all, meta_all
