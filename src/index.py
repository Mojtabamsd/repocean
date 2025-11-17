from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.io import load_run_config
from src.metadata import load_run_metadata
from src.stream import open_h5


REQUIRED = {
    "features": "features_*.h5",
    "preds": "predictions_with_top3_scores.csv",
    "model_cfg": "model_config.yaml",
    "run_cfg": "config.yaml",
}


def build_run_index(parent_dir: str | Path) -> pd.DataFrame:
    """
    Original behaviour: one row per run directory.

    Columns:
      - run_id
      - run_dir
      - features
      - preds
      - model_cfg
      - run_cfg
    """
    parent = Path(parent_dir)
    rows = []
    for sub in parent.iterdir():
        if not sub.is_dir():
            continue
        hit = {"run_id": sub.name, "run_dir": str(sub)}
        ok = True
        for k, pat in REQUIRED.items():
            matches = list(sub.glob(pat))
            if not matches:
                ok = False
                break
            hit[k] = str(matches[0])
        if ok:
            rows.append(hit)
    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    return df


def _build_name_to_index(h5f) -> dict[str, int]:
    """
    Map image filename (basename) -> row index in H5.

    Assumes H5 has a dataset "image_names" with paths or filenames.
    """
    names = h5f["image_names"][:].astype(str)
    keys = [n.replace("\\", "/").split("/")[-1] for n in names]
    return {k: i for i, k in enumerate(keys)}


def build_group_index(
    parent_dir: str | Path,
    mode: Literal["run", "meta"] = "run",
    group_col: str = "sample_id",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Higher-level index that can return:
      - mode="run": one group per run (compatible with build_run_index)
      - mode="meta": within each run, one group per metadata value
                     in column `group_col` (e.g. sample_id, station, etc.)

    Returned columns:
      - run_id
      - group_id        (== run_id if mode="run")
      - run_dir
      - features
      - preds
      - model_cfg
      - run_cfg
      - indices         (np.ndarray of H5 row indices for this group; None for mode="run")
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        return runs  # empty DataFrame

    # --- Simple case: one group per run (backwards-compatible) ---
    if mode == "run":
        groups = runs.copy()
        groups["group_id"] = groups["run_id"]
        groups["indices"] = None
        return groups.reset_index(drop=True)

    # --- Metadata grouping: split each run by group_col (e.g. sample_id) ---
    rows = []

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        features_path = r["features"]
        run_cfg_path = r["run_cfg"]

        # Load run config + metadata to get group_col (e.g. sample_id)
        cfg = load_run_config(run_cfg_path)
        input_path = cfg.get("input_path")
        if not input_path:
            # No metadata path in config; skip this run for meta grouping
            continue

        meta = load_run_metadata(input_path)
        if meta.empty or group_col not in meta.columns:
            # No usable metadata / column; skip
            continue

        # Open H5 and build filename -> index mapping
        with open_h5(features_path) as h5f:
            name_to_idx = _build_name_to_index(h5f)

        # Build a (idx, group) table by matching metadata filenames to H5 rows
        idxs = []
        groups = []
        for img, row in meta.iterrows():
            i = name_to_idx.get(img)
            if i is not None:
                idxs.append(i)
                groups.append(row[group_col])

        if not idxs:
            # No overlap between metadata and H5 for this run
            if not drop_empty:
                # Optionally keep a "dummy" group row
                rows.append({
                    "run_id": run_id,
                    "group_id": None,
                    "run_dir": r["run_dir"],
                    "features": features_path,
                    "preds": r["preds"],
                    "model_cfg": r["model_cfg"],
                    "run_cfg": run_cfg_path,
                    "indices": np.array([], dtype=np.int64),
                })
            continue

        dfm = pd.DataFrame({"idx": idxs, "group_id": groups})

        for g_val, sub in dfm.groupby("group_id", dropna=False):
            grp_idx = np.sort(sub["idx"].values.astype(np.int64))
            if grp_idx.size == 0 and drop_empty:
                continue

            rows.append({
                "run_id": run_id,
                "group_id": str(g_val),
                "run_dir": r["run_dir"],
                "features": features_path,
                "preds": r["preds"],
                "model_cfg": r["model_cfg"],
                "run_cfg": run_cfg_path,
                "indices": grp_idx,
            })

    if not rows:
        return pd.DataFrame(columns=list(runs.columns) + ["group_id", "indices"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["run_id", "group_id"], na_position="last").reset_index(drop=True)
    return out
