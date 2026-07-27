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
    # "features": "features_contrastive20250326162033.h5",
    "preds": "predictions_with_top3_scores.csv",
    # "features": "features_contrastive20250326162033_s.h5",
    # "preds": "predictions_with_top3_scores_s.csv",
    "model_cfg": "model_config.yaml",
    "run_cfg": "config.yaml",
}


def build_run_index(parent_dir: str | Path) -> pd.DataFrame:
    parent = Path(parent_dir)
    rows = []

    def check_run(run_dir: Path):
        hit = {"run_id": run_dir.name, "run_dir": str(run_dir)}
        for k, pat in REQUIRED.items():
            matches = list(run_dir.glob(pat))
            if not matches:
                return None
            hit[k] = str(matches[0])
        return hit

    # ---- Case 1: parent_dir itself is a run directory
    hit = check_run(parent)
    if hit is not None:
        rows.append(hit)

    # ---- Case 2: parent_dir contains run subdirectories
    else:
        for sub in parent.iterdir():
            if not sub.is_dir():
                continue
            hit = check_run(sub)
            if hit is not None:
                rows.append(hit)

    return (
        pd.DataFrame(rows)
        .sort_values("run_id")
        .reset_index(drop=True)
    )


def _build_name_to_index(h5f) -> dict[str, int]:
    """
    Map image filename (basename) -> row index in H5.

    Assumes H5 has a dataset "image_names" with paths or filenames.
    """
    names = h5f["image_names"][:].astype(str)
    # keys = [n.replace("\\", "/").split("/")[-1] for n in names]
    keys = [n.replace("\\", "/") for n in names]
    return {k: i for i, k in enumerate(keys)}


def _format_depth_bin(val: float, bin_size: float | int) -> str:
    if pd.isna(val):
        return "nan"
    lo = int(np.floor(float(val) / float(bin_size)) * float(bin_size))
    hi = int(lo + float(bin_size))
    return f"{lo:03d}-{hi:03d}m"


def _load_pred_name_to_pos(preds_path: str | Path, name_col: str | None = None) -> dict[str, int]:
    df = pd.read_csv(preds_path)
    if name_col is None:
        for col in ("image_name", "image", "Image Name", "file_name", "img_name"):
            if col in df.columns:
                name_col = col
                break
        else:
            name_col = df.columns[0]
    names = df[name_col].astype(str).str.replace("\\", "/", regex=False)
    name_to_pos: dict[str, int] = {}
    for pos, n in enumerate(names):
        if n not in name_to_pos:      # keep first occurrence if dup filenames
            name_to_pos[n] = pos
    return name_to_pos


def _aligned_h5_pred_indices(features_path, preds_path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (h5_idx, pred_idx) parallel arrays for images present in BOTH
    the H5 file and the predictions CSV, sorted by h5_idx.
    """
    with open_h5(features_path) as h5f:
        name_to_h5idx = _build_name_to_index(h5f)
    name_to_predpos = _load_pred_name_to_pos(preds_path)

    common = name_to_h5idx.keys() & name_to_predpos.keys()
    if not common:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pairs = sorted((name_to_h5idx[n], name_to_predpos[n]) for n in common)
    h5_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    pred_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    return h5_idx, pred_idx


def build_group_index(
    parent_dir: str | Path,
    mode: Literal["run", "meta"] = "run",
    group_col: str = "sample_id",
    drop_empty: bool = True,
    depth_bin_size: float | int | None = None,
    profile_col: str = "sample_id",
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
        return runs

    if mode == "run":
        groups = runs.copy()
        groups["group_id"] = groups["run_id"]

        indices_col, pred_indices_col, n_img_col = [], [], []
        for _, r in groups.iterrows():
            h5_idx, pred_idx = _aligned_h5_pred_indices(r["features"], r["preds"])
            indices_col.append(h5_idx)
            pred_indices_col.append(pred_idx)
            n_img_col.append(int(h5_idx.size))

        groups["indices"] = indices_col
        groups["pred_indices"] = pred_indices_col
        groups["group_lat"] = np.nan
        groups["group_lon"] = np.nan
        groups["n_images_in_group"] = n_img_col
        return groups.reset_index(drop=True)

    rows = []

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        features_path = r["features"]
        run_cfg_path = r["run_cfg"]

        cfg = load_run_config(run_cfg_path)
        input_path = cfg.get("input_path")
        if not input_path:
            continue

        need_cols = [group_col, profile_col, "sample_id", "object_lat", "object_lon"]
        need_cols = list(dict.fromkeys(need_cols))  # keep order, remove duplicates

        numeric_cols = ["object_lat", "object_lon"]
        if group_col == "object_depth_min":
            numeric_cols.append("object_depth_min")

        meta = load_run_metadata(
            input_path,
            cols=need_cols,
            numeric_cols=numeric_cols,
        )
        if meta.empty or group_col not in meta.columns:
            continue

        with open_h5(features_path) as h5f:
            name_to_idx = _build_name_to_index(h5f)
        name_to_predpos = _load_pred_name_to_pos(r["preds"])

        idxs, pred_idxs, gids, profiles, depth_bins, lats, lons = [], [], [], [], [], [], []

        # IMPORTANT: meta.index is image_name already
        for img, row in meta.iterrows():
            i = name_to_idx.get(img)
            if i is None:
                continue

            p = name_to_predpos.get(img)
            if p is None:
                continue

            raw_gid = row.get(group_col, None)

            # special case: group by profile + depth bin
            if group_col == "object_depth_min" and depth_bin_size is not None:
                prof = row.get(profile_col, row.get("sample_id", None))
                depth_val = pd.to_numeric(pd.Series([raw_gid]), errors="coerce").iloc[0]
                depth_bin = _format_depth_bin(depth_val, depth_bin_size)
                gid = f"{prof}__{depth_bin}"
            else:
                prof = row.get(profile_col, row.get("sample_id", None))
                depth_bin = np.nan
                gid = raw_gid

            idxs.append(i)
            pred_idxs.append(p)
            gids.append(gid)
            profiles.append(prof)
            depth_bins.append(depth_bin)
            lats.append(row.get("object_lat", np.nan))
            lons.append(row.get("object_lon", np.nan))

        if not idxs:
            if not drop_empty:
                rows.append({
                    "run_id": run_id,
                    "group_id": None,
                    "run_dir": r["run_dir"],
                    "features": features_path,
                    "preds": r["preds"],
                    "model_cfg": r["model_cfg"],
                    "run_cfg": run_cfg_path,
                    "indices": np.array([], dtype=np.int64),
                    "group_lat": np.nan,
                    "group_lon": np.nan,
                    "n_images_in_group": 0,
                })
            continue

        dfm = pd.DataFrame({
            "idx": np.asarray(idxs, dtype=np.int64),
            "pred_idx": np.asarray(pred_idxs, dtype=np.int64),
            "group_id": gids,
            "profile_id": profiles,
            "depth_bin": depth_bins,
            "lat": pd.to_numeric(pd.Series(lats), errors="coerce"),
            "lon": pd.to_numeric(pd.Series(lons), errors="coerce"),
        })

        for g_val, sub in dfm.groupby("group_id", dropna=False):
            sub_sorted = sub.sort_values("idx")
            grp_idx = sub_sorted["idx"].values.astype(np.int64)
            grp_pred_idx = sub_sorted["pred_idx"].values.astype(np.int64)
            if grp_idx.size == 0 and drop_empty:
                continue

            # ✅ group-level mean location (safe if missing)
            glat = float(sub["lat"].mean()) if sub["lat"].notna().any() else np.nan
            glon = float(sub["lon"].mean()) if sub["lon"].notna().any() else np.nan

            rows.append({
                "run_id": run_id,
                "group_id": str(g_val),
                "profile_id": str(sub["profile_id"].iloc[0]) if "profile_id" in sub.columns else np.nan,
                "depth_bin": str(sub["depth_bin"].iloc[0]) if "depth_bin" in sub.columns else np.nan,
                "run_dir": r["run_dir"],
                "features": features_path,
                "preds": r["preds"],
                "model_cfg": r["model_cfg"],
                "run_cfg": run_cfg_path,
                "indices": grp_idx,
                "pred_indices": grp_pred_idx,
                "group_lat": glat,
                "group_lon": glon,
                "n_images_in_group": int(grp_idx.size),
            })

    if not rows:
        return pd.DataFrame(columns=list(runs.columns) + ["group_id", "indices", "group_lat", "group_lon", "n_images_in_group"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["run_id", "group_id"], na_position="last").reset_index(drop=True)
    return out
