from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.index import build_run_index
from src.utils.io import load_run_config
from src.metadata import load_run_metadata
from src.stream import load_predictions_map
from src.visual.tools import shorten_label


def run_class_mix_by_sample(
    parent_dir: str,
    out_dir: str,
    group_col: str = "sample_id",
    save_plots: bool = True,
    top_k_classes: int = 30,
) -> Dict[str, str]:
    """
    Summarise predicted class distribution per sample (e.g. per sample_id)
    and globally.

    Per-sample output (class_mix_per_sample.csv):
        run_id, sample_id, class, count, frac

    Per-sample summary (class_mix_per_sample_summary.csv):
        run_id, sample_id, n_images, n_classes

    Global output (class_mix_global.csv):
        class, total_count, frac_global, n_samples_present

    Optionally:
        - global_class_bar.png
        - heatmap_sample_class.png (samples × classes, fractions for top-K classes)
    """
    parent = Path(parent_dir)
    out_root = Path(out_dir) / "class_mix"
    out_root.mkdir(parents=True, exist_ok=True)

    runs = build_run_index(parent)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    all_rows: List[pd.DataFrame] = []

    # ----------------------------------------------------------------------
    # 1. Collect per-image info: [run_id, image_name, sample_id, pred1_label]
    # ----------------------------------------------------------------------
    for _, r in runs.iterrows():
        run_id = r["run_id"]
        cfg = load_run_config(r["run_cfg"])
        input_path = cfg.get("input_path", "")

        if not input_path:
            print(f"[WARN] run {run_id}: no input_path in config; skipping.")
            continue

        # Metadata with sample_id (indexed by image_name)
        meta = load_run_metadata(
            input_path,
            cols=[group_col]
        )
        if meta.empty or group_col not in meta.columns:
            print(f"[WARN] run {run_id}: no '{group_col}' in metadata; skipping.")
            continue

        # Predictions (indexed by image_name)
        preds = load_predictions_map(
            r["preds"],
            cols=["Image Name", "Top-1 Predicted Label"],
        )

        # Join on image_name index
        preds.index = preds.index.str.replace("\\", "/")
        df = preds.join(meta[[group_col]], how="inner")
        if df.empty:
            print(f"[WARN] run {run_id}: no overlap between preds and metadata; skipping.")
            continue

        # df = df.reset_index().rename(columns={"index": "image_name"})
        df = df.rename(columns={"pred1_label": "class"})
        df = df.dropna(subset=[group_col, "class"])
        if df.empty:
            continue

        df["run_id"] = run_id
        all_rows.append(df[["run_id", "image_name", group_col, "class"]])

    if not all_rows:
        raise RuntimeError("No data collected (check metadata, preds, and sample_col).")

    data = pd.concat(all_rows, ignore_index=True)

    # ----------------------------------------------------------------------
    # 2. Per-sample class distribution
    # ----------------------------------------------------------------------
    grp = data.groupby(["run_id", group_col, "class"])
    counts = grp.size().rename("count").reset_index()

    # frac within each (run_id, sample_id)
    counts["frac"] = counts["count"] / counts.groupby(["run_id", group_col])["count"].transform("sum")

    # per-sample summary: number of images, number of classes
    per_sample_summary = (
        counts.groupby(["run_id", group_col])
        .agg(
            n_images=("count", "sum"),
            n_classes=("class", "nunique"),
        )
        .reset_index()
    )

    # ----------------------------------------------------------------------
    # 3. Global class distribution
    # ----------------------------------------------------------------------
    global_counts = (
        data.groupby("class")
        .size()
        .rename("total_count")
        .reset_index()
    )
    total_images = float(global_counts["total_count"].sum())
    global_counts["frac_global"] = global_counts["total_count"] / max(total_images, 1.0)

    # in how many (run_id, sample_id) each class appears
    sample_keys = data[["run_id", group_col, "class"]].drop_duplicates()
    sample_presence = (
        sample_keys.groupby("class")
        .size()
        .rename("n_samples_present")
        .reset_index()
    )

    global_summary = global_counts.merge(sample_presence, on="class", how="left")

    # ----------------------------------------------------------------------
    # 4. Save CSVs
    # ----------------------------------------------------------------------
    per_sample_csv = out_root / "class_mix_per_sample.csv"
    per_sample_summary_csv = out_root / "class_mix_per_sample_summary.csv"
    global_csv = out_root / "class_mix_global.csv"

    counts.to_csv(per_sample_csv, index=False)
    per_sample_summary.to_csv(per_sample_summary_csv, index=False)
    global_summary.to_csv(global_csv, index=False)

    # ----------------------------------------------------------------------
    # 5. Plots (optional)
    # ----------------------------------------------------------------------
    if save_plots:
        # 5a. Global bar chart
        try:
            gs = global_summary.sort_values("total_count", ascending=False)
            plt.figure(figsize=(max(6, len(gs) * 0.25), 4))
            labels = [shorten_label(c) for c in gs["class"].astype(str).tolist()]
            plt.bar(np.arange(len(gs)), gs["total_count"].values)
            plt.xticks(np.arange(len(gs)), labels, rotation=90, fontsize=8)
            plt.ylabel("Total images")
            plt.title("Global predicted class distribution")
            plt.tight_layout()
            plt.savefig(out_root / "global_class_bar.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot global class bar: {e}")

        # 5b. Heatmap: samples × top-K classes
        try:
            # pick top-K classes globally
            top_classes = (
                global_summary.sort_values("total_count", ascending=False)["class"]
                .astype(str)
                .head(top_k_classes)
                .tolist()
            )

            # Create a pivot table of frac for (run_id + sample_id) vs class
            counts["sample_key"] = counts["run_id"].astype(str) + "|" + counts[sample_col].astype(str)
            pivot = counts[counts["class"].isin(top_classes)].pivot_table(
                index="sample_key",
                columns="class",
                values="frac",
                fill_value=0.0,
            )

            if not pivot.empty:
                # sort samples and classes for nicer view
                pivot = pivot.sort_index()
                pivot = pivot[top_classes]

                # shorten labels for display
                sample_labels = [shorten_label(k) for k in pivot.index.tolist()]
                class_labels = [shorten_label(c) for c in pivot.columns.astype(str).tolist()]

                plt.figure(figsize=(max(8, pivot.shape[1] * 0.3), max(6, pivot.shape[0] * 0.12)))
                plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
                plt.colorbar(label="Fraction per sample")
                plt.xticks(
                    ticks=np.arange(pivot.shape[1]),
                    labels=class_labels,
                    rotation=90,
                    fontsize=7,
                )
                plt.yticks(
                    ticks=np.arange(pivot.shape[0]),
                    labels=sample_labels,
                    fontsize=7,
                )
                plt.title(f"Class mix per sample (top {top_k_classes} classes)")
                plt.xlabel("Class")
                plt.ylabel("Sample (run|sample_id)")
                plt.tight_layout()
                plt.savefig(out_root / "heatmap_sample_class.png", dpi=180)
                plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot sample-class heatmap: {e}")

    return {
        "per_sample_csv": str(per_sample_csv),
        "per_sample_summary_csv": str(per_sample_summary_csv),
        "global_csv": str(global_csv),
    }
