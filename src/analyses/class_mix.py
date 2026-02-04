from __future__ import annotations
from pathlib import Path
from typing import Dict, List

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
        run_id, group_col (e.g. sample_id), class, count, frac

    Per-sample summary (class_mix_per_sample_summary.csv):
        run_id, group_col, n_images, n_classes

    Global output (class_mix_global.csv):
        class, total_count, frac_global, n_samples_present

    Plots (if save_plots=True):
        - global_class_bar.png
            Top-K classes (by total_count) plus an "Other" bucket.
        - global_class_bar_log.png
            All classes, log-scale y-axis.
        - heatmap_sample_class.png
            Samples × classes (fractions for top-K classes).
        - per_sample_hist_n_images.png
            Histogram of number of images per sample.
        - per_sample_hist_n_classes.png
            Histogram of number of classes per sample.
        - global_class_presence_bar.png
            Top-K classes vs number of samples they appear in.
    """
    parent = Path(parent_dir)
    out_root = Path(out_dir) / "class_mix"
    out_root.mkdir(parents=True, exist_ok=True)

    runs = build_run_index(parent)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    all_rows: List[pd.DataFrame] = []

    # ----------------------------------------------------------------------
    # 1. Collect per-image info: [run_id, image_name, group_col, class]
    # ----------------------------------------------------------------------
    for _, r in runs.iterrows():
        run_id = r["run_id"]
        cfg = load_run_config(r["run_cfg"])
        input_path = cfg.get("input_path", "")

        if not input_path:
            print(f"[WARN] run {run_id}: no input_path in config; skipping.")
            continue

        # Metadata with group_col (e.g. sample_id), indexed by image_name
        meta = load_run_metadata(
            input_path,
            cols=[group_col],
        )
        if meta.empty or group_col not in meta.columns:
            print(f"[WARN] run {run_id}: no '{group_col}' in metadata; skipping.")
            continue

        # Predictions (indexed by image_name)
        preds = load_predictions_map(
            r["preds"],
            cols=["Image Name", "Top-1 Predicted Label"],
        )

        # Normalise paths (slash vs backslash)
        preds.index = preds.index.to_series().astype(str).str.replace("\\", "/")
        meta.index = meta.index.to_series().astype(str).str.replace("\\", "/")

        # Join on image_name index
        df = preds.join(meta[[group_col]], how="inner")
        if df.empty:
            print(f"[WARN] run {run_id}: no overlap between preds and metadata; skipping.")
            continue

        df = df.rename(columns={"pred1_label": "class"})
        df = df.dropna(subset=[group_col, "class"])
        if df.empty:
            continue

        df["run_id"] = run_id
        # df = df.reset_index().rename(columns={"index": "image_name"})
        all_rows.append(df[["run_id", "image_name", group_col, "class"]])

    if not all_rows:
        raise RuntimeError("No data collected (check metadata, preds, and group_col).")

    data = pd.concat(all_rows, ignore_index=True)

    # ----------------------------------------------------------------------
    # 2. Per-sample class distribution
    # ----------------------------------------------------------------------
    grp = data.groupby(["run_id", group_col, "class"])
    counts = grp.size().rename("count").reset_index()

    # frac within each (run_id, group_col)
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

    # in how many (run_id, group_col) each class appears
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
        # 5a. Global bar chart (top-K + "Other")
        try:
            gs = global_summary.sort_values("total_count", ascending=False).reset_index(drop=True)

            # Take top-K and group rest into "Other"
            top_k = min(top_k_classes, len(gs))
            head = gs.iloc[:top_k].copy()

            if len(gs) > top_k:
                other_count = gs["total_count"].iloc[top_k:].sum()
                other_frac = other_count / max(total_images, 1.0)
                other_samples = gs["n_samples_present"].iloc[top_k:].sum()
                head = pd.concat(
                    [
                        head,
                        pd.DataFrame(
                            {
                                "class": ["__OTHER__"],
                                "total_count": [other_count],
                                "frac_global": [other_frac],
                                "n_samples_present": [other_samples],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            labels = [shorten_label(str(c)) for c in head["class"].tolist()]
            x = np.arange(len(head))

            plt.figure(figsize=(max(6, len(head) * 0.4), 4))
            plt.bar(x, head["total_count"].values)
            plt.xticks(x, labels, rotation=90, fontsize=8)
            plt.ylabel("Total images")
            plt.title(f"Global predicted class distribution (top {top_k_classes} + Other)")
            plt.tight_layout()
            plt.savefig(out_root / "global_class_bar.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot global class bar: {e}")

        # 5b. Global bar chart (all classes, log-scale)
        try:
            gs = global_summary.sort_values("total_count", ascending=False)
            labels_all = [shorten_label(str(c)) for c in gs["class"].tolist()]
            x_all = np.arange(len(gs))

            plt.figure(figsize=(max(6, len(gs) * 0.25), 4))
            plt.bar(x_all, gs["total_count"].values)
            plt.yscale("log")
            plt.xticks(x_all, labels_all, rotation=90, fontsize=7)
            plt.ylabel("Total images (log scale)")
            plt.title("Global predicted class distribution (log scale)")
            plt.tight_layout()
            plt.savefig(out_root / "global_class_bar_log.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot global class bar (log): {e}")

        # 5c. Heatmap: samples × top-K classes (fractions)
        try:
            # pick top-K classes globally
            top_classes = (
                global_summary.sort_values("total_count", ascending=False)["class"]
                .astype(str)
                .head(top_k_classes)
                .tolist()
            )

            # create a sample key
            counts["sample_key"] = counts["run_id"].astype(str) + "|" + counts[group_col].astype(str)

            pivot = counts[counts["class"].isin(top_classes)].pivot_table(
                index="sample_key",
                columns="class",
                values="frac",
                fill_value=0.0,
            )

            if not pivot.empty:
                pivot = pivot.sort_index()
                # Ensure columns in desired order
                pivot = pivot[[c for c in top_classes if c in pivot.columns]]

                sample_labels = [shorten_label(k) for k in pivot.index.tolist()]
                class_labels = [shorten_label(str(c)) for c in pivot.columns.tolist()]

                plt.figure(
                    figsize=(
                        max(8, pivot.shape[1] * 0.3),
                        max(6, pivot.shape[0] * 0.12),
                    )
                )
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
                plt.ylabel(f"Sample (run|{group_col})")
                plt.tight_layout()
                plt.savefig(out_root / "heatmap_sample_class.png", dpi=180)
                plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot sample-class heatmap: {e}")

        # 5d. Per-sample histograms (n_images, n_classes)
        try:
            # Histogram of number of images per sample
            plt.figure(figsize=(6, 4))
            plt.hist(per_sample_summary["n_images"].values, bins=100)
            plt.xlabel("Number of images per sample")
            plt.ylabel("Count of samples")
            plt.title("Distribution of images per sample")
            plt.tight_layout()
            plt.savefig(out_root / "per_sample_hist_n_images.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot per-sample n_images histogram: {e}")

        try:
            # Histogram of number of classes per sample
            plt.figure(figsize=(6, 4))
            plt.hist(per_sample_summary["n_classes"].values, bins=global_counts.shape[0])
            plt.xlabel("Number of classes per sample")
            plt.ylabel("Count of samples")
            plt.title("Distribution of classes per sample")
            plt.tight_layout()
            plt.savefig(out_root / "per_sample_hist_n_classes.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot per-sample n_classes histogram: {e}")

        # 5e. Global class presence bar (top-K)
        try:
            gs_presence = global_summary.sort_values("n_samples_present", ascending=False).head(top_k_classes)
            labels = [shorten_label(str(c)) for c in gs_presence["class"].tolist()]
            x = np.arange(len(gs_presence))

            plt.figure(figsize=(max(6, len(gs_presence) * 0.4), 4))
            plt.bar(x, gs_presence["n_samples_present"].values)
            plt.xticks(x, labels, rotation=90, fontsize=8)
            plt.ylabel(f"Number of samples with class (by {group_col})")
            plt.title(f"Class presence across samples (top {top_k_classes})")
            plt.tight_layout()
            plt.savefig(out_root / "global_class_presence_bar.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot global class presence bar: {e}")

    return {
        "per_sample_csv": str(per_sample_csv),
        "per_sample_summary_csv": str(per_sample_summary_csv),
        "global_csv": str(global_csv),
    }
