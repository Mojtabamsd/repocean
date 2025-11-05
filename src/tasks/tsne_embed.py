from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

from src.index import build_run_index
from src.stream import open_h5, get_h5_shapes, sample_indices_uniform, read_rows_by_indices, load_predictions_map


def tsne_across_runs(parent_dir: str,
                     out_csv: str,
                     out_png: Optional[str] = None,
                     sample_per_run: int = 2000,
                     pca_dim: int = 50,
                     random_state: int = 42,
                     perplexity: float = 30.0,
                     learning_rate: float = 200.0):
    """
    Global t-SNE over many runs, with memory-efficient sampling and no feature duplication.
    - Samples up to 'sample_per_run' rows per run.
    - Applies Incremental PCA to D -> pca_dim.
    - Runs t-SNE on the concatenated PCA features.
    Saves a small CSV (embedding + minimal metadata) and optional PNG plot.
    """
    rng = np.random.default_rng(random_state)
    index = build_run_index(parent_dir)
    if index.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    # ---- Pass 1: collect samples (small, RAM-friendly) ----
    samples = []
    meta_rows = []
    for _, row in index.iterrows():
        run_id = row["run_id"]
        feats_path = row["features"]
        preds_path = row["preds"]

        with open_h5(feats_path) as h5f:
            n, d = get_h5_shapes(h5f)
            if n == 0:
                continue
            idx = sample_indices_uniform(n, sample_per_run, rng)
            part = read_rows_by_indices(h5f, idx)
            X = part["features"]  # shape [k, D]
            names = part["image_names"]

        # attach minimal labels (optional)
        # Only join on sampled image_names to avoid loading large tables twice
        preds_df = load_predictions_map(preds_path, cols=[
            "Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"
        ])
        # map
        pred1_label = preds_df.reindex(names)["pred1_label"].values
        pred1_conf  = preds_df.reindex(names)["pred1_conf"].values

        samples.append(X)
        meta_rows.append(pd.DataFrame({
            "run_id": run_id,
            "image_name": names,
            "pred1_label": pred1_label,
            "pred1_conf": pred1_conf
        }))

    if not samples:
        raise RuntimeError("No features sampled; check your runs.")

    X_all = np.vstack(samples)  # small concatenated sample only
    meta_all = pd.concat(meta_rows, ignore_index=True)

    # ---- Pass 2: PCA (incremental, but here sample is already small) ----
    if X_all.shape[1] > pca_dim:
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
        X_pca = ipca.fit_transform(X_all)
    else:
        X_pca = X_all

    # ---- Pass 3: t-SNE on PCA features ----
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
        n_iter=1000,
        verbose=1
    )
    emb = tsne.fit_transform(X_pca)

    # ---- Save output (tiny CSV; no duplication of features) ----
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = meta_all.copy()
    out_df["tsne_x"] = emb[:, 0]
    out_df["tsne_y"] = emb[:, 1]
    out_df.to_csv(out_csv_path, index=False)

    # ---- Optional quick plot (matplotlib) ----
    if out_png:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        # light plotting: color by run_id count or by pred1_label if not too many
        # Here we color by run to visualize domain clustering
        runs = out_df["run_id"].astype("category")
        plt.scatter(out_df["tsne_x"], out_df["tsne_y"], s=3, alpha=0.7, c=runs.cat.codes)
        plt.title("t-SNE of sampled features across runs")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
