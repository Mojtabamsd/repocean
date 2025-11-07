from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

from src.index import build_run_index
from src.stream import open_h5, get_h5_shapes, sample_indices_uniform, read_rows_by_indices, load_predictions_map


def run_tsne(parent_dir: str,
             out_csv: str,
             out_png: str | None = None,
             sample_per_run: int = 2000,
             pca_dim: int = 50,
             perplexity: float = 30.0,
             learning_rate: float = 200.0,
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under {parent_dir}")

    # collect small sample per run
    X_parts, metas = [], []
    for _, r in runs.iterrows():
        with open_h5(r["features"]) as h5f:
            n, _ = get_h5_shapes(h5f)
            if n == 0:
                continue
            idx = sample_indices_uniform(n, sample_per_run, rng)
            part = read_rows_by_indices(h5f, idx)
            X = part["features"]
            names = part["image_names"]

        # minimal join for coloring
        preds = load_predictions_map(r["preds"], cols=["Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"])
        meta = pd.DataFrame({
            "run_id": r["run_id"],
            "image_name": names,
            "pred1_label": preds.reindex(names)["pred1_label"].values,
            "pred1_conf": preds.reindex(names)["pred1_conf"].values,
        })

        X_parts.append(X)
        metas.append(meta)

    X_all = np.vstack(X_parts)
    meta_all = pd.concat(metas, ignore_index=True)

    # PCA first for speed/stability
    if X_all.shape[1] > pca_dim:
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
        X_pca = ipca.fit_transform(X_all)
    else:
        X_pca = X_all

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=seed,
        n_iter=1000,
        verbose=1
    )
    emb = tsne.fit_transform(X_pca)
    out_df = meta_all.copy()
    out_df["tsne_x"] = emb[:, 0]
    out_df["tsne_y"] = emb[:, 1]

    # save small artifact
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    if out_png:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        runs_cat = out_df["run_id"].astype("category")
        plt.scatter(out_df["tsne_x"], out_df["tsne_y"], s=3, alpha=0.7, c=runs_cat.cat.codes)
        plt.title("t-SNE of sampled features across runs")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()

    return out_df
