from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

from src.index import build_group_index
from src.stream import (
    open_h5,
    get_h5_shapes,
    sample_indices_uniform,
    read_rows_by_indices,
    load_predictions_map,
)


def run_tsne(
    parent_dir: str,
    out_csv: str,
    out_png: str | None = None,
    group_mode: str = "run",          # "run" | "meta"
    group_col: str = "sample_id",     # used when group_mode == "meta"
    sample_per_group: int = 2000,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run t-SNE on a sample of features, grouped either by run or by metadata.

    group_mode = "run":
        - One group per run (backwards-compatible behaviour).
        - group_id == run_id, and all rows in the H5 are eligible for sampling.

    group_mode = "meta":
        - Within each run, split by metadata column `group_col` (e.g. sample_id).
        - Each (run_id, group_id) is a logical group, with `indices` into the H5.
        - We sample up to `sample_per_group` rows per group.
    """
    rng = np.random.default_rng(seed)

    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
    )
    if groups.empty:
        raise RuntimeError(f"No groups found under {parent_dir} (mode={group_mode})")

    X_parts: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        group_id = g["group_id"]
        features_path = g["features"]
        preds_path = g["preds"]

        with open_h5(features_path) as h5f:
            n, _ = get_h5_shapes(h5f)
            if n == 0:
                continue

            # Decide which indices belong to this group
            if group_mode == "run":
                # Use all rows in the H5, like before
                idx_all = np.arange(n, dtype=np.int64)
            else:
                # group_mode == "meta": we have a subset of indices for this group
                idx_group = g["indices"]
                if idx_group is None:
                    continue
                idx_all = np.asarray(idx_group, dtype=np.int64)
                if idx_all.size == 0:
                    continue

            # Sample within this group
            k = min(sample_per_group, idx_all.size)
            if k <= 0:
                continue

            # sample_indices_uniform works on [0, ..., len-1], so map back
            rel_idx = sample_indices_uniform(idx_all.size, k, rng)
            sel_idx = np.sort(idx_all[rel_idx])

            part = read_rows_by_indices(h5f, sel_idx)
            X = part["features"]
            names = part["image_names"]

        # Minimal join for coloring / metadata
        preds = load_predictions_map(
            preds_path,
            cols=["Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"],
        )

        meta = pd.DataFrame({
            "run_id": run_id,
            "group_id": group_id if group_mode == "meta" else run_id,
            "image_name": names,
            "pred1_label": preds.reindex(names)["pred1_label"].values,
            "pred1_conf": preds.reindex(names)["pred1_conf"].values,
        })

        X_parts.append(X)
        metas.append(meta)

    if not X_parts:
        raise RuntimeError("No features collected for t-SNE (check grouping and sampling settings).")

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

    # Save CSV
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    # Optional plot
    if out_png:
        import matplotlib.pyplot as plt

        color_key = "group_id" if group_mode == "meta" else "run_id"
        cats = out_df[color_key].astype("category")
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            out_df["tsne_x"],
            out_df["tsne_y"],
            s=3,
            alpha=0.7,
            c=cats.cat.codes,
            cmap="tab20"
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(color_key)
        cbar.set_ticks(np.arange(len(cats.cat.categories)))
        cbar.set_ticklabels(cats.cat.categories.to_list())

        title_label = f"t-SNE grouped by {color_key}"
        plt.title(title_label)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()

    return out_df
