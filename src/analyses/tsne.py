from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.index import build_group_index
from src.analyses.common import collect_group_samples


def run_tsne(
    parent_dir: str,
    out_dir: str,
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

    out_root = Path(out_dir) / "tsne"
    out_root.mkdir(parents=True, exist_ok=True)

    X_all, meta_all = collect_group_samples(
        groups=groups,
        group_mode=group_mode,
        sample_per_group=sample_per_group,
        rng=rng,
        attach_preds=True,
    )

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
    out_csv_path = out_root / f"tsne_{group_mode}.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    # Optional plot
    save_plot = True
    if save_plot:
        color_key = "group_id" if group_mode == "meta" else "run_id"
        cats = out_df[color_key].astype("category")

        codes = cats.cat.codes.to_numpy()
        n_cats = len(cats.cat.categories)
        # cmap = plt.get_cmap("tab20", n_cats)
        cmap = plt.get_cmap("gist_ncar", n_cats)

        plt.figure(figsize=(20, 18))
        sc = plt.scatter(
            out_df["tsne_x"],
            out_df["tsne_y"],
            s=3,
            alpha=0.7,
            c=codes,
            cmap=cmap,
            vmin=0,
            vmax=n_cats - 1,
        )

        cbar = plt.colorbar(sc, ticks=np.arange(n_cats))
        cbar.set_label(color_key)
        cbar.set_ticklabels(cats.cat.categories)

        title_label = f"t-SNE grouped by {color_key}"
        plt.title(title_label)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(out_root / f"tsne_{group_mode}.png", dpi=400)
        plt.close()

    return out_df
