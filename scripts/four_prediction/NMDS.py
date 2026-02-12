import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bray_curtis_heatmap(
    df: pd.DataFrame,
    dataset_col: str = "run_id",
    label_col: str = "label",
    count_col: str = None,
    out_png: str = "bray_curtis_heatmap.png",
    figsize=(10, 8),
    cmap="viridis",
):
    """
    Compute Bray–Curtis distance between datasets and save heatmap as PNG.
    """

    # ---- Build dataset x label table ----
    if count_col is None:
        counts = (
            df.groupby([dataset_col, label_col])
              .size()
              .unstack(label_col, fill_value=0)
              .astype(float)
        )
    else:
        counts = (
            df.groupby([dataset_col, label_col])[count_col]
              .sum()
              .unstack(label_col, fill_value=0)
              .astype(float)
        )

    X = counts.to_numpy()
    n = X.shape[0]
    row_sum = X.sum(axis=1)

    # ---- Compute Bray–Curtis matrix ----
    bc = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            shared = np.minimum(X[i], X[j]).sum()
            denom = row_sum[i] + row_sum[j]
            d = 0.0 if denom == 0 else 1.0 - (2.0 * shared / denom)
            bc[i, j] = bc[j, i] = d

    bc_df = pd.DataFrame(bc, index=counts.index, columns=counts.index)

    # ---- Plot heatmap ----
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(bc_df.values, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(bc_df.index, rotation=90)
    ax.set_yticklabels(bc_df.index)

    ax.set_title("Bray–Curtis Dissimilarity Between Datasets")
    cbar = plt.colorbar(im)
    cbar.set_label("Bray–Curtis Distance (0 = identical, 1 = different)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved heatmap to {out_png}")

    return bc_df



path = r'C:\alr4\ai_predict_all\prediction_parti20260119121141'
df_name_ = r'\predictions_with_top3_scores_s.csv'
df = pd.read_csv(path + df_name_)
df["run_id"] = (
    df["Image Name"]
      .str.extract(r"(d\d{4})", expand=False)   # extract d0001, d0002, ...
)

bc_df = bray_curtis_heatmap(
    df,
    dataset_col="run_id",
    label_col="Top-1 Predicted Label",
    count_col=None,
    out_png=path + r"\bc_deployments.png"
)