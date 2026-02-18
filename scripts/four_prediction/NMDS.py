import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform


# -----------------------------
# 1) Build dataset x label table
# -----------------------------
def build_label_table(df: pd.DataFrame, dataset_col: str, label_col: str) -> pd.DataFrame:
    return (
        df.groupby([dataset_col, label_col])
          .size()
          .unstack(label_col, fill_value=0)
          .astype(float)
    )


# -----------------------------
# 2) Shannon + effective species
# -----------------------------
def shannon_and_effective(counts: pd.DataFrame) -> pd.DataFrame:
    totals = counts.sum(axis=1).replace(0, np.nan)
    p = counts.div(totals, axis=0)

    # Shannon H = -sum p log p (ignore p=0)
    p = p.replace(0, np.nan)
    H = -(p * np.log(p)).sum(axis=1).fillna(0.0)

    eff = np.exp(H)  # "effective number of species/classes"
    out = pd.DataFrame({"Shannon": H, "Effective_species": eff})
    return out


# -----------------------------
# 3) Bray–Curtis distance matrix
# -----------------------------
def bray_curtis_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    X = counts.to_numpy()
    n = X.shape[0]
    row_sum = X.sum(axis=1)

    bc = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            shared = np.minimum(X[i], X[j]).sum()
            denom = row_sum[i] + row_sum[j]
            d = 0.0 if denom == 0 else 1.0 - (2.0 * shared / denom)
            bc[i, j] = bc[j, i] = d

    return pd.DataFrame(bc, index=counts.index, columns=counts.index)


# -----------------------------
# 4) Cluster order from distance
# -----------------------------
def cluster_order_from_distance(dist_df: pd.DataFrame, method: str = "average"):
    condensed = squareform(dist_df.values, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    return Z, order


# -----------------------------
# 5) Plot: dendrogram + clustered heatmap
# -----------------------------
def save_dendrogram_and_heatmap(
    dist_df: pd.DataFrame,
    out_png: str = "bc_clustered_dendrogram.png",
    method: str = "average",
    figsize=(12, 9),
    cmap="viridis",
):
    Z, order = cluster_order_from_distance(dist_df, method=method)
    d = dist_df.iloc[order, order]

    fig = plt.figure(figsize=figsize)

    # Dendrogram (top)
    ax_d = fig.add_axes([0.12, 0.78, 0.78, 0.18])  # [left, bottom, width, height]
    dendrogram(Z, labels=d.index.tolist(), leaf_rotation=90, ax=ax_d)
    ax_d.set_yticks([])
    ax_d.set_ylabel("")
    ax_d.set_title("Hierarchical clustering (based on Bray–Curtis)")

    # Heatmap (bottom)
    ax_h = fig.add_axes([0.12, 0.12, 0.78, 0.62])
    im = ax_h.imshow(d.values, cmap=cmap, vmin=0, vmax=1)
    ax_h.set_xticks(range(len(d)))
    ax_h.set_yticks(range(len(d)))
    ax_h.set_xticklabels(d.index.tolist(), rotation=90, fontsize=8)
    ax_h.set_yticklabels(d.index.tolist(), fontsize=8)
    ax_h.set_title("Clustered Bray–Curtis dissimilarity")

    # Colorbar
    ax_c = fig.add_axes([0.92, 0.12, 0.02, 0.62])
    cbar = plt.colorbar(im, cax=ax_c)
    cbar.set_label("Distance (0=identical, 1=different)")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    return d, Z, order


# -----------------------------
# 6) Plot: Shannon + Effective species (two PNGs or one)
# -----------------------------
def save_shannon_and_effective_plots(
    sh: pd.DataFrame,
    out_png: str = "shannon_effective.png",
    figsize=(12, 6),
):
    # Sort by Shannon (descending)
    shs = sh.sort_values("Shannon", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(shs.index.astype(str), shs["Shannon"].values)
    ax.set_title("Alpha diversity per dataset (Shannon entropy)")
    ax.set_ylabel("Shannon (H)")
    ax.set_xticklabels(shs.index.astype(str), rotation=90, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png.replace(".png", "_shannon.png"), dpi=300)
    plt.close(fig)
    print(f"Saved {out_png.replace('.png','_shannon.png')}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(shs.index.astype(str), shs["Effective_species"].values)
    ax.set_title("Alpha diversity per dataset (Effective #classes = exp(H))")
    ax.set_ylabel("exp(Shannon)")
    ax.set_xticklabels(shs.index.astype(str), rotation=90, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png.replace(".png", "_effective.png"), dpi=300)
    plt.close(fig)
    print(f"Saved {out_png.replace('.png','_effective.png')}")


# -----------------------------
# 7) Plot: clustered heatmap + Shannon sidebar
# -----------------------------
def save_clustered_heatmap_with_shannon_sidebar(
    dist_df: pd.DataFrame,
    sh: pd.DataFrame,
    out_png: str = "bc_clustered_with_shannon_sidebar.png",
    method: str = "average",
    figsize=(12, 9),
    cmap="viridis",
):
    Z, order = cluster_order_from_distance(dist_df, method=method)

    d = dist_df.iloc[order, order]
    sh_ord = sh.loc[d.index]

    fig = plt.figure(figsize=figsize)

    # Heatmap axes
    ax_h = fig.add_axes([0.12, 0.12, 0.66, 0.78])
    im = ax_h.imshow(d.values, cmap=cmap, vmin=0, vmax=1)

    ax_h.set_xticks(range(len(d)))
    ax_h.set_yticks(range(len(d)))
    ax_h.set_xticklabels(d.index.tolist(), rotation=90, fontsize=8)
    ax_h.set_yticklabels(d.index.tolist(), fontsize=8)
    ax_h.set_title("Clustered Bray–Curtis (with Shannon sidebar)")

    # Shannon sidebar axes (same y order)
    ax_s = fig.add_axes([0.80, 0.12, 0.10, 0.78])
    ax_s.barh(np.arange(len(sh_ord)), sh_ord["Shannon"].values)
    ax_s.set_yticks(np.arange(len(sh_ord)))
    ax_s.set_yticklabels([])  # labels already on heatmap
    ax_s.invert_yaxis()       # match imshow orientation
    ax_s.set_xlabel("Shannon")
    ax_s.set_title("Alpha\n(Shannon)", fontsize=10)

    # Colorbar
    ax_c = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    cbar = plt.colorbar(im, cax=ax_c)
    cbar.set_label("Bray–Curtis distance")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    return d


# -----------------------------
# 8) One-call runner
# -----------------------------
def run_ecology_outputs(
    df: pd.DataFrame,
    dataset_col: str,
    label_col: str,
    out_prefix: str = "ecology",
    clustering_method: str = "average",
    root_dir: str = None,
):
    counts = build_label_table(df, dataset_col, label_col)
    sh = shannon_and_effective(counts)
    bc = bray_curtis_matrix(counts)

    # 1) dendrogram + heatmap
    save_dendrogram_and_heatmap(
        bc,
        out_png=root_dir + f"\{out_prefix}_bc_clustered_dendrogram.png",
        method=clustering_method
    )

    # 2) Shannon + effective plots
    save_shannon_and_effective_plots(
        sh,
        out_png=root_dir + f"\{out_prefix}_alpha.png"
    )

    # 3) Heatmap with Shannon sidebar
    save_clustered_heatmap_with_shannon_sidebar(
        bc,
        sh,
        out_png=root_dir + f"\{out_prefix}_bc_with_shannon_sidebar.png",
        method=clustering_method
    )

    # 4) Save Shannon dataframe
    sh_out_path = root_dir + f"\{out_prefix}_shannon_table.csv"
    sh.to_csv(sh_out_path)
    print(f"Saved {sh_out_path}")

    return bc, sh, counts


path = r'C:\alr4\ai_predict_all\prediction_parti20260119121141'
# df_name_ = r'\predictions_with_top3_scores_s.csv'
df_name_ = r'\predictions_with_top3_scores.csv'
df = pd.read_csv(path + df_name_)
df["run_id"] = df["Image Name"].str.split("_").str[2].astype(int)

df = df.rename(columns={"run_id": "group_id"})


bc, sh, counts = run_ecology_outputs(
    df,
    dataset_col="group_id",
    label_col="Top-1 Predicted Label",
    out_prefix="ALR4",
    root_dir=path
)


