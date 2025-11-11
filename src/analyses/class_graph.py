from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import importlib

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from src.index import build_run_index
from src.stream import (
    open_h5, get_h5_shapes, read_rows_by_indices, iter_feature_chunks,
    load_predictions_map,
)

# -----------------------
# Utilities
# -----------------------


def _faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None or importlib.util.find_spec("faiss_cpu") is not None


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def _fit_ipca_for_run(h5_path: str, pca_dim: int, bootstrap: int = 4000) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
    with open_h5(h5_path) as h5f:
        n, _ = get_h5_shapes(h5f)
        if n == 0:
            return ipca
        k = min(bootstrap, n)
        idx = np.linspace(0, n - 1, num=k, dtype=np.int64)  # sorted for h5
        part = read_rows_by_indices(h5f, idx)
        X = part["features"]
        if X.shape[0] > 4096:
            for j in range(0, X.shape[0], 4096):
                ipca.partial_fit(X[j:j+4096])
        else:
            ipca.partial_fit(X)
    return ipca


def _sample_for_graph(h5_path: str, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, features) sampled evenly along capture order (sorted).
    """
    with open_h5(h5_path) as h5f:
        n, _ = get_h5_shapes(h5f)
        if n == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0))
        k = min(max_points, n)
        idx = np.linspace(0, n - 1, num=k, dtype=np.int64)
        part = read_rows_by_indices(h5f, idx)
        X = part["features"]
    return idx, X


def _build_knn(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (distances, indices) for kNN. Prefers FAISS (L2) if available.
    X is assumed L2-normalized; euclidean ~ cosine distance.
    First neighbor is the point itself.
    """
    k = min(k, max(1, X.shape[0]))
    if _faiss_available():
        try:
            import faiss  # type: ignore
            X32 = X.astype(np.float32, copy=False)
            index = faiss.IndexFlatL2(X32.shape[1])
            index.add(X32)
            D, I = index.search(X32, k)
            return D, I
        except Exception:
            pass
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    D, I = nn.kneighbors(X, return_distance=True)
    return D, I


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / max(counts.sum(), eps)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + eps)))


def _purity_from_counts(counts: np.ndarray) -> float:
    s = counts.sum()
    return float(0.0 if s <= 0 else counts.max() / s)

# -----------------------
# Main per-run logic
# -----------------------


def _class_graph_for_run(
    run_row: pd.Series,
    out_dir: Path,
    k: int,
    pca_dim: int,
    max_points_per_run: int,
    include_self: bool = False,
    save_plots: bool = True,
) -> None:
    """
    Build k-NN graph on a sampled set of points for one run, compute node entropy/purity
    and a class-confusion matrix (row-normalised).
    """
    run_id = run_row["run_id"]
    run_out = out_dir / run_id / "class_graph"
    run_out.mkdir(parents=True, exist_ok=True)

    # --- sample embeddings for graph
    idx, X = _sample_for_graph(run_row["features"], max_points_per_run)
    if X.size == 0:
        return

    # --- labels for sampled nodes
    preds = load_predictions_map(
        run_row["preds"],
        cols=["Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"]
    )

    # get image names for sampled indices
    with open_h5(run_row["features"]) as h5f:
        names = h5f["image_names"][idx].astype(str)

    sub = preds.reindex(names)
    labels = sub["pred1_label"].fillna("unknown").astype(str).values
    conf1  = sub["pred1_conf"].astype(float).fillna(0.0).values

    # --- PCA -> L2-normalize
    ipca = _fit_ipca_for_run(run_row["features"], pca_dim=pca_dim, bootstrap=min(4000, max_points_per_run))
    if X.shape[1] > ipca.n_components:
        Xp = ipca.transform(X)
    else:
        Xp = X - ipca.mean_
    Xp = _l2_normalize(Xp)

    # --- kNN
    D, I = _build_knn(Xp, k=k)
    # Optionally drop self neighbor
    if not include_self and I.shape[1] > 1:
        I = I[:, 1:]
        D = D[:, 1:]
    k_eff = I.shape[1]

    # --- Node-level metrics: entropy & purity of neighbour label distribution
    uniq = np.unique(labels)
    label_to_idx = {c: i for i, c in enumerate(uniq)}
    node_ent = np.zeros(Xp.shape[0], dtype=np.float32)
    node_pur = np.zeros(Xp.shape[0], dtype=np.float32)

    for i in range(Xp.shape[0]):
        nbr_lbls = labels[I[i]]
        counts = np.zeros(uniq.size, dtype=np.int64)
        for lab in nbr_lbls:
            counts[label_to_idx[lab]] += 1
        node_ent[i] = _entropy_from_counts(counts)
        node_pur[i] = _purity_from_counts(counts)

    # --- Class↔Class confusion (row-normalised)
    C = pd.DataFrame(0, index=uniq, columns=uniq, dtype=np.int64)
    for i in range(Xp.shape[0]):
        a = labels[i]
        for b in labels[I[i]]:
            C.loc[a, b] += 1
    # normalise rows to probabilities
    C_prob = C.div(C.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # --- Save nodes table
    nodes = pd.DataFrame({
        "run_id": run_id,
        "node_idx": np.arange(Xp.shape[0]),
        "image_name": names,
        "pred1_label": labels,
        "pred1_conf": conf1,
        "entropy": node_ent,
        "purity": node_pur,
        "k": k_eff,
    })
    nodes.to_csv(run_out / "nodes.csv", index=False)

    # --- Save class confusion & per-class summary
    C_prob.to_csv(run_out / "class_confusion.csv")
    cls_summary = pd.DataFrame({
        "run_id": run_id,
        "class": uniq,
        "n_nodes": [int((labels == c).sum()) for c in uniq],
        "mean_entropy": [float(nodes.loc[nodes.pred1_label == c, "entropy"].mean()) for c in uniq],
        "mean_purity":  [float(nodes.loc[nodes.pred1_label == c, "purity"].mean()) for c in uniq],
    })
    cls_summary.to_csv(run_out / "summary.csv", index=False)

    # --- Optional heatmap
    if save_plots and C_prob.shape[0] <= 60:  # avoid giant images by default
        plt.figure(figsize=(max(6, C_prob.shape[0]*0.3), max(4, C_prob.shape[1]*0.3)))
        plt.imshow(C_prob.values, aspect="auto", interpolation="nearest")
        plt.title(f"Class–Class Neighbour Matrix — {run_id}")
        plt.xlabel("Neighbour class")
        plt.ylabel("Source class")
        plt.colorbar(label="P(neighbour class | source class)")
        plt.xticks(ticks=np.arange(len(uniq)), labels=uniq, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(len(uniq)), labels=uniq, fontsize=8)
        plt.tight_layout()
        plt.savefig(run_out / "confusion_heatmap.png", dpi=180)
        plt.close()

# -----------------------
# Public API
# -----------------------


def run_class_graph(
    parent_dir: str,
    out_dir: str,
    k: int = 10,
    pca_dim: int = 50,
    max_points_per_run: int = 8000,
    include_self: bool = False,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Build a k-NN class graph per run on a sampled set of points and export:
      - class_graph/nodes.csv         (node entropy/purity)
      - class_graph/class_confusion.csv (row-normalised)
      - class_graph/summary.csv
      - class_graph/confusion_heatmap.png (optional)
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, r in runs.iterrows():
        _class_graph_for_run(
            run_row=r,
            out_dir=out_root,
            k=k,
            pca_dim=pca_dim,
            max_points_per_run=max_points_per_run,
            include_self=include_self,
            save_plots=save_plots,
        )

    return {"out_dir": str(out_root)}
