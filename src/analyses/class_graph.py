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


def _fit_ipca_for_many(samples: List[np.ndarray], pca_dim: int) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
    for X in samples:
        if X.shape[0] == 0:
            continue
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


def _mutual_neighbor_lists(I: np.ndarray) -> List[List[int]]:
    """
    Given neighbor index array I (n,k), return a list of mutual neighbors per node.
    mutual(i) = { j in I[i] | i in I[j] }.
    """
    n, k = I.shape
    neighbor_sets = [set(I[i].tolist()) for i in range(n)]
    mutual = []
    for i in range(n):
        mi = [j for j in I[i] if i in neighbor_sets[j]]
        mutual.append(mi)
    return mutual

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
    mutual_only: bool = False,
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
    Xp = ipca.transform(X) if X.shape[1] > ipca.n_components else (X - ipca.mean_)
    Xp = _l2_normalize(Xp)

    # --- kNN
    D, I = _build_knn(Xp, k=k)
    # Optionally drop self neighbor
    if not include_self and I.shape[1] > 1:
        I = I[:, 1:]
        D = D[:, 1:]

    # Mutual filter?
    if mutual_only:
        mutual_lists = _mutual_neighbor_lists(I)
        k_eff_per_node = [len(mi) for mi in mutual_lists]
    else:
        mutual_lists = [row.tolist() for row in I]
        k_eff_per_node = [I.shape[1]] * I.shape[0]

    # Node metrics
    uniq = np.unique(labels)
    label_to_idx = {c: i for i, c in enumerate(uniq)}
    node_ent = np.zeros(Xp.shape[0], dtype=np.float32)
    node_pur = np.zeros(Xp.shape[0], dtype=np.float32)

    for i in range(Xp.shape[0]):
        nbr_idx = mutual_lists[i]
        nbr_lbls = labels[nbr_idx] if len(nbr_idx) else np.array([], dtype=labels.dtype)
        counts = np.zeros(uniq.size, dtype=np.int64)
        for lab in nbr_lbls:
            counts[label_to_idx[lab]] += 1
        node_ent[i] = _entropy_from_counts(counts) if counts.sum() > 0 else 0.0
        node_pur[i] = _purity_from_counts(counts) if counts.sum() > 0 else 0.0

    # Class vs Class confusion
    C = pd.DataFrame(0, index=uniq, columns=uniq, dtype=np.int64)
    for i in range(Xp.shape[0]):
        a = labels[i]
        for j in mutual_lists[i]:
            b = labels[j]
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
        "k_effective": k_eff_per_node,
        "mutual_only": mutual_only,
    })
    nodes.to_csv(run_out / ("nodes_mutual.csv" if mutual_only else "nodes.csv"), index=False)
    C_prob.to_csv(run_out / ("class_confusion_mutual.csv" if mutual_only else "class_confusion.csv"))
    cls_summary = pd.DataFrame({
        "run_id": run_id,
        "class": uniq,
        "n_nodes": [int((labels == c).sum()) for c in uniq],
        "mean_entropy": [float(nodes.loc[nodes.pred1_label == c, "entropy"].mean()) for c in uniq],
        "mean_purity":  [float(nodes.loc[nodes.pred1_label == c, "purity"].mean()) for c in uniq],
    })
    cls_summary.to_csv(run_out / ("summary_mutual.csv" if mutual_only else "summary.csv"), index=False)

    # Heatmap
    if save_plots and C_prob.shape[0] <= 60:
        plt.figure(figsize=(max(6, C_prob.shape[0]*0.3), max(4, C_prob.shape[1]*0.3)))
        plt.imshow(C_prob.values, aspect="auto", interpolation="nearest")
        title_mode = "Mutual kNN" if mutual_only else "kNN"
        plt.title(f"Class–Class Neighbour Matrix ({title_mode}) — {run_id}")
        plt.xlabel("Neighbour class"); plt.ylabel("Source class")
        plt.colorbar(label="P(neighbour | source)")
        plt.xticks(ticks=np.arange(len(uniq)), labels=uniq, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(len(uniq)), labels=uniq, fontsize=8)
        plt.tight_layout()
        plt.savefig(run_out / ("confusion_heatmap_mutual.png" if mutual_only else "confusion_heatmap.png"), dpi=180)
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
    mutual_only: bool = False,
    save_plots: bool = True,
) -> Dict[str, str]:
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    out_root = Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)
    for _, r in runs.iterrows():
        _class_graph_for_run(
            run_row=r, out_dir=out_root, k=k, pca_dim=pca_dim,
            max_points_per_run=max_points_per_run, include_self=include_self,
            mutual_only=mutual_only, save_plots=save_plots,
        )
    return {"out_dir": str(out_root)}

# -----------------------
# Global graph across runs
# -----------------------


def run_class_graph_global(
    parent_dir: str,
    out_dir: str,
    k: int = 10,
    pca_dim: int = 50,
    max_points_total: int = 20000,
    per_run_cap: int = 4000,
    include_self: bool = False,
    mutual_only: bool = False,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Build one kNN graph across *all runs* using an even per-run sample.
    Exports under: <out_dir>/_global/class_graph/
      - nodes_global.csv (run_id, image_name, label, entropy, purity, k_effective)
      - class_confusion_global.csv
      - summary_global.csv
      - confusion_heatmap_global.png (optional)
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    # sample evenly per run
    n_runs = len(runs)
    per_run = min(per_run_cap, max(1, max_points_total // max(1, n_runs)))

    samples_X, samples_names, samples_labels, samples_conf, samples_runid = [], [], [], [], []

    # first collect samples for PCA fitting
    for _, r in runs.iterrows():
        idx, X = _sample_for_graph(r["features"], per_run)
        if X.size == 0:
            continue
        # get names + labels
        with open_h5(r["features"]) as h5f:
            names = h5f["image_names"][idx].astype(str)
        preds = load_predictions_map(
            r["preds"],
            cols=["Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"]
        ).reindex(names)
        labs = preds["pred1_label"].fillna("unknown").astype(str).values
        conf = preds["pred1_conf"].astype(float).fillna(0.0).values

        samples_X.append(X)
        samples_names.append(names)
        samples_labels.append(labs)
        samples_conf.append(conf)
        samples_runid.append(np.array([r["run_id"]]*len(names), dtype=object))

    if not samples_X:
        raise RuntimeError("No samples available across runs.")

    # fit PCA across all samples (incremental)
    ipca = _fit_ipca_for_many(samples_X, pca_dim=pca_dim)

    # project + normalize and stack
    Xp_list = []
    for X in samples_X:
        Xp = ipca.transform(X) if X.shape[1] > ipca.n_components else (X - ipca.mean_)
        Xp_list.append(_l2_normalize(Xp))
    Xp_all = np.vstack(Xp_list)
    names_all = np.concatenate(samples_names)
    labels_all = np.concatenate(samples_labels)
    conf_all = np.concatenate(samples_conf)
    runid_all = np.concatenate(samples_runid)

    # kNN
    D, I = _build_knn(Xp_all, k=k)
    if not include_self and I.shape[1] > 1:
        I = I[:, 1:]
        D = D[:, 1:]

    # Mutual?
    if mutual_only:
        mutual_lists = _mutual_neighbor_lists(I)
        k_eff = [len(mi) for mi in mutual_lists]
    else:
        mutual_lists = [row.tolist() for row in I]
        k_eff = [I.shape[1]] * I.shape[0]

    # node metrics
    uniq = np.unique(labels_all)
    label_to_idx = {c: i for i, c in enumerate(uniq)}
    node_ent = np.zeros(Xp_all.shape[0], dtype=np.float32)
    node_pur = np.zeros(Xp_all.shape[0], dtype=np.float32)
    for i in range(Xp_all.shape[0]):
        nbr_idx = mutual_lists[i]
        nbr_lbls = labels_all[nbr_idx] if len(nbr_idx) else np.array([], dtype=labels_all.dtype)
        counts = np.zeros(uniq.size, dtype=np.int64)
        for lab in nbr_lbls:
            counts[label_to_idx[lab]] += 1
        node_ent[i] = _entropy_from_counts(counts) if counts.sum() > 0 else 0.0
        node_pur[i] = _purity_from_counts(counts) if counts.sum() > 0 else 0.0

    # class↔class across runs
    C = pd.DataFrame(0, index=uniq, columns=uniq, dtype=np.int64)
    for i in range(Xp_all.shape[0]):
        a = labels_all[i]
        for j in mutual_lists[i]:
            b = labels_all[j]
            C.loc[a, b] += 1
    C_prob = C.div(C.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # outputs
    global_out = Path(out_dir) / "_global" / "class_graph"
    global_out.mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame({
        "run_id": runid_all,
        "node_idx": np.arange(Xp_all.shape[0]),
        "image_name": names_all,
        "pred1_label": labels_all,
        "pred1_conf": conf_all,
        "entropy": node_ent,
        "purity": node_pur,
        "k_effective": k_eff,
        "mutual_only": mutual_only,
    })
    nodes.to_csv(global_out / ("nodes_global_mutual.csv" if mutual_only else "nodes_global.csv"), index=False)
    C_prob.to_csv(global_out / ("class_confusion_global_mutual.csv" if mutual_only else "class_confusion_global.csv"))

    cls_summary = pd.DataFrame({
        "class": uniq,
        "n_nodes": [int((labels_all == c).sum()) for c in uniq],
        "mean_entropy": [float(nodes.loc[nodes.pred1_label == c, "entropy"].mean()) for c in uniq],
        "mean_purity":  [float(nodes.loc[nodes.pred1_label == c, "purity"].mean()) for c in uniq],
    })
    cls_summary.to_csv(global_out / ("summary_global_mutual.csv" if mutual_only else "summary_global.csv"), index=False)

    if save_plots and C_prob.shape[0] <= 60:
        plt.figure(figsize=(max(6, C_prob.shape[0]*0.3), max(4, C_prob.shape[1]*0.3)))
        plt.imshow(C_prob.values, aspect="auto", interpolation="nearest")
        title_mode = "Mutual kNN" if mutual_only else "kNN"
        plt.title(f"Global Class–Class Neighbour Matrix ({title_mode})")
        plt.xlabel("Neighbour class"); plt.ylabel("Source class")
        plt.colorbar(label="P(neighbour | source)")
        plt.xticks(ticks=np.arange(len(uniq)), labels=uniq, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(len(uniq)), labels=uniq, fontsize=8)
        plt.tight_layout()
        plt.savefig(global_out / ("confusion_heatmap_global_mutual.png" if mutual_only else "confusion_heatmap_global.png"), dpi=180)
        plt.close()

    return {"out_dir": str(global_out)}
