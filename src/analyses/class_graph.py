from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import importlib

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from src.index import build_group_index
from src.stream import (
    open_h5, get_h5_shapes, read_rows_by_indices, iter_feature_chunks,
    load_predictions_map,
)
from src.utils.paths import _safe_slug

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


def _sample_for_group(
    h5_path: str,
    max_points: int,
    idx_subset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (indices, features, image_names) sampled evenly.

    If idx_subset is None:
        sample over the full [0..n-1] range.
    Else:
        sample only within the given subset indices.
    """
    with open_h5(h5_path) as h5f:
        n, _ = get_h5_shapes(h5f)
        if n == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 0)),
                np.empty((0,), dtype=object),
            )

        if idx_subset is None:
            k = min(max_points, n)
            idx = np.linspace(0, n - 1, num=k, dtype=np.int64)
        else:
            idx_subset = np.asarray(idx_subset, dtype=np.int64)
            if idx_subset.size == 0:
                return (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0, 0)),
                    np.empty((0,), dtype=object),
                )
            idx_subset = np.unique(idx_subset)
            k = min(max_points, idx_subset.size)
            sel = np.linspace(0, idx_subset.size - 1, num=k, dtype=np.int64)
            idx = idx_subset[sel]

        part = read_rows_by_indices(h5f, idx)
        X = part["features"]
        names = part["image_names"].astype(str)

    return idx, X, names


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


def _class_graph_for_group(
    run_id: str,
    group_id: str,
    features_path: str,
    preds_path: str,
    ipca: IncrementalPCA,
    out_root: Path,
    k: int,
    pca_dim: int,
    max_points_per_group: int,
    idx_subset: Optional[np.ndarray] = None,
    include_self: bool = False,
    mutual_only: bool = False,
    save_plots: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build k-NN class graph for a *single group* (run or profile).

    Returns:
      Xp_group    [n, d]  – projected, L2-normalised features used in the graph
      labels      [n]
      conf1       [n]
      names       [n]
      run_ids     [n]  (all = run_id)
      group_ids   [n]  (all = group_id)
    """
    # Output dir: class_graph/<run_id>/<group_slug>
    group_slug = _safe_slug(str(group_id))
    group_out = out_root / "class_graph" / run_id / group_slug
    group_out.mkdir(parents=True, exist_ok=True)

    # --- sample embeddings for this group
    idx, X, names = _sample_for_group(features_path, max_points_per_group, idx_subset=idx_subset)
    if X.size == 0:
        return (
            np.empty((0, 0)),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )

    # --- labels for sampled nodes
    preds = load_predictions_map(
        preds_path,
        cols=["Image Name", "Top-1 Predicted Label", "Top-1 Confidence Score"],
    )
    sub = preds.reindex(names)
    labels = sub["pred1_label"].fillna("unknown").astype(str).values
    conf1 = sub["pred1_conf"].astype(float).fillna(0.0).values

    # --- PCA -> L2-normalize (IPCA is per-run, already fitted)
    if X.shape[1] > ipca.n_components:
        Xp = ipca.transform(X)
    else:
        Xp = X - ipca.mean_
    Xp = _l2_normalize(Xp)

    # --- kNN
    D, I = _build_knn(Xp, k=k)
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

    # Class vs Class confusion (within this group)
    C = pd.DataFrame(0, index=uniq, columns=uniq, dtype=np.int64)
    for i in range(Xp.shape[0]):
        a = labels[i]
        for j in mutual_lists[i]:
            b = labels[j]
            C.loc[a, b] += 1
    C_prob = C.div(C.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # --- Save per-group nodes table
    nodes = pd.DataFrame(
        {
            "run_id": run_id,
            "group_id": group_id,
            "node_idx": np.arange(Xp.shape[0]),
            "image_name": names,
            "pred1_label": labels,
            "pred1_conf": conf1,
            "entropy": node_ent,
            "purity": node_pur,
            "k_effective": k_eff_per_node,
            "mutual_only": mutual_only,
        }
    )
    nodes.to_csv(
        group_out / ("nodes_mutual.csv" if mutual_only else "nodes.csv"),
        index=False,
    )
    C_prob.to_csv(
        group_out
        / ("class_confusion_mutual.csv" if mutual_only else "class_confusion.csv")
    )

    cls_summary = pd.DataFrame(
        {
            "run_id": run_id,
            "group_id": group_id,
            "class": uniq,
            "n_nodes": [int((labels == c).sum()) for c in uniq],
            "mean_entropy": [
                float(nodes.loc[nodes.pred1_label == c, "entropy"].mean())
                for c in uniq
            ],
            "mean_purity": [
                float(nodes.loc[nodes.pred1_label == c, "purity"].mean())
                for c in uniq
            ],
        }
    )
    cls_summary.to_csv(
        group_out / ("summary_mutual.csv" if mutual_only else "summary.csv"),
        index=False,
    )

    # Heatmap (small-ish)
    if save_plots and C_prob.shape[0] <= 60:
        plt.figure(
            figsize=(
                max(6, C_prob.shape[0] * 0.3),
                max(4, C_prob.shape[1] * 0.3),
            )
        )
        plt.imshow(C_prob.values, aspect="auto", interpolation="nearest")
        title_mode = "Mutual kNN" if mutual_only else "kNN"
        plt.title(
            f"Class–Class Neighbour Matrix ({title_mode}) — {run_id} | {group_id}"
        )
        plt.xlabel("Neighbour class")
        plt.ylabel("Source class")
        plt.colorbar(label="P(neighbour | source)")
        plt.xticks(
            ticks=np.arange(len(uniq)),
            labels=uniq,
            rotation=90,
            fontsize=8,
        )
        plt.yticks(ticks=np.arange(len(uniq)), labels=uniq, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            group_out
            / (
                "confusion_heatmap_mutual.png"
                if mutual_only
                else "confusion_heatmap.png"
            ),
            dpi=180,
        )
        plt.close()

    # For global graph: return these samples
    run_ids = np.array([run_id] * Xp.shape[0], dtype=object)
    group_ids = np.array([group_id] * Xp.shape[0], dtype=object)

    return Xp, labels, conf1, names, run_ids, group_ids


# -----------------------
# Public API
# -----------------------


def run_class_graph(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",          # "run" | "meta"
    group_col: str = "sample_id",     # used when group_mode == "meta"
    k: int = 10,
    pca_dim: int = 50,
    max_points_per_group: int = 8000,
    global_max_points: Optional[int] = None,  # optional extra cap for global graph
    include_self: bool = False,
    mutual_only: bool = False,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Build class kNN graphs:

      - Per group (run or profile/sample_id):
          <out_dir>/class_graph/<run_id>/<group_id_slug>/...

      - Global across all groups:
          <out_dir>/_global/class_graph/...

    group_mode="run":
        each run is one group (original behaviour, but now also global).

    group_mode="meta":
        group by metadata column group_col (e.g. 'sample_id', 'acq_id');
        indices for each group are taken from build_group_index().
    """
    # Build group index (reuses the same mechanism as t-SNE / purity)
    groups = build_group_index(
        parent_dir=parent_dir,
        mode="run" if group_mode == "run" else "meta",
        group_col=group_col,
    )
    if groups.empty:
        raise RuntimeError(
            f"No groups found under {parent_dir} (group_mode={group_mode}, group_col={group_col})"
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Cache IPCA per run to avoid refitting for each group
    ipca_cache: Dict[str, IncrementalPCA] = {}

    # Collect samples for global graph
    all_Xp, all_labels, all_conf, all_names, all_run_ids, all_group_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for _, g in groups.iterrows():
        run_id = g["run_id"]
        group_id = (
            g["group_id"] if group_mode == "meta" else run_id
        )  # for 'run', group_id = run_id
        features_path = g["features"]
        preds_path = g["preds"]

        # indices only for meta grouping
        idx_subset = None
        if group_mode == "meta":
            idx_subset = g.get("indices", None)
            if idx_subset is not None and not isinstance(idx_subset, np.ndarray):
                # in case it's a list/object; normalise
                idx_subset = np.asarray(idx_subset, dtype=np.int64)

        # Fit / reuse run-level IPCA
        if run_id not in ipca_cache:
            ipca_cache[run_id] = _fit_ipca_for_run(
                features_path,
                pca_dim=pca_dim,
                bootstrap=min(4000, max_points_per_group),
            )
        ipca = ipca_cache[run_id]

        # Per-group graph + samples for global
        Xp, labels, conf1, names, run_ids, group_ids = _class_graph_for_group(
            run_id=run_id,
            group_id=str(group_id),
            features_path=features_path,
            preds_path=preds_path,
            ipca=ipca,
            out_root=out_root,
            k=k,
            pca_dim=pca_dim,
            max_points_per_group=max_points_per_group,
            idx_subset=idx_subset,
            include_self=include_self,
            mutual_only=mutual_only,
            save_plots=save_plots,
        )

        if Xp.size == 0:
            continue

        all_Xp.append(Xp)
        all_labels.append(labels)
        all_conf.append(conf1)
        all_names.append(names)
        all_run_ids.append(run_ids)
        all_group_ids.append(group_ids)

    # ---------------- Global graph ----------------
    if not all_Xp:
        raise RuntimeError("No samples collected from any group for global graph.")

    Xp_all = np.vstack(all_Xp)
    labels_all = np.concatenate(all_labels)
    conf_all = np.concatenate(all_conf)
    names_all = np.concatenate(all_names)
    runid_all = np.concatenate(all_run_ids)
    groupid_all = np.concatenate(all_group_ids)

    # Optional extra cap: downsample for global if huge
    if global_max_points is not None and Xp_all.shape[0] > global_max_points:
        idx = np.linspace(
            0, Xp_all.shape[0] - 1, num=global_max_points, dtype=np.int64
        )
        Xp_all = Xp_all[idx]
        labels_all = labels_all[idx]
        conf_all = conf_all[idx]
        names_all = names_all[idx]
        runid_all = runid_all[idx]
        groupid_all = groupid_all[idx]

    # kNN
    D, I = _build_knn(Xp_all, k=k)
    if not include_self and I.shape[1] > 1:
        I = I[:, 1:]
        D = D[:, 1:]

    if mutual_only:
        mutual_lists = _mutual_neighbor_lists(I)
        k_eff = [len(mi) for mi in mutual_lists]
    else:
        mutual_lists = [row.tolist() for row in I]
        k_eff = [I.shape[1]] * I.shape[0]

    # Node metrics for global graph
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

    # Global class↔class
    C = pd.DataFrame(0, index=uniq, columns=uniq, dtype=np.int64)
    for i in range(Xp_all.shape[0]):
        a = labels_all[i]
        for j in mutual_lists[i]:
            b = labels_all[j]
            C.loc[a, b] += 1
    C_prob = C.div(C.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Outputs
    global_out = out_root / "class_graph" / "_global"
    global_out.mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame(
        {
            "run_id": runid_all,
            "group_id": groupid_all,
            "node_idx": np.arange(Xp_all.shape[0]),
            "image_name": names_all,
            "pred1_label": labels_all,
            "pred1_conf": conf_all,
            "entropy": node_ent,
            "purity": node_pur,
            "k_effective": k_eff,
            "mutual_only": mutual_only,
        }
    )
    nodes.to_csv(
        global_out / ("nodes_global_mutual.csv" if mutual_only else "nodes_global.csv"),
        index=False,
    )
    C_prob.to_csv(
        global_out
        / (
            "class_confusion_global_mutual.csv"
            if mutual_only
            else "class_confusion_global.csv"
        )
    )

    cls_summary = pd.DataFrame(
        {
            "class": uniq,
            "n_nodes": [int((labels_all == c).sum()) for c in uniq],
            "mean_entropy": [
                float(nodes.loc[nodes.pred1_label == c, "entropy"].mean())
                for c in uniq
            ],
            "mean_purity": [
                float(nodes.loc[nodes.pred1_label == c, "purity"].mean())
                for c in uniq
            ],
        }
    )
    cls_summary.to_csv(
        global_out
        / (
            "summary_global_mutual.csv"
            if mutual_only
            else "summary_global.csv"
        ),
        index=False,
    )

    if save_plots and C_prob.shape[0] <= 60:
        plt.figure(
            figsize=(
                max(6, C_prob.shape[0] * 0.3),
                max(4, C_prob.shape[1] * 0.3),
            )
        )
        plt.imshow(C_prob.values, aspect="auto", interpolation="nearest")
        title_mode = "Mutual kNN" if mutual_only else "kNN"
        plt.title(f"Global Class–Class Neighbour Matrix ({title_mode})")
        plt.xlabel("Neighbour class")
        plt.ylabel("Source class")
        plt.colorbar(label="P(neighbour | source)")
        plt.xticks(
            ticks=np.arange(len(uniq)),
            labels=uniq,
            rotation=90,
            fontsize=8,
        )
        plt.yticks(ticks=np.arange(len(uniq)), labels=uniq, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            global_out
            / (
                "confusion_heatmap_global_mutual.png"
                if mutual_only
                else "confusion_heatmap_global.png"
            ),
            dpi=180,
        )
        plt.close()

    return {
        "per_group_dir": str(out_root / "class_graph"),
        "global_dir": str(global_out),
    }
