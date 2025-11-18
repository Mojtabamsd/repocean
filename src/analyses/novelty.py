from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import heapq
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

from src.index import build_run_index
from src.utils.io import load_run_config
from src.metadata import load_run_metadata
from src.stream import (
    open_h5,
    get_h5_shapes,
    iter_feature_chunks,
    load_predictions_map,
    read_rows_by_indices,
)
from src.utils.paths import _safe_slug

# ----------------------------
# Small helpers
# ----------------------------


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(nrm, eps, None)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # assumes both a, b are already L2-normalized
    sim = a @ b.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim  # in [0,2]


def _fit_ipca_basis_for_run(h5_path: str, pca_dim: int, bootstrap: int = 4000) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
    with open_h5(h5_path) as h5f:
        n, _ = get_h5_shapes(h5f)
        if n == 0:
            return ipca
        k = min(bootstrap, n)
        # evenly spaced, sorted indices for stable h5 reads
        idx = np.linspace(0, n - 1, num=k, dtype=np.int64)
        part = read_rows_by_indices(h5f, idx)
        X = part["features"]
        if X.shape[0] > 4096:
            for j in range(0, X.shape[0], 4096):
                ipca.partial_fit(X[j:j+4096])
        else:
            ipca.partial_fit(X)
    return ipca


def _index_images_recursive(root: Path, exts=(".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> Dict[str, Path]:
    """Map lowercased filename -> full path (scan once per run)."""
    mapping: Dict[str, Path] = {}
    if not root.exists():
        return mapping
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.name.lower()] = p
    return mapping

# ----------------------------
# Core per-run logic
# ----------------------------


def _build_centroids_and_reservoir(
    h5_path: str,
    preds_csv: str,
    ipca: IncrementalPCA,
    reservoir_cap: int = 20000,
    batch_size: int = 4096,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Pass A: stream features, project -> PCA, L2-normalize; accumulate per-class centroids
    and build a reservoir sample for local density approximation.
    Returns:
      - centroids: {label -> centroid vector in PCA space (L2-normalized)}
      - reservoir: [R, pca_dim] L2-normalized matrix
    """
    # running sums per class (in PCA space)
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    # simple FIFO reservoir (bounded memory)
    reservoir_parts: List[np.ndarray] = []
    total_in_reservoir = 0

    preds = load_predictions_map(
        preds_csv,
        cols=[
            "Image Name",
            "Top-1 Predicted Label",
        ],
    )

    with open_h5(h5_path) as h5f:
        for blk in iter_feature_chunks(h5f, batch_size=batch_size):
            X = blk["features"]
            names = blk["image_names"]
            # project -> PCA
            if X.shape[1] > ipca.n_components:
                Xp = ipca.transform(X)
            else:
                Xp = X - ipca.mean_
            Xp = _l2_normalize(Xp)

            # labels (top-1)
            sub = preds.reindex(names)
            labs = sub["pred1_label"].fillna("unknown").astype(str).values

            # accumulate sums per label
            for lab in np.unique(labs):
                mask = (labs == lab)
                if not mask.any():
                    continue
                Xm = Xp[mask]
                if lab not in sums:
                    sums[lab] = Xm.sum(axis=0)
                    counts[lab] = Xm.shape[0]
                else:
                    sums[lab] += Xm.sum(axis=0)
                    counts[lab] += Xm.shape[0]

            # append to reservoir
            if reservoir_cap > 0:
                remain = reservoir_cap - total_in_reservoir
                if remain > 0:
                    take = min(remain, Xp.shape[0])
                    reservoir_parts.append(Xp[:take])
                    total_in_reservoir += take

    # finalize centroids (L2-normalize)
    centroids: Dict[str, np.ndarray] = {}
    for lab, s in sums.items():
        mu = s / max(counts[lab], 1)
        mu = mu.reshape(1, -1)
        mu = _l2_normalize(mu)[0]
        centroids[lab] = mu

    reservoir = np.vstack(reservoir_parts) if reservoir_parts else np.empty((0, ipca.n_components), dtype=np.float32)
    return centroids, reservoir


def _build_density_index(reservoir: np.ndarray, knn_k: int) -> Optional[NearestNeighbors]:
    if reservoir.shape[0] == 0:
        return None
    # kNN index on reservoir (cosine via euclidean on unit vectors)
    nn = NearestNeighbors(n_neighbors=min(knn_k, max(1, reservoir.shape[0])), algorithm="auto", metric="euclidean")
    nn.fit(reservoir)  # on L2-normalized vectors, this ~ cosine distance
    return nn

# ----------------------------
# Public API
# ----------------------------


def run_novelty_inbox(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",  # "run" | "meta"
    group_col: str = "sample_id",  # used when group_mode == "meta"
    top_n: int = 200,
    pca_dim: int = 50,
    reservoir_cap: int = 20000,
    knn_k: int = 10,
    w_centroid: float = 0.4,
    w_density: float = 0.4,
    w_margin: float = 0.2,
    batch_size: int = 4096,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Novelty scoring for each run, optionally split by metadata groups.

    Original behaviour (group_mode="run"):
      For each run, write:
        <out_dir>/novelty/<run_id>/novelty_inbox.csv

      Columns:
        run_id, image_name, abs_path, pred1_label, pred1_conf, pred2_label, pred2_conf,
        centroid_dist, knn_mean, margin_penalty, score

      Score = w_centroid * centroid_dist
              + w_density * knn_mean
              + w_margin * (1 - (conf1 - conf2)_clip).

    Metadata behaviour (group_mode="meta"):
      - Still fit PCA + reservoir per run (same model).
      - But keep a separate top-N heap per group_id (e.g. per sample_id).
      - Write:
          <out_dir>/novelty/<run_id>/<group_id>/novelty_inbox.csv
    """
    rng = np.random.default_rng(seed)
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    out_root = Path(out_dir) / "novelty"
    out_root.mkdir(parents=True, exist_ok=True)

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        run_out = out_root / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        # 0) Load run config + metadata (for grouping)
        cfg = load_run_config(r["run_cfg"])
        input_path = cfg.get("input_path")
        if group_mode == "meta" and input_path:
            meta = load_run_metadata(input_path, cols=[group_col])
            if not meta.empty and group_col in meta.columns:
                # normalise index to bare filenames to match H5 image_names
                meta = meta.copy()
                meta.index = (
                    meta.index.astype(str)
                    .str.replace("\\", "/", regex=False)
                    .str.split("/")
                    .str[-1]
                )
            else:
                # no usable metadata; gracefully fall back to run-level grouping
                meta = pd.DataFrame()
                group_mode_effective = "run"
            group_mode_effective = "meta" if not meta.empty and group_col in meta.columns else "run"
        else:
            meta = pd.DataFrame()
            group_mode_effective = "run"

        # 1) Fit PCA basis per run
        ipca = _fit_ipca_basis_for_run(
            r["features"],
            pca_dim=pca_dim,
            bootstrap=4000,
        )

        # 2) Build centroids + reservoir (per run)
        centroids, reservoir = _build_centroids_and_reservoir(
            h5_path=r["features"],
            preds_csv=r["preds"],
            ipca=ipca,
            reservoir_cap=reservoir_cap,
            batch_size=batch_size,
        )
        nn = _build_density_index(reservoir, knn_k=knn_k)

        # Absolute path mapping (for CSV convenience)
        file_map = _index_images_recursive(Path(input_path)) if input_path else {}

        # Load predictions (now with top2 for margin)
        preds = load_predictions_map(
            r["preds"],
            cols=[
                "Image Name",
                "Top-1 Predicted Label", "Top-1 Confidence Score",
                "Top-2 Predicted Label", "Top-2 Confidence Score",
            ],
        )

        # 3) Pass B: stream again, compute scores, keep top-N
        if group_mode_effective == "run":
            # single heap per run (original behaviour)
            heap: List[Tuple[float, Dict[str, object]]] = []
            heap_dict = None
        else:
            # separate heap per group_id
            heap = None
            heap_dict: Dict[str, List[Tuple[float, Dict[str, object]]]] = {}

        with open_h5(r["features"]) as h5f:
            for blk in iter_feature_chunks(h5f, batch_size=batch_size):
                X = blk["features"]
                names = blk["image_names"]

                # project -> PCA -> L2-normalize
                if X.shape[1] > ipca.n_components:
                    Xp = ipca.transform(X)
                else:
                    Xp = X - ipca.mean_
                Xp = _l2_normalize(Xp)

                sub = preds.reindex(names)
                pred1_label = sub["pred1_label"].fillna("unknown").astype(str).values
                pred1_conf  = sub["pred1_conf"].astype(float).fillna(0.0).values
                pred2_label = sub["pred2_label"].fillna("").astype(str).values
                pred2_conf  = sub["pred2_conf"].astype(float).fillna(0.0).values

                # distances to class centroid (cosine)
                default_centroid = (
                    reservoir.mean(axis=0, keepdims=True)
                    if reservoir.shape[0] > 0
                    else np.zeros((1, Xp.shape[1]), dtype=np.float32)
                )
                if default_centroid.shape[0] == 1:
                    default_centroid = _l2_normalize(default_centroid)
                C = np.vstack([centroids.get(l, default_centroid[0]) for l in pred1_label])
                d_centroid = _cosine_distance(Xp, C).diagonal()

                # local density approximation via reservoir kNN
                if nn is not None and reservoir.shape[0] > 0:
                    D_knn, _ = nn.kneighbors(
                        Xp,
                        n_neighbors=min(knn_k, reservoir.shape[0]),
                        return_distance=True,
                    )
                    knn_mean = D_knn.mean(axis=1)
                else:
                    knn_mean = np.zeros(Xp.shape[0], dtype=np.float32)

                # margin penalty (higher = more ambiguous)
                margin = np.clip(pred1_conf - pred2_conf, 0.0, 1.0)
                margin_pen = 1.0 - margin  # in [0,1]

                # combined score
                score = (
                    w_centroid * d_centroid
                    + w_density * knn_mean
                    + w_margin * margin_pen
                )

                # push rows into fixed-size heap(s)
                for i in range(Xp.shape[0]):
                    name = str(names[i])
                    name_key = name.replace("\\", "/").split("/")[-1]

                    # assign group
                    if group_mode_effective == "meta":
                        if name_key in meta.index:
                            group_id = str(meta.loc[name_key, group_col])
                        else:
                            # if no metadata match, you can skip or assign to a special group
                            group_id = "unknown"
                    else:
                        group_id = run_id  # single group per run

                    rec = {
                        "run_id": run_id,
                        "group_id": group_id if group_mode_effective == "meta" else run_id,
                        "image_name": name,
                        "abs_path": str(file_map.get(name_key.lower(), "")),
                        "pred1_label": pred1_label[i],
                        "pred1_conf": float(pred1_conf[i]),
                        "pred2_label": pred2_label[i],
                        "pred2_conf": float(pred2_conf[i]),
                        "centroid_dist": float(d_centroid[i]),
                        "knn_mean": float(knn_mean[i]),
                        "margin_penalty": float(margin_pen[i]),
                        "score": float(score[i]),
                    }

                    if group_mode_effective == "run":
                        # one heap for the whole run
                        if len(heap) < top_n:
                            heapq.heappush(heap, (rec["score"], rec))
                        else:
                            if score[i] > heap[0][0]:
                                heapq.heapreplace(heap, (rec["score"], rec))
                    else:
                        # one heap per group_id
                        h = heap_dict.setdefault(group_id, [])
                        if len(h) < top_n:
                            heapq.heappush(h, (rec["score"], rec))
                        else:
                            if score[i] > h[0][0]:
                                heapq.heapreplace(h, (rec["score"], rec))

        # 4) Save CSV(s), highest score first
        if group_mode_effective == "run":
            heap.sort(key=lambda x: x[0], reverse=True)
            rows = [rec for _, rec in heap]
            out_csv = run_out / "novelty_inbox.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        else:
            # one CSV per group inside this run
            for g_id, h in heap_dict.items():
                h.sort(key=lambda x: x[0], reverse=True)
                rows = [rec for _, rec in h]
                safe_group = _safe_slug(str(g_id))
                group_out = run_out / safe_group
                group_out.mkdir(parents=True, exist_ok=True)
                out_csv = group_out / "novelty_inbox.csv"
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    return {"out_dir": str(out_root)}
