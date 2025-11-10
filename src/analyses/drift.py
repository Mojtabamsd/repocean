from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from src.index import build_run_index
from src.stream import open_h5, get_h5_shapes, read_rows_by_indices
from src.utils.io import load_run_config
from src.metadata import load_run_metadata


# -------------------------
# Utilities
# -------------------------

def _fit_ipca_basis(
    runs_df: pd.DataFrame,
    pca_dim: int,
    bootstrap_per_run: int = 2000,
    seed: int = 42,
) -> IncrementalPCA:
    """
    Fit an IncrementalPCA basis using a small bootstrap from each run.
    Keeps RAM small and creates a common space for drift comparison.
    """
    rng = np.random.default_rng(seed)
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)

    for _, r in runs_df.iterrows():
        with open_h5(r["features"]) as h5f:
            n, d = get_h5_shapes(h5f)
            if n == 0:
                continue
            k = min(bootstrap_per_run, n)
            # evenly spaced for stability & sorted for h5py
            idx = np.linspace(0, n - 1, num=k, dtype=np.int64)
            X = read_rows_by_indices(h5f, idx)["features"]
            if X.shape[0] > 4096:
                for j in range(0, X.shape[0], 4096):
                    ipca.partial_fit(X[j : j + 4096])
            else:
                ipca.partial_fit(X)
    return ipca


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Unbiased MMD^2 with RBF kernel (stream-friendly for mid-size windows).
    If gamma is None, use median heuristic on pooled sample.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    n, m = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    if gamma is None:
        # median heuristic on a small subsample
        subs = min(2000, Z.shape[0])
        idx = np.random.choice(Z.shape[0], size=subs, replace=False)
        d2 = np.sum((Z[idx, None, :] - Z[None, idx, :]) ** 2, axis=-1)
        med = np.median(d2[np.triu_indices_from(d2, k=1)])
        gamma = 1.0 / (2.0 * max(med, 1e-8))

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    # Unbiased estimators: remove diagonal terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = Kxx.sum() / (n * (n - 1) + 1e-8) + Kyy.sum() / (m * (m - 1) + 1e-8) - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))


def _window_indices_by_count(n: int, window_size: int, hop: int) -> List[np.ndarray]:
    """
    Produce overlapping windows of *indices* by count along capture order.
    Example: window_size=500, hop=250 -> 0:500, 250:750, 500:1000, ...
    """
    idxs = []
    if n <= 0 or window_size <= 1:
        return idxs
    for start in range(0, n - window_size + 1, hop):
        end = start + window_size
        idxs.append(np.arange(start, end, dtype=np.int64))
    return idxs


def _build_name_to_index(h5f) -> Dict[str, int]:
    names = h5f["image_names"][:].astype(str)
    # map filename only (strip dirs in H5 if present)
    keys = [n.replace("\\", "/").split("/")[-1] for n in names]
    return {k: i for i, k in enumerate(keys)}


def _depth_bins(values: np.ndarray, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_ids per value, bin_edges)."""
    if np.isnan(values).all():
        return np.full(values.shape, -1, dtype=int), np.array([])
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or bin_size <= 0:
        return np.full(values.shape, -1, dtype=int), np.array([])
    edges = np.arange(vmin, vmax + bin_size, bin_size)
    ids = np.digitize(values, edges, right=False) - 1
    return ids.astype(int), edges

# ---------- Core per-run computation ----------


def _project_window(h5f, indices: np.ndarray, sample_per_window: int, ipca: IncrementalPCA) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, ipca.n_components), dtype=np.float32)
    k = min(sample_per_window, indices.size)
    # evenly spaced sampling preserves order and saves IO
    if k < indices.size:
        sel = np.linspace(indices[0], indices[-1], num=k, dtype=np.int64)
    else:
        sel = indices
    X = read_rows_by_indices(h5f, sel)["features"]
    if X.shape[1] > ipca.n_components:
        Xp = ipca.transform(X)
    else:
        Xp = X - ipca.mean_
    return Xp


def _drift_from_windows(h5f, windows: List[np.ndarray], sample_per_window: int, ipca: IncrementalPCA) -> List[float]:
    proj = []
    for w in windows:
        Xp = _project_window(h5f, w, sample_per_window, ipca)
        proj.append(Xp)
    mmds = []
    for i in range(len(proj) - 1):
        if proj[i].shape[0] == 0 or proj[i+1].shape[0] == 0:
            mmds.append(np.nan)
        else:
            mmds.append(_mmd_rbf(proj[i], proj[i+1], gamma=None))
    return mmds

# ---------- Public API ----------

def run_drift(
    parent_dir: str,
    out_dir: str,
    mode: str = "count",                  # 'count' | 'depth' | 'profile'
    window_size: int = 1000,              # for count-mode
    hop: int = 500,                       # for count-mode
    sample_per_window: int = 400,
    pca_dim: int = 50,
    bootstrap_per_run: int = 2000,
    depth_bin_size: float = 5.0,          # meters (or same units as object_depth)
    seed: int = 42,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Drift over windows defined by:
      - mode='count': sliding windows over capture sequence (index order)
      - mode='depth': grouped contiguous bins of object_depth (bin size)
      - mode='profile': grouped by acq_id (ordered by first appearance)

    Writes per-run CSV & optional PNG under out_dir/<run_id>/.
    """
    assert mode in {"count", "depth", "profile"}
    rng = np.random.default_rng(seed)
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    ipca = _fit_ipca_basis(runs, pca_dim=pca_dim, bootstrap_per_run=bootstrap_per_run, seed=seed)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open_h5(r["features"]) as h5f:
            n, _ = get_h5_shapes(h5f)
            if n == 0:
                continue

            if mode == "count":
                windows = _window_indices_by_count(n, window_size, hop)
                label_positions = list(range(len(windows)))  # integers for plotting x-axis

            else:
                # load metadata from input_path (via config.yaml)
                cfg = load_run_config(r["run_cfg"])
                meta = load_run_metadata(cfg["input_path"]) if cfg.get("input_path") else pd.DataFrame()
                name_to_idx = _build_name_to_index(h5f)

                # map metadata rows to H5 indices
                if not meta.empty:
                    idxs = []
                    depths = []
                    profs = []
                    for img, row in meta.iterrows():
                        i = name_to_idx.get(img)
                        if i is not None:
                            idxs.append(i)
                            depths.append(row.get("object_depth", np.nan))
                            profs.append(row.get("acq_id", None))
                    if not idxs:
                        # no overlap between metadata and H5 names
                        windows, label_positions = [], []
                    else:
                        idxs = np.asarray(sorted(idxs), dtype=np.int64)
                        depths = np.asarray(depths, dtype=float)
                        profs = np.asarray(profs, dtype=object)

                        if mode == "depth":
                            # depth bins; within each bin, keep capture order by index
                            bin_ids, edges = _depth_bins(depths, depth_bin_size)
                            windows = []
                            labels = []
                            for b in np.unique(bin_ids):
                                if b < 0:
                                    continue
                                mask = (bin_ids == b)
                                group = idxs[mask]
                                if group.size > 0:
                                    windows.append(group)
                                    # use mid-edge (bin center) as x-label
                                    if edges.size >= (b + 2):
                                        mid = 0.5 * (edges[b] + edges[b+1])
                                    else:
                                        mid = b
                                    labels.append(mid)
                            # sort by depth label
                            order = np.argsort(labels)
                            windows = [windows[i] for i in order]
                            label_positions = [labels[i] for i in order]

                        elif mode == "profile":
                            # group by acq_id; sort groups by first index appearance
                            dfm = pd.DataFrame({"idx": idxs, "acq_id": profs})
                            groups = []
                            labels = []
                            for g, sub in dfm.groupby("acq_id", dropna=False):
                                group = np.sort(sub["idx"].values.astype(np.int64))
                                if group.size > 0:
                                    groups.append(group)
                                    labels.append(str(g))
                            # order groups by first occurrence in capture sequence
                            first_idx = [g[0] for g in groups]
                            order = np.argsort(first_idx)
                            windows = [groups[i] for i in order]
                            label_positions = [labels[i] for i in order]
                else:
                    windows, label_positions = [], []

            # compute drift across consecutive windows
            mmds = _drift_from_windows(h5f, windows, sample_per_window, ipca)

        # save CSV
        df = pd.DataFrame({
            "run_id": run_id,
            "window_from": label_positions[:len(mmds)],
            "window_to":   label_positions[1:len(mmds)+1],
            "mmd2": mmds
        })
        csv_path = run_dir / f"drift_{mode}.csv"
        df.to_csv(csv_path, index=False)

        # optional plot
        if save_plots and not df.empty:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            x = np.arange(len(df))
            plt.plot(x, df["mmd2"], marker="o", linewidth=1)
            plt.title(f"Drift ({mode}) — {run_id}")
            plt.xlabel("Window step")
            plt.ylabel("MMD² (consecutive windows)")
            plt.tight_layout()
            plt.savefig(run_dir / f"drift_{mode}.png", dpi=180)
            plt.close()

    return {"per_run_dir": str(out_root)}
