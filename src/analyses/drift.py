from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from src.index import build_run_index
from src.stream import open_h5, get_h5_shapes, read_rows_by_indices


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
            # strictly increasing indices help h5py performance
            idx = np.linspace(0, n - 1, num=k, dtype=np.int64)
            part = read_rows_by_indices(h5f, idx)
            X = part["features"]
            # partial fit on chunks if very large
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


# -------------------------
# Main API
# -------------------------

def run_drift_by_count(
    parent_dir: str,
    out_dir: str,
    window_size: int = 1000,
    hop: int = 500,
    sample_per_window: int = 400,
    pca_dim: int = 50,
    bootstrap_per_run: int = 2000,
    seed: int = 42,
    save_plots: bool = True,
) -> Dict[str, str]:
    """
    Representation drift across each run, using *count-based* sliding windows
    (no need for timestamps). Good default when chronological filenames/ordering
    reflect capture sequence in HDF5.

    Steps per run:
      1) Build sliding windows of size `window_size` with step `hop`
      2) From each window, sample up to `sample_per_window` rows (streamed)
      3) Project with a global IPCA basis (fit once across runs)
      4) Compute MMD^2 between consecutive windows
      5) Save a small CSV of drift vs. window index; optional PNG plot

    Returns a dict with 'per_run_dir'.
    """
    rng = np.random.default_rng(seed)
    runs = build_run_index(parent_dir)
    if runs.empty:
        raise RuntimeError(f"No runs found under: {parent_dir}")

    # Fit a common PCA basis once (small bootstrap per run)
    ipca = _fit_ipca_basis(runs, pca_dim=pca_dim, bootstrap_per_run=bootstrap_per_run, seed=seed)

    out_dirp = Path(out_dir)
    out_dirp.mkdir(parents=True, exist_ok=True)

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        feats_path = r["features"]

        with open_h5(feats_path) as h5f:
            n, d = get_h5_shapes(h5f)
            if n < window_size:
                # too small to window — skip gracefully
                continue

            win_idxs = _window_indices_by_count(n, window_size, hop)
            if not win_idxs:
                continue

            # For each window, sample subset and project with IPCA
            proj_windows: List[np.ndarray] = []
            for widx in win_idxs:
                k = min(sample_per_window, widx.size)
                # choose evenly spaced indices to keep reads sorted + diverse
                if k < widx.size:
                    sel = np.linspace(widx[0], widx[-1], num=k, dtype=np.int64)
                else:
                    sel = widx
                part = read_rows_by_indices(h5f, sel)
                Xw = part["features"]
                # project
                if Xw.shape[1] > ipca.n_components:
                    Xw = ipca.transform(Xw)
                else:
                    # if features already <= pca_dim, still center via ipca.mean_
                    Xw = (Xw - ipca.mean_)  # simple centering
                proj_windows.append(Xw)

        # Compute drift between consecutive projected windows
        records = []
        for i in range(len(proj_windows) - 1):
            X = proj_windows[i]
            Y = proj_windows[i + 1]
            mmd2 = _mmd_rbf(X, Y, gamma=None)
            records.append({"run_id": run_id, "window_i": i, "window_j": i + 1, "mmd2": mmd2})

        df_run = pd.DataFrame(records)
        run_dir = out_dirp / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        df_path = run_dir / "drift_by_count.csv"
        df_run.to_csv(df_path, index=False)

        # Optional plot
        if save_plots and not df_run.empty:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            plt.plot(df_run["window_j"], df_run["mmd2"], marker="o", linewidth=1)
            plt.title(f"Representation Drift (MMD²) — {run_id}")
            plt.xlabel("Window index")
            plt.ylabel("MMD² (consecutive windows)")
            plt.tight_layout()
            plt.savefig(run_dir / "drift_by_count.png", dpi=180)
            plt.close()

    return {"per_run_dir": str(out_dirp)}
