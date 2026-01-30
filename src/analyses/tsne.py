from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.index import build_group_index
from src.analyses.common import collect_group_samples


TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12
CBAR_FONTSIZE  = 12


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


def compute_extent(lon, lat, expand_factor=1.5):
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    return [
        lon_min - expand_factor * lon_range,
        lon_max + expand_factor * lon_range,
        lat_min - expand_factor * lat_range,
        lat_max + expand_factor * lat_range,
    ]


def _plot_transect_map_tiles(
    gplot: pd.DataFrame,
    out_png: Path,
    title: str,
    zoom: int = 8,
    dot_size: float = 60.0,
):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # Cartopy tile sources
    from cartopy.io.img_tiles import OSM  # street-map tiles (OpenStreetMap)

    lon = gplot["group_lon"].to_numpy()
    lat = gplot["group_lat"].to_numpy()

    pad_lon = max(0.05, 0.05 * (lon.max() - lon.min() + 1e-9))
    pad_lat = max(0.05, 0.05 * (lat.max() - lat.min() + 1e-9))
    extent = [lon.min() - pad_lon, lon.max() + pad_lon, lat.min() - pad_lat, lat.max() + pad_lat]

    tiler = OSM()
    mercator = tiler.crs  # WebMercator

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=mercator)

    extent = compute_extent(
        gplot["group_lon"].to_numpy(),
        gplot["group_lat"].to_numpy(),
        expand_factor=1.2,  # try 1.2, 1.5, 2.0
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add the actual tile imagery (this is the "google-map-ish" part)
    ax.add_image(tiler, zoom)

    # Overlay your points (still in PlateCarree lon/lat)
    sc = ax.scatter(
        gplot["group_lon"], gplot["group_lat"],
        s=dot_size,
        c=gplot["transect_c"].values,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        alpha=0.95,
        edgecolors="k",
        linewidths=0.2,
        zorder=5,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Transect position (0→1)")

    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run_tsne_map(
    parent_dir: str,
    out_dir: str,
    group_mode: str = "run",
    group_col: str = "sample_id",
    sample_per_group: int = 2000,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    seed: int = 42,
    dot_size: float = 12.0,
) -> pd.DataFrame:
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

    # ------------------------------------------------------------------
    # ✅ Build a "transect colour" per group_id using lat/lon ordering
    # ------------------------------------------------------------------
    transect_df = None
    if group_mode == "meta" and ("group_lat" in groups.columns) and ("group_lon" in groups.columns):
        transect_df = groups[["run_id", "group_id", "group_lat", "group_lon"]].copy()

        # numeric safety
        transect_df["group_lat"] = pd.to_numeric(transect_df["group_lat"], errors="coerce")
        transect_df["group_lon"] = pd.to_numeric(transect_df["group_lon"], errors="coerce")

        # define along-track ordering:
        # simplest: sort by lon then lat (often OK for a transect)
        # (if your transect is mostly N-S, switch to sort by lat then lon)
        transect_df = transect_df.sort_values(["run_id", "group_lon", "group_lat"]).reset_index(drop=True)

        # rank within each run so colours are consistent within a run
        transect_df["transect_rank"] = transect_df.groupby("run_id").cumcount()

        # normalise to 0..1 within each run
        def _norm01(x: pd.Series) -> pd.Series:
            if len(x) <= 1:
                return pd.Series(np.full(len(x), 0.5), index=x.index)
            mn, mx = float(x.min()), float(x.max())
            if mx <= mn:
                return pd.Series(np.full(len(x), 0.5), index=x.index)
            return (x - mn) / (mx - mn)

        transect_df["transect_c"] = transect_df.groupby("run_id")["transect_rank"].transform(_norm01)

    # ------------------------------------------------------------------
    # Sample features per group + attach preds (your existing pipeline)
    # ------------------------------------------------------------------
    X_all, meta_all = collect_group_samples(
        groups=groups,
        group_mode=group_mode,
        sample_per_group=sample_per_group,
        rng=rng,
        attach_preds=True,
    )

    # PCA
    if X_all.shape[1] > pca_dim:
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
        X_pca = ipca.fit_transform(X_all)
    else:
        X_pca = X_all

    # t-SNE
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

    if transect_df is not None and ("group_id" in out_df.columns):
        out_df = out_df.merge(
            transect_df[["run_id", "group_id", "group_lat", "group_lon", "transect_rank", "transect_c"]],
            on=["run_id", "group_id"],
            how="left",
        )

    # Save CSV
    out_csv_path = out_root / f"tsne_{group_mode}.csv"
    out_df.to_csv(out_csv_path, index=False)

    # ------------------------------------------------------------------
    # ✅ Plots
    # ------------------------------------------------------------------
    save_plot = True
    if save_plot:
        # 1) t-SNE plot
        plt.figure(figsize=(14, 10))

        if group_mode == "meta" and ("transect_c" in out_df.columns) and out_df["transect_c"].notna().any():
            sc = plt.scatter(
                out_df["tsne_x"], out_df["tsne_y"],
                s=dot_size, alpha=0.75,
                c=out_df["transect_c"].values,
                cmap="viridis",
            )
            cbar = plt.colorbar(sc)
            cbar.set_label("Transect position (0→1)")
            title = f"t-SNE coloured by transect position (based on {group_col} lat/lon)"
        else:
            # fallback: categorical colouring
            color_key = "group_id" if group_mode == "meta" else "run_id"
            cats = out_df[color_key].astype("category")
            codes = cats.cat.codes.to_numpy()
            n_cats = len(cats.cat.categories)
            cmap = plt.get_cmap("gist_ncar", n_cats)

            sc = plt.scatter(
                out_df["tsne_x"], out_df["tsne_y"],
                s=dot_size, alpha=0.75,
                c=codes, cmap=cmap, vmin=0, vmax=max(0, n_cats - 1),
            )
            cbar = plt.colorbar(sc, ticks=np.arange(n_cats))
            cbar.set_label(color_key)
            cbar.set_ticklabels(cats.cat.categories)
            title = f"t-SNE grouped by {color_key}"

        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(out_root / f"tsne_{group_mode}.png", dpi=300)
        plt.close()

        # 2) Real map plot (Cartopy basemap)
        if group_mode == "meta" and ("group_lat" in out_df.columns) and ("group_lon" in out_df.columns):
            gplot = out_df.dropna(subset=["group_lat", "group_lon", "transect_c"]).drop_duplicates(
                subset=["run_id", "group_id"]
            )
            if not gplot.empty:
                try:
                    _plot_transect_map_tiles(
                        gplot=gplot,
                        out_png=out_root / f"transect_map_{group_mode}_cartopy.png",
                        title=f"ALR track coloured by {group_col} (same colours as t-SNE)",
                        dot_size=70.0,
                    )
                except Exception as e:
                    print(f"[WARN] Cartopy map failed, falling back to plain lon/lat scatter: {e}")
                    # fallback: plain scatter (no basemap)
                    plt.figure(figsize=(10, 8))
                    sc2 = plt.scatter(
                        gplot["group_lon"], gplot["group_lat"],
                        s=70, alpha=0.9,
                        c=gplot["transect_c"].values,
                        cmap="viridis",
                    )
                    plt.colorbar(sc2, label="Transect position (0→1)")
                    plt.xlabel("Longitude");
                    plt.ylabel("Latitude")
                    plt.title(f"Track coloured by {group_col} (fallback)")
                    plt.tight_layout()
                    plt.savefig(out_root / f"transect_map_{group_mode}_fallback.png", dpi=300)
                    plt.close()

    return out_df

