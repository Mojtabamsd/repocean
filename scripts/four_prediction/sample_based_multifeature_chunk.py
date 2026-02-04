import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

INPUT_PATH = r"C:\alr4\subset_d_samples_path_corrected.tsv"
OUTPUT_PATH = r"C:\alr4\sample_multifeature_representative.csv"


# INPUT_PATH = r'C:\alr4\ecotaxa_export__TSV_19605_20260105_0951_path_corrected.tsv'
# OUTPUT_PATH = r"C:\alr4\log_binned_sample_all.csv"

SEP = "\t"
CHUNKSIZE = 1_000_000
RANDOM_SEED = 42

# - size: object_major (log)
# - shape: object_elongation, object_circ.
# - brightness/texture proxies: object_mean, object_stddev
FEATURES = ["object_major", "object_elongation", "object_mean", "object_stddev", "object_circ."]

# Number of clusters in feature space (controls diversity granularity)
N_CLUSTERS = 20

# Sampling plan:
# Option A: fixed samples per cluster (gives equal coverage of clusters)
SAMPLES_PER_CLUSTER = 100

# Optional cap on total samples (keeps output size bounded)
# Set to None to keep all cluster samples
MAX_TOTAL_SAMPLES = None


def _prepare_features(chunk: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return cleaned feature matrix with log-size and numeric coercion, dropping rows with NaNs."""
    tmp = chunk.copy()

    # ensure all features exist
    for c in features:
        if c not in tmp.columns:
            tmp[c] = np.nan

    # numeric coercion
    for c in features:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # log-transform object_major for heavy tails
    if "object_major" in features:
        tmp["object_major"] = np.log10(np.clip(tmp["object_major"].to_numpy(dtype=float), 1e-6, None))

    # drop rows with missing feature values
    tmp = tmp.dropna(subset=features)

    X = tmp[features].to_numpy(dtype=float)
    return X, tmp.index


def train_kmeans_streaming(path: str) -> tuple[StandardScaler, MiniBatchKMeans]:
    """
    Pass 1:
    - Fit a StandardScaler incrementally
    - Fit MiniBatchKMeans incrementally
    """
    scaler = StandardScaler()
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        batch_size=50_000,
        n_init="auto",
        reassignment_ratio=0.01,
    )

    # First fit scaler (partial_fit) over chunks, then kmeans (partial_fit) over scaled chunks.
    # We do both in one pass by: partial_fit scaler -> transform -> partial_fit kmeans.
    for chunk in pd.read_csv(path, sep=SEP, chunksize=CHUNKSIZE, low_memory=False):
        X, _ = _prepare_features(chunk, FEATURES)
        if X.size == 0:
            continue
        scaler.partial_fit(X)
        Xs = scaler.transform(X)
        kmeans.partial_fit(Xs)

    return scaler, kmeans


def reservoir_sample_per_cluster(path: str, scaler: StandardScaler, kmeans: MiniBatchKMeans):
    """
    Pass 2:
    - Predict cluster for each row (on scaled features)
    - Reservoir sample per cluster (uniform) to get SAMPLES_PER_CLUSTER each
    """
    rng = np.random.default_rng(RANDOM_SEED)

    reservoirs = {c: [] for c in range(N_CLUSTERS)}
    seen = np.zeros(N_CLUSTERS, dtype=np.int64)

    for chunk in pd.read_csv(path, sep=SEP, chunksize=CHUNKSIZE, low_memory=False):
        # Build feature matrix for valid rows
        X, valid_idx = _prepare_features(chunk, FEATURES)
        if X.size == 0:
            continue

        Xs = scaler.transform(X)
        clusters = kmeans.predict(Xs)

        valid_rows = chunk.loc[valid_idx].copy()

        # stream over valid rows and do reservoir per cluster
        for j, c in enumerate(clusters):
            c = int(c)
            seen[c] += 1
            row = valid_rows.iloc[j].to_dict()
            row["_cluster"] = c

            n = seen[c]
            if len(reservoirs[c]) < SAMPLES_PER_CLUSTER:
                reservoirs[c].append(row)
            else:
                r = rng.integers(0, n)
                if r < SAMPLES_PER_CLUSTER:
                    reservoirs[c][r] = row

    # flatten
    sampled = [row for c in range(N_CLUSTERS) for row in reservoirs[c]]

    # Optional: cap total size while keeping diversity (sample across clusters)
    if MAX_TOTAL_SAMPLES is not None and len(sampled) > MAX_TOTAL_SAMPLES:
        # group by cluster then sample evenly across clusters
        df_tmp = pd.DataFrame(sampled)
        per = max(1, MAX_TOTAL_SAMPLES // max(1, df_tmp["_cluster"].nunique()))
        df_tmp = (
            df_tmp.groupby("_cluster", group_keys=False)
                  .apply(lambda x: x.sample(min(len(x), per), random_state=RANDOM_SEED))
        )
        sampled = df_tmp.to_dict(orient="records")

    return sampled, seen


def main():
    print("[INFO] Pass 1: training scaler + MiniBatchKMeans on streaming chunks...")
    scaler, kmeans = train_kmeans_streaming(INPUT_PATH)

    print("[INFO] Pass 2: reservoir sampling per cluster...")
    sampled_rows, seen = reservoir_sample_per_cluster(INPUT_PATH, scaler, kmeans)

    out_df = pd.DataFrame(sampled_rows)

    # Optional: sort by cluster for readability
    if not out_df.empty and "_cluster" in out_df.columns:
        out_df = out_df.sort_values("_cluster")

    # out_df.to_csv(OUTPUT_PATH, sep=SEP, index=False)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved {len(out_df):,} rows to: {OUTPUT_PATH}")

    # Quick report
    report = pd.DataFrame({
        "cluster": np.arange(N_CLUSTERS),
        "seen": seen,
        "sampled": [min(SAMPLES_PER_CLUSTER, int(s)) for s in seen],
    })
    print(report.describe(include="all").to_string())
    print("[INFO] Example cluster counts (top 10 seen):")
    print(report.sort_values("seen", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
