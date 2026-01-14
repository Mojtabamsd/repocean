import pandas as pd
import numpy as np

# path_eco_fine = r'C:\alr4\subset_d_samples_correct.tsv'
INPUT_PATH = r'C:\alr4\ecotaxa_export__TSV_19605_20260105_0951_path_corrected.tsv'
OUTPUT_PATH = r"C:\alr4\binned_sample_all.csv"


# SEP = ","
SEP = "\t"          # use "\t" for TSV
CHUNKSIZE = 1000_000

COL = "object_major"
KEY_COL = "image_name"
N_BINS = 20
K = 100                 # samples per bin
RANDOM_SEED = 42

# ----------------------------------------


def compute_min_max(path, sep, col, chunksize):
    min_v, max_v = None, None
    for chunk in pd.read_csv(path, sep=sep, chunksize=chunksize):
        s = pd.to_numeric(chunk[col], errors="coerce").dropna()
        if s.empty:
            continue
        cmin, cmax = float(s.min()), float(s.max())
        min_v = cmin if min_v is None else min(min_v, cmin)
        max_v = cmax if max_v is None else max(max_v, cmax)
    if min_v is None or max_v is None:
        raise ValueError(f"Could not compute min/max: column '{col}' has no numeric values.")
    if min_v == max_v:
        # Degenerate case: everything same value -> one bin effectively
        # We'll still build edges, but all values land in the last bin.
        max_v = min_v + 1e-9
    return min_v, max_v


def reservoir_sample_bins(path, sep, col, edges, k, chunksize, seed):
    rng = np.random.default_rng(seed)

    # Reservoir storage: one list per bin
    reservoirs = [[] for _ in range(len(edges) - 1)]
    seen_counts = np.zeros(len(edges) - 1, dtype=np.int64)

    for chunk in pd.read_csv(path, sep=sep, chunksize=chunksize):
        # Ensure numeric
        vals = pd.to_numeric(chunk[col], errors="coerce")
        mask = vals.notna()
        if not mask.any():
            continue

        chunk = chunk.loc[mask].copy()
        vals = vals.loc[mask].to_numpy()

        # Assign bin indices (0..N_BINS-1)
        # right=False -> [edge_i, edge_{i+1})
        bin_idx = np.searchsorted(edges, vals, side="right") - 1

        # clamp values that might fall on max edge due to floating error
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

        # Stream each row and apply per-bin reservoir sampling
        for i, b in enumerate(bin_idx):
            seen_counts[b] += 1
            row = chunk.iloc[i].to_dict()

            # Optional: keep bin info in output
            row["_bin_id"] = int(b)
            row["_bin_left"] = float(edges[b])
            row["_bin_right"] = float(edges[b + 1])

            n = seen_counts[b]
            if len(reservoirs[b]) < k:
                reservoirs[b].append(row)
            else:
                j = rng.integers(0, n)  # uniform in [0, n-1]
                if j < k:
                    reservoirs[b][j] = row

    return reservoirs, seen_counts


def main():
    # Pass 1: min/max for equal-width bins
    min_v, max_v = compute_min_max(INPUT_PATH, SEP, COL, CHUNKSIZE)
    edges = np.linspace(min_v, max_v, N_BINS + 1)

    print(f"Min={min_v}, Max={max_v}")
    print(f"Bin edges: [{edges[0]}, ..., {edges[-1]}] ({N_BINS} bins)")

    # Pass 2: reservoir sample per bin
    reservoirs, seen_counts = reservoir_sample_bins(
        INPUT_PATH, SEP, COL, edges, K, CHUNKSIZE, RANDOM_SEED
    )

    # Flatten reservoirs to one dataframe
    sampled_rows = [r for bin_res in reservoirs for r in bin_res]
    out_df = pd.DataFrame(sampled_rows)

    # Sort nicely by bin then (optionally) by object_major
    if not out_df.empty:
        out_df = out_df.sort_values(by=["_bin_id"])

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows to: {OUTPUT_PATH}")

    # Quick report: how many total seen per bin vs sampled
    report = pd.DataFrame({
        "bin_id": np.arange(N_BINS),
        "seen_in_bin": seen_counts,
        "sampled": [len(r) for r in reservoirs],
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
    })
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()



