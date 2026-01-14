import pandas as pd

# -------- CONFIG --------
tsv_path1 = r"C:\alr4\ecodata\merged_all.tsv"
tsv_path2 = r"C:\alr4\ecotaxa_export__TSV_19605_20260105_0951.tsv"
out_path  = r"C:\alr4\ecotaxa_export__TSV_19605_20260105_0951_path_corrected.tsv"

CHUNKSIZE1 = 500_000   # for df1 (mapping build)
CHUNKSIZE2 = 500_000   # for df2 (processing)
# ------------------------

# 1) Build object_id -> image_name lookup from df1 using chunks
# df1 has img_file_name; you previously removed first row: df1.iloc[1:]
# When streaming, we mimic that by skipping the first data row.
# NOTE: This skips the FIRST DATA ROW after header (same as df.iloc[1:]).
df1_iter = pd.read_csv(
    tsv_path1,
    sep="\t",
    usecols=["object_id", "img_file_name"],
    chunksize=CHUNKSIZE1,
    skiprows=[1],          # skip first data row (row index 1 in file)
    low_memory=False,
)

mapping = {}  # object_id -> image_name

for chunk in df1_iter:
    chunk = chunk.rename(columns={"img_file_name": "image_name"})
    # drop missing ids
    chunk = chunk.dropna(subset=["object_id"])
    # if duplicates exist, keep the first one seen
    # (dict update would keep last; so do "setdefault")
    for oid, img in zip(chunk["object_id"].astype(str), chunk["image_name"].astype(str)):
        mapping.setdefault(oid, img)

print(f"Built mapping for {len(mapping):,} object_id values.")

# 2) Stream df2, attach image_name, and write incrementally
first_write = True

df2_iter = pd.read_csv(
    tsv_path2,
    sep="\t",
    chunksize=CHUNKSIZE2,
    low_memory=False,
)

for chunk in df2_iter:
    # Rename column as you did
    chunk = chunk.rename(columns={"object_annotation_category": "class"})

    # Make sure object_id is comparable to mapping keys
    # (often object_id may be int; mapping uses str)
    chunk["object_id"] = chunk["object_id"].astype(str)

    # Attach image_name via map
    chunk["image_name"] = chunk["object_id"].map(mapping)

    # Write out
    chunk.to_csv(out_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
    first_write = False

print(f"Done. Output written to: {out_path}")
