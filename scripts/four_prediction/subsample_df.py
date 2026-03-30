import pandas as pd
import h5py
import numpy as np


def collect_feature_indices(h5f, keep_names):
    names = h5f["image_names"][:].astype(str)
    names = [n.replace("\\", "/") for n in names]

    keep_idx = [
        i for i, n in enumerate(names)
        if n in keep_names
    ]

    return np.array(keep_idx, dtype=int)


def write_trimmed_h5(
    src_h5_path: str,
    dst_h5_path: str,
    keep_idx: np.ndarray,
):
    with h5py.File(src_h5_path, "r") as src, \
         h5py.File(dst_h5_path, "w") as dst:

        for key in src.keys():
            data = src[key]

            # only trim datasets with matching first dimension
            if data.shape[0] == src["image_names"].shape[0]:
                dst.create_dataset(
                    key,
                    data=data[keep_idx],
                    compression="gzip",
                )
            else:
                # copy untouched (e.g. metadata)
                src.copy(key, dst)

        # copy root attributes
        for k, v in src.attrs.items():
            dst.attrs[k] = v



path = r'C:\alr4\ai_predict_all\prediction_parti20260119121141'
df_name = r'\predictions_with_top3_scores.csv'
df_name_ = r'\predictions_with_top3_scores_s.csv'
# labels_keep = {"detritus", "copepoda eggs", "Rhizaria"}
labels_keep = {"Calanoida", "Rhizaria"}

df = pd.read_csv(path + df_name)
mask = (
    df["Image Name"].str.contains(r"d0001|d0002|d0003|d0004", regex=True, na=False)
    & (df["Top-1 Confidence Score"] >= 0.4)
    & (df["Top-1 Predicted Label"].isin(labels_keep))
)

new_df = df.loc[mask].reset_index(drop=True)


new_df.to_csv(path + df_name_, index=False)


keep_names = set(
    new_df["Image Name"]
        .str.replace("\\", "/", regex=False)
        .tolist()
)

features_name = r'\features_contrastive20250326162033.h5'
out_features_name = r'\features_contrastive20250326162033_s.h5'


with h5py.File(path + features_name, "r") as h5f:
    keep_idx = collect_feature_indices(h5f, keep_names)

write_trimmed_h5(
    path + features_name,
    path + out_features_name,
    keep_idx,
)

print(f"Kept {len(keep_idx)} feature vectors")







