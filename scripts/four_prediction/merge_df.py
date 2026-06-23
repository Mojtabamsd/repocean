
import pandas as pd

# path_eco_ref = r'C:\alr4\subset_d_samples_path_corrected.tsv'
path_eco_fine = r'C:\alr4\subset_d_samples_path_corrected.tsv'

path_root = r'C:\alr4\ai_predict\ai_predict_d_all'

# path_pro_ref = r"C:\alr4\ai_predict_all\prediction_parti20260106124204"
path_pro_ref = path_root + r"\prediction_parti20260119121141"
path_pro_fine = path_root + r"\prediction_parti20260324132632_fine"

df_name = r'\predictions_with_top3_scores.csv'
out_dir = path_root + r"\merge_three_prediction_all.csv"

# df_e_r = pd.read_csv(path_eco_ref, delimiter='\t')
df_e_f = pd.read_csv(path_eco_fine, delimiter='\t')

df_p_r = pd.read_csv(path_pro_ref + df_name)
df_p_f = pd.read_csv(path_pro_fine + df_name)

df_p_r = df_p_r.rename(columns={"Top-1 Predicted Label": "class_p_r"})
df_p_r = df_p_r.rename(columns={"Top-1 Confidence Score": "score_p_r"})
df_p_r = df_p_r.rename(columns={"Image Name": "image_name"})

df_p_f = df_p_f.rename(columns={"Top-1 Predicted Label": "class_p_f"})
df_p_f = df_p_f.rename(columns={"Top-1 Confidence Score": "score_p_f"})
df_p_f = df_p_f.rename(columns={"Image Name": "image_name"})


df_e_f = df_e_f.rename(columns={"class": "class_e_f"})

df_p_r["image_name"] = df_p_r["image_name"].astype(str).str.replace("\\", "/").str.strip()
df_p_f["image_name"] = df_p_f["image_name"].astype(str).str.replace("\\", "/").str.strip()
df_e_f["class_e_f"] = df_e_f["class_e_f"].astype(str).str.lower().str.strip()


df_merged = (
    df_e_f
    .merge(
        df_p_r[['image_name', 'class_p_r', 'score_p_r']],
        on='image_name',
        how='left'
    )
    .merge(
        df_p_f[['image_name', 'class_p_f', 'score_p_f']],
        on='image_name',
        how='left'
    )
)


# Stage 2 merge score ecotaxa as well
path_api = path_root + r"\ecotaxa_sample_d_export_api.csv"
df_api = pd.read_csv(path_api)


df_api = df_api.rename(columns={"obj.classif_auto_score": "score_e_f"})
df_api = df_api.rename(columns={"obj.orig_id": "object_id"})

df_merged = (
    df_merged
    .merge(
        df_api[['object_id', 'score_e_f']],
        on='object_id',
        how='left'
    )
)


df_merged.to_csv(out_dir, index=False)

unique_labels = pd.DataFrame(sorted(df_merged['class_e_f'].unique()))
unique_labels.to_csv(path_root + r"\class_unique_e.csv", index=False)


df_merged.to_csv(out_dir, index=False)





