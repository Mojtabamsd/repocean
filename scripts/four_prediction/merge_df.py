
import pandas as pd

# path_eco_ref = r'C:\alr4\subset_d_samples_correct.tsv'
path_eco_fine = r'C:\alr4\subset_d_samples_correct.tsv'

path_pro_ref = r"C:\alr4\ai_predict_all\prediction_parti20260106124204"
path_pro_fine = r"C:\alr4\ai_predict_all\prediction_parti20260113104208"

df_name = r'\predictions_with_top3_scores.csv'
out_dir = r"C:\alr4\ai_predict_all\merge_four_prediction.csv"

# df_e_r = pd.read_csv(path_eco_ref, delimiter='\t')
df_e_f = pd.read_csv(path_eco_fine, delimiter='\t')

df_p_r = pd.read_csv(path_pro_ref + df_name)
df_p_f = pd.read_csv(path_pro_ref + df_name)

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


class_name = "copepoda<maxillopoda"
# class_name = "fiber<detritus"
# class_name = "eumalacostraca"
df2_filtered = df_p_r.merge(
    df_e_f[df_e_f["class"] == class_name][["image_name"]],
    on="image_name",
    how="inner"
)

summary = df2_filtered.groupby('class').size().reset_index(name='count')
print(summary)

df2_filtered.to_csv(out_dir)
df1=1




