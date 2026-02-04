import pandas as pd

path = r'C:\alr4\ai_predict_all\prediction_parti20260119121141'
df_name = r'\predictions_with_top3_scores.csv'
df_name_ = r'\predictions_with_top3_scores_s.csv'

df = pd.read_csv(path + df_name)
mask = (
    df["Image Name"].str.contains(r"d0001|d0002", regex=True, na=False)
    & (df["Top-1 Confidence Score"] >= 0.5)
)

new_df = df.loc[mask].reset_index(drop=True)


new_df.to_csv(path + df_name_, index=False)
