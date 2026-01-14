import pandas as pd

csv_path1 = r"C:\alr4\ecodata\d\merged.tsv"
csv_path2 = r"C:\alr4\subset_d_samples.tsv"

df1 = pd.read_csv(csv_path1, sep='\t')
df2 = pd.read_csv(csv_path2, sep='\t')

df2 = df2.rename(columns={"object_annotation_category": "class"})
# df2["image_name"] = df1["object_id"].copy()

df1 = df1.iloc[1:].reset_index(drop=True)
df1 = df1.rename(columns={"img_file_name": "image_name"})

df_merged = (
    df2
    .merge(
        df1[['object_id', 'image_name']],
        on='object_id',
        how='left'
    )
)

df_results_filename_tsv = r'C:\alr4\subset_d_samples_path_corrected.tsv'
df_merged.to_csv(df_results_filename_tsv, sep="\t", index=False)
