# This version reconstructs 165 missing file names from folder pattern

import pandas as pd

csv_path1 = r"C:\alr4\ecodata\d\merged.tsv"
csv_path2 = r"C:\alr4\subset_d_samples.tsv"

df1 = pd.read_csv(csv_path1, sep='\t', low_memory=False)
df2 = pd.read_csv(csv_path2, sep='\t')

df2 = df2.rename(columns={"object_annotation_category": "class"})
df1 = df1.rename(columns={"img_file_name": "image_name"})

df_merged = df2.merge(
    df1[['object_id', 'image_name']],
    on='object_id',
    how='left'
)

# Reconstruct image_name for _1 entries missing from df1
df1['base_id'] = df1['object_id'].str.rsplit('_', n=1).str[0]
df1_2s = df1[df1['object_id'].str.endswith('_2')][['base_id', 'image_name']].copy()
df1_2s['folder'] = df1_2s['image_name'].str.rsplit('/', n=1).str[0]
folder_lookup = df1_2s.set_index('base_id')['folder']

mask = df_merged['image_name'].isna()
df_merged.loc[mask, 'image_name'] = (
    df_merged.loc[mask, 'object_id']
    .map(lambda oid: folder_lookup.get('_'.join(oid.split('_')[:-1]), None))
    .astype(str)
    + '/'
    + df_merged.loc[mask, 'object_id']
    + '.png'
)

df_results_filename_tsv = r'C:\alr4\subset_d_samples_path_corrected.tsv'
df_merged.to_csv(df_results_filename_tsv, sep="\t", index=False)

print(f"Done. Total rows: {len(df_merged)}, Missing image_name: {df_merged['image_name'].isna().sum()}")