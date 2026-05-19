import pandas as pd
import glob
import os

folder = "chunks/topK/new"   # change this
output_file = os.path.join(folder, "dataOverlap_combined.csv")

csv_files = glob.glob(os.path.join(folder, "*.csv"))

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv(output_file, index=False)

print(f"Combined {len(csv_files)} files into {output_file}")
