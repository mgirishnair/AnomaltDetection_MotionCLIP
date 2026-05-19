import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("../dataOverlap_combined.csv")

# sort by overlap score (LOW = bad overlap)
#df_sorted = df.sort_values("overlap_score")
df_low = df.sort_values(
    ["mean_best_similarity", "proportion_above_0.75"],
    ascending=True
).head(25)
# take worst 25
#df_low = df_sorted.head(25)
plt.figure(figsize=(10, 7))
plt.barh(
    df_low["ntu_class"].astype(str),
    df_low["mean_best_similarity"]
)

plt.xlabel("Mean Best Similarity")
plt.ylabel("NTU Class")
plt.title("Lowest Overlap NTU Classes")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("plots/lowest_overlap_mean_similarity.png", dpi=300)
plt.show()
#
#plt.figure(figsize=(10, 7))
#plt.barh(df_low["ntu_class"].astype(str), df_low["overlap_score"])
#plt.xlabel("Overlap Score")
#plt.ylabel("NTU Class")
#plt.title("Lowest Overlap NTU Classes")
#plt.gca().invert_yaxis()
#plt.tight_layout()
#plt.savefig("plots/lowest_overlap_classes.png", dpi=300)
#plt.show()
