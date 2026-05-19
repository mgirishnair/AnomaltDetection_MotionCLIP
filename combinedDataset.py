import os
import numpy as np

dir_1_60 = "/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_1-60"
dir_61_120 = "/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_61-120"
out_dir = "/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_1-120"

os.makedirs(out_dir, exist_ok=True)

X1 = np.load(os.path.join(dir_1_60, "X.npy"), mmap_mode="r")
y1 = np.load(os.path.join(dir_1_60, "y.npy"))

X2 = np.load(os.path.join(dir_61_120, "X.npy"), mmap_mode="r")
y2 = np.load(os.path.join(dir_61_120, "y.npy"))

print("X1:", X1.shape, "y1:", y1.shape, "labels:", y1.min(), y1.max())
print("X2:", X2.shape, "y2:", y2.shape, "labels:", y2.min(), y2.max())

assert X1.shape[1:] == X2.shape[1:], f"Shape mismatch: {X1.shape} vs {X2.shape}"

# Only shift y2 if it is still labelled 0-59 or 1-60
if y2.min() == y1.min() and y2.max() == y1.max():
    print("Shifting y2 labels by +60")
    y2 = y2 + 60
else:
    print("Not shifting y2 labels")

X_combined = np.concatenate([X1, X2], axis=0)
y_combined = np.concatenate([y1, y2], axis=0)

print("Combined X:", X_combined.shape)
print("Combined y:", y_combined.shape)
print("Combined labels:", y_combined.min(), y_combined.max())
print("Unique labels:", len(np.unique(y_combined)))

np.save(os.path.join(out_dir, "X.npy"), X_combined)
np.save(os.path.join(out_dir, "y.npy"), y_combined)

for fname in ["filenames.txt", "label_names.txt"]:
    with open(os.path.join(dir_1_60, fname), "r") as f:
        lines1 = f.readlines()
    with open(os.path.join(dir_61_120, fname), "r") as f:
        lines2 = f.readlines()

    with open(os.path.join(out_dir, fname), "w") as f:
        f.writelines(lines1)
        f.writelines(lines2)

print("Saved combined dataset to:", out_dir)
