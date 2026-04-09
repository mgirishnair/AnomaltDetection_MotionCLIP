#!/bin/bash
#SBATCH --job-name=check_npy
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-DSAIT


module load 2024r1
module load python
module load py-numpy

DATASET_DIR="/scratch/mgirishnair/nturgb+d_skeletons_npy"
OUTFILE="npy_check_report.txt"

python - <<'EOF' > "$OUTFILE"
import os
import glob
import numpy as np

dataset_dir = "/scratch/mgirishnair/nturgb+d_skeletons_npy"
files = sorted(glob.glob(os.path.join(dataset_dir, "*.npy")))

print(f"Checking {len(files)} files in {dataset_dir}\n")

bad = 0

for path in files:
    fname = os.path.basename(path)
    try:
        x = np.load(path)

        issues = []

        if x.ndim != 3:
            issues.append(f"wrong_ndim={x.ndim}")

        if x.ndim == 3 and x.shape[1:] != (25, 3):
            issues.append(f"wrong_shape={x.shape}")

        if x.size == 0:
            issues.append("empty_array")

        if np.isnan(x).any():
            issues.append("contains_nan")

        if np.isinf(x).any():
            issues.append("contains_inf")

        if np.all(x == 0):
            issues.append("all_zero")

        x_min = np.min(x)
        x_max = np.max(x)
        x_std = np.std(x)

        if x_std == 0:
            issues.append("zero_std")

        if np.allclose(x, x.flat[0]):
            issues.append("constant_array")

        if x.ndim == 3 and x.shape[0] == 0:
            issues.append("zero_frames")

        if issues:
            bad += 1
            print(f"{fname}")
            print(f"  shape={x.shape}, dtype={x.dtype}")
            print(f"  min={x_min}, max={x_max}, std={x_std}")
            print(f"  issues={', '.join(issues)}")
            print()

    except Exception as e:
        bad += 1
        print(f"{fname}")
        print(f"  issues=load_error: {e}")
        print()

print(f"Done. Suspicious files: {bad} / {len(files)}")
EOF

