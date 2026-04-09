#!/bin/bash
#
#SBATCH --job-name="pca_plot_means"
#SBATCH --partition=compute-p1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-19
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/plot_pca/%x_%A_%a.out

module load miniconda3
module load 2024r1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $HOME/.conda/envs/motionclip_viz || exit 1

which python
python --version

#module load py-numpy
#module load py-matplotlib/3.7.1
#module load py-scikit-learn

SPLITS_TXT="/scratch/mgirishnair/MotionCLIP_experiment/splits.txt"

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NUM}p" "$SPLITS_TXT")

if [ -z "$LINE" ]; then
    echo "Error: no line found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Raw split line: $LINE"

SPLIT_NAME=$(echo "$LINE" | cut -d':' -f1 | xargs)
CLASS_STR=$(echo "$LINE" | cut -d':' -f2- | xargs)

if [ -z "$SPLIT_NAME" ] || [ -z "$CLASS_STR" ]; then
    echo "Error: could not parse split line: $LINE"
    exit 1
fi

echo "Parsed split name: $SPLIT_NAME"
echo "Parsed normal classes: $CLASS_STR"

EMB_DIR="/scratch/mgirishnair/MotionCLIP_experiment/embeddings"
OUT_DIR="/scratch/mgirishnair/MotionCLIP_experiment/plots/all"

EMB_PATH="${EMB_DIR}/motionclip_embeddings_${SPLIT_NAME}_allData.npz"
OUT_PATH="${OUT_DIR}/pca_${SPLIT_NAME}.png"

if [ ! -f "$EMB_PATH" ]; then
    echo "Error: embeddings file not found: $EMB_PATH"
    exit 1
fi

mkdir -p "$OUT_DIR"

python /scratch/mgirishnair/MotionCLIP_experiment/pca_plots/plot_pca_class_means.py \
    --embeddings_path "$EMB_PATH" \
    --output_path "$OUT_PATH" \
    --fit_on train \
    --max_abnormal_points 200 \
    --make_normals_only_plot
