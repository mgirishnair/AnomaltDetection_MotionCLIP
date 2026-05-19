#!/bin/sh

#SBATCH --job-name="NTU_BABEL_overlap"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/dataAnalysis/%x_%A_%a.out

set -e

echo "Job started on $(hostname)"
echo "Job started at $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

module load miniconda3
module load cuda/11.7

source ~/.bashrc
conda activate motionclip_clone

unset PYTHONPATH
unset PYTHONHOME

SCRIPT_PATH="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/dataAnalysis/dataLabelOverlap.py"

BABEL_LABELS_PATH="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/babel60_labels.txt"
NTU_LABELS_PATH="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/ntu_labels.txt"

OUTPUT_DIR="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/dataAnalysis/ntu_babel_overlap/chunks/topK/new/labelOverlap"
OUTPUT_CSV="${OUTPUT_DIR}/ntu_babel_overlap_chunk_${SLURM_ARRAY_TASK_ID}.csv"


mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/mgirishnair/Thesis/SLURM_logs/dataAnalysis"

python "${SCRIPT_PATH}" \
  --ntu_labels_path "${NTU_LABELS_PATH}" \
  --babel_labels_path "${BABEL_LABELS_PATH}" \
  --output_csv "${OUTPUT_CSV}" \
  --top_k 5 \
  --entropy_temperature 0.05 \
  --use_prompt_averaging true \
  --use_hub_correction true
echo "Job finished at $(date)"
echo "Saved chunk CSV: ${OUTPUT_CSV}"
