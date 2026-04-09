#!/bin/sh

#SBATCH --job-name="motionclip_computeEmbeddings_split_contrastive"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-5
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/embeddings/%x_%A_%a.out

echo "Job started on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd /scratch/mgirishnair/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

SPLITS_TXT="/scratch/mgirishnair/MotionCLIP_experiment/finetune_splits.txt"

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NUM}p" "${SPLITS_TXT}")

if [ -z "${LINE}" ]; then
  echo "Error: no line found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "Raw split line: ${LINE}"

SPLIT_NAME=$(echo "${LINE}" | cut -d':' -f1 | xargs)
CLASS_STR=$(echo "${LINE}" | cut -d':' -f2-)

if [ -z "${SPLIT_NAME}" ] || [ -z "${CLASS_STR}" ]; then
  echo "Error: could not parse split line: ${LINE}"
  exit 1
fi

NORMAL_CLASSES=$(echo "${CLASS_STR}" | tr ',' ' ' | xargs)

echo "Parsed split name: ${SPLIT_NAME}"
echo "Parsed normal classes: ${NORMAL_CLASSES}"

X_PATH="/scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/X.npy"
Y_PATH="/scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/y.npy"

FINETUNE_DIR="/scratch/mgirishnair/MotionCLIP_experiment/finetune/contrastive_splitwise/final"
SPLIT_INDICES_PATH="${FINETUNE_DIR}/${SPLIT_NAME}/${SPLIT_NAME}_split_indices.npz"
CHECKPOINT_PATH="${FINETUNE_DIR}/${SPLIT_NAME}/motionclip_finetuned_${SPLIT_NAME}.pth"

EMBEDDINGS_DIR="/scratch/mgirishnair/MotionCLIP_experiment/embeddings/contrastive_splitwise/final"
mkdir -p "${EMBEDDINGS_DIR}"

OUTPUT_PATH="${EMBEDDINGS_DIR}/motionclip_embeddings_${SPLIT_NAME}.npz"

if [ ! -f "${SPLIT_INDICES_PATH}" ]; then
  echo "Error: split indices file not found: ${SPLIT_INDICES_PATH}"
  exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Error: checkpoint file not found: ${CHECKPOINT_PATH}"
  exit 1
fi

python /scratch/mgirishnair/MotionCLIP_experiment/finetuned_embeddings.py \
  --x_path "${X_PATH}" \
  --y_path "${Y_PATH}" \
  --split_indices_path "${SPLIT_INDICES_PATH}" \
  --motionclip_repo MotionCLIP \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --batch_size 32 \
  --output_path "${OUTPUT_PATH}"
