#!/bin/sh

#SBATCH --job-name="computeAnomaly_allsplit_fullContras"
#SBATCH --partition=compute-p2
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-13
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/computeAnomaly/allSplits/%x_%A_%a.out

echo "Job started on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd /scratch/mgirishnair/MotionCLIP_experiment

module load miniconda3
module load 2024r1

source ~/.bashrc
conda activate motionclip

SPLITS_TXT="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/splits/finetune_allSplits.txt"

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

echo "Parsed split name: ${SPLIT_NAME}"

EMBEDDINGS_DIR="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/embeddings/allData/fullContras/allSplits"
EMBEDDINGS_PATH="${EMBEDDINGS_DIR}/motionclip_embeddings_${SPLIT_NAME}.npz"

if [ ! -f "${EMBEDDINGS_PATH}" ]; then
  echo "Error: embeddings file not found: ${EMBEDDINGS_PATH}"
  exit 1
fi

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/anomalyDetection/motionCLIP_anomalyDetection_computeAnomaly_multiModal.py \
  --embeddings_path "${EMBEDDINGS_PATH}" \
  --split "${SPLIT_NAME}" \
  --threshold_percentile 95 \
  --output_path "/scratch/mgirishnair/Thesis/MotionCLIP_experiment/results/finetune/allSplits/NTU60/fullContras/anomaly_results_${SPLIT_NAME}.txt"
