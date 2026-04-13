#!/bin/sh

#SBATCH --job-name="optuna_motionclip_contrastive"
#SBATCH --partition=gpu-a100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-5
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/optuna/testSplitwise/%x_%A_%a.out

echo "Job started on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

SPLITS_TXT="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/splits/finetune_splits.txt"

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

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/tune_motionclip_optuna_splitwise.py \
  --mode contrastive \
  --study_name "motionclip_contrastive_${SPLIT_NAME}" \
  --storage "sqlite:////scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_dbs/positiveLoss/testSplitwise/motionclip_contrastive_${SPLIT_NAME}.db" \
  --n_trials 30 \
  --split_name "${SPLIT_NAME}" \
  --normal_classes ${NORMAL_CLASSES} \
  --contrastive_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/finetune_contrastive_split.py \
  --x_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/X.npy \
  --y_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --work_dir optuna_runs/testSplitwise \
  --train_fraction 0.8 \
  --val_fraction 0.1 \
  --epochs 100 \
  --patience 8 \
  --min_delta 0.0 \
  --num_trainable_blocks_fixed 2 \
  --n_classes_per_batch 3 \
  --scheduler_patience 2 \
  --scheduler_factor 0.5 \
  --scheduler_min_lr 1e-4 \
  --num_workers 4 \
  --pin_memory \
  --seed 42
