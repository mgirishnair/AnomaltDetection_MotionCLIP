#!/bin/sh

#SBATCH --job-name="motionclip_finetune_contrastiveSplit_positiveLoss"
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-5
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/finetune/%x_%A_%a.out

echo "Job started on $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# go to project folder
cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

# activate environment
source ~/.bashrc
conda activate motionclip

SPLITS_TXT="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/splits/finetune_splits.txt"
HPARAMS_TXT="/scratch/mgirishnair/Thesis/SLURM_logs/optuna/contrastive_splitwise_positiveLoss/output_summary.txt"


# 0-based: task 0 reads line 1, task 1 reads line 2, ...
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

HP_LINE=$(grep -E "^${SPLIT_NAME}:" "${HPARAMS_TXT}" || true)

if [ -z "${HP_LINE}" ]; then
  echo "Error: no hyperparameter line found for split '${SPLIT_NAME}' in ${HPARAMS_TXT}"
  exit 1
fi

echo "Raw hyperparameter line: ${HP_LINE}"

HP_STR=$(echo "${HP_LINE}" | cut -d':' -f2-)

LR_ENCODER=$(echo "${HP_STR}" | sed -n 's/.*lr_encoder=\([^,]*\).*/\1/p' | xargs)
CONTRASTIVE_TEMP=$(echo "${HP_STR}" | sed -n 's/.*contrastive_temp=\([^,]*\).*/\1/p' | xargs)
N_SAMPLES_PER_CLASS=$(echo "${HP_STR}" | sed -n 's/.*n_samples_per_class=\([^,]*\).*/\1/p' | xargs)
WEIGHT_DECAY=$(echo "${HP_STR}" | sed -n 's/.*weight_decay=\([^,]*\).*/\1/p' | xargs)

if [ -z "${LR_ENCODER}" ] || [ -z "${CONTRASTIVE_TEMP}" ] || [ -z "${N_SAMPLES_PER_CLASS}" ] || [ -z "${WEIGHT_DECAY}" ]; then
  echo "Error: failed to parse one or more hyperparameters for split '${SPLIT_NAME}'"
  exit 1
fi

# keep this fixed unless you also tuned it
N_CLASSES_PER_BATCH=3
BATCH_SIZE=$((N_CLASSES_PER_BATCH * N_SAMPLES_PER_CLASS))

echo "Parsed hyperparameters:"
echo "  lr_encoder=${LR_ENCODER}"
echo "  contrastive_temp=${CONTRASTIVE_TEMP}"
echo "  n_samples_per_class=${N_SAMPLES_PER_CLASS}"
echo "  weight_decay=${WEIGHT_DECAY}"
echo "  n_classes_per_batch=${N_CLASSES_PER_BATCH}"
echo "  batch_size=${BATCH_SIZE}"

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/finetune_contrastive_split.py \
  --x_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/X.npy \
  --y_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --split_name "${SPLIT_NAME}" \
  --normal_classes ${NORMAL_CLASSES} \
  --train_fraction 0.8 \
  --epochs 150 \
  --batch_size ${BATCH_SIZE} \
  --num_trainable_blocks 2 \
  --lr_encoder ${LR_ENCODER} \
  --weight_decay ${WEIGHT_DECAY} \
  --contrastive_temp ${CONTRASTIVE_TEMP} \
  --use_class_aware_sampler \
  --n_classes_per_batch ${N_CLASSES_PER_BATCH} \
  --n_samples_per_class ${N_SAMPLES_PER_CLASS} \
  --val_fraction 0.1 \
  --patience 8 \
  --pin_memory \
  --scheduler_patience 3 \
  --scheduler_factor 0.7 \
  --scheduler_min_lr 1e-7 \
  --min_delta 0.0 \
  --num_workers 2 \
  --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/finetune/contrastive_splitwise_positiveLoss \
  --save_checkpoint motionclip_finetuned_${SPLIT_NAME}.pth \
  --save_metrics_npz finetune_metrics_${SPLIT_NAME}.npz \
  --save_summary finetune_summary_${SPLIT_NAME}.json
