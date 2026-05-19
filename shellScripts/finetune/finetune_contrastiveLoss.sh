#!/bin/sh

#SBATCH --job-name="motionclip_finetune_contrastiveLoss_fullContras"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/finetune/allData/fullContras/%x_%j.out

echo "Job started on $(hostname)"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

HPARAMS_TXT="/scratch/mgirishnair/Thesis/SLURM_logs/optuna/allData/fullContras/output_summary.txt"

if [ ! -f "${HPARAMS_TXT}" ]; then
  echo "Error: hyperparameter summary file not found: ${HPARAMS_TXT}"
  exit 1
fi

# Expect a line like:
# global_all_classes: lr_encoder=..., contrastive_temp=..., n_samples_per_class=..., weight_decay=...
HP_LINE=$(grep -E "^global_all_classes:" "${HPARAMS_TXT}" || true)

if [ -z "${HP_LINE}" ]; then
  echo "Error: no hyperparameter line found for 'global_all_classes' in ${HPARAMS_TXT}"
  echo "File contents:"
  cat "${HPARAMS_TXT}"
  exit 1
fi

echo "Raw hyperparameter line: ${HP_LINE}"

HP_STR=$(echo "${HP_LINE}" | cut -d':' -f2-)

LR_ENCODER=$(echo "${HP_STR}" | sed -n 's/.*lr_encoder=\([^,]*\).*/\1/p' | xargs)
CONTRASTIVE_TEMP=$(echo "${HP_STR}" | sed -n 's/.*contrastive_temp=\([^,]*\).*/\1/p' | xargs)
N_SAMPLES_PER_CLASS=$(echo "${HP_STR}" | sed -n 's/.*n_samples_per_class=\([^,]*\).*/\1/p' | xargs)
WEIGHT_DECAY=$(echo "${HP_STR}" | sed -n 's/.*weight_decay=\([^,]*\).*/\1/p' | xargs)

if [ -z "${LR_ENCODER}" ] || [ -z "${CONTRASTIVE_TEMP}" ] || [ -z "${N_SAMPLES_PER_CLASS}" ] || [ -z "${WEIGHT_DECAY}" ]; then
  echo "Error: failed to parse one or more hyperparameters from:"
  echo "${HP_LINE}"
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

GPU_LOG="/scratch/mgirishnair/Thesis/SLURM_logs/finetune/allData/fullContras/gpu_usage_${SLURM_JOB_ID}.log"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "${GPU_LOG}" &
GPU_MONITOR_PID=$!

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/finetune_contrastive_allData.py \
  --x_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/X.npy \
  --y_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --run_name global_all_classes \
  --train_fraction 0.8 \
  --val_fraction 0.1 \
  --epochs 150 \
  --batch_size ${BATCH_SIZE} \
  --num_trainable_blocks 2 \
  --lr_encoder ${LR_ENCODER} \
  --weight_decay ${WEIGHT_DECAY} \
  --contrastive_temp ${CONTRASTIVE_TEMP} \
  --use_class_aware_sampler \
  --n_classes_per_batch ${N_CLASSES_PER_BATCH} \
  --n_samples_per_class ${N_SAMPLES_PER_CLASS} \
  --patience 8 \
  --pin_memory \
  --scheduler_patience 3 \
  --scheduler_factor 0.7 \
  --scheduler_min_lr 1e-7 \
  --min_delta 0.0 \
  --num_workers 4 \
  --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/finetune/contrastive_allData/fullContras \
  --save_checkpoint motionclip_finetuned_global_all_classes.pth \
  --save_metrics_npz finetune_metrics_global_all_classes.npz \
  --save_summary finetune_summary_global_all_classes.json \
  --save_indices_npz global_split_indices.npz

PY_EXIT_CODE=$?

kill "${GPU_MONITOR_PID}" 2>/dev/null || true
wait "${GPU_MONITOR_PID}" 2>/dev/null || true

echo "Max GPU memory usage:"
awk 'max<$1 {max=$1} END {print max " MiB"}' "${GPU_LOG}"

exit ${PY_EXIT_CODE}
