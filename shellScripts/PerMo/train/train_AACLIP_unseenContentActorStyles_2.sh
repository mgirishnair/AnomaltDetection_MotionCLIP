#!/bin/bash
#SBATCH --job-name="Train_AACLIP_SeedArray_unseenContentActorStyles_2"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/train/%x_%A_%a.out

# Use an explicitly supplied seed file, or seeds.txt beside this script.

SEED_FILE="seeds.txt"

if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: Seed file not found: $SEED_FILE"
    echo "Usage: sbatch $0 [path/to/seeds.txt]"
    exit 1
fi

# Accept both comma-separated and newline-separated seed files.
mapfile -t SEEDS < <(
    tr ',' '\n' < "$SEED_FILE" |
    sed 's/^[[:space:]]*//;s/[[:space:]]*$//' |
    grep -v '^$'
)

if [ "${#SEEDS[@]}" -eq 0 ]; then
    echo "ERROR: No seeds found in $SEED_FILE"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: Submit this script as a Slurm array job using sbatch."
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID, but only ${#SEEDS[@]} seeds were found."
    echo "Fix #SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))"
    exit 1
fi

ORDER_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_SEED=42
SPLIT_SEED=0

if ! [[ "$TRAIN_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Seed is not a non-negative integer: $TRAIN_SEED"
    exit 1
fi

OUT_DIR="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/train_differentSplitSeeds/noContrastiveLoss/unseenContentActorStyles/2_UnseenStyles/Seed${ORDER_SEED}"

mkdir -p "$OUT_DIR"

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"
echo "Array task ID=${SLURM_ARRAY_TASK_ID}"
echo "Seed file=${SEED_FILE}"
echo "Split seed=${SPLIT_SEED}"
echo "Train seed=${TRAIN_SEED}"
echo "Order seed=${ORDER_SEED}"
echo "Output dir=${OUT_DIR}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/train/train_AACLIP_noContras.py \
  --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
  --output_dir "$OUT_DIR" \
  --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
  --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --actor_col actor \
  --split_seed "$SPLIT_SEED" \
  --train_seed "$TRAIN_SEED" \
  --order_seed "$ORDER_SEED" \
  --normal_target 200 \
  --anomaly_target 200 \
  --test_fraction 0.20 \
  --val_fraction 0.10 \
  --stage1_epochs 8 \
  --stage2_epochs 25 \
  --stage1_lr 2.596e-06 \
  --stage2_lr 0.000109 \
  --stage1_weight_decay 0.000483 \
  --stage2_weight_decay 0.000931 \
  --stage1_disentangle_weight 0.6604 \
  --temperature 0.0107 \
  --batch_size 8 \
  --num_workers 2 \
  --stage1_templates_per_batch 1 \
  --early_stopping_patience 8 \
  --unseen_actions kick "kick something" throw wave \
  --unseen_actors A01 A05 \
  --unseen_styles drunken exhausted

#  --stage2_contrastive_weight 0.00861 \
