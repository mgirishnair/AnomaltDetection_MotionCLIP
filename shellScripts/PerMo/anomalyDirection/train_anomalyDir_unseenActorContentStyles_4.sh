#!/bin/bash
#SBATCH --job-name="Train_anomalyDir_e2eMLP_unseenContentActorStyles_4"
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

OUT_DIR="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/anomalyDirection/e2eMLP/unseenContentActorStyles/4_UnseenStyles/Seed${ORDER_SEED}"

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

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/anomalyDirection/train_anomalyDir_e2eMLP.py \
  --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
  --output_dir "$OUT_DIR" \
  --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
  --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --split_seed "$SPLIT_SEED" \
  --train_seed "$TRAIN_SEED" \
  --order_seed "$ORDER_SEED" \
  --normal_target 200 \
  --anomaly_target 200 \
  --test_fraction 0.20 \
  --val_fraction 0.10 \
  --batch_size 32 \
  --pairs_per_batch 16 \
  --num_workers 2 \
  --epochs 26 \
  --lr 7.059e-05 \
  --weight_decay 0.000105 \
  --lambda_binary 0.882 \
  --lambda_direction 0.041 \
  --lambda_contrastive 0.01 \
  --lambda_residual_mlp 0.1 \
  --residual_mlp_hidden_dim 16 \
  --residual_mlp_dropout 0.1 \
  --temperature 0.0138 \
  --adapter_ratio 0.075 \
  --weak_pair_weight 0.776 \
  --max_pairs_per_anomaly 2 \
  --direction_init paired_raw \
  --alpha_init 0.301 \
  --grad_clip 1.0 \
  --early_stopping_patience 8 \
  --actor_col actor \
  --unseen_actions kick "kick something" throw wave \
  --unseen_actors A01 A05 \
  --unseen_styles drunken exhausted legaching textnecked

#  --lambda_action 0.421 \
