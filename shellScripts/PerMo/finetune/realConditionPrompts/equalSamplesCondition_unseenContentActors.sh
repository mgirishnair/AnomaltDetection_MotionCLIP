#!/bin/bash
#SBATCH --job-name=Finetune_unseenContentActors
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-4
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/finetune/realCondition/equalSamplesCondition/differentOrderSeeds/%x_%A_%a.out


SEED_FILE="seeds.txt"

if [ -z "$SEED_FILE" ]; then
    echo "ERROR: Missing seed file."
    echo "Usage: sbatch $0 /path/to/seeds.txt"
    exit 1
fi

if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: Seed file not found: $SEED_FILE"
    exit 1
fi

# Read comma-separated or newline-separated seeds into a bash array.
mapfile -t SEEDS < <(tr ',' '\n' < "$SEED_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$')

if [ "${#SEEDS[@]}" -eq 0 ]; then
    echo "ERROR: No seeds found in $SEED_FILE"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script should be submitted with sbatch as an array job."
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#SEEDS[@]} seeds found."
    echo "Fix #SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))"
    exit 1
fi

ORDER_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_SEED=42
SPLIT_SEED=0

if ! [[ "$SPLIT_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Seed is not a non-negative integer: $SPLIT_SEED"
    exit 1
fi

#JOB_TAG="UnseenContentActorStyles_4UnseenStyles_TrainSeed${SPLIT_SEED}"
OUT_DIR="/scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/finetune/realConditionPrompts/equalSamplesCondition/differentOrderSeeds/unseenContentActors/Seed_${ORDER_SEED}"

#mkdir -p /scratch/mgirishnair/Thesis/SLURM_logs/PerMo/finetune/realCondition/equalSamplesCondition/unseenContentActorStyles

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"
echo "Array task ID=${SLURM_ARRAY_TASK_ID}"
echo "Job tag=${JOB_TAG}"
echo "Split seed=${SPLIT_SEED}"
echo "Train seed=${TRAIN_SEED}"
echo "Output dir=${OUT_DIR}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/finetune/realConditionPrompts/equalSamplesCondition.py \
    --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
    --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
    --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
    --output_dir "${OUT_DIR}" \
    --epochs 150 \
    --num_workers 2 \
    --batch_size 32 \
    --lr 1e-5 \
    --trainable_layers 2 \
    --normal_target 200 \
    --anomaly_target 200 \
    --test_fraction 0.2 \
    --val_fraction 0.1 \
    --train_seed "${TRAIN_SEED}" \
    --split_seed "${SPLIT_SEED}" \
    --order_seed "${ORDER_SEED}" \
    --condition_col condition_label \
    --actor_col actor \
    --label_col is_anomaly \
    --unseen_actions kick "kick something" throw wave \
    --unseen_actors A01 A05
#    --unseen_styles drunken exhausted textnecked legaching


##!/bin/sh
##SBATCH --job-name="UnseenContentActorStyles_4UnseenStyles_Seed0"
##SBATCH --partition=gpu-a100-small
##SBATCH --time=1:00:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=2
##SBATCH --gpus-per-task=1
##SBATCH --mem-per-gpu=10G
##SBATCH --account=Education-EEMCS-MSc-DSAIT
##SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/finetune/realCondition/equalSamplesCondition/unseenContentActorStyles/%x_%j.out
#
#echo "Job started on $(hostname)"
#echo "Job ID=${SLURM_JOB_ID}"
#
#cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1
#
#module load miniconda3
#module load 2024r1
#module load cuda/11.7
#
#source ~/.bashrc
#conda activate motionclip
#
#python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/finetune/realConditionPrompts/equalSamplesCondition.py \
#    --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
#    --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
#    --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
#    --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/finetune/realConditionPrompts/equalSamplesCondition/differentSplitSeeds/Seed0/unseenContentActorStyles/4_UnseenStyles \
#    --epochs 150 \
#    --num_workers 2 \
#    --batch_size 32 \
#    --lr 1e-5 \
#    --trainable_layers 2 \
#    --normal_target 200 \
#    --anomaly_target 200 \
#    --test_fraction 0.2 \
#    --val_fraction 0.1 \
#    --train_seed 42 \
#    --split_seed 0 \
#    --condition_col condition_label \
#    --actor_col actor \
#    --label_col is_anomaly \
#    --unseen_actions kick "kick something" throw wave \
#    --unseen_actors A01 A05 \
#    --unseen_styles drunken exhausted textnecked legaching
