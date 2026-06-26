#!/bin/bash
#SBATCH --job-name="Tune_AACLIP_valLoss_seen"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/train/optuna_%x_%j.out
#set -euo pipefail

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

# Optuna must be installed in the motionclip environment, for example:
# python -m pip install optuna

python -c "import optuna; print('Optuna version:', optuna.__version__)"

# Optional unseen holdouts. Leave this array empty for the seen experiment.

UNSEEN_ARGS=(
#   --unseen_actions kick "kick something" throw wave
#   --unseen_actors A01 A05
#   --unseen_styles drunken exhausted legaching textnecked
)

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/train/optuna_AACLIP.py \
  --train_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/train/train_AACLIP.py \
  --study_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/optuna/valLoss/train_seen \
  --study_name AACLIP_valLoss_seen \
  --n_trials 8 \
  --study_timeout_hours 46 \
  --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
  --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
  --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --actor_col actor \
  --split_seed 0 \
  --train_seed 42 \
  --normal_target 200 \
  --anomaly_target 200 \
  --test_fraction 0.20 \
  --val_fraction 0.10 \
  --num_workers 2 \
  --stage1_templates_per_batch 1 \
  --early_stopping_patience 8 \
  "${UNSEEN_ARGS[@]}" \
  --stage1_epochs_min 3 \
  --stage1_epochs_max 12 \
  --stage2_epochs_min 10 \
  --stage2_epochs_max 40 \
  --stage1_lr_min 1e-6 \
  --stage1_lr_max 1e-4 \
  --stage2_lr_min 1e-5 \
  --stage2_lr_max 2e-3 \
  --stage1_weight_decay_min 1e-6 \
  --stage1_weight_decay_max 1e-2 \
  --stage2_weight_decay_min 1e-6 \
  --stage2_weight_decay_max 1e-2 \
  --disentangle_weight_min 1e-3 \
  --disentangle_weight_max 1.0 \
  --contrastive_weight_min 1e-3 \
  --contrastive_weight_max 1.0 \
  --temperature_min 1e-2 \
  --temperature_max 2e-1 \
  --batch_sizes 8 16 32 \
  --disable_prompted_unseen_style_eval \
  --cleanup_nonbest_checkpoints
#
#
#python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/train/optuna_AACLIP.py \
#  --train_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/train/train_AACLIP.py \
#  --study_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/optuna/train_seen \
#  --study_name AACLIP_seen \
#  --n_trials 8 \
#  --study_timeout_hours 46 \
#  --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
#  --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
#  --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
#  --actor_col actor \
#  --split_seed 0 \
#  --train_seed 42 \
#  --normal_target 200 \
#  --anomaly_target 200 \
#  --test_fraction 0.20 \
#  --val_fraction 0.10 \
#  --num_workers 2 \
#  --stage1_templates_per_batch 1 \
#  --early_stopping_patience 8 \
##  --unseen_actions kick "kick something" throw wave \
##  --unseen_actors A01 A05 \
##  --unseen_styles drunken exhausted\
#  --stage1_epochs_min 3 \
#  --stage1_epochs_max 12 \
#  --stage2_epochs_min 10 \
#  --stage2_epochs_max 40 \
#  --stage1_lr_min 1e-6 \
#  --stage1_lr_max 1e-4 \
#  --stage2_lr_min 1e-5 \
#  --stage2_lr_max 2e-3 \
#  --stage1_weight_decay_min 1e-6 \
#  --stage1_weight_decay_max 1e-2 \
#  --stage2_weight_decay_min 1e-6 \
#  --stage2_weight_decay_max 1e-2 \
#  --disentangle_weight_min 1e-3 \
#  --disentangle_weight_max 1.0 \
#  --contrastive_weight_min 1e-3 \
#  --contrastive_weight_max 1.0 \
#  --temperature_min 1e-2 \
#  --temperature_max 2e-1 \
#  --batch_sizes 8 16 32 \
#  --disable_prompted_unseen_style_eval \
#  --cleanup_nonbest_checkpoints
