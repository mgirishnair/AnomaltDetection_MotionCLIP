#!/bin/bash
#SBATCH --job-name="Tune_anomalyDir_valLoss_seen"
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

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/anomalyDirection/optuna_anomalyDir.py \
  --train_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/anomalyDirection/train_anomalyDir.py \
  --study_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/optuna/anomalyDirection \
  --study_name shared_direction_permo \
  --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_condition_metadata.csv \
  --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
  --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --actor_col actor \
  --split_seed 0 \
  --train_seed 0 \
  --order_seed 0 \
  --n_trials 8 \
  --objective val_loss \
  --normal_target 200 \
  --anomaly_target 200 \
  --test_fraction 0.20 \
  --val_fraction 0.10 \
  --num_workers 2 \
  --early_stopping_patience 8 \
  --batch_sizes 16 32 \
  --cleanup_nonbest_checkpoints
