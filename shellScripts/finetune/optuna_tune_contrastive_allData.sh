#!/bin/sh

#SBATCH --job-name="optuna_motionclip_global_FullcontrastiveLoss_ntu120"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/optuna/allData/ntu120/fullContras/%x_%j.out

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

mkdir -p /scratch/mgirishnair/Thesis/SLURM_logs/optuna
mkdir -p /scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_dbs/global/fullContrastiveLoss/ntu120
mkdir -p /scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_runs/global/fullContrastiveLoss/ntu120

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/tune_motionclip_optuna_allData.py \
  --study_name "motionclip_global_fullContrastiveLoss" \
  --storage "sqlite:////scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_dbs/global/fullContrastiveLoss/ntu120/motionclip_global_contrastive.db" \
  --n_trials 30 \
  --global_contrastive_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/finetune_contrastive_allData.py \
  --x_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_120/X.npy \
  --y_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_120/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --work_dir optuna_runs/global/fullContrastiveLoss/ntu120 \
  --run_name global_all_classes_train100 \
  --train_fraction 1 \
  --val_fraction 0.1 \
  --epochs 100 \
  --patience 8 \
  --min_delta 0.0 \
  --num_trainable_blocks_fixed 2 \
  --scheduler_patience 3 \
  --scheduler_factor 0.7 \
  --scheduler_min_lr 1e-7 \
  --num_workers 2 \
  --pin_memory \
  --seed 42 \
  --cleanup_trial_dir
