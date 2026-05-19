#!/bin/sh

#SBATCH --job-name="optuna_motionclip_global_lessNeg"
#SBATCH --partition=gpu-v100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/optuna/allData/%x_%j.out

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

mkdir -p /scratch/mgirishnair/Thesis/SLURM_logs/optuna
mkdir -p /scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_dbs/global/lessNegContrastive
mkdir -p /scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_runs/global/lessNegContrastive

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/tune_motionclip_optuna_allData.py \
  --study_name "motionclip_global_fullContrastiveLoss" \
  --storage "sqlite:////scratch/mgirishnair/Thesis/MotionCLIP_experiment/optuna_dbs/global/lessNegContrastive/motionclip_global_contrastive.db" \
  --n_trials 10 \
  --global_contrastive_script /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/finetune/finetune_contrastive_allData_lessNeg.py \
  --x_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/X.npy \
  --y_path /scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --work_dir optuna_runs/global/lessNegContrastive \
  --run_name global_all_classes_train80 \
  --train_fraction 0.8 \
  --val_fraction 0.1 \
  --epochs 100 \
  --patience 8 \
  --min_delta 0.0 \
  --num_trainable_blocks_fixed 2 \
  --scheduler_patience 3 \
  --scheduler_factor 0.7 \
  --scheduler_min_lr 1e-7 \
  --num_workers 4 \
  --pin_memory \
  --seed 42 \
  --cleanup_trial_dir
