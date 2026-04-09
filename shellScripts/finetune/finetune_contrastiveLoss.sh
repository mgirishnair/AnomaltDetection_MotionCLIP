#!/bin/sh

#SBATCH --job-name="motionclip_finetune_contrastiveLoss"
#SBATCH --partition=gpu-a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/finetune/%x_%j.out

echo "Job started on $(hostname)"


# go to project folder
cd /scratch/mgirishnair/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

# activate environment
source ~/.bashrc
conda activate motionclip

python /scratch/mgirishnair/MotionCLIP_experiment/motionCLIP_finetune_contrastiveLoss.py \
  --x_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/X.npy \
  --y_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/y.npy \
  --motionclip_repo MotionCLIP \
  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
  --epochs 100 \
  --batch_size 48 \
  --num_trainable_blocks 2 \
  --lr_encoder 1e-5 \
  --contrastive_temp 0.07 \
  --use_class_aware_sampler \
  --n_classes_per_batch 8 \
  --n_samples_per_class 6 \
  --val_fraction 0.1 \
  --pin_memory \
  --patience 5 \
  --min_delta 0.0 \
  --num_workers 4 \
  --scheduler_patience 2 \
  --scheduler_factor 0.5 \
  --scheduler_min_lr 1e-7 \
  --output_dir /scratch/mgirishnair/MotionCLIP_experiment/finetune/contrastiveLoss \
  --save_checkpoint motionclip_finetuned_contrastive_only_1.pth \
  --save_metrics_npz finetune_metrics_contrastive_only_1.npz \
  --save_summary finetune_summary_contrastive_only_1.json
