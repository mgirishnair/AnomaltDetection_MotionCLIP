#!/bin/bash
#SBATCH --job-name="plot_losses"
#SBATCH --partition=compute
#SBATCH --time=00:00:15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/%x_%j.out

module load 2024r1
module load python
module load py-numpy
module load py-matplotlib/3.7.1


python /scratch/mgirishnair/MotionCLIP_experiment/plot_losses.py \
    --json_path /scratch/mgirishnair/MotionCLIP_experiment/finetune/contrastiveLoss/finetune_summary_contrastive_only_1.json \
    --output_dir /scratch/mgirishnair/MotionCLIP_experiment/finetune/plots \
    --filename "finetune_summary_contrastive_only_1.png"
