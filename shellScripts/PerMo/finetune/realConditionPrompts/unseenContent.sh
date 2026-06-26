#!/bin/sh
#SBATCH --job-name="8unseenContent"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/finetune/realCondition/%x_%j.out

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/finetune/realConditionPrompts/unseenContent.py \
    --csv_path /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_metadata.csv \
    --project_root /scratch/mgirishnair/Thesis/MotionCLIP_experiment \
    --checkpoint /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
    --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/finetune/realConditionPrompts/unseenContent/8Unseen \
    --epochs 100 \
    --num_workers 2 \
    --batch_size 32 \
    --lr 1e-5 \
    --trainable_layers 2 \
    --unseen_actions kick "kick something" throw wave hop punch run walk
