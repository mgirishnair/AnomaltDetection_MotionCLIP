#!/bin/sh
#SBATCH --job-name="cosSimAnomalyDetection_updatedFlawed"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/baseline/%x_%j.out

echo "Job started on $(hostname)"
echo "Job ID=${SLURM_JOB_ID}"

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment || exit 1

module load miniconda3
module load 2024r1
module load cuda/11.7

source ~/.bashrc
conda activate motionclip

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/PerMo/baseline/cosSimAnomalyDetection_updatedFlawed.py \
    --metadata_csv /scratch/mgirishnair/Thesis/PerMoConverted/PerMo_metadata.csv \
    --motionclip_repo /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP \
    --checkpoint_path /scratch/mgirishnair/Thesis/MotionCLIP_experiment/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
    --output_dir /scratch/mgirishnair/Thesis/MotionCLIP_experiment/PerMo/baseline \
    --run_name "Cosine Similatiry Baseline Updated FlawedHealthy" \
    --num_workers 2 \
    --save_embeddings
