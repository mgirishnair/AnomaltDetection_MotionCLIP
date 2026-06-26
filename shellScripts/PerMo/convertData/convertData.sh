#!/bin/sh
#
#SBATCH --job-name="PerMo_to_motionclip"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/PerMo/convertData/%x.out


module load miniconda3
module load 2024r1

conda activate motionclip

cd /scratch/mgirishnair/Thesis/MotionCLIP_experiment

python pythonFiles/PerMo/convertData/convertData.py --input_root /scratch/mgirishnair/Thesis/PerMo/Traits --output_root /scratch/mgirishnair/Thesis/PerMoConverted/Traits
