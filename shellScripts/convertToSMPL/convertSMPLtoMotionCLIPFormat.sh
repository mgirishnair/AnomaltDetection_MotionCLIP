#!/bin/sh
#
#SBATCH --job-name="smpl_to_motionclip"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/dataProcessing/toMotionCLIP/%x.out


module load miniconda3
module load 2024r1

cd /scratch/mgirishnair/Pose_to_SMPL

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smplpytorch

python /scratch/mgirishnair/Thesis/MotionCLIP_experiment/pythonFiles/convertToSMPL/convertSMPLtoMotionCLIPFormat.py
