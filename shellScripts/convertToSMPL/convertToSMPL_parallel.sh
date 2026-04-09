#!/bin/sh
#SBATCH --job-name="pose_to_smpl"
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=6G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=61-75
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/%x_%A_%a.out

module load miniconda3
module load 2024r1
module load cuda/11.7

cd /scratch/mgirishnair/Pose_to_SMPL

mkdir -p /scratch/mgirishnair/SLURM_logs
mkdir -p /scratch/mgirishnair/Pose_to_SMPL/chunk_lists

FILELIST=/scratch/mgirishnair/ntu_files.txt
CHUNK_SIZE=569

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))

sed -n "${START},${END}p" "$FILELIST" > /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/files_${SLURM_ARRAY_TASK_ID}.txt
conda activate smplpytorch
python fit/tools/main.py \
    --dataset_name NTU \
    --dataset_path /scratch/mgirishnair/nturgb+d_skeletons_npy \
    --file_list /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/files_${SLURM_ARRAY_TASK_ID}.txt \
    --exp "array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
