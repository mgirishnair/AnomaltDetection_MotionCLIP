#!/bin/sh
#SBATCH --job-name="pose_to_smpl"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=70-100
#SBATCH --output=/scratch/mgirishnair/Thesis/SLURM_logs/dataProcessing/%x_%A_%a.out

module load miniconda3
module load 2024r1
module load cuda/11.7

cd /scratch/mgirishnair/Pose_to_SMPL

mkdir -p /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/last60/

FILELIST=/scratch/mgirishnair/Thesis/ntu_60-120.txt
CHUNK_SIZE=569

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
TOTAL=$(wc -l < $FILELIST)
END=$((START + CHUNK_SIZE - 1))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

sed -n "${START},${END}p" "$FILELIST" > /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/last60/files_${SLURM_ARRAY_TASK_ID}.txt
conda activate smplpytorch
echo "START=$START END=$END TOTAL=$TOTAL"
echo "Processing file list:"
head /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/last60/files_${SLURM_ARRAY_TASK_ID}.txt
python fit/tools/main.py \
    --dataset_name NTU_120 \
    --dataset_path /scratch/mgirishnair/Thesis/ntu_60-120_npy \
    --file_list /scratch/mgirishnair/Pose_to_SMPL/chunk_lists/last60/files_${SLURM_ARRAY_TASK_ID}.txt \
    --exp "array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
