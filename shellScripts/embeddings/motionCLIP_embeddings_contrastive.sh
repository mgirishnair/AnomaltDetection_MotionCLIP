#!/bin/sh
#SBATCH --job-name="motionclip_embed_contrastive"
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/%x_%A_%a.out
#SBATCH --array=1-10

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# go to project folder
cd /scratch/mgirishnair/MotionCLIP_experiment

module load miniconda3
module load 2024r1
module load cuda/11.7

# activate environment
source ~/.bashrc
conda activate motionclip

SPLITS_FILE=/scratch/mgirishnair/MotionCLIP_experiment/splits.txt

mkdir -p embeddings

#while IFS= read -r line || [ -n "$line" ]; do
#    [ -z "$line" ] && continue
#    case "$line" in \#*) continue ;; esac

#    split_name="${line%%:*}"
#    normal_classes="${line#*:}"


line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SPLITS_FILE")
[ -z "$line" ] && { echo "No split found for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

split_name="${line%%:*}"
normal_classes="${line#*:}"

echo "Running split: $split_name"
echo "Normal classes: $normal_classes"

python motionCLIP_anomalyDetection_emebddings.py \
     --x_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/X.npy \
     --y_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/y.npy \
     --motionclip_repo MotionCLIP \
     --checkpoint_path /scratch/mgirishnair/MotionCLIP_experiment/finetune/contrastiveLoss/motionclip_finetuned_contrastive_only.pth \
     --normal_classes $normal_classes \
     --train_fraction 0.8 \
     --seed 42 \
     --batch_size 64 \
     --output_path "/scratch/mgirishnair/MotionCLIP_experiment/embeddings/contrastive/motionclip_embeddings_${split_name}_allData.npz"



#python motionCLIP_anomalyDetection_emebddings.py \
#  --x_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/X.npy \
#  --y_path /scratch/mgirishnair/MotionCLIP_ready_datasetFinalAll/y.npy \
#  --motionclip_repo MotionCLIP \
#  --checkpoint_path MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar \
#  --normal_classes 50 51 52 53 \
#  --train_fraction 0.8 \
#  --seed 42 \
#  --batch_size 64 \
#  --output_path /scratch/mgirishnair/MotionCLIP_experiment/embeddings/motionclip_embeddings_fighting_allData.npz
