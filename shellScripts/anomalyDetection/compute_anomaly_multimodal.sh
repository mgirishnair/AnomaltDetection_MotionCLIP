#!/bin/bash
#
#SBATCH --job-name="compute_anomaly_multiModal_contrastive"
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=1-20
#SBATCH --output=/scratch/mgirishnair/SLURM_logs/%x_%A_%a.out
module load miniconda3
module load 2024r1

cd /scratch/mgirishnair/MotionCLIP_experiment

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate motionclip

SPLITS_FILE=/scratch/mgirishnair/MotionCLIP_experiment/splits_names.txt

split=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SPLITS_FILE")

[ -z "$split" ] && { echo "No split found for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

echo "Running split: $split"

python motionCLIP_anomalyDetection_computeAnomaly_multiModal.py \
    --embeddings_path "/scratch/mgirishnair/MotionCLIP_experiment/embeddings/contrastive/motionclip_embeddings_${split}_allData.npz" \
    --threshold_percentile 95 \
    --split "$split" \
    --output_path "/scratch/mgirishnair/MotionCLIP_experiment/results/multimodal/classwiseGaussian/contrastive/anomaly_results_multimodalClasswiseGaussian_${split}.txt"
