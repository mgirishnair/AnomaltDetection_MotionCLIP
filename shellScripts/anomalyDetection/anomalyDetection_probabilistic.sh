#!/bin/sh
#
#SBATCH --job-name="anomaly_detection"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=8G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

# ================ OUTPUT FILES ================
# compute a small incremental index based on existing files
base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs_anomalyDetection"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"

# redirect stdout and stderr
exec >"$outfile" 2>&1

module load miniconda3
module load 2024r1
module load cuda/11.7
cd /scratch/mgirishnair/MotionCLIP_experiment
#conda run -n smplpytorch python fit/tools/main.py --dataset_name NTU --dataset_path "/scratch/mgirishnair/nturgb+d_skeletons_npy"
conda activate motionclip
python motionCLIP_anomalyDetection_probabilistic.py
