#!/bin/sh
#
#SBATCH --job-name="smpl_to_motionclip"
#SBATCH --partition=compute-p1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3400M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

# ================ OUTPUT FILES ================
base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"

# redirect stdout and stderr
exec >"$outfile" 2>&1

module load miniconda3
module load 2024r1

cd /scratch/mgirishnair/Pose_to_SMPL

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smplpytorch

python /scratch/mgirishnair/convertSMPLtoMotionCLIPFormat.py
