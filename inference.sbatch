#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=inference            # Name of the job 
#SBATCH -o %x_%j.log
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --ntasks=1                   # Number of tasks to run
#SBATCH --cpus-per-task=8         # Number of CPU cores to allocate
#SBATCH --time=5:00:00              # Max execution time
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --mem=20G
#SBATCH --partition=a5000
#

conda activate banner-replacement
cd /scratch/users/vgaspar/banner-replacement
echo "Starting job at $(date)"
pwd

# Set parameters

# path to the video
VIDEO=/scratch/users/vgaspar/camera_calibration_old/evaluation/data/sn-gamestate/test/SNGS-142/img1
# Path to logo
LOGO=SoccerNet_resized.png
# N workers
N_WORKERS=8
# Use TTA (1) or not (0)
TTA=1
# Use image sequence (0) or video (1)
SEQUENCE_MODE=1

# print all parameters
echo "Parameters:"
echo "VIDEO: $VIDEO"
echo "LOGO: $LOGO"
echo "N_WORKERS: $N_WORKERS"
echo "TTA: $TTA"
echo "SEQUENCE_MODE: $SEQUENCE_MODE"

# Run the inference
if [ $TTA -eq 1 ]; then
    if [ $SEQUENCE_MODE -eq 1 ]; then
        echo "Running inference with TTA and image sequence mode"
        python inference.py $VIDEO $LOGO --n_workers $N_WORKERS --tta --sequence
    else
        echo "Running inference with TTA and video mode"
        python inference.py $VIDEO $LOGO --n_workers $N_WORKERS --tta
    fi
else
    if [ $SEQUENCE_MODE -eq 1 ]; then
        echo "Running inference without TTA and image sequence mode"
        python inference.py $VIDEO $LOGO --n_workers $N_WORKERS --sequence
    else
        echo "Running inference without TTA and video mode"
        python inference.py $VIDEO $LOGO --n_workers $N_WORKERS
    fi
fi

echo "Job finished at $(date)"