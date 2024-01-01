#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=mask2former            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=mask2former-output.log   # Log-file (important!)
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=48:00:00              # Max execution time
#

# Activate your Anaconda environment
conda activate mmseg

# Run your Python script
cd /scratch/users/vgaspar
python train.py configs/mask2former/mask2former_r50_4xb2-90k_soccernet-1080x1920.py --launcher=slurm
# python test_pytorch.py