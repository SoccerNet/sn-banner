#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=soccerNetSegVic22            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=soccerNetSegVic22-output.log   # Log-file (important!)
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=1:00:00              # Max execution time
#

echo "Starting sbatch script myscript.sh at:`date`"

# Activate your Anaconda environment
conda activate mmseg

# Run your Python script
cd /scratch/users/vgaspar
python soccerNetSegVic2.py
# python test_pytorch.py