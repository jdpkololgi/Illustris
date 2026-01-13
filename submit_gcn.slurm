#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=gcn_pipeline_50k
#SBATCH --output=logs/gcn_pipeline_%j.out
#SBATCH --error=logs/gcn_pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate cosmic_env

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run the script
# Note: The script uses mp.spawn internally, so we run it as a single process
echo "Starting training job on $(hostname) with 4 GPUs..."
srun python /global/homes/d/dkololgi/TNG/Illustris/gcn_pipeline.py
