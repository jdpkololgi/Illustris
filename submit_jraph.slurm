#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=jraph_pipeline
#SBATCH --output=logs/jraph_pipeline_%j.out
#SBATCH --error=logs/jraph_pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate base

# Set environment variables for JAX
# Use platform allocator to avoid fragmentation issues
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run the script
echo "Starting Jraph training job on $(hostname) with 4 GPUs (80GB VRAM requested)..."
srun python /global/homes/d/dkololgi/TNG/Illustris/jraph_pipeline.py
