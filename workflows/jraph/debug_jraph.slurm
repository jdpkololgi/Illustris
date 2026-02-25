#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:15:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --account=desi
#SBATCH --job-name=jraph_debug
#SBATCH --output=logs/jraph_debug_%j.out
#SBATCH --error=logs/jraph_debug_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate base

# Set environment variables for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run the script with few epochs for testing
echo "Starting Jraph debug job on $(hostname)..."
srun python /global/homes/d/dkololgi/TNG/Illustris/jraph_pipeline.py \
    --epochs 5 \
    --lr 0.005 \
    --weight_decay 0.06 \
    --dropout 0.13 \
    --latent_size 80 \
    --num_heads 4
