#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=jraph_optuna
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

# Create logs directory
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate base
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Starting Optuna Hyperparameter Tuning..."
# Run the tuning script
# It uses JIT on a single GPU (CUDA_VISIBLE_DEVICES=0 implicit or managed by Slurm)
python hyperparameter_tuning.py
