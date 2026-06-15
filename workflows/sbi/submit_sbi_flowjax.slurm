#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --job-name=flowjax_sbi
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/flowjax_sbi_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/flowjax_sbi_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate cosmic_env

# Set environment variables for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2"

# Run the script
echo "Starting Flowjax SBI training job on $(hostname)..."
srun python -u /global/homes/d/dkololgi/TNG/Illustris/jraph_sbi_flowjax.py
