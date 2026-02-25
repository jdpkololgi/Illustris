#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=jraph_ensemble
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

# Create logs directory
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate base
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Starting Ensemble Training..."

# Run 4 in parallel
echo "Launching seeds 42-45..."
CUDA_VISIBLE_DEVICES=0 python jraph_pipeline.py --seed 42 --epochs 4000 --latent_size 80 --dropout 0.1 > logs/seed_42.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python jraph_pipeline.py --seed 43 --epochs 4000 --latent_size 80 --dropout 0.1 > logs/seed_43.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python jraph_pipeline.py --seed 44 --epochs 4000 --latent_size 80 --dropout 0.1 > logs/seed_44.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python jraph_pipeline.py --seed 45 --epochs 4000 --latent_size 80 --dropout 0.1 > logs/seed_45.log 2>&1 &

wait

echo "Batch 1 finished. Launching Seed 46..."
# Run the last one
CUDA_VISIBLE_DEVICES=0 python jraph_pipeline.py --seed 46 --epochs 4000 --latent_size 80 --dropout 0.1 > logs/seed_46.log 2>&1 &

wait

echo "All 5 models trained."
