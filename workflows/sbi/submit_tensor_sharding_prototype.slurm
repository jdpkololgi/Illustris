#!/bin/bash
#SBATCH --job-name=abacus_tensor_proto
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_tensor_proto_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_tensor_proto_%j.err

source ~/.bashrc
conda activate cosmic_env

CACHE_PATH="${CACHE_PATH:-/pscratch/sd/d/dkololgi/abacus/SBI_cache files/abacus_alpha_sbi_cache_transformed.pkl}"
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
LOG_PATH="${LOG_PATH:-/pscratch/sd/d/dkololgi/logs/tensor_shard_prototype_${SLURM_JOB_ID}.json}"

# Optional coordinator override for multi-process JAX.
export COORDINATOR_ADDRESS="${COORDINATOR_ADDRESS:-127.0.0.1:12345}"

srun python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/prototype_tensor_sharding_fullgraph.py" \
  --sbi-cache-path "$CACHE_PATH" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-path "$LOG_PATH"
