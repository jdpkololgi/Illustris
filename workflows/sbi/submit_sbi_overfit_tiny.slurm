#!/bin/bash
#SBATCH --job-name=abacus_sbi_overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_sbi_overfit_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_sbi_overfit_%j.err

set -euo pipefail

source ~/.bashrc
conda activate cosmic_env

MANIFEST="${MANIFEST:-/pscratch/sd/d/dkololgi/abacus/partitions_alpha_h4_50k/partition_manifest.json}"
SBI_CACHE="${SBI_CACHE:-/pscratch/sd/d/dkololgi/abacus/SBI_cache files/abacus_alpha_sbi_cache_transformed.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/tng_illustris/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/sbi_overfit_tiny_${SLURM_JOB_ID}}"

# Tiny overfit defaults (intentionally low regularization).
EPOCHS="${EPOCHS:-300}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
DROPOUT="${DROPOUT:-0.0}"
NUM_PASSES="${NUM_PASSES:-4}"
LATENT_SIZE="${LATENT_SIZE:-32}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_FLOW_LAYERS="${NUM_FLOW_LAYERS:-3}"
FLOW_HIDDEN_SIZE="${FLOW_HIDDEN_SIZE:-128}"
SEED="${SEED:-42}"

# Step-1 core controls: train on a tiny fixed subset.
TRAIN_PARTITION_LIMIT="${TRAIN_PARTITION_LIMIT:-2}"
VAL_PARTITION_LIMIT="${VAL_PARTITION_LIMIT:-2}"
FULL_VAL_PARTITION_LIMIT="${FULL_VAL_PARTITION_LIMIT:-2}"
TRAIN_PARTITIONS_PER_EPOCH="${TRAIN_PARTITIONS_PER_EPOCH:-0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
FULL_VAL_EVERY="${FULL_VAL_EVERY:-5}"
TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-1}"

# Keep this simple: single process, no distributed/pmap.
MIXED_PRECISION="${MIXED_PRECISION:-none}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

EXTRA_ARGS=""
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --activation-checkpointing"
else
  EXTRA_ARGS="${EXTRA_ARGS} --no-activation-checkpointing"
fi

echo "=============================================================="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Manifest: ${MANIFEST}"
echo "SBI cache: ${SBI_CACHE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Tiny overfit config: train_limit=${TRAIN_PARTITION_LIMIT}, val_limit=${VAL_PARTITION_LIMIT}, epochs=${EPOCHS}"
echo "=============================================================="

srun python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/jraph_sbi_flowjax_partitioned.py" \
  --partition-manifest "${MANIFEST}" \
  --sbi-cache-path "${SBI_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --dropout "${DROPOUT}" \
  --seed "${SEED}" \
  --num-passes "${NUM_PASSES}" \
  --latent-size "${LATENT_SIZE}" \
  --num-heads "${NUM_HEADS}" \
  --num-flow-layers "${NUM_FLOW_LAYERS}" \
  --flow-hidden-size "${FLOW_HIDDEN_SIZE}" \
  --mixed-precision "${MIXED_PRECISION}" \
  --train-partition-limit "${TRAIN_PARTITION_LIMIT}" \
  --val-partition-limit "${VAL_PARTITION_LIMIT}" \
  --full-val-partition-limit "${FULL_VAL_PARTITION_LIMIT}" \
  --train-partitions-per-epoch "${TRAIN_PARTITIONS_PER_EPOCH}" \
  --eval-every "${EVAL_EVERY}" \
  --full-val-every "${FULL_VAL_EVERY}" \
  --train-progress-every "${TRAIN_PROGRESS_EVERY}" \
  ${EXTRA_ARGS}

