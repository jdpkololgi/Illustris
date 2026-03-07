#!/bin/bash
#SBATCH --job-name=abacus_sbi_part_dp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_sbi_part_dp_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_sbi_part_dp_%j.err

set -euo pipefail

source ~/.bashrc
conda activate cosmic_env

MANIFEST="${MANIFEST:-/pscratch/sd/d/dkololgi/abacus/partitions_alpha_h4_50k/partition_manifest.json}"
SBI_CACHE="${SBI_CACHE:-/pscratch/sd/d/dkololgi/abacus/SBI_cache files/abacus_alpha_sbi_cache_transformed.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/d/dkololgi/tng_illustris/outputs/sbi_partitioned}"

EPOCHS="${EPOCHS:-50}"
NUM_PASSES="${NUM_PASSES:-2}"
LATENT_SIZE="${LATENT_SIZE:-16}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_FLOW_LAYERS="${NUM_FLOW_LAYERS:-2}"
FLOW_HIDDEN_SIZE="${FLOW_HIDDEN_SIZE:-64}"
TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-10}"
TRAIN_PARTITION_LIMIT="${TRAIN_PARTITION_LIMIT:-0}"
VAL_PARTITION_LIMIT="${VAL_PARTITION_LIMIT:-8}"
BUCKET_SPAN_MULTIPLIER="${BUCKET_SPAN_MULTIPLIER:-8}"
BUCKET_SORT_KEY="${BUCKET_SORT_KEY:-max}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

EXTRA_ARGS=""
if [[ "${SLURM_NTASKS:-1}" -gt 1 ]]; then
  # Multi-node launch (one process per node): enable distributed initialization.
  if [[ -z "${COORDINATOR_ADDRESS:-}" ]]; then
    FIRST_HOST="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
    export COORDINATOR_ADDRESS="${FIRST_HOST}:12355"
  fi
  EXTRA_ARGS="${EXTRA_ARGS} --distributed --coordinator-address ${COORDINATOR_ADDRESS}"
fi

if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --activation-checkpointing"
fi

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Manifest: ${MANIFEST}"
echo "SBI cache: ${SBI_CACHE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Distributed world size: ${SLURM_NTASKS:-1}"

srun python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/jraph_sbi_flowjax_partitioned.py" \
  --partition-manifest "${MANIFEST}" \
  --sbi-cache-path "${SBI_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --num-passes "${NUM_PASSES}" \
  --latent-size "${LATENT_SIZE}" \
  --num-heads "${NUM_HEADS}" \
  --num-flow-layers "${NUM_FLOW_LAYERS}" \
  --flow-hidden-size "${FLOW_HIDDEN_SIZE}" \
  --train-progress-every "${TRAIN_PROGRESS_EVERY}" \
  --train-partition-limit "${TRAIN_PARTITION_LIMIT}" \
  --val-partition-limit "${VAL_PARTITION_LIMIT}" \
  --bucket-span-multiplier "${BUCKET_SPAN_MULTIPLIER}" \
  --bucket-sort-key "${BUCKET_SORT_KEY}" \
  --data-parallel \
  ${EXTRA_ARGS}
