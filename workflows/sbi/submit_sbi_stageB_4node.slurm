#!/bin/bash
#SBATCH --job-name=abacus_sbi_stageB_4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_sbi_stageB_4n_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_sbi_stageB_4n_%j.err

set -euo pipefail

source ~/.bashrc
conda activate cosmic_env

MANIFEST="${MANIFEST:-/pscratch/sd/d/dkololgi/abacus/partitions_alpha_h4_50k/partition_manifest.json}"
SBI_CACHE="${SBI_CACHE:-/pscratch/sd/d/dkololgi/abacus/SBI_cache files/abacus_alpha_sbi_cache_transformed.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/tng_illustris/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/sbi_partitioned_stageB_4node_${SLURM_JOB_ID}}"

# Stage-B defaults (override via env at submit time).
EPOCHS="${EPOCHS:-120}"
NUM_PASSES="${NUM_PASSES:-4}"
LATENT_SIZE="${LATENT_SIZE:-32}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_FLOW_LAYERS="${NUM_FLOW_LAYERS:-3}"
FLOW_HIDDEN_SIZE="${FLOW_HIDDEN_SIZE:-64}"
TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-1}"

# Stochastic epoch / validation schedule.
TRAIN_PARTITIONS_PER_EPOCH="${TRAIN_PARTITIONS_PER_EPOCH:-16}"
EVAL_EVERY="${EVAL_EVERY:-2}"
VAL_PARTITION_LIMIT="${VAL_PARTITION_LIMIT:-16}"
FULL_VAL_EVERY="${FULL_VAL_EVERY:-40}"
FULL_VAL_PARTITION_LIMIT="${FULL_VAL_PARTITION_LIMIT:-48}"

# Throughput/memory tuning.
BUCKET_SPAN_MULTIPLIER="${BUCKET_SPAN_MULTIPLIER:-8}"
BUCKET_SORT_KEY="${BUCKET_SORT_KEY:-max}"
PAD_NODE_MULTIPLE="${PAD_NODE_MULTIPLE:-1024}"
PAD_EDGE_MULTIPLE="${PAD_EDGE_MULTIPLE:-65536}"
PARTITION_CACHE_SIZE="${PARTITION_CACHE_SIZE:-256}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
PREFETCH_LOOKAHEAD_STEPS="${PREFETCH_LOOKAHEAD_STEPS:-4}"

# Default precision/checkpointing behavior in trainer is bf16 + enabled.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

if [[ -z "${COORDINATOR_ADDRESS:-}" ]]; then
  FIRST_HOST="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export COORDINATOR_ADDRESS="${FIRST_HOST}:12355"
fi

EXTRA_ARGS=""
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --activation-checkpointing"
else
  EXTRA_ARGS="${EXTRA_ARGS} --no-activation-checkpointing"
fi

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Coordinator: ${COORDINATOR_ADDRESS}"
echo "Manifest: ${MANIFEST}"
echo "SBI cache: ${SBI_CACHE}"
echo "Output dir: ${OUTPUT_DIR}"

srun --gpu-bind=none python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/jraph_sbi_flowjax_partitioned.py" \
  --partition-manifest "${MANIFEST}" \
  --sbi-cache-path "${SBI_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --num-passes "${NUM_PASSES}" \
  --latent-size "${LATENT_SIZE}" \
  --num-heads "${NUM_HEADS}" \
  --num-flow-layers "${NUM_FLOW_LAYERS}" \
  --flow-hidden-size "${FLOW_HIDDEN_SIZE}" \
  --mixed-precision "${MIXED_PRECISION}" \
  --train-progress-every "${TRAIN_PROGRESS_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --train-partitions-per-epoch "${TRAIN_PARTITIONS_PER_EPOCH}" \
  --val-partition-limit "${VAL_PARTITION_LIMIT}" \
  --full-val-every "${FULL_VAL_EVERY}" \
  --full-val-partition-limit "${FULL_VAL_PARTITION_LIMIT}" \
  --bucket-span-multiplier "${BUCKET_SPAN_MULTIPLIER}" \
  --bucket-sort-key "${BUCKET_SORT_KEY}" \
  --pad-node-multiple "${PAD_NODE_MULTIPLE}" \
  --pad-edge-multiple "${PAD_EDGE_MULTIPLE}" \
  --partition-cache-size "${PARTITION_CACHE_SIZE}" \
  --prefetch-workers "${PREFETCH_WORKERS}" \
  --prefetch-lookahead-steps "${PREFETCH_LOOKAHEAD_STEPS}" \
  --data-parallel \
  --distributed \
  --coordinator-address "${COORDINATOR_ADDRESS}" \
  ${EXTRA_ARGS}
