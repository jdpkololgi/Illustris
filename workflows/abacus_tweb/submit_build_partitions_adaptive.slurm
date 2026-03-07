#!/bin/bash
#SBATCH --job-name=abacus_part_adapt
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=11:50:00
#SBATCH --exclusive
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_part_adapt_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_part_adapt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

PROJECT_DIR="${TNG_ILLUSTRIS_PROJECT_DIR:-/global/homes/d/dkololgi/TNG/Illustris}"
LOG_DIR="${TNG_LOG_DIR:-/pscratch/sd/d/dkololgi/logs}"
INPUT_CACHE_PATH="${INPUT_CACHE_PATH:-/pscratch/sd/d/dkololgi/abacus/SBI_cache files/abacus_alpha_sbi_cache_transformed.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/d/dkololgi/abacus/partitions_alpha_h4_adaptive_parallel}"

HALO_HOPS="${HALO_HOPS:-4}"
CORE_PARTITION_SIZE="${CORE_PARTITION_SIZE:-50000}"
MIN_CORE_NODES="${MIN_CORE_NODES:-5000}"
MAX_CORE_NODES="${MAX_CORE_NODES:-50000}"
TARGET_TOTAL_NODES="${TARGET_TOTAL_NODES:-700000}"
TARGET_EDGES="${TARGET_EDGES:-12000000}"
EDGE_SELECTION_CHUNK_SIZE="${EDGE_SELECTION_CHUNK_SIZE:-10000000}"
MAX_PARTITIONS_PER_SPLIT="${MAX_PARTITIONS_PER_SPLIT:-0}"
# Use all allocated CPU cores by default on one Perlmutter CPU node.
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-128}}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Some user bashrc files reference unset vars (e.g. PYTHONPATH).
set +u
source ~/.bashrc
set -u
conda activate cosmic_env

cd "${PROJECT_DIR}"

# Avoid nested BLAS/OpenMP oversubscription when using many workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Input cache: ${INPUT_CACHE_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Adaptive: halo_hops=${HALO_HOPS}, min_core=${MIN_CORE_NODES}, max_core=${MAX_CORE_NODES}"
echo "Budgets: target_total_nodes=${TARGET_TOTAL_NODES}, target_edges=${TARGET_EDGES}"
echo "Workers: ${NUM_WORKERS}"

srun --cpu-bind=cores python -u "/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb/build_abacus_partition_batches.py" \
  --input-cache-path "${INPUT_CACHE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --core-partition-size "${CORE_PARTITION_SIZE}" \
  --halo-hops "${HALO_HOPS}" \
  --adaptive-core-size \
  --min-core-nodes "${MIN_CORE_NODES}" \
  --max-core-nodes "${MAX_CORE_NODES}" \
  --target-total-nodes "${TARGET_TOTAL_NODES}" \
  --target-edges "${TARGET_EDGES}" \
  --edge-selection-chunk-size "${EDGE_SELECTION_CHUNK_SIZE}" \
  --max-partitions-per-split "${MAX_PARTITIONS_PER_SPLIT}" \
  --num-workers "${NUM_WORKERS}"

echo "Adaptive partition build completed."
