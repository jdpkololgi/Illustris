#!/bin/bash
#SBATCH --job-name=abacus_sbi_post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_sbi_post_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_sbi_post_%j.err

set -euo pipefail

source ~/.bashrc
conda activate cosmic_env

SCRIPT="/global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/plot_flowjax_posteriors_partitioned.py"
MANIFEST="${MANIFEST:-/pscratch/sd/d/dkololgi/abacus/partitions_alpha_h4_50k/partition_manifest.json}"
MODEL="${MODEL:-/pscratch/sd/d/dkololgi/tng_illustris/outputs/sbi_stageB_chain_4node/partitioned_model_seed_42_20260315_124637.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/tng_illustris/outputs/sbi_stageB_chain_4node}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/posteriors_${SLURM_JOB_ID}}"

SEED="${SEED:-42}"

# Throughput controls (override at submit time if needed).
NUM_SAMPLES="${NUM_SAMPLES:-512}"
NUM_CALIBRATION="${NUM_CALIBRATION:-50000}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-512}"
TARP_BATCH_SIZE="${TARP_BATCH_SIZE:-512}"
TARP_BOOTSTRAP_RESAMPLES="${TARP_BOOTSTRAP_RESAMPLES:-300}"
TARP_BOOTSTRAP_CI="${TARP_BOOTSTRAP_CI:-95}"
STRUCTURAL_NUM_NODES="${STRUCTURAL_NUM_NODES:-30000}"
STRUCTURAL_BATCH_SIZE="${STRUCTURAL_BATCH_SIZE:-512}"
STRUCTURAL_COVERAGE_ALPHA="${STRUCTURAL_COVERAGE_ALPHA:-0.9}"
VECTOR_POS_COLS="${VECTOR_POS_COLS:-0,1}"
VECTOR_MAX_POINTS="${VECTOR_MAX_POINTS:-8000}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async

TEST_NODES=$(python - <<'PY'
import json
import os
from pathlib import Path
m = Path(os.environ["MANIFEST"])
with m.open("r", encoding="utf-8") as f:
    d = json.load(f)
print(sum(int(p["n_core_nodes"]) for p in d["partitions"] if p.get("split") == "test"))
PY
)
TARP_NODES=$(( TEST_NODES / 10 ))
if [[ "${TARP_NODES}" -lt 10000 ]]; then
  TARP_NODES=10000
fi

echo "=============================================================="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Script: ${SCRIPT}"
echo "Model: ${MODEL}"
echo "Manifest: ${MANIFEST}"
echo "Output dir: ${OUTPUT_DIR}"
echo "TEST_NODES=${TEST_NODES}"
echo "TARP_NODES=max(10000,10% test)=${TARP_NODES}"
echo "=============================================================="

srun python -u "${SCRIPT}" \
  --model-path "${MODEL}" \
  --partition-manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-test-nodes "${TARP_NODES}" \
  --num-tarp "${TARP_NODES}" \
  --num-samples "${NUM_SAMPLES}" \
  --num-calibration "${NUM_CALIBRATION}" \
  --calibration-batch-size "${CALIBRATION_BATCH_SIZE}" \
  --tarp-batch-size "${TARP_BATCH_SIZE}" \
  --tarp-bootstrap-resamples "${TARP_BOOTSTRAP_RESAMPLES}" \
  --tarp-bootstrap-ci "${TARP_BOOTSTRAP_CI}" \
  --structural-num-nodes "${STRUCTURAL_NUM_NODES}" \
  --structural-batch-size "${STRUCTURAL_BATCH_SIZE}" \
  --structural-coverage-alpha "${STRUCTURAL_COVERAGE_ALPHA}" \
  --vector-pos-cols "${VECTOR_POS_COLS}" \
  --vector-max-points "${VECTOR_MAX_POINTS}" \
  --seed "${SEED}"

