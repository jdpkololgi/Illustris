#!/bin/bash
#SBATCH --job-name=abacus_feat_cpu
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_feat_cpu_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_feat_cpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

# Compute Abacus graph features on CPU with networkit.

PROJECT_DIR="${TNG_ILLUSTRIS_PROJECT_DIR:-/global/homes/d/dkololgi/TNG/Illustris}"
LOG_DIR="${TNG_LOG_DIR:-/pscratch/sd/d/dkololgi/logs}"
GRAPH_DIR="${ABACUS_GRAPH_DIR:-/pscratch/sd/d/dkololgi/abacus/graph_constructions}"
PREFIX="${ABACUS_GRAPH_PREFIX:-abacus_alpha}"
OUTPUT_PREFIX="${ABACUS_FEATURES_CPU_OUTPUT_PREFIX:-${PREFIX}_graph_features}"

mkdir -p "${LOG_DIR}" "${GRAPH_DIR}"

# Some user bashrc files reference unset vars (e.g. PYTHONPATH).
set +u
source ~/.bashrc
set -u
conda activate cosmic_env

cd "${PROJECT_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Points: auto-resolve (<prefix>_points.npy preferred)"
echo "Artifacts dir: ${GRAPH_DIR}"
echo "Prefix: ${PREFIX}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Starting CPU graph feature extraction..."

srun --cpu-bind=cores python workflows/abacus_tweb/abacus_graph_features.py \
  --artifacts-dir "${GRAPH_DIR}" \
  --prefix "${PREFIX}" \
  --output-prefix "${OUTPUT_PREFIX}"

echo "CPU graph feature extraction completed."
