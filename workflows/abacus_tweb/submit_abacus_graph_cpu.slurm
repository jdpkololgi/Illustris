#!/bin/bash
#SBATCH --job-name=abacus_alpha_graph
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_alpha_graph_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_alpha_graph_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

# Build Abacus mock alpha-complex graph artifacts on a CPU node.
# Notes:
# - Uses all Y1/Y5 mock galaxies from FITS.
# - Uses observed Z (RSD-preserving) via build_abacus_graph.py defaults.
# - Override walltime/QOS here if queue policy requires different limits.

PROJECT_DIR="${TNG_ILLUSTRIS_PROJECT_DIR:-/global/homes/d/dkololgi/TNG/Illustris}"
LOG_DIR="${TNG_LOG_DIR:-/pscratch/sd/d/dkololgi/logs}"
CATALOG_PATH="${ABACUS_MOCK_CATALOG_PATH:-/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits}"
OUTPUT_DIR="${ABACUS_GRAPH_OUTPUT_DIR:-/pscratch/sd/d/dkololgi/abacus}"
OUTPUT_PREFIX="${ABACUS_GRAPH_PREFIX:-abacus_alpha}"
BOXSIZE_MPC="${ABACUS_GRAPH_BOXSIZE_MPC:-2000.0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

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
echo "Catalog: ${CATALOG_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Boxsize (Mpc): ${BOXSIZE_MPC}"
echo "Starting Abacus graph build..."

srun --cpu-bind=cores python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "${CATALOG_PATH}" \
  --mode alpha \
  --boxsize-mpc "${BOXSIZE_MPC}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}"

echo "Abacus graph build completed."
