#!/bin/bash
#SBATCH --job-name=abacus_tweb_3414
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_tweb_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_tweb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

# -----------------------------------------------------------------------------
# Abacus T-Web MPI job for 3414^3 density field
#
# Uses run_tweb_memory_optimized (from abacus_process_particles2.py) which
# aggressively frees intermediate arrays, reducing peak per-rank memory from
# ~200+ GB (original mpi_run_tweb) to ~90 GB.
#
# Memory budget: 16 nodes x 2 tasks/node = 32 ranks.
#   512 GB/node / 2 tasks = ~250 GB available per rank.
#   Peak usage ~90 GB/rank → >150 GB headroom.
#
# Prerequisites:
#   - 32 density slab files at:
#     /pscratch/sd/d/dkololgi/AbscusSummit_densities/abacus_z0200_density_slab_rankXXXX.npz
#   - Slab count MUST match total MPI tasks (32).
# -----------------------------------------------------------------------------

PROJECT_DIR="/global/homes/d/dkololgi/TNG/Illustris"
SLAB_DIR="/pscratch/sd/d/dkololgi/AbscusSummit_densities"
LOG_DIR="/pscratch/sd/d/dkololgi/logs"
SCRIPT="${PROJECT_DIR}/abacus_cactus_tweb.py"

mkdir -p "${LOG_DIR}"

# Some user bashrc files reference unset vars (e.g. PYTHONPATH).
# Temporarily disable nounset while sourcing shell init.
set +u
source ~/.bashrc
set -u
conda activate cosmic_env

cd "${PROJECT_DIR}"

# Keep one thread per MPI rank to avoid hidden oversubscription and memory spikes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NTASKS=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))

echo "Host: $(hostname)"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Tasks/node: ${SLURM_NTASKS_PER_NODE}"
echo "Total MPI tasks: ${NTASKS}"
echo "Using script: ${SCRIPT}"
echo "Using slab dir: ${SLAB_DIR}"

SLAB_COUNT=$(ls "${SLAB_DIR}"/abacus_z0200_density_slab_rank*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "Detected slab files: ${SLAB_COUNT}"

if [[ "${SLAB_COUNT}" -ne "${NTASKS}" ]]; then
    echo "ERROR: slab file count (${SLAB_COUNT}) does not match MPI task count (${NTASKS})."
    echo "Regenerate slabs with the same MPI size, or submit with matching tasks."
    exit 1
fi

echo "Starting MPI T-Web run..."
srun --cpu-bind=cores python "${SCRIPT}"
echo "MPI T-Web run completed."

