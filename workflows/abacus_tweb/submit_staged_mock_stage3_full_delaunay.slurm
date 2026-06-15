# DO NOT SBATCH - use salloc only (interactive, -A desi, max 4h)
#!/bin/bash
#SBATCH --job-name=mock_s3_full_dln
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/staged_mock_stage3_full_delaunay_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/staged_mock_stage3_full_delaunay_%j.err

set -euo pipefail

REPO="/global/homes/d/dkololgi/TNG/Illustris"
GRAPH_DIR="/pscratch/sd/d/dkololgi/abacus/graph_constructions"
PREFIX="staged_mock_stage3_postcollision_full_rs7"
POINTS="${GRAPH_DIR}/${PREFIX}_points_xyz.npy"

mkdir -p /pscratch/sd/d/dkololgi/logs "${GRAPH_DIR}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosmic_env
PY="${CONDA_PREFIX}/bin/python"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO}"

if [[ ! -f "${POINTS}" ]]; then
  echo "ERROR: points file missing: ${POINTS}" >&2
  echo "Run submit_staged_mock_stage3_full_points_export.slurm first." >&2
  exit 2
fi

echo "Host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Points: ${POINTS}"
echo "NPY size bytes: $(stat -c%s "${POINTS}")"

echo "=== Build Delaunay graph (alpha_sq=inf, single complex) ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "" \
  --points-path "${POINTS}" \
  --mode delaunay \
  --no-split-hemispheres \
  --output-dir "${GRAPH_DIR}" \
  --output-prefix "${PREFIX}"

echo "Done. Artifacts: ${GRAPH_DIR}/${PREFIX}_*"
