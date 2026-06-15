# DO NOT SBATCH - use salloc only (interactive, -A desi, max 4h)
#!/bin/bash
#SBATCH --job-name=mock_s3_full_pts
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/staged_mock_stage3_full_points_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/staged_mock_stage3_full_points_%j.err

set -euo pipefail

REPO="/global/homes/d/dkololgi/TNG/Illustris"
STAGE3_FITS="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/datcomb_brightwdup.fits"
GRAPH_DIR="/pscratch/sd/d/dkololgi/abacus/graph_constructions"
PREFIX="staged_mock_stage3_postcollision_full_rs7"

mkdir -p /pscratch/sd/d/dkololgi/logs "${GRAPH_DIR}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosmic_env
PY="${CONDA_PREFIX}/bin/python"

cd "${REPO}"

echo "=== Export full stage-3 post-collision points ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/export_staged_mock_stage3_full_points.py \
  --stage3-fits "${STAGE3_FITS}" \
  --out-dir "${GRAPH_DIR}" \
  --out-prefix "${PREFIX}"

POINTS="${GRAPH_DIR}/${PREFIX}_points_xyz.npy"
test -f "${POINTS}"
echo "Export done: ${POINTS}"
