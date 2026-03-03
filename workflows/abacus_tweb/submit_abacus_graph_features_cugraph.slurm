#!/bin/bash
#SBATCH --job-name=abacus_feat_gpu
#SBATCH --account=desi
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/abacus_feat_gpu_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/abacus_feat_gpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dakshesh.kololgi.23@ucl.ac.uk

set -euo pipefail

# Compute Abacus graph features on GPU with cuGraph.

PROJECT_DIR="${TNG_ILLUSTRIS_PROJECT_DIR:-/global/homes/d/dkololgi/TNG/Illustris}"
LOG_DIR="${TNG_LOG_DIR:-/pscratch/sd/d/dkololgi/logs}"
GRAPH_DIR="${ABACUS_GRAPH_DIR:-/pscratch/sd/d/dkololgi/abacus/graph_constructions}"
PREFIX="${ABACUS_GRAPH_PREFIX:-abacus_alpha}"
OUTPUT_PREFIX="${ABACUS_FEATURES_CUGRAPH_OUTPUT_PREFIX:-${PREFIX}_cugraph}"
BATCH_SIZE="${ABACUS_CUGRAPH_BATCH_SIZE:-2000000}"
RAPIDS_ENV_PATH="${ABACUS_RAPIDS_ENV_PATH:-/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn}"

mkdir -p "${LOG_DIR}" "${GRAPH_DIR}"

# Some user bashrc files reference unset vars (e.g. PYTHONPATH).
set +u
source ~/.bashrc
# Prevent DESI Python path/library leakage into RAPIDS env.
unset PYTHONPATH PYTHONHOME LD_PRELOAD
set -u
source /global/homes/d/dkololgi/miniforge3/bin/activate "${RAPIDS_ENV_PATH}"

cd "${PROJECT_DIR}"

echo "Host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "RAPIDS env: ${RAPIDS_ENV_PATH}"
echo "Points: auto-resolve (<prefix>_points.npy preferred)"
echo "Artifacts dir: ${GRAPH_DIR}"
echo "Prefix: ${PREFIX}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Batch size: ${BATCH_SIZE}"
echo "Starting GPU graph feature extraction..."

srun python workflows/abacus_tweb/abacus_graph_features_cugraph.py \
  --artifacts-dir "${GRAPH_DIR}" \
  --prefix "${PREFIX}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --batch-size "${BATCH_SIZE}"

echo "GPU graph feature extraction completed."
