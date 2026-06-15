#!/bin/bash
#SBATCH -A m4657
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --job-name=mock_wedge_jraph
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/staged_mock_wedge_jraph_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/staged_mock_wedge_jraph_%j.err

set -euo pipefail

REPO="/global/homes/d/dkololgi/TNG/Illustris"
TRUTH_NPZ="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_wedge_stage3_postcollision_rs7.npz"
GRAPH_DIR="/pscratch/sd/d/dkololgi/abacus/graph_constructions"
PREFIX="staged_mock_wedge_stage3_postcollision_rs7"
CACHE_OUT="/pscratch/sd/d/dkololgi/abacus/sbi_caches/${PREFIX}_sbi_cache_15d.pkl"

mkdir -p /pscratch/sd/d/dkololgi/logs "${GRAPH_DIR}" "$(dirname "${CACHE_OUT}")"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosmic_env
PY="${CONDA_PREFIX}/bin/python"

cd "${REPO}"

echo "=== Export points from truth NPZ ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/export_staged_mock_wedge_points.py \
  --truth-npz "${TRUTH_NPZ}" \
  --out-dir "${GRAPH_DIR}" \
  --out-prefix "${PREFIX}"

POINTS="${GRAPH_DIR}/${PREFIX}_points_xyz.npy"

echo "=== Build Delaunay graph (alpha_sq=inf) ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "" \
  --points-path "${POINTS}" \
  --mode delaunay \
  --no-split-hemispheres \
  --output-dir "${GRAPH_DIR}" \
  --output-prefix "${PREFIX}"

echo "=== cuGraph node/edge features ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/abacus_graph_features_cugraph.py \
  --graph-metadata "${GRAPH_DIR}/${PREFIX}_metadata.json" \
  --output-prefix "${PREFIX}_cugraph"

echo "=== SBI cache (15-d if derivatives absent; else 3-d increments) ==="
srun --cpu-bind=cores "${PY}" workflows/abacus_tweb/build_staged_mock_wedge_sbi_cache.py \
  --gnn-metadata-path "${GRAPH_DIR}/${PREFIX}_cugraph_gnn_metadata.json" \
  --truth-npz-path "${TRUTH_NPZ}" \
  --output-cache-path "${CACHE_OUT}"

echo "Done. Cache: ${CACHE_OUT}"
