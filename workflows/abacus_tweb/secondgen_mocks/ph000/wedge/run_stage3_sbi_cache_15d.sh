#!/bin/bash
set -euxo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module purge || true
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd /global/homes/d/dkololgi/TNG/Illustris
$PY workflows/abacus_tweb/build_abacus_sbi_cache.py \
  --gnn-metadata-path /pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json \
  --targets-catalog-path /pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_wedge_targets.fits \
  --no-apply-y1y5-filter --no-exclude-invalid-box-index \
  --output-cache-path /pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl
