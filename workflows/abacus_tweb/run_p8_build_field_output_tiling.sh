#!/usr/bin/env bash
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PYTHON=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
OUT=/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/field_output_tiling

cd "$REPO"
mkdir -p "$OUT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD || true
exec "$PYTHON" -u -m workflows.abacus_tweb.p8_build_field_output_tiling \
  --output "$OUT"
