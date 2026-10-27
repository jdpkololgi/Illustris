#!/bin/bash
set -uo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/restart_repair_20261026_v1
python -m outputs.field_restart_repair_20261026.diagnose --scope endpoint --output "$root/endpoint_strict"
a=$?
python -m outputs.field_restart_repair_20261026.diagnose --scope encoder --output "$root/encoder_strict"
b=$?
exit $((a || b))
