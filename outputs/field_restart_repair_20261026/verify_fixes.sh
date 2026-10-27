#!/bin/bash
set -uo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/restart_repair_20261026_v1
python -m workflows.sbi.e2e_wholefield_unet_readiness --mode compute --output "$root/endpoint_fixed" --source-revision configs/field_restart_numerics_20261026.json
a=$?
python -m workflows.sbi.p12a_field_training_readiness --mode compute --output "$root/encoder_fixed"
b=$?
exit $((a || b))
