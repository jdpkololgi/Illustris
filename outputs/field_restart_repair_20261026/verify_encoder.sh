#!/bin/bash
set -euo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
python -m workflows.sbi.p12a_field_training_readiness --mode compute --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/restart_repair_20261026_v1/encoder_fixed_pool
