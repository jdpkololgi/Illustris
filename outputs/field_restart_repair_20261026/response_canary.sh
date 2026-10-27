#!/bin/bash
set -euo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
timeout 900 python -m workflows.sbi.p12a_prepare_response_inputs --phase ph007
