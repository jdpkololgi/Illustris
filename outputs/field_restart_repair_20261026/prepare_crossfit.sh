#!/bin/bash
set -euo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
timeout 1800 python -m workflows.sbi.p12_prepare_crossfit_contracts --full-contract /pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/training_contract_20261021 --source-root /pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020 --output-root /pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/p12a_crossfit_20261026_v1 --workers 4
