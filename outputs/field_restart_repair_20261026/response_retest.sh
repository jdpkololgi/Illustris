#!/bin/bash
set -euo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
python -m unittest tests.phase4.test_random_response_sequential_read tests.phase4.test_p12_parent_response tests.phase4.test_p12_expanded_inputs
python -m outputs.field_restart_repair_20261026.verify_parent_response
python -m outputs.field_restart_repair_20261026.verify_endpoint_entrypoint
timeout 600 python -m workflows.sbi.p12a_prepare_response_inputs --phase ph007
