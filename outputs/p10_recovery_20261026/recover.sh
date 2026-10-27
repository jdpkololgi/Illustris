#!/bin/bash
set -euo pipefail
campaign=/global/homes/d/dkololgi/TNG/Illustris/outputs/p10_recovery_20261026
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python "$campaign/verify_source.py"
exec bash "$campaign/source/workflows/abacus_tweb/run_p10_v2_phase_stage.sh" ph012 p3-p4
