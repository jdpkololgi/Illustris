#!/bin/bash
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_unattended_20261021
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python "$campaign/verify_source.py"
exec bash "$campaign/source/workflows/abacus_tweb/run_p10_unattended_stage_20261021.sh" "$@"
