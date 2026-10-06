#!/usr/bin/env bash
set -eo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${CFM_SCORE_THREADS:-4}
export OPENBLAS_NUM_THREADS=${CFM_SCORE_THREADS:-1} MKL_NUM_THREADS=${CFM_SCORE_THREADS:-1}
set -u
exec python "$@"
