#!/bin/bash
# One bounded technical step within an existing approved allocation; no salloc.
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$1"
exec python -u -m workflows.sbi.e2e_coupled_postprocess_benchmark
