#!/bin/bash
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
cd "$1"
srun --jobid="$2" --exact -N1 -n1 -c32 --mem=128G --cpu-bind=cores \
  python -u -m workflows.sbi.e2e_coupled_legacy_native --seconds 10500
