#!/bin/bash
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$1"
srun --jobid="$2" --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --cpu-bind=cores \
    python -u -m workflows.sbi.e2e_coupled_condition_smoke --phase ph007 --seconds 9000
