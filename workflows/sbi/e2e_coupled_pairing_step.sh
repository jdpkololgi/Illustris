#!/bin/bash
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
snapshot="$1"
cd "$snapshot"
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --hours 1
exec srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
    python -u -m workflows.sbi.e2e_coupled_pairing --stop-after-seconds 3300 \
    --phases ph007 ph008 ph009 ph010 ph011 ph020 ph021 ph022 ph023 ph024 \
    ph012 ph013 ph014 ph015 ph016 ph017 ph018 ph019 ph000 ph002 ph003
