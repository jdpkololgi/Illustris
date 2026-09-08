#!/bin/bash
# Foreground interactive preparation: no idle allocation shell, no model fits.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this inside the authorized salloc allocation}"
E2E_REPO=/global/u2/d/dkololgi/TNG/Illustris
E2E_DATA=/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/data_20260908
E2E_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$E2E_REPO"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  --output="$E2E_DATA/native_labels_resume_${SLURM_JOB_ID}.log" \
  "$E2E_PY" -m workflows.sbi.e2e_field_package_native_truth
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  --output="$E2E_DATA/validate_${SLURM_JOB_ID}.log" \
  "$E2E_PY" -m workflows.sbi.e2e_field_validate_products
