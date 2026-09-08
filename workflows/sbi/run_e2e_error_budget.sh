#!/bin/bash
# Foreground, user-authorized interactive audit; no model fits or D2 changes.
set -euo pipefail
: "${SLURM_JOB_ID:?Use the authorized interactive allocation}"
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1
E2E_AUDIT_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  --output="/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/error_budget_20260908_${SLURM_JOB_ID}.log" \
  "$E2E_AUDIT_PY" -u -m workflows.sbi.e2e_field_error_budget
