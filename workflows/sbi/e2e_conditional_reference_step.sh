#!/bin/bash
# Finite GPU step; call only within the separately approved allocation.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
RUN_ROOT=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/conditional_reference_20260919_v1
cd "$RUN_ROOT/source"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REF_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$REF_PY" -u -m workflows.sbi.e2e_conditional_reference \
  --config configs/e2e_conditional_reference_v1.json --output "$RUN_ROOT/smoke" --mode smoke
"$REF_PY" -u -m workflows.sbi.e2e_conditional_reference \
  --config configs/e2e_conditional_reference_v1.json --output "$RUN_ROOT/results" --mode run
