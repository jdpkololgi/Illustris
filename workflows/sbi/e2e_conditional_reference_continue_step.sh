#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
CONT_ROOT=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/conditional_reference_continuation_20260919_v1
cd "$CONT_ROOT/source"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONT_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --nodes=1 --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=single:1 \
  --cpu-bind=cores --kill-on-bad-exit=1 --export=ALL \
  "$CONT_PY" -u -m workflows.sbi.e2e_conditional_reference_continue worker \
  --output "$CONT_ROOT" --mode smoke
srun --nodes=1 --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=single:1 \
  --cpu-bind=cores --kill-on-bad-exit=1 --export=ALL \
  "$CONT_PY" -u -m workflows.sbi.e2e_conditional_reference_continue worker \
  --output "$CONT_ROOT" --mode run
"$CONT_PY" -u -m workflows.sbi.e2e_conditional_reference_continue collect --output "$CONT_ROOT"
