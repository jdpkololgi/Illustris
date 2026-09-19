#!/bin/bash
# One approved technical GPU hour; this script never requests a successor.
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$1"
python -m workflows.sbi.e2e_coupled_prepare_ops register --job-id "$SLURM_JOB_ID" --kind gpu --gpus 1 --hours 1
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1
srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores --time=00:55:00 \
  python -u -m workflows.sbi.e2e_coupled_gpu_benchmark \
  > "$run_root/gpu_benchmark_${SLURM_JOB_ID}.log" 2>&1
