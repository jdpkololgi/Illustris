#!/bin/bash
set -euo pipefail
job_id=${1:?allocation ID required}
run_root=${2:?frozen run root required}
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --jobid="$job_id" --nodes=1 --ntasks=4 --cpus-per-task=16 \
    --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 \
    "$python_bin" -u -m workflows.sbi.e2e_reference_resolution worker --output "$run_root"
srun --jobid="$job_id" --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
    "$python_bin" -u -m workflows.sbi.e2e_reference_resolution collect --output "$run_root"
