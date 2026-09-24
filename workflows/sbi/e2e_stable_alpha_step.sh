#!/bin/bash
set -euo pipefail
job_id=${1:?allocation required}
run_root=${2:?root required}
trap 'status=$?; printf "%s\n" "$status" > "$run_root/LAUNCH_EXIT_CODE"' EXIT
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --jobid="$job_id" -N1 -n4 -c16 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 \
 "$python_bin" -u -m workflows.sbi.e2e_stable_alpha worker --output "$run_root"
srun --jobid="$job_id" -N1 -n1 -c8 --gpus=1 "$python_bin" -u -m workflows.sbi.e2e_stable_alpha collect --output "$run_root"
