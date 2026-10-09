#!/bin/bash
set -euo pipefail
segment=${1:?segment required}
workers=${2:?worker count required}
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1
cd "$root/source"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$py" -m workflows.sbi.e2e_wholefield_fullcap_run start --segment "$segment"
pids=()
for ((worker=0; worker<workers; worker++)); do
    srun --exclusive --mpi=none --resv-ports=0 -N1 -n1 -c16 --gpus=1 --cpu-bind=cores \
        "$py" -u -m workflows.sbi.e2e_wholefield_fullcap_run worker \
        --segment "$segment" --worker "$worker" &
    pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do
    wait "$pid" || result=1
done
exit "$result"
