#!/bin/bash
set -euo pipefail
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coarse_regularization_20260926_v1
cd "$root/source"
if [[ ${1:-} != compute ]]; then
    set +e
    salloc --job-name=coarse_reg --account=desi_g --constraint=gpu --nodes=1 \
        --ntasks=4 --cpus-per-task=16 --gpus=4 --qos=interactive \
        --time=04:00:00 --licenses=scratch --immediate=600 \
        bash workflows/sbi/run_e2e_coarse_regularization.sh compute > "$root/allocation.log" 2>&1
    result=$?
    printf 'EXIT_CODE=%s\n' "$result" >> "$root/allocation.log"
    exit "$result"
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
srun --exclusive -N1 -n1 -c16 --gpus=1 python -m unittest tests.test_e2e_coarse_regularization tests.test_e2e_coarse_controls
srun --exclusive -N1 -n1 -c16 --gpus=1 python -u -m workflows.sbi.e2e_classical_control_audit --root "$root"
pids=()
for seed in 17 29; do
    for arm in baseline decay; do
        srun --exclusive -N1 -n1 -c16 --gpus=1 python -u -m workflows.sbi.e2e_coarse_regularization \
            --root "$root" --seed "$seed" --arm "$arm" > "$root/${arm}_${seed}.log" 2>&1 &
        pids+=("$!")
    done
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
if [[ $result == 0 ]]; then
    python -m workflows.sbi.e2e_coarse_regularization_report --root "$root" || result=1
fi
exit "$result"
