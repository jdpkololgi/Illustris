#!/bin/bash
set -euo pipefail
pilot_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_20260924_v1
cd "$pilot_root/source"
if [[ ${1:-} != compute ]]; then
    set +e
    salloc --job-name=cfm_pilot --account=desi_g --constraint=gpu --nodes=1 \
        --ntasks=4 --cpus-per-task=16 --gpus=4 --qos=interactive \
        --time=02:00:00 --licenses=scratch --immediate=600 \
        bash workflows/sbi/run_e2e_coupled_cfm_pilot.sh compute > "$pilot_root/allocation.log" 2>&1
    status=$?
    printf 'EXIT_CODE=%s\n' "$status" >> "$pilot_root/allocation.log"
    exit "$status"
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
srun --exclusive -N1 -n1 -c16 --gpus=1 python -m unittest tests.test_e2e_coupled_cfm_pilot
pids=()
for seed in 17 29; do
    for stage in coarse fine; do
        srun --exclusive -N1 -n1 -c16 --gpus=1 python -u -m workflows.sbi.e2e_coupled_cfm_pilot \
            --stage "$stage" --seed "$seed" --output "$pilot_root/${stage}_${seed}" \
            --seconds 6600 > "$pilot_root/${stage}_${seed}.log" 2>&1 &
        pids+=("$!")
    done
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
