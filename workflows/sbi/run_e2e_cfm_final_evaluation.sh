#!/bin/bash
set -euo pipefail
eval_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_final_eval_20260925_v1
cd "$eval_root/source"
if [[ ${1:-} != compute ]]; then
    set +e
    salloc --job-name=cfm_final_eval --account=desi_g --constraint=gpu --nodes=1 \
        --ntasks=2 --cpus-per-task=32 --gpus=2 --qos=shared_interactive \
        --time=04:00:00 --licenses=scratch --immediate=600 \
        bash workflows/sbi/run_e2e_cfm_final_evaluation.sh compute > "$eval_root/allocation.log" 2>&1
    result=$?
    printf 'EXIT_CODE=%s\n' "$result" >> "$eval_root/allocation.log"
    exit "$result"
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
srun --exclusive -N1 -n1 -c32 --gpus=1 python -m unittest tests.test_e2e_cfm_pilot_evaluate
pids=()
for seed in 17 29; do
    srun --exclusive -N1 -n1 -c32 --gpus=1 python -u -m workflows.sbi.e2e_cfm_pilot_evaluate \
        --seed "$seed" --step 26624 --output "$eval_root/results" \
        --seconds 13800 > "$eval_root/seed${seed}.log" 2>&1 &
    pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
if [[ $result == 0 ]]; then
    python -m workflows.sbi.e2e_cfm_pilot_eval_report --root "$eval_root/results" \
        --baseline-root /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_eval_20260924_v1/results || result=1
fi
exit "$result"
