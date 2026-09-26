#!/bin/bash
set -euo pipefail
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coarse_controls_20260926_v1
cd "$root/source"
if [[ ${1:-} != compute ]]; then
    set +e
    salloc --job-name=coarse_controls --account=desi_g --constraint=gpu --nodes=1 \
        --ntasks=4 --cpus-per-task=16 --gpus=4 --qos=interactive \
        --time=04:00:00 --licenses=scratch --immediate=600 \
        bash workflows/sbi/run_e2e_coarse_controls.sh compute > "$root/allocation.log" 2>&1
    result=$?
    printf 'EXIT_CODE=%s\n' "$result" >> "$root/allocation.log"
    exit "$result"
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
srun --exclusive -N1 -n1 -c16 --gpus=1 python -m unittest tests.test_e2e_coarse_controls tests.test_e2e_cfm_pilot_evaluate
srun --exclusive -N1 -n1 -c16 --gpus=1 python -u -m workflows.sbi.e2e_coarse_controls --root "$root" --mode fit
# Full-size classical smoke and tightened solver tolerance before the panel.
srun --exclusive -N1 -n1 -c16 --gpus=1 python -u -m workflows.sbi.e2e_coarse_controls --root "$root" --mode classical --shard 0 --limit 1
pids=();shard=0
for seed in 17 29; do
    for step in 13312 26624; do
        srun --exclusive -N1 -n1 -c16 --gpus=1 bash -c '
            set -e
            python -u -m workflows.sbi.e2e_coarse_controls --root "$1" --mode neural --seed "$2" --step "$3"
            python -u -m workflows.sbi.e2e_coarse_controls --root "$1" --mode classical --shard "$4"
        ' _ "$root" "$seed" "$step" "$shard" > "$root/worker${shard}.log" 2>&1 &
        pids+=("$!");shard=$((shard+1))
    done
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
if [[ $result == 0 ]]; then
    python -m workflows.sbi.e2e_coarse_controls_report --root "$root" || result=1
fi
exit "$result"
