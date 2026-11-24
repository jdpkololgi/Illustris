#!/bin/bash
set -eu
r=/pscratch/sd/d/dkololgi/abacus
out=$r/encoder_prod_window_attention_61664810
d=$out/sweep_window_attention_production/unet/seed_42
here=/global/u2/d/dkololgi/TNG/Illustris/workflows/abacus_tweb
if [[ ${1:-} == step ]]; then
    cd "$out/src"
    test "$(git rev-parse HEAD)" = 71307f51349fd71263d5db8daa01e1d40ce7c010
    jq -e '.epoch==2 and .cursor==300912 and .global_step==647276' "$d/ALLOCATION_PAUSED.json" >/dev/null
    jq -r '.source_contract|to_entries[]|.value+"  "+.key' "$d/run_manifest.json" | sha256sum --check --status
    unset PYTHONHOME PYTHONUSERBASE LD_PRELOAD
    export PYTHONPATH="$out/src" PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
    exec srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u "$here/window_epoch2_interactive.py" window_attention --ordered-prefetch --epochs 4 --min-epochs 4 --contract-root "$r/p12a_lowz_20261029_v1/training_contract_v1" --model unet --seed 42 --disable-early-stopping --auto-resume --run-name sweep_window_attention_production --output-root "$out" --checkpoint-every 1000 --max-runtime-seconds 13800 --validation-reserve-seconds 6500
fi
test "$(squeue -h -j 61786757 -o %T)" = PENDING
scontrol hold 61786757
[[ "$(squeue -h -j 61786757 -o '%T %r')" == 'PENDING JobHeldUser' ]]
marker="$r/window-epoch2-granted-61786757"
test ! -e "$marker"
if ! salloc -N1 -n1 -c32 --constraint='gpu&a100' --gpus=1 --qos=shared_interactive --account=desi_g --licenses=scratch --time=04:00:00 --immediate=600 --job-name=window-e2-eval bash -c 'echo "$SLURM_JOB_ID" > "$1"; exec bash "$2" step' _ "$marker" "$here/window_epoch2_window.sh"; then
    if [[ ! -f "$marker" ]]; then scontrol release 61786757; fi
    exit 1
fi
