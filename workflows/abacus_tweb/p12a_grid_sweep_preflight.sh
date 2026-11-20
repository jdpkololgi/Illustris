#!/bin/bash
# Run as one approved GPU job step; only a throughput/compatibility gate.
set -euo pipefail
: "${SLURM_STEP_ID:?requires compute step}" "${SLURM_JOB_ID:?requires allocation}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd /global/u2/d/dkololgi/TNG/Illustris
r=/pscratch/sd/d/dkololgi/abacus
out="$r/p12a_grid_probe_$SLURM_JOB_ID"
mkdir "$out"
git rev-parse HEAD > "$out/revision"
for arch in reference large_kernel window_attention; do
    name=sweep_${arch}_preflight
    /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.abacus_tweb.p12a_encoder_sweep "$arch" \
        --contract-root "$r/p12a_lowz_20261029_v1/training_contract_v1" --output-root "$out" \
        --run-name "$name" --model unet --seed 42 --ordered-prefetch --disable-early-stopping \
        --epochs 4 --min-epochs 4 --loss-log-every 8 --stop-after-updates 64 >"$out/$arch.log" 2>&1
    jq -e '.pass and .global_step==64 and .epoch==1 and .cursor==64' \
        "$out/$name/unet/seed_42/TECHNICAL_CANARY_COMPLETE.json"
done
