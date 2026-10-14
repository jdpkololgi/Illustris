#!/bin/bash
# Explicitly approved 2026-10-14: one fixed stability job per architecture.
# No fits, successor submissions, retries, cancellations or checkpoint selection.
set -euo pipefail
architecture=${1:?architecture required}
[[ $architecture == unet || $architecture == transformer ]] || exit 64
repo=/global/u2/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1/stability_v1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
cd "$repo"
"$py" /global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
mkdir -p "$root"
set -C
exec > "$root/${architecture}_OWNER.log" 2>&1
sha256sum workflows/sbi/e2e_wholefield_pilot_{models,train,data,smoke,worker,exterior,evaluate,campaign,stability}.py \
    workflows/sbi/e2e_wholefield_pilot_stability_gpu.sh > "$root/${architecture}_SOURCES.sha256"
date -u '+owner_start=%Y-%m-%dT%H:%M:%SZ'
set +e
salloc -N1 -n1 -c32 --gpus=1 --constraint='gpu&hbm80g' \
    --qos=shared_interactive --account=desi_g --licenses=scratch \
    --time=03:00:00 --immediate=600 --job-name="wf_stability_${architecture}" \
    srun --mpi=none --resv-ports=0 --kill-on-bad-exit=1 -N1 -n1 -c32 \
    --gpus=1 --cpu-bind=cores "$py" -u -m workflows.sbi.e2e_wholefield_pilot_stability \
    --architecture "$architecture" --seconds 10200
result=$?
sha256sum -c "$root/${architecture}_SOURCES.sha256"
source_result=$?
printf 'allocation_exit_code=%s source_check_exit_code=%s\n' "$result" "$source_result"
date -u '+owner_end=%Y-%m-%dT%H:%M:%SZ'
(( result == 0 && source_result == 0 ))
