#!/bin/bash
# One explicitly authorized interactive segment. No retry, scheduler chain,
# autonomous agent, cancellation, or batch fallback.
set -euo pipefail
architecture=${1:?architecture required}
segment=${2:?unique segment required}
engineering=${3:-no}
[[ $architecture == unet || $architecture == transformer ]] || exit 64
[[ $segment =~ ^[a-z][0-9]{3}$ ]] || exit 64
[[ $engineering == yes || $engineering == no ]] || exit 64
repo=/global/u2/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
cd "$repo"
# Read-only slot check immediately before requesting resources. A full slot
# count exits rather than waiting in a hidden launcher or creating a third job.
"$py" /global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
set -C
exec > "$root/${architecture}_${segment}_OWNER.log" 2>&1
options=()
[[ $engineering == no ]] || options+=(--engineering)
sha256sum workflows/sbi/e2e_wholefield_pilot_{models,train,smoke,worker,exterior,evaluate}.py > "$root/${architecture}_${segment}_SOURCES.sha256"
date -u '+owner_start=%Y-%m-%dT%H:%M:%SZ'
set +e
salloc -N1 -n1 -c32 --gpus=1 --constraint='gpu&hbm80g' \
    --qos=shared_interactive --account=desi_g --licenses=scratch \
    --time=03:00:00 --immediate=600 --job-name="wf_pilot_${architecture}" \
    srun --mpi=none --resv-ports=0 --kill-on-bad-exit=1 -N1 -n1 -c32 \
    --gpus=1 --cpu-bind=cores "$py" -u -m workflows.sbi.e2e_wholefield_pilot_worker \
    --architecture "$architecture" --seconds 10200 "${options[@]}"
result=$?
sha256sum -c "$root/${architecture}_${segment}_SOURCES.sha256"
source_result=$?
printf 'allocation_exit_code=%s source_check_exit_code=%s\n' "$result" "$source_result"
date -u '+owner_end=%Y-%m-%dT%H:%M:%SZ'
(( result == 0 && source_result == 0 ))
