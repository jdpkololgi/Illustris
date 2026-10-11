#!/bin/bash
# User-authorized tmux owner for ONE bounded interactive allocation.
# No agent, scheduler retry, allocation chain or numerical-source modification.
set -euo pipefail
segment=${1:?segment required}
gpus=${2:?GPU count required}
seconds=${3:?wall seconds required}
mode=${4:-generation}
[[ $segment =~ ^s[0-9]+$ ]] || exit 64
[[ $gpus == 1 || $gpus == 4 ]] || exit 64
[[ $mode == generation || $mode == extract-s014-s015 ]] || exit 64
[[ $mode != extract-s014-s015 || $gpus == 1 ]] || exit 64
case "$seconds" in
    600) wall=00:10:00 ;;
    1800) wall=00:30:00 ;;
    10800) wall=03:00:00 ;;
    *) exit 64 ;;
esac
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1
repo=/global/u2/d/dkololgi/TNG/Illustris
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
dir="$root/allocations/$segment"
[[ -f $dir/RESERVED.json && -n ${TMUX:-} ]] || exit 64
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["gpus"]==int(sys.argv[2]) and r["wall_seconds"]==int(sys.argv[3])' "$dir/RESERVED.json" "$gpus" "$seconds"
# Exclusive logs prevent silently reusing a segment/attempt.
set -o noclobber
exec > "$dir/TMUX_OWNER.log" 2>&1
sha256sum "$0" "$repo/workflows/sbi/e2e_wholefield_fullcap_parallel.py" \
    "$repo/workflows/sbi/e2e_wholefield_fullcap_batch.py" \
    "$repo/workflows/sbi/e2e_wholefield_fullcap_batch.sh" \
    "$repo/workflows/sbi/e2e_wholefield_extract_s014_s015.sh" > "$dir/TMUX_SOURCES.sha256"
date -u '+owner_start=%Y-%m-%dT%H:%M:%SZ'
printf 'owner_pid=%s host=%s segment=%s gpus=%s\n' "$$" "$(hostname)" "$segment" "$gpus"
set +e
if [[ $mode == extract-s014-s015 ]]; then
    salloc -N1 -n1 -c32 --gpus=1 --constraint='gpu&hbm80g' \
        --qos=shared_interactive --account=desi_g --licenses=scratch \
        --time="$wall" --immediate=600 --job-name="wf_extract_$segment" \
        srun --mpi=none --resv-ports=0 --kill-on-bad-exit=1 -N1 -n1 -c32 \
        --gpus=1 --cpu-bind=cores bash "$repo/workflows/sbi/e2e_wholefield_extract_s014_s015.sh" "$segment"
elif [[ $gpus == 1 ]]; then
    salloc -N1 -n1 -c32 --gpus=1 --constraint='gpu&hbm80g' \
        --qos=shared_interactive --account=desi_g --licenses=scratch \
        --time="$wall" --immediate=600 --job-name="wf_tmux_$segment" \
        bash "$repo/workflows/sbi/e2e_wholefield_fullcap_batch.sh" "$segment"
else
    salloc -N1 -n4 -c32 --gpus=4 --constraint='gpu&hbm80g' \
        --qos=interactive --account=desi_g --licenses=scratch \
        --time="$wall" --immediate=600 --job-name="wf_tmux_$segment" \
        srun --mpi=none --resv-ports=0 --kill-on-bad-exit=1 \
        -N1 -n4 -c32 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
        "$py" -u "$repo/workflows/sbi/e2e_wholefield_fullcap_parallel.py" --segment "$segment"
fi
result=$?
printf '{"allocation_command_exit_code":%s}\n' "$result" > "$dir/TMUX_OWNER_EXIT.json"
sha256sum -c "$dir/TMUX_SOURCES.sha256"
source_result=$?
if [[ -f $dir/STARTED.json ]]; then
    cd "$root/source" || exit 1
    # Slurm accounting may lag allocation release. Retry metadata only,
    # for at most 30 seconds; never resubmit a scheduler job.
    accounting_result=1
    for attempt in 1 2 3 4 5 6; do
        sleep 5
        "$py" -m workflows.sbi.e2e_wholefield_fullcap_run account --segment "$segment"
        accounting_result=$?
        (( accounting_result == 0 )) && break
    done
else
    accounting_result=1
fi
date -u '+owner_end=%Y-%m-%dT%H:%M:%SZ'
(( result == 0 && source_result == 0 && accounting_result == 0 ))
