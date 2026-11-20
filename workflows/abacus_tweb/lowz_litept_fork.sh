#!/bin/bash
# One approved GPU step, no retry. Keep prior trajectories and packages read-only.
set -euo pipefail
: "${SLURM_STEP_ID:?}" "${SLURM_JOB_ID:?}"
repo=/global/u2/d/dkololgi/TNG/Illustris
r=/pscratch/sd/d/dkololgi/abacus
d="$r/lowz_litept_fork_$SLURM_JOB_ID"
mkdir "$d"
exec >"$d/probe.log" 2>&1
git -C "$repo" show "${1:?diagnostic revision}:workflows/abacus_tweb/lowz_litept_fork_probe.py" >"$d/probe.py"
git clone -q --shared --no-checkout "$repo" "$d/src"
cd "$d/src"
git sparse-checkout init --cone
git sparse-checkout set shared workflows configs
git -c advice.detachedHead=false checkout -q --detach fc0b407371dba9d164b29f103755fdbc6dc86b81
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 CUMM_CUDA_ARCH_LIST=8.0
export XDG_CACHE_HOME="$d/cache" CUDA_CACHE_PATH="$d/cache/cuda"
echo "diagnostic=$1 trainer=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID"
"$r/lowz_litept_preflight_61662989/venv/bin/python" -u "$d/probe.py" run "$d"
jq '{strict_pass,steps_total,timed_updates,timed_seconds,observed_updates_per_second}' "$d/FORK_DIAGNOSTIC_COMPLETE.json"
