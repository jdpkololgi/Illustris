#!/usr/bin/env bash
#SBATCH -A desi_g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH --licenses=scratch
#SBATCH -t 04:00:00
#SBATCH -J p12f3_d2
#SBATCH -o /pscratch/sd/d/dkololgi/abacus/p12f3_d2_%x_%j.out
#SBATCH -e /pscratch/sd/d/dkololgi/abacus/p12f3_d2_%x_%j.err

# One reviewed, single-GPU production stage.  Submission must explicitly export
# D2_ACTION, D2_SOURCE_ROOT, EXPECTED_GIT_REVISION, and
# EXPECTED_CONFIG_SHA256.  Official stages use a dedicated immutable worktree;
# unrelated P12-A/log commits in the canonical checkout must not strand D2.
# The frozen D2
# contract then verifies every registered source and data artifact transitively.
set -euo pipefail

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO="${D2_SOURCE_ROOT:?submit with the dedicated immutable D2 worktree}"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || {
  echo "D2_SOURCE_ROOT is not a Git worktree" >&2
  exit 3
}
CONFIG="$REPO/configs/p12f3_d2_diffusion_v1.json"
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
LAUNCHER="$REPO/workflows/sbi/run_p12f3_d2_in_allocation.sh"

: "${D2_ACTION:?submit with --export=ALL,D2_ACTION=...}"
: "${EXPECTED_GIT_REVISION:?submit with the reviewed committed Git revision}"
: "${EXPECTED_CONFIG_SHA256:?submit with the reviewed D2 config SHA256}"

cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_GIT_REVISION" ]] || {
  echo "D2 Git revision guard failed" >&2
  exit 3
}
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] || {
  echo "D2 config hash guard failed" >&2
  exit 3
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dedicated D2 source worktree is not globally clean; refusing production stage" >&2
  exit 3
}

# shellcheck disable=SC2086
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --gpus-per-task=1 \
  "$LAUNCHER" $D2_ACTION
