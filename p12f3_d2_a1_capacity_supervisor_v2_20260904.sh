#!/usr/bin/env bash
set -euo pipefail

repo=/global/u2/d/dkololgi/TNG/Illustris_d2_467f442
expected=467f442c5c54864658fdfaf948335d6e11a647fe
output=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1
launcher=${repo}/workflows/sbi/run_p12f3_d2_in_allocation.sh
a1_marker=${output}/training/modern_base8/seed42_v1/D2_CANARY_COMPLETE.json
capacity_marker=${output}/D2_CAPACITY_SELECTION.json
log=${output}/d2_a1_capacity_supervisor_20260904.log
lock=${output}/d2_a1_capacity_supervisor_20260904.lock

worker() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "Refusing D2 A1 outside Slurm" >&2
    exit 2
  }
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
  export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
  export D2_SOURCE_ROOT="${repo}" D2_OUTPUT_ROOT="${output}"

  if [[ ! -f "${a1_marker}" ]]; then
    "${launcher}" a1
  else
    echo "D2 A1 marker already exists; not retraining"
  fi
  [[ -f "${a1_marker}" ]] || {
    echo "D2 A1 did not produce its completion marker" >&2
    exit 4
  }

  if [[ ! -f "${capacity_marker}" ]]; then
    "${launcher}" select-capacity
  else
    echo "D2 capacity marker already exists; not overwriting"
  fi
  [[ -f "${capacity_marker}" ]] || {
    echo "D2 capacity selection did not produce its marker" >&2
    exit 5
  }
  echo "D2 A1 and registered train-only capacity selection complete; stopping before A2"
}

if [[ "${1:-}" == worker ]]; then
  worker
  exit 0
fi

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "D2 A1/capacity supervisor already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

[[ "$(git -C "${repo}" rev-parse HEAD)" == "${expected}" ]] || {
  echo "D2 pinned worktree revision changed" >&2
  exit 6
}
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || {
  echo "D2 pinned worktree is dirty" >&2
  exit 7
}
[[ -f "${output}/D2_CONTRACT_FROZEN.json" ]] || {
  echo "D2 frozen contract is absent" >&2
  exit 8
}
[[ -f "${output}/D2_TRANSFORM_ROUNDTRIP.json" && -f "${output}/D2_GPU_SMOKE.json" ]] || {
  echo "D2 preflight markers are absent" >&2
  exit 9
}
[[ -f "${output}/D2_MATCHED_REFERENCE_REPORTS.json" ]] || {
  echo "D2 matched-reference marker is absent" >&2
  exit 10
}
[[ -f "${output}/training/modern_base4/seed42_v1/D2_CANARY_COMPLETE.json" ]] || {
  echo "D2 A0 completion marker is absent" >&2
  exit 11
}

echo "[$(date -u +%FT%TZ)] requesting one bounded GPU interactive allocation"
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --constraint=gpu \
  --gpus=1 --qos=shared_interactive --time=02:00:00 --account=desi_g \
  --licenses=scratch,u2 --immediate=600 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
  --cpu-bind=cores --export=ALL "${BASH_SOURCE[0]}" worker
