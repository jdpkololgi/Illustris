#!/usr/bin/env bash
# One fail-closed JEPA continuation step inside an existing interactive allocation.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
RUN_NAME=${P11_RUN_NAME:-paired_degrade_jepa_m25_v2}
RUN_DIR=${P11_RUN_DIR:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/training/paired_degrade_jepa_v2/${RUN_NAME}/jepa/seed_42}
CONTRACT=${P11_CONTRACT:-${REPO}/configs/p11_paired_degrade_jepa_v2.json}
GUARD=${REPO}/workflows/abacus_tweb/p11_jepa_supervisor_guard.py
WORKER=${REPO}/workflows/abacus_tweb/run_p11_jepa_interactive.sh
PY=${P11_PYTHON:-${COSMIC_ENV_PYTHON:-/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python}}
STATUS_FILE=${P11_SUPERVISOR_STATUS_FILE:-}
STARTED_FILE=${P11_SUPERVISOR_STARTED_FILE:-}

if [[ -n "${STARTED_FILE}" ]]; then
  temporary_started="${STARTED_FILE}.tmp.$$"
  printf 'started_utc=%s\nslurm_job_id=%s\n' \
    "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-missing}" > "${temporary_started}"
  mv "${temporary_started}" "${STARTED_FILE}"
fi

write_status() {
  local worker_code=$1
  local checkpoint_valid=$2
  local complete_valid=$3
  local reason=$4
  if [[ -z "${STATUS_FILE}" ]]; then
    return
  fi
  local temporary="${STATUS_FILE}.tmp.$$"
  printf '{"created_utc":"%s","slurm_job_id":"%s","worker_exit_code":%d,"checkpoint_valid":%s,"complete_valid":%s,"reason":"%s"}\n' \
    "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-missing}" "${worker_code}" \
    "${checkpoint_valid}" "${complete_valid}" "${reason}" > "${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "P11 science worker requires an interactive allocation" >&2
  write_status 64 false false no_slurm_job
  exit 64
fi
if [[ "${SLURM_JOB_PARTITION:-}" != *gpu* && "${SLURM_JOB_CONSTRAINTS:-}" != *gpu* ]]; then
  echo "P11 science worker requires a GPU allocation" >&2
  write_status 64 false false no_gpu_allocation
  exit 64
fi
if [[ -z "${PY}" || ! -x "${PY}" || "${PY}" != /global/cfs/* ]]; then
  echo "P11 science worker requires the executable CFS recovery interpreter" >&2
  write_status 64 false false invalid_cfs_interpreter
  exit 64
fi
if [[ "${P11_ARM:-jepa}" != "jepa" ]]; then
  echo "This persistent worker is intentionally restricted to JEPA" >&2
  write_status 64 false false non_jepa_arm_refused
  exit 64
fi

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export P11_PYTHON="${PY}"
export P11_ARM=jepa
export P11_RUN_NAME="${RUN_NAME}"
# Empty means: resume beyond the already-passed 500-update technical canary and
# registered 0/250/500 latent diagnostic gate.
export P11_STOP_AFTER_UPDATES=
cd "${REPO}"

if ! "${PY}" -m unittest tests.phase4.test_p11_jepa_supervisor_guard; then
  echo "P11 supervisor-guard focused tests failed" >&2
  write_status 64 false false supervisor_guard_tests_failed
  exit 64
fi
if ! "${PY}" -u "${GUARD}" --mode checkpoint --run-dir "${RUN_DIR}" --contract "${CONTRACT}"; then
  echo "P11 pre-resume checkpoint/provenance guard failed" >&2
  write_status 64 false false pre_resume_guard_failed
  exit 64
fi

set +e
bash "${WORKER}"
code=$?
set -e

if [[ ${code} -eq 75 ]]; then
  if "${PY}" -u "${GUARD}" --mode checkpoint --run-dir "${RUN_DIR}" --contract "${CONTRACT}"; then
    write_status 75 true false checkpointed_allocation_pause
    exit 75
  fi
  echo "Trainer requested continuation but its checkpoint guard failed" >&2
  write_status 65 false false post_pause_checkpoint_invalid
  exit 65
fi

if [[ ${code} -eq 0 ]]; then
  if "${PY}" -u "${GUARD}" --mode complete --run-dir "${RUN_DIR}" --contract "${CONTRACT}"; then
    write_status 0 true true scientific_jepa_complete
    exit 0
  fi
  echo "Trainer exited zero without a valid scientific completion marker" >&2
  write_status 66 false false zero_exit_without_valid_completion
  exit 66
fi

echo "P11 JEPA worker failed with unexpected code ${code}" >&2
write_status "${code}" false false unexpected_trainer_exit
exit "${code}"
