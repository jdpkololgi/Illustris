#!/usr/bin/env bash
# Persistent, fail-closed salloc supervisor for the post-canary JEPA arm only.
#
# Run this from a durable login-node tmux session.  It never uses sbatch, never
# requests more than one allocation itself, and queries the shared two-job
# interactive limit before every request.  The later controls remain a
# registered plan in this order:
#   1. supervised_masked
#   2. masked_reconstruction
#   3. response_only
# They are deliberately not auto-launched: each needs an objective-specific
# canary and scientific review after the JEPA arm completes.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1
RUN_NAME=${P11_RUN_NAME:-paired_degrade_jepa_v1}
RUN_DIR=${P11_RUN_DIR:-${ROOT}/training/paired_degrade_jepa_v1/${RUN_NAME}/jepa/seed_42}
CONTRACT=${P11_CONTRACT:-${REPO}/configs/p11_paired_degrade_jepa_v1.json}
PY=${P11_PYTHON:-${COSMIC_ENV_PYTHON:-/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python}}
GUARD=${REPO}/workflows/abacus_tweb/p11_jepa_supervisor_guard.py
WORKER=${REPO}/workflows/abacus_tweb/run_p11_jepa_science_worker.sh
ALLOCATION_STATUS=/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
LOG_ROOT=${ROOT}/supervisor_logs/p11_jepa_science
MAX_ALLOCATION_ATTEMPTS=${P11_MAX_ALLOCATION_ATTEMPTS:-24}
SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
SESSION_DIR=${LOG_ROOT}/${SESSION_ID}
SUPERVISOR_LOG=${SESSION_DIR}/supervisor.log
LOCK=/global/homes/d/dkololgi/.p11_jepa_science_supervisor.lock

mkdir -p "${SESSION_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another P11 JEPA science supervisor is already active" >&2
  exit 2
fi

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

fail() {
  log "supervisor_fail reason=$1"
  exit "${2:-1}"
}

if [[ -z "${PY}" || ! -x "${PY}" || "${PY}" != /global/cfs/* ]]; then
  fail invalid_cfs_recovery_interpreter 2
fi
if [[ ! "${MAX_ALLOCATION_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  fail invalid_max_allocation_attempts 2
fi
if [[ "${P11_ARM:-jepa}" != "jepa" ]]; then
  fail automatic_control_launch_refused 2
fi

export P11_PYTHON="${PY}"
export P11_RUN_DIR="${RUN_DIR}"
export P11_CONTRACT="${CONTRACT}"
export P11_ARM=jepa
export P11_RUN_NAME="${RUN_NAME}"
export P11_STOP_AFTER_UPDATES=

log "supervisor_start session=${SESSION_ID} pid=$$ interpreter=${PY}"
log "registered_control_plan=supervised_masked,masked_reconstruction,response_only auto_launch=false"

# This is intentionally checked before the first allocation request.  A missing,
# stale, or failed technical canary is a terminal error, not a reason to spend a
# GPU allocation trying to repair scientific state.
if ! "${PY}" -u "${GUARD}" --mode preallocation --run-dir "${RUN_DIR}" \
  --contract "${CONTRACT}" >> "${SUPERVISOR_LOG}" 2>&1; then
  fail passing_canary_marker_required 3
fi

if [[ -f "${RUN_DIR}/P11_MATCHED_ARM_COMPLETE.json" ]]; then
  if "${PY}" -u "${GUARD}" --mode terminal --run-dir "${RUN_DIR}" \
    --contract "${CONTRACT}" >> "${SUPERVISOR_LOG}" 2>&1; then
    log scientific_jepa_already_complete
    exit 0
  fi
  fail existing_completion_marker_invalid 4
fi

attempt=0
while [[ ! -f "${RUN_DIR}/P11_MATCHED_ARM_COMPLETE.json" ]]; do
  # Recheck marker, exact Git revision, contract hash and source/data inventories
  # every time: no allocation can be requested after a stale/manual replacement.
  if ! "${PY}" -u "${GUARD}" --mode preallocation --run-dir "${RUN_DIR}" \
    --contract "${CONTRACT}" >> "${SUPERVISOR_LOG}" 2>&1; then
    fail canary_marker_changed 5
  fi

  allocation_snapshot=${SESSION_DIR}/allocation_status_$(printf '%03d' $((attempt + 1))).json
  set +e
  "${ALLOCATION_STATUS}" --max-interactive 2 > "${allocation_snapshot}" 2>> "${SUPERVISOR_LOG}"
  allocation_status_code=$?
  set -e
  if [[ ${allocation_status_code} -eq 2 ]]; then
    log "allocation_capacity_full snapshot=${allocation_snapshot}; waiting=60s"
    sleep 60
    continue
  fi
  if [[ ${allocation_status_code} -ne 0 ]]; then
    fail allocation_status_query_failed 6
  fi

  attempt=$((attempt + 1))
  if (( attempt > MAX_ALLOCATION_ATTEMPTS )); then
    fail allocation_attempt_limit_reached 7
  fi
  tag=$(printf '%03d' "${attempt}")
  status_file=${SESSION_DIR}/attempt_${tag}_worker_status.json
  started_file=${SESSION_DIR}/attempt_${tag}_worker_started.txt
  allocation_log=${SESSION_DIR}/attempt_${tag}_allocation.log
  export P11_SUPERVISOR_STATUS_FILE="${status_file}"
  export P11_SUPERVISOR_STARTED_FILE="${started_file}"

  log "allocation_request attempt=${attempt} account=desi_g gpu=1 constraint=gpu_and_hbm80g time=04:00:00"
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --constraint="gpu&hbm80g" --gpus=1 --qos=interactive \
    --time=04:00:00 --account=desi_g --immediate=600 \
    --job-name=p11_jepa_sci \
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
      --cpu-bind=cores --export=ALL bash "${WORKER}" \
      2>&1 | tee "${allocation_log}"
  pipeline_codes=("${PIPESTATUS[@]}")
  allocation_code=${pipeline_codes[0]}
  tee_code=${pipeline_codes[1]}
  set -e
  log "allocation_exit attempt=${attempt} code=${allocation_code} log=${allocation_log}"
  if [[ ${tee_code} -ne 0 ]]; then
    fail allocation_log_write_failed 12
  fi

  if [[ ! -f "${started_file}" ]]; then
    # salloc did not start the worker (for example --immediate timed out).  No
    # scientific state was touched, so another request is safe and does not
    # count as a trainer resume.
    log "allocation_not_started attempt=${attempt}; waiting=60s"
    sleep 60
    continue
  fi
  if [[ ! -f "${status_file}" ]]; then
    fail started_worker_has_no_atomic_status 8
  fi

  worker_code=$("${PY}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["worker_exit_code"]))' \
    "${status_file}") || fail malformed_worker_status 9
  checkpoint_valid=$("${PY}" -c \
    'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["checkpoint_valid"])).lower())' \
    "${status_file}") || fail malformed_worker_checkpoint_status 9
  complete_valid=$("${PY}" -c \
    'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["complete_valid"])).lower())' \
    "${status_file}") || fail malformed_worker_completion_status 9

  if [[ ${allocation_code} -eq 75 && ${worker_code} -eq 75 && "${checkpoint_valid}" == true ]]; then
    log "valid_checkpoint_pause attempt=${attempt}; chaining_next_allocation_after=30s"
    sleep 30
    continue
  fi
  if [[ ${allocation_code} -eq 0 && ${worker_code} -eq 0 && "${complete_valid}" == true ]]; then
    if "${PY}" -u "${GUARD}" --mode terminal --run-dir "${RUN_DIR}" \
      --contract "${CONTRACT}" >> "${SUPERVISOR_LOG}" 2>&1; then
      log "scientific_jepa_complete attempt=${attempt}"
      break
    fi
    fail final_completion_guard_failed 10
  fi

  # In particular, code 0 plus a checkpoint is never treated as resumable.  The
  # only continuation signal is code 75 plus an independently reloaded and
  # validated checkpoint produced inside the just-finished allocation.
  fail "unexpected_worker_exit allocation_code=${allocation_code} worker_code=${worker_code} checkpoint_valid=${checkpoint_valid} complete_valid=${complete_valid}" 11
done

log "supervisor_complete session=${SESSION_ID}; controls_require_explicit_review"
