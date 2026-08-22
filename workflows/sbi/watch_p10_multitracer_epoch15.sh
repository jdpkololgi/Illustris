#!/usr/bin/env bash
# Lightweight login-node guard for a supervisor that was launched before the
# epoch-15 stop contract existed. It performs no scientific compute.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
RUN_ROOT=${ROOT}/p12_and_multitracer_training
FREEZE=${REPO}/workflows/abacus_tweb/p10_freeze_multitracer_epoch15.py
LOG=${ROOT}/p12_and_multitracer_logs/epoch15_guard.log

history_for() {
  local view=$1
  echo "${RUN_ROOT}/p10_bf_${view}_v1/unet_multitracer/seed_42/epoch_history.jsonl"
}

marker_for() {
  local view=$1
  echo "${RUN_ROOT}/p10_bf_${view}_v1/unet_multitracer/seed_42/EPOCH15_FROZEN.json"
}

while true; do
  newly_complete=()
  for view in proxy null; do
    history=$(history_for "${view}")
    marker=$(marker_for "${view}")
    if [[ ! -f "${marker}" && -f "${history}" ]] && grep -q '"epoch": 15' "${history}"; then
      newly_complete+=("${view}")
    fi
  done
  if [[ ${#newly_complete[@]} -gt 0 ]]; then
    echo "$(date -u +%FT%TZ) epoch15_detected views=${newly_complete[*]}" >> "${LOG}"
    # The trainer writes history immediately before atomic best/cursor checkpoints.
    sleep 30
    mapfile -t jobs < <(squeue -h -n p10mtp12 -o '%A' | sort -u)
    if [[ ${#jobs[@]} -gt 0 ]]; then
      scancel "${jobs[@]}"
      echo "$(date -u +%FT%TZ) cancelled_jobs=${jobs[*]}" >> "${LOG}"
      sleep 45
    fi
    for view in "${newly_complete[@]}"; do
      "${PY}" "${FREEZE}" --run-root "${RUN_ROOT}" --view "${view}" >> "${LOG}" 2>&1
    done
  fi
  if [[ -f "$(marker_for proxy)" && -f "$(marker_for null)" ]]; then
    "${PY}" "${FREEZE}" --run-root "${RUN_ROOT}" >> "${LOG}" 2>&1
    echo "$(date -u +%FT%TZ) guard_complete" >> "${LOG}"
    exit 0
  fi
  sleep 30
done
