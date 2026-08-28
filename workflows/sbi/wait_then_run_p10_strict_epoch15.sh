#!/usr/bin/env bash
# Wait for the recovered epoch-10 strict-control allocation to release, then
# resume the same frozen checkpoints to their registered epoch-15 terminal
# markers without exceeding the user's two submitted interactive-job limit.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
RUNNER=${REPO}/workflows/sbi/run_p10_strict_multitracer_controls_interactive.sh
TRAIN_ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/strict_control_training

gate10_ready() {
  local run_name seed history
  for run_name in p10_r3_rf_dm_seed1701_v1 p10_bf_xphase_forward_v1; do
    for seed in 42 43; do
      history=${TRAIN_ROOT}/${run_name}/unet_multitracer/seed_${seed}/epoch_history.jsonl
      [[ -f "${history}" ]] || return 1
      (( $(wc -l < "${history}") >= 10 )) || return 1
    done
  done
}


submitted_job_count() {
  squeue -h -u "${USER}" -o '%A' 2>/dev/null | sort -u | wc -l
}

strict_job_present() {
  [[ -n "$(squeue -h -u "${USER}" -n p10strict -o '%A' 2>/dev/null)" ]]
}

while ! gate10_ready || strict_job_present || (( $(submitted_job_count) >= 2 )); do
  sleep 60
done

cd "${REPO}"
export P10_STRICT_GATE_EPOCH=0
exec bash "${RUNNER}"
