#!/usr/bin/env bash
# Wait for the recovered epoch-10 strict-control allocation to release, then
# resume the same frozen checkpoints to their registered epoch-15 terminal
# markers without exceeding the user's two submitted interactive-job limit.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
RUNNER=${REPO}/workflows/sbi/run_p10_strict_multitracer_controls_interactive.sh

submitted_job_count() {
  squeue -h -u "${USER}" -o '%A' 2>/dev/null | sort -u | wc -l
}

strict_job_present() {
  [[ -n "$(squeue -h -u "${USER}" -n p10strict -o '%A' 2>/dev/null)" ]]
}

while strict_job_present || (( $(submitted_job_count) >= 2 )); do
  sleep 60
done

cd "${REPO}"
export P10_STRICT_GATE_EPOCH=0
exec bash "${RUNNER}"
