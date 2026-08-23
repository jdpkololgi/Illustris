#!/bin/bash
# Build the visible-phase R2 assignment overlays within one CPU salloc shell.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
SCRIPT=${REPO}/workflows/abacus_tweb/p10_build_r2_assignment_overlays.py
LOG_ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_response_v1/logs
PHASES=(ph000 ph002 ph003 ph004 ph005 ph006)

mkdir -p "${LOG_ROOT}"

run_phase() {
  local phase=$1
  local log=${LOG_ROOT}/${phase}.log
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
    --export=NONE --output="${log}" --error="${log}" \
    /bin/bash -lc "unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD; \
      export PYTHONNOUSERSITE=1; cd '${REPO}'; \
      '${PY}' '${SCRIPT}' component --phase '${phase}' --cap NGC; \
      '${PY}' '${SCRIPT}' component --phase '${phase}' --cap SGC"
}

pids=()
for phase in "${PHASES[@]}"; do
  run_phase "${phase}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

srun --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores --export=NONE \
  /bin/bash -lc "unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD; \
    export PYTHONNOUSERSITE=1; cd '${REPO}'; \
    '${PY}' '${SCRIPT}' aggregate; \
    '${PY}' -m unittest tests.phase4.test_p10_r2_assignment_overlays"
