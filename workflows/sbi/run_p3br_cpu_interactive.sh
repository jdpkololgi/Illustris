#!/usr/bin/env bash
# Persistent interactive-CPU supervisor for all visible-phase P3b-R products.
# Run this inside tmux.  It requests only one CPU allocation at a time and the
# Python pipeline resumes exact random-count progress and phase/cap products.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
PIPELINE=${REPO}/workflows/abacus_tweb/run_p3br_pipeline.py
LOG_ROOT=${ROOT}/p3br_logs
MARKER=${ROOT}/training_contract/P3BR_PIPELINE_COMPLETE.json
mkdir -p "${LOG_ROOT}"

echo "$(date -u +%FT%TZ) cpu_supervisor_start pid=$$" >> "${LOG_ROOT}/cpu_supervisor.log"
attempt=0
while [[ ! -f "${MARKER}" ]]; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/cpu_supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu --qos=interactive \
    --time=02:00:00 --account=desi --immediate=600 --job-name=p3brcpu bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      cd '${REPO}'
      srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
        --export=ALL '${PY}' '${PIPELINE}' --map-workers 4 --overlay-workers 3
    " >> "${LOG_ROOT}/cpu_supervisor_attempt_${attempt}.log" 2>&1
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/cpu_supervisor.log"
  [[ -f "${MARKER}" ]] && break
  # NERSC rejects a third interactive request immediately while the user's
  # two allowed allocations are live; it does not necessarily hold the salloc
  # request until one exits.  Keep retrying long enough to bridge an allocation
  # boundary while retaining a finite supervisor contract.
  if [[ ${attempt} -ge 96 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/cpu_supervisor.log"
    exit 1
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) cpu_supervisor_complete" >> "${LOG_ROOT}/cpu_supervisor.log"
