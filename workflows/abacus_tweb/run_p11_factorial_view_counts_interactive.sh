#!/usr/bin/env bash
# Persistent CPU materialization chain for the P11 factorial count products.
# P12-A is deliberately given priority.  This supervisor requests no allocation
# until P12A_COMPLETE exists and fewer than two user allocations are submitted.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
P12=${ROOT}/p12a_base_response_v1/fmpe_seed42/P12A_COMPLETE.json
OUTPUT=${ROOT}/p11_factorial_views_v1/FACTORIAL_VIEW_PRODUCTS_READY.json
LOG_ROOT=${ROOT}/p11_factorial_views_v1/logs
BUILDER=${REPO}/workflows/abacus_tweb/p11_build_factorial_view_counts.py
mkdir -p "${LOG_ROOT}"

allocation_slot_available() {
  local count
  count=$(squeue -h -u "${USER}" -o '%A' 2>/dev/null | sort -u | wc -l) || return 1
  [[ ${count} -lt 2 ]]
}

echo "$(date -u +%FT%TZ) factorial_supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
while [[ ! -f "${P12}" ]]; do
  echo "$(date -u +%FT%TZ) waiting_for_p12a" >> "${LOG_ROOT}/supervisor.log"
  sleep 180
done

attempt=0
while [[ ! -f "${OUTPUT}" ]]; do
  while ! allocation_slot_available; do sleep 60; done
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu \
    --qos=interactive --time=04:00:00 --account=desi --immediate=600 \
    --job-name=p11views bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      cd '${REPO}'
      srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
        --export=ALL '${PY}' -u '${BUILDER}' \
        >> '${LOG_ROOT}/builder.log' 2>&1
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/supervisor.log"
  [[ -f "${OUTPUT}" ]] && break
  if [[ ${attempt} -ge 8 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/supervisor.log"
    exit 1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) factorial_supervisor_complete" >> "${LOG_ROOT}/supervisor.log"
