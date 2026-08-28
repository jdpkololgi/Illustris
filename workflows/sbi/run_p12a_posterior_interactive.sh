#!/usr/bin/env bash
# Persistent P12-A posterior supervisor.  It deliberately does not request an
# allocation until all OOF summaries exist and fewer than two user allocations
# are submitted.  Heavy preparation, canary fitting, and the full FMPE fit all
# run on an interactive GPU node, never on a login node.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
SUMMARY_ROOT=${ROOT}/p12_oof_summaries
OUTPUT_ROOT=${ROOT}/p12a_base_response_v1
CANARY_DATA=${ROOT}/p12a_base_response_canary_v1
CANARY_OUTPUT=${CANARY_DATA}/fmpe_seed42
FULL_OUTPUT=${OUTPUT_ROOT}/fmpe_seed42
LOG_ROOT=${ROOT}/p12a_logs
PREP=${REPO}/workflows/sbi/p12_prepare_base_response_dataset.py
TRAIN=${REPO}/workflows/sbi/p12_train_base_response_fmpe.py
mkdir -p "${LOG_ROOT}"

# A reconnect must not create two allocation watchers for the same posterior.
# Keep the lock descriptor open for the lifetime of this supervisor; flock
# releases it automatically if the shell exits or its tmux session is killed.
LOCK_PATH=${LOG_ROOT}/supervisor.lock
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) duplicate_supervisor_refused pid=$$" >> "${LOG_ROOT}/supervisor.log"
  exit 0
fi

all_summaries_ready() {
  local phase
  for phase in ph000 ph002 ph003 ph004 ph005 ph006; do
    [[ -f "${SUMMARY_ROOT}/${phase}/OOF_SUMMARY_COMPLETE.json" ]] || return 1
  done
}

allocation_slot_available() {
  local count
  count=$(squeue -h -u "${USER}" -o '%A' 2>/dev/null | sort -u | wc -l) || return 1
  [[ ${count} -lt 2 ]]
}

echo "$(date -u +%FT%TZ) p12a_supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
while ! all_summaries_ready; do
  echo "$(date -u +%FT%TZ) waiting_for_oof_summaries" >> "${LOG_ROOT}/supervisor.log"
  sleep 120
done

# Let the crossfit allocation release before enforcing the two-allocation gate.
sleep 60
while ! allocation_slot_available; do
  echo "$(date -u +%FT%TZ) waiting_for_allocation_slot" >> "${LOG_ROOT}/supervisor.log"
  sleep 60
done

attempt=0
while [[ ! -f "${FULL_OUTPUT}/P12A_COMPLETE.json" ]]; do
  while ! allocation_slot_available; do sleep 60; done
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --constraint='gpu&hbm80g' --gpus=1 --qos=interactive \
    --time=04:00:00 --account=desi_g --immediate=600 \
    --job-name=p12afmpe bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      cd '${REPO}'
      srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
        --gpu-bind=single:1 --cpu-bind=cores --export=ALL bash -lc '
          set -euo pipefail
          "${PY}" "${PREP}" \
            --output-root "${CANARY_DATA}" --train-rows 50000 --validation-rows 20000
          "${PY}" "${TRAIN}" \
            --dataset-root "${CANARY_DATA}" --output-root "${CANARY_OUTPUT}" \
            --batch-size 2048 --hidden-features 64 --num-layers 3 \
            --stop-after-epochs 1 --max-epochs 2 --n-posterior-samples 64 \
            --calibration-rows 500 --evaluation-rows 1000 --score-rows 100 \
            --sample-chunk 256
          "${PY}" "${PREP}" --output-root "${OUTPUT_ROOT}"
          "${PY}" "${TRAIN}" \
            --dataset-root "${OUTPUT_ROOT}" --output-root "${FULL_OUTPUT}"
        ' >> '${LOG_ROOT}/posterior.log' 2>&1
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/supervisor.log"
  [[ -f "${FULL_OUTPUT}/P12A_COMPLETE.json" ]] && break
  if [[ ${attempt} -ge 8 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/supervisor.log"
    exit 1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) p12a_supervisor_complete" >> "${LOG_ROOT}/supervisor.log"
