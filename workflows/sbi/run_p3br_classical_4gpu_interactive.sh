#!/usr/bin/env bash
# Matched random-response CIC/DTFE evaluation on four independent one-GPU workers.
# Run in tmux only after P3b-R products are frozen.  This uses the existing exact
# DTFE density rasters and changes only the P3b-R M/mu response contract.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
CLASSICAL=${REPO}/workflows/abacus_tweb/p10_classical_fullcap.py
CONTRACT=${ROOT}/training_contract_r1_random
DTFE_BUILD=${ROOT}/classical/dtfe_build_v1
CIC_ROOT=${ROOT}/classical/cic_random_response_v1
DTFE_ROOT=${ROOT}/classical/dtfe_random_response_v1
LOG_ROOT=${ROOT}/p3br_classical_logs
mkdir -p "${LOG_ROOT}"

run_raw() {
  local estimator=$1
  shift
  local output phase
  output="${CIC_ROOT}"
  [[ "${estimator}" == dtfe ]] && output="${DTFE_ROOT}"
  for phase in "$@"; do
    "${PY}" "${CLASSICAL}" raw --phase "${phase}" --estimator "${estimator}" \
      --contract-root "${CONTRACT}" --output-root "${output}" \
      --dtfe-build-root "${DTFE_BUILD}"
  done
}

export PY CLASSICAL CONTRACT DTFE_BUILD CIC_ROOT DTFE_ROOT
export -f run_raw

all_terminal() {
  [[ -f "${CIC_ROOT}/P10_CIC_PH006_COMPLETE.json" ]] &&
  [[ -f "${DTFE_ROOT}/P10_DTFE_PH006_COMPLETE.json" ]]
}

echo "$(date -u +%FT%TZ) classical_supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
while [[ ! -f "${ROOT}/training_contract/P3BR_PIPELINE_COMPLETE.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting_for_p3br_contract" >> "${LOG_ROOT}/supervisor.log"
  sleep 60
done

attempt=0
while ! all_terminal; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=16 --constraint='gpu&hbm80g' \
    --gpus=4 --qos=interactive --time=04:00:00 --account=desi_g \
    --immediate=600 --job-name=p3brclass bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export XLA_PYTHON_CLIENT_ALLOCATOR=platform
      cd '${REPO}'
      groups=('ph000 ph005' 'ph002 ph006' 'ph003' 'ph004')
      code=0
      for estimator in cic dtfe; do
        for worker in 0 1 2 3; do
          srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
            --cpu-bind=cores --export=ALL bash -lc \
            "run_raw \${estimator} \${groups[\${worker}]}" \
            >> '${LOG_ROOT}/'\${estimator}'_worker_'\${worker}'.log' 2>&1 &
          pids[\${worker}]=\$!
        done
        for pid in \${pids[@]}; do wait \${pid} || code=\$?; done
        output='${CIC_ROOT}'
        [[ \${estimator} == dtfe ]] && output='${DTFE_ROOT}'
        if [[ \${code} -eq 0 ]]; then
          srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
            --cpu-bind=cores --export=ALL '${PY}' '${CLASSICAL}' finalize \
            --estimator \${estimator} --contract-root '${CONTRACT}' \
            --output-root \${output} --dtfe-build-root '${DTFE_BUILD}' \
            >> '${LOG_ROOT}/'\${estimator}'_finalize.log' 2>&1 || code=\$?
        fi
      done
      [[ \${code} -eq 0 ]]
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/supervisor.log"
  all_terminal && break
  if [[ ${attempt} -ge 12 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/supervisor.log"
    exit 1
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) classical_supervisor_complete" >> "${LOG_ROOT}/supervisor.log"
