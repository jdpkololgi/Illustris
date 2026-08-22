#!/usr/bin/env bash
# Persistent four-GPU interactive supervisor for the production random-response pivot.
#
# GPU 0: exactly 1,000-patch R1 canary, then the capacity-matched R1 U-PATCH.
# GPUs 1-3: outstanding P12 leave-one-phase-out encoders and OOF exports.
#
# This intentionally uses four independent one-GPU tasks.  It does not introduce
# data parallelism into the frozen one-patch-per-update scientific objective.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
R1_CANARY=${REPO}/workflows/abacus_tweb/p3br_run_r1_throughput_canary.py
R1_TRAIN=${REPO}/workflows/abacus_tweb/p10_train_random_response.py
BASE_TRAIN=${REPO}/workflows/abacus_tweb/p10_train_arm_a.py
EXPORT=${REPO}/workflows/sbi/p12_export_unet_summaries.py
CLASSICAL=${REPO}/workflows/abacus_tweb/p10_classical_fullcap.py
R1_CONTRACT=${ROOT}/training_contract_r1_random
R1_RUN_ROOT=${ROOT}/response_training
XFIT_ROOT=${ROOT}/p12_crossfit_contracts
P12_RUN_ROOT=${ROOT}/p12_and_multitracer_training
SUMMARY_ROOT=${ROOT}/p12_oof_summaries
CIC_ROOT=${ROOT}/classical/cic_random_response_v1
DTFE_ROOT=${ROOT}/classical/dtfe_random_response_v1
DTFE_BUILD=${ROOT}/classical/dtfe_build_v1
LOG_ROOT=${ROOT}/p3br_r1_p12_logs
mkdir -p "${R1_RUN_ROOT}" "${SUMMARY_ROOT}" "${LOG_ROOT}"

run_r1() {
  "${PY}" "${R1_CANARY}" \
    --contract-root "${R1_CONTRACT}" --output-root "${R1_RUN_ROOT}"
  "${PY}" "${R1_TRAIN}" --model unet --contract-root "${R1_CONTRACT}" \
    --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
    --loss-log-every 25 --checkpoint-every 250 --max-runtime-seconds 12600 \
    --validation-reserve-seconds 1200 --run-name p3br_r1_v1 \
    --output-root "${R1_RUN_ROOT}" --auto-resume
}

run_xfit() {
  local omitted=$1
  local contract="${XFIT_ROOT}/omit_${omitted}"
  local name="p12_xfit_omit_${omitted}_v1"
  local canary="p12_xfit_omit_${omitted}_canary_v1"
  local canary_marker="${P12_RUN_ROOT}/${canary}/unet/seed_42/TECHNICAL_CANARY_COMPLETE.json"
  local complete="${P12_RUN_ROOT}/${name}/unet/seed_42/ARM_A_TRAINING_COMPLETE.json"
  local checkpoint="${P12_RUN_ROOT}/${name}/unet/seed_42/best_checkpoint.pt"
  local summary="${SUMMARY_ROOT}/${omitted}/OOF_SUMMARY_COMPLETE.json"
  if [[ ! -f "${canary_marker}" ]]; then
    "${PY}" "${BASE_TRAIN}" --model unet --contract-root "${contract}" \
      --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
      --loss-log-every 1 --checkpoint-every 1 --stop-after-updates 2 \
      --run-name "${canary}" --output-root "${P12_RUN_ROOT}" --auto-resume || return $?
  fi
  if [[ ! -f "${complete}" ]]; then
    "${PY}" "${BASE_TRAIN}" --model unet --contract-root "${contract}" \
      --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
      --loss-log-every 25 --checkpoint-every 250 --max-runtime-seconds 12600 \
      --validation-reserve-seconds 1200 --run-name "${name}" \
      --output-root "${P12_RUN_ROOT}" --auto-resume || return $?
  fi
  if [[ ! -f "${summary}" ]]; then
    "${PY}" "${EXPORT}" --contract-root "${contract}" \
      --checkpoint "${checkpoint}" --phase "${omitted}" \
      --output-root "${SUMMARY_ROOT}" --device cuda
  fi
}

export_ph006() {
  local summary="${SUMMARY_ROOT}/ph006/OOF_SUMMARY_COMPLETE.json"
  [[ -f "${summary}" ]] && return 0
  "${PY}" "${EXPORT}" --contract-root "${ROOT}/training_contract" \
    --checkpoint "${ROOT}/arm_a_training/arm_a_r0_v1/unet/seed_42/best_checkpoint.pt" \
    --phase ph006 --output-root "${SUMMARY_ROOT}" --device cuda
}

run_classical_group() {
  local estimator=$1
  shift
  local output="${CIC_ROOT}"
  [[ "${estimator}" == dtfe ]] && output="${DTFE_ROOT}"
  local phase
  for phase in "$@"; do
    "${PY}" "${CLASSICAL}" raw --phase "${phase}" --estimator "${estimator}" \
      --contract-root "${R1_CONTRACT}" --output-root "${output}" \
      --dtfe-build-root "${DTFE_BUILD}"
  done
}

worker1() {
  run_xfit ph003 && run_classical_group cic ph000 ph005 \
    && run_classical_group dtfe ph000 ph005
}
worker2() {
  run_xfit ph004 && run_classical_group cic ph002 ph006 \
    && run_classical_group dtfe ph002 ph006
}
worker3() {
  run_xfit ph005 && export_ph006 && run_classical_group cic ph003 ph004 \
    && run_classical_group dtfe ph003 ph004
}
export REPO PY ROOT R1_CANARY R1_TRAIN BASE_TRAIN EXPORT CLASSICAL R1_CONTRACT R1_RUN_ROOT
export XFIT_ROOT P12_RUN_ROOT SUMMARY_ROOT CIC_ROOT DTFE_ROOT DTFE_BUILD LOG_ROOT
export -f run_r1 run_xfit export_ph006 run_classical_group worker1 worker2 worker3

all_terminal() {
  [[ -f "${R1_RUN_ROOT}/p3br_r1_v1/unet/seed_42/ARM_A_TRAINING_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph000/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph002/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph003/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph004/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph005/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph006/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${CIC_ROOT}/P10_CIC_PH006_COMPLETE.json" ]] &&
  [[ -f "${DTFE_ROOT}/P10_DTFE_PH006_COMPLETE.json" ]]
}

echo "$(date -u +%FT%TZ) supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
while [[ ! -f "${ROOT}/training_contract/P3BR_PIPELINE_COMPLETE.json" || \
         ! -f "${R1_CONTRACT}/TRAINING_LOADER_READY.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting_for_p3br_contract" >> "${LOG_ROOT}/supervisor.log"
  sleep 60
done

attempt=0
while ! all_terminal; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=16 \
    --constraint='gpu&hbm80g' --gpus=4 --qos=interactive --time=04:00:00 \
    --account=desi_g --immediate=600 --job-name=p3brp12 bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export XLA_PYTHON_CLIENT_ALLOCATOR=platform
      cd '${REPO}'
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
        --cpu-bind=cores --export=ALL bash -lc run_r1 \
        >> '${LOG_ROOT}/r1.log' 2>&1 & pids[0]=\$!
      for worker in 1 2 3; do
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL bash -lc "worker\${worker}" \
          >> '${LOG_ROOT}/worker_'\${worker}'.log' 2>&1 & pids[\${worker}]=\$!
      done
      code=0
      for worker in 1 2 3; do wait \${pids[\${worker}]} || code=\$?; done
      if [[ \${code} -eq 0 ]]; then
        for estimator in cic dtfe; do
          output='${CIC_ROOT}'
          [[ \${estimator} == dtfe ]] && output='${DTFE_ROOT}'
          srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
            --cpu-bind=cores --export=ALL '${PY}' '${CLASSICAL}' finalize \
            --estimator \${estimator} --contract-root '${R1_CONTRACT}' \
            --output-root \${output} --dtfe-build-root '${DTFE_BUILD}' \
            >> '${LOG_ROOT}/classical_'\${estimator}'_finalize.log' 2>&1 || code=\$?
        done
      fi
      wait \${pids[0]} || code=\$?
      [[ \${code} -eq 0 || \${code} -eq 75 ]]
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/supervisor.log"
  all_terminal && break
  if [[ ${attempt} -ge 24 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/supervisor.log"
    exit 1
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) supervisor_complete" >> "${LOG_ROOT}/supervisor.log"
