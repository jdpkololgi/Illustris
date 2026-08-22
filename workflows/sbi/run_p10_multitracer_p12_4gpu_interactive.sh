#!/usr/bin/env bash
# Persistent four-GPU supervisor for P10 BF Proxy/Null and P12 U cross-fitting.
# Each GPU runs one scientifically independent one-patch-per-update task. This
# uses all four GPUs without changing the frozen optimizer/batch semantics.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
TRAIN=${REPO}/workflows/abacus_tweb/p10_train_arm_a.py
EXPORT=${REPO}/workflows/sbi/p12_export_unet_summaries.py
MT_ROOT=${ROOT}/multitracer/v1
XFIT_ROOT=${ROOT}/p12_crossfit_contracts
RUN_ROOT=${ROOT}/p12_and_multitracer_training
SUMMARY_ROOT=${ROOT}/p12_oof_summaries
LOG_ROOT=${ROOT}/p12_and_multitracer_logs
mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}" "${LOG_ROOT}"

train_mt() {
  local view=$1
  local name="p10_bf_${view}_v1"
  local frozen="${RUN_ROOT}/${name}/unet_multitracer/seed_42/EPOCH15_FROZEN.json"
  [[ -f "${frozen}" ]] && return 0
  local canary="p10_bf_${view}_canary_v1"
  local canary_marker="${RUN_ROOT}/${canary}/unet_multitracer/seed_42/TECHNICAL_CANARY_COMPLETE.json"
  if [[ ! -f "${canary_marker}" ]]; then
    "${PY}" "${TRAIN}" --model unet_multitracer --multitracer-view "${view}" \
      --multitracer-root "${MT_ROOT}" --seed 42 --epochs 20 --min-epochs 10 \
      --disable-early-stopping --lr 0.002 --loss-log-every 1 \
      --checkpoint-every 1 --stop-after-updates 2 --run-name "${canary}" \
      --output-root "${RUN_ROOT}" --auto-resume || return $?
  fi
  "${PY}" "${TRAIN}" --model unet_multitracer --multitracer-view "${view}" \
    --multitracer-root "${MT_ROOT}" --seed 42 --epochs 20 --min-epochs 10 \
    --disable-early-stopping --lr 0.002 --loss-log-every 25 \
    --checkpoint-every 250 --max-runtime-seconds 12600 \
    --validation-reserve-seconds 1200 --run-name "${name}" \
    --output-root "${RUN_ROOT}" --auto-resume
}

train_xfit() {
  local omitted=$1
  local contract="${XFIT_ROOT}/omit_${omitted}"
  local name="p12_xfit_omit_${omitted}_v1"
  local canary="p12_xfit_omit_${omitted}_canary_v1"
  local canary_marker="${RUN_ROOT}/${canary}/unet/seed_42/TECHNICAL_CANARY_COMPLETE.json"
  local complete="${RUN_ROOT}/${name}/unet/seed_42/ARM_A_TRAINING_COMPLETE.json"
  local checkpoint="${RUN_ROOT}/${name}/unet/seed_42/best_checkpoint.pt"
  local summary="${SUMMARY_ROOT}/${omitted}/OOF_SUMMARY_COMPLETE.json"
  if [[ ! -f "${canary_marker}" ]]; then
    "${PY}" "${TRAIN}" --model unet --contract-root "${contract}" \
      --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
      --loss-log-every 1 --checkpoint-every 1 --stop-after-updates 2 \
      --run-name "${canary}" --output-root "${RUN_ROOT}" --auto-resume || return $?
  fi
  if [[ ! -f "${complete}" ]]; then
    "${PY}" "${TRAIN}" --model unet --contract-root "${contract}" \
      --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
      --loss-log-every 25 --checkpoint-every 250 --max-runtime-seconds 12600 \
      --validation-reserve-seconds 1200 --run-name "${name}" \
      --output-root "${RUN_ROOT}" --auto-resume || return $?
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
  "${PY}" "${EXPORT}" \
    --contract-root "${ROOT}/training_contract" \
    --checkpoint "${ROOT}/arm_a_training/arm_a_r0_v1/unet/seed_42/best_checkpoint.pt" \
    --phase ph006 --output-root "${SUMMARY_ROOT}" --device cuda
}

worker0() { train_mt proxy && train_xfit ph003; }
worker1() { train_mt null && train_xfit ph004; }
worker2() { train_xfit ph000 && train_xfit ph005; }
worker3() { train_xfit ph002 && export_ph006; }
export REPO PY ROOT TRAIN EXPORT MT_ROOT XFIT_ROOT RUN_ROOT SUMMARY_ROOT LOG_ROOT
export -f train_mt train_xfit export_ph006 worker0 worker1 worker2 worker3

all_terminal() {
  [[ -f "${RUN_ROOT}/p10_bf_proxy_v1/unet_multitracer/seed_42/EPOCH15_FROZEN.json" ]] &&
  [[ -f "${RUN_ROOT}/p10_bf_null_v1/unet_multitracer/seed_42/EPOCH15_FROZEN.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph000/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph002/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph003/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph004/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph005/OOF_SUMMARY_COMPLETE.json" ]] &&
  [[ -f "${SUMMARY_ROOT}/ph006/OOF_SUMMARY_COMPLETE.json" ]]
}

echo "$(date -u +%FT%TZ) supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
while [[ ! -f "${MT_ROOT}/P10_MULTITRACER_VIEWS_READY.json" || \
         ! -f "${XFIT_ROOT}/P12_CROSSFIT_CONTRACTS_READY.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting_for_cpu_contracts" >> "${LOG_ROOT}/supervisor.log"
  sleep 60
done

attempt=0
while ! all_terminal; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=16 \
    --constraint='gpu&hbm80g' --gpus=4 --qos=interactive \
    --time=04:00:00 --account=desi_g --immediate=600 \
    --job-name=p10mtp12 bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      cd '${REPO}'
      for worker in 0 1 2 3; do
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL bash -lc \"worker\${worker}\" \
          >> '${LOG_ROOT}/worker_'\${worker}'.log' 2>&1 &
        pids[\${worker}]=\$!
      done
      code=0
      for pid in \${pids[@]}; do wait \${pid} || code=\$?; done
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
