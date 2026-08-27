#!/usr/bin/env bash
# Persistent interactive supervisor for strict FAINT/random controls.
#
# Stage 1 uses one CPU allocation to build input-only visible-phase products.
# Stage 2 uses one four-GPU allocation at a time for four independent frozen
# U-PATCH trajectories.  Never submit this script with sbatch.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
PRODUCT_ROOT=${ROOT}/multitracer/strict_controls
TRAIN_ROOT=${ROOT}/strict_control_training
LOG_ROOT=${ROOT}/strict_control_logs
BUILDER=${REPO}/workflows/abacus_tweb/p10_build_strict_multitracer_controls.py
VALIDATOR=${REPO}/workflows/abacus_tweb/p10_validate_strict_multitracer_controls.py
TRAINER=${REPO}/workflows/abacus_tweb/p10_train_arm_a.py
SCRIPT=${REPO}/workflows/sbi/run_p10_strict_multitracer_controls_interactive.sh
PHASES=(ph000 ph002 ph003 ph004 ph005 ph006)
# Optional external science gate.  This does not change trainer arguments,
# optimizer state, or the frozen 15-epoch cosine schedule; it only prevents the
# interactive supervisor from requesting another allocation after every worker
# has written the requested complete-epoch validation row.
GATE_EPOCH=${P10_STRICT_GATE_EPOCH:-0}
mkdir -p "${PRODUCT_ROOT}" "${TRAIN_ROOT}" "${LOG_ROOT}"

clean_environment() {
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
}

product_root() {
  case "$1" in
    dm1701) echo "${PRODUCT_ROOT}/r3_rf_dm_seed1701_v1" ;;
    dm2718) echo "${PRODUCT_ROOT}/r3_rf_dm_seed2718_v1" ;;
    xforward) echo "${PRODUCT_ROOT}/bf_xphase_forward_v1" ;;
    xreverse) echo "${PRODUCT_ROOT}/bf_xphase_reverse_v1" ;;
    *) return 2 ;;
  esac
}

build_one() {
  local arm=$1 phase=$2
  case "${arm}" in
    dm1701)
      "${PY}" -u "${BUILDER}" --control r3_rf_dm --seed 1701 --phase "${phase}"
      ;;
    dm2718)
      "${PY}" -u "${BUILDER}" --control r3_rf_dm --seed 2718 --phase "${phase}"
      ;;
    xforward)
      "${PY}" -u "${BUILDER}" --control bf_xphase --derangement forward --phase "${phase}"
      ;;
    xreverse)
      "${PY}" -u "${BUILDER}" --control bf_xphase --derangement reverse --phase "${phase}"
      ;;
    *) return 2 ;;
  esac
}

cpu_worker() {
  local worker=$1
  clean_environment
  cd "${REPO}"
  local tasks=()
  local arm phase
  for arm in dm1701 dm2718 xforward xreverse; do
    for phase in "${PHASES[@]}"; do tasks+=("${arm} ${phase}"); done
  done
  local index spec
  for ((index=worker; index<${#tasks[@]}; index+=4)); do
    spec=${tasks[index]}
    arm=${spec% *}
    phase=${spec#* }
    echo "$(date -u +%FT%TZ) build_start arm=${arm} phase=${phase}"
    build_one "${arm}" "${phase}"
    echo "$(date -u +%FT%TZ) build_complete arm=${arm} phase=${phase}"
  done
}

finalize_products() {
  clean_environment
  cd "${REPO}"
  "${PY}" -u "${BUILDER}" --control r3_rf_dm --seed 1701 --finalize
  "${PY}" -u "${BUILDER}" --control r3_rf_dm --seed 2718 --finalize
  "${PY}" -u "${BUILDER}" --control bf_xphase --derangement forward --finalize
  "${PY}" -u "${BUILDER}" --control bf_xphase --derangement reverse --finalize
  local arm root
  for arm in dm1701 dm2718 xforward xreverse; do
    root=$(product_root "${arm}")
    "${PY}" -u "${VALIDATOR}" --root "${root}"
  done
}

cpu_terminal() {
  local arm root
  for arm in dm1701 dm2718 xforward xreverse; do
    root=$(product_root "${arm}")
    [[ -f "${root}/STRICT_CONTROL_LOADER_SMOKE.json" ]] || return 1
  done
}

run_gpu_worker() {
  local arm=$1 seed=$2 root run_name canary
  case "${arm}" in
    dm)
      root=$(product_root dm1701)
      run_name=p10_r3_rf_dm_seed1701_v1
      ;;
    xphase)
      root=$(product_root xforward)
      run_name=p10_bf_xphase_forward_v1
      ;;
    *) return 2 ;;
  esac
  clean_environment
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  cd "${REPO}"
  local complete=${TRAIN_ROOT}/${run_name}/unet_multitracer/seed_${seed}/ARM_A_TRAINING_COMPLETE.json
  [[ -f "${complete}" ]] && return 0
  if (( GATE_EPOCH > 0 )); then
    local history=${TRAIN_ROOT}/${run_name}/unet_multitracer/seed_${seed}/epoch_history.jsonl
    if [[ -f "${history}" ]] && (( $(wc -l < "${history}") >= GATE_EPOCH )); then
      return 0
    fi
  fi

  canary=${run_name}_canary1000
  local canary_marker=${TRAIN_ROOT}/${canary}/unet_multitracer/seed_${seed}/TECHNICAL_CANARY_COMPLETE.json
  if [[ ! -f "${canary_marker}" ]]; then
    "${PY}" -u "${TRAINER}" \
      --model unet_multitracer --multitracer-view proxy --multitracer-root "${root}" \
      --seed "${seed}" --epochs 15 --min-epochs 10 --disable-early-stopping \
      --lr 0.002 --loss-log-every 25 --checkpoint-every 250 \
      --stop-after-updates 1000 --run-name "${canary}" \
      --output-root "${TRAIN_ROOT}" --auto-resume || return $?
  fi
  "${PY}" -u "${TRAINER}" \
    --model unet_multitracer --multitracer-view proxy --multitracer-root "${root}" \
    --seed "${seed}" --epochs 15 --min-epochs 10 --disable-early-stopping \
    --lr 0.002 --loss-log-every 25 --checkpoint-every 250 \
    --max-runtime-seconds 12600 --validation-reserve-seconds 1200 \
    --run-name "${run_name}" --output-root "${TRAIN_ROOT}" --auto-resume
}

gpu_terminal() {
  local arm run_name seed
  for arm in dm xphase; do
    [[ "${arm}" == dm ]] && run_name=p10_r3_rf_dm_seed1701_v1 || run_name=p10_bf_xphase_forward_v1
    for seed in 42 43; do
      local run_dir=${TRAIN_ROOT}/${run_name}/unet_multitracer/seed_${seed}
      if [[ -f "${run_dir}/ARM_A_TRAINING_COMPLETE.json" ]]; then
        continue
      fi
      if (( GATE_EPOCH > 0 )) && [[ -f "${run_dir}/epoch_history.jsonl" ]] &&
        (( $(wc -l < "${run_dir}/epoch_history.jsonl") >= GATE_EPOCH )); then
        continue
      fi
      return 1
    done
  done
}

case "${1:-}" in
  --cpu-worker) cpu_worker "$2"; exit $? ;;
  --finalize-products) finalize_products; exit $? ;;
  --gpu-worker) run_gpu_worker "$2" "$3"; exit $? ;;
esac

SUPERVISOR_LOG=${LOG_ROOT}/supervisor.log
echo "$(date -u +%FT%TZ) supervisor_start pid=$$ host=$(hostname)" >> "${SUPERVISOR_LOG}"
echo "$(date -u +%FT%TZ) supervisor_gate_epoch=${GATE_EPOCH}" >> "${SUPERVISOR_LOG}"

attempt=0
while ! cpu_terminal; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 8 ]]; then
    echo "$(date -u +%FT%TZ) cpu_retry_exhausted" >> "${SUPERVISOR_LOG}"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) cpu_allocation_request attempt=${attempt}" >> "${SUPERVISOR_LOG}"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=16 --constraint=cpu \
    --qos=interactive --time=04:00:00 --account=desi --immediate=600 \
    --job-name=p10strictcpu bash -lc "
      code=0
      pids=()
      for worker in 0 1 2 3; do
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
          --export=ALL '${SCRIPT}' --cpu-worker \${worker} \
          >> '${LOG_ROOT}/cpu_worker_'\${worker}'.log' 2>&1 &
        pids+=(\$!)
      done
      for pid in \${pids[@]}; do wait \${pid} || code=\$?; done
      if [[ \${code} -eq 0 ]]; then
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
          --export=ALL '${SCRIPT}' --finalize-products \
          >> '${LOG_ROOT}/cpu_finalize.log' 2>&1 || code=\$?
      fi
      exit \${code}
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) cpu_allocation_exit attempt=${attempt} code=${code}" >> "${SUPERVISOR_LOG}"
  cpu_terminal && break
  sleep 30
done

attempt=0
while ! gpu_terminal; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 36 ]]; then
    echo "$(date -u +%FT%TZ) gpu_retry_exhausted" >> "${SUPERVISOR_LOG}"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) gpu_allocation_request attempt=${attempt}" >> "${SUPERVISOR_LOG}"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=32 --constraint='gpu&hbm80g' \
    --gpus=4 --qos=interactive --time=04:00:00 --account=desi_g \
    --immediate=600 --job-name=p10strict bash -lc "
      code=0
      pids=()
      for spec in 'dm 42' 'dm 43' 'xphase 42' 'xphase 43'; do
        arm=\${spec% *}
        seed=\${spec#* }
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
          --cpu-bind=cores --export=ALL '${SCRIPT}' --gpu-worker \${arm} \${seed} \
          >> '${LOG_ROOT}/gpu_'\${arm}'_seed_'\${seed}'.log' 2>&1 &
        pids+=(\$!)
      done
      for pid in \${pids[@]}; do wait \${pid} || code=\$?; done
      [[ \${code} -eq 0 || \${code} -eq 75 ]]
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) gpu_allocation_exit attempt=${attempt} code=${code}" >> "${SUPERVISOR_LOG}"
  gpu_terminal && break
  sleep 30
done
echo "$(date -u +%FT%TZ) supervisor_complete" >> "${SUPERVISOR_LOG}"
