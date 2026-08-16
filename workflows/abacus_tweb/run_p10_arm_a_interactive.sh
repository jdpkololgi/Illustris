#!/usr/bin/env bash
# Persistent two-hour interactive-allocation supervisor for one P10 Arm-A model.
# Run one copy for U-PATCH and one for G-PATCH; never start a third allocation.
set -uo pipefail

MODEL="${1:-}"
if [[ "${MODEL}" != "unet" && "${MODEL}" != "graph" ]]; then
  echo "usage: $0 {unet|graph}" >&2
  exit 2
fi

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
TRAINER=${REPO}/workflows/abacus_tweb/p10_train_arm_a.py
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training
RUN_NAME=arm_a_r0_v1
CANARY_NAME=arm_a_gpu_canary_v2
RUN_DIR=${ROOT}/${RUN_NAME}/${MODEL}/seed_42
CANARY_DIR=${ROOT}/${CANARY_NAME}/${MODEL}/seed_42
LOG_DIR=${ROOT}/launcher_logs
SUPERVISOR_LOG=${LOG_DIR}/${MODEL}_supervisor.log
SCIENTIFIC_LOG=${LOG_DIR}/${MODEL}_scientific.log
CANARY_LOG=${LOG_DIR}/${MODEL}_canary.log
VALIDATION_GROUP_CORES=8
GPU_CONSTRAINT=gpu
if [[ "${MODEL}" == "graph" ]]; then
  # The ph000 G-PATCH peak was 30.8 GiB with groups of eight.  Groups of four
  # retain exact row coverage.  Multi-phase training nevertheless encountered
  # a larger individual training patch that exceeded a 40-GiB A100, so resume
  # the unchanged scientific run on an 80-GiB A100.
  VALIDATION_GROUP_CORES=4
  GPU_CONSTRAINT='gpu&hbm80g'
fi
mkdir -p "${LOG_DIR}"

echo "$(date -u +%FT%TZ) supervisor_start model=${MODEL} pid=$$" >> "${SUPERVISOR_LOG}"
attempt=0
while [[ ! -f "${RUN_DIR}/ARM_A_TRAINING_COMPLETE.json" ]]; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request model=${MODEL} attempt=${attempt}" >> "${SUPERVISOR_LOG}"
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --constraint="${GPU_CONSTRAINT}" --gpus=1 --qos=interactive \
    --time=02:00:00 --account=desi_g --immediate=600 \
    --job-name="p10A_${MODEL}" \
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
      --cpu-bind=cores --export=ALL bash -lc "
        unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
        export PYTHONNOUSERSITE=1
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform
        cd '${REPO}'
        if [[ ! -f '${CANARY_DIR}/TECHNICAL_CANARY_COMPLETE.json' ]]; then
          '${PY}' '${TRAINER}' \
            --model '${MODEL}' --seed 42 \
            --epochs 1 --min-epochs 1 --disable-early-stopping \
            --stop-after-updates 2 \
            --run-name '${CANARY_NAME}' \
            --output-root '${ROOT}' >> '${CANARY_LOG}' 2>&1 || exit \$?
        fi
        '${PY}' '${TRAINER}' \
          --model '${MODEL}' --seed 42 \
          --epochs 20 --min-epochs 10 --patience 5 --min-delta 0.002 \
          --disable-early-stopping --lr 0.002 \
          --validation-group-cores '${VALIDATION_GROUP_CORES}' \
          --loss-log-every 25 --checkpoint-every 250 \
          --max-runtime-seconds 6600 --validation-reserve-seconds 1200 \
          --run-name '${RUN_NAME}' --output-root '${ROOT}' \
          --auto-resume >> '${SCIENTIFIC_LOG}' 2>&1
      "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit model=${MODEL} attempt=${attempt} code=${code}" >> "${SUPERVISOR_LOG}"
  if [[ -f "${RUN_DIR}/ARM_A_TRAINING_COMPLETE.json" ]]; then
    break
  fi
  if [[ ${code} -eq 75 ]]; then
    sleep 30
    continue
  fi
  if [[ ${code} -eq 0 && -f "${RUN_DIR}/arm_a_checkpoint.pt" ]]; then
    sleep 30
    continue
  fi
  # Interactive nodes can be temporarily unavailable. Retry a bounded number
  # of allocation failures, but never hide a trainer failure after a checkpoint.
  if [[ ${code} -ne 0 && ! -f "${RUN_DIR}/arm_a_checkpoint.pt" && ${attempt} -lt 6 ]]; then
    sleep 60
    continue
  fi
  echo "$(date -u +%FT%TZ) supervisor_stop_unexpected model=${MODEL} code=${code}" >> "${SUPERVISOR_LOG}"
  exit "${code}"
done

echo "$(date -u +%FT%TZ) supervisor_complete model=${MODEL}" >> "${SUPERVISOR_LOG}"
