#!/usr/bin/env bash
# Resume R1 on the otherwise idle fourth GPU of the legacy p10mtp12 allocation.
#
# This sidecar never requests an allocation.  It attaches one exclusive one-GPU
# step only to an existing running p10mtp12 job, waits if all resources are busy,
# and retries after allocation rollover.  Once the integrated p3brp12 supervisor
# replaces the legacy job name, this script remains idle rather than duplicating
# its R1/P12 work.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
TRAIN=${REPO}/workflows/abacus_tweb/p10_train_random_response.py
CONTRACT=${ROOT}/training_contract_r1_random
RUN_ROOT=${ROOT}/response_training
RUN_DIR=${RUN_ROOT}/p3br_r1_v1/unet/seed_42
COMPLETE=${RUN_DIR}/ARM_A_TRAINING_COMPLETE.json
LOG_ROOT=${ROOT}/p3br_r1_p12_logs
LOG=${LOG_ROOT}/r1_existing_gpu_sidecar.log
mkdir -p "${LOG_ROOT}"

echo "$(date -u +%FT%TZ) sidecar_start pid=$$" >> "${LOG}"
while [[ ! -f "${COMPLETE}" ]]; do
  # Use only the legacy allocation.  The production p3brp12 supervisor already
  # owns and schedules all four GPUs itself.
  job_id=$(squeue -h -u "${USER}" -n p10mtp12 -t R -o '%A' | head -n 1)
  if [[ -z "${job_id}" ]]; then
    sleep 30
    continue
  fi
  echo "$(date -u +%FT%TZ) attach job=${job_id}" >> "${LOG}"
  set +e
  srun --jobid="${job_id}" --exclusive --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --gpus=1 --cpu-bind=cores --export=ALL \
    /bin/bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export XLA_PYTHON_CLIENT_ALLOCATOR=platform
      cd '${REPO}'
      '${PY}' '${TRAIN}' --model unet --contract-root '${CONTRACT}' \
        --seed 42 --epochs 20 --min-epochs 10 --disable-early-stopping --lr 0.002 \
        --loss-log-every 25 --checkpoint-every 250 --max-runtime-seconds 12600 \
        --validation-reserve-seconds 1200 --run-name p3br_r1_v1 \
        --output-root '${RUN_ROOT}' --auto-resume
    " >> "${LOG_ROOT}/r1.log" 2>&1
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) detach job=${job_id} code=${code}" >> "${LOG}"
  [[ -f "${COMPLETE}" ]] && break
  sleep 15
done
echo "$(date -u +%FT%TZ) sidecar_complete" >> "${LOG}"
