#!/usr/bin/env bash
# Persistent login-node tmux supervisor for the P10 R2/R3-RF response ladder.
# Requests one four-GPU interactive allocation at a time and resumes four independent
# one-GPU tasks. Never submit with sbatch.
set -uo pipefail

REPO=/global/u2/d/dkololgi/TNG/Illustris
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
SCRIPT=${REPO}/workflows/sbi/run_p10_response_ladder_interactive.sh
OUT=${ROOT}/response_training
LOG_ROOT=${OUT}/logs/p10_response_ladder_supervisor
R2_CONTRACT=${ROOT}/training_contract_r2_assignment
R3_CONTRACT=${ROOT}/training_contract_r3_random_field
mkdir -p "${LOG_ROOT}"

run_worker() {
  local arm=$1
  local seed=$2
  local trainer contract run_name
  case "${arm}" in
    r2)
      trainer=${REPO}/workflows/abacus_tweb/p10_train_assignment_response.py
      contract=${R2_CONTRACT}
      run_name=p10_r2_assignment_v1
      ;;
    r3rf)
      trainer=${REPO}/workflows/abacus_tweb/p10_train_r3_random_field.py
      contract=${R3_CONTRACT}
      run_name=p10_r3_rf_v1
      ;;
    *)
      echo "unknown arm: ${arm}" >&2
      return 2
      ;;
  esac
  local run_dir=${OUT}/${run_name}/unet/seed_${seed}
  if [[ -f "${run_dir}/ARM_A_TRAINING_COMPLETE.json" ]]; then
    echo "$(date -u +%FT%TZ) ${arm} seed=${seed} already_complete"
    return 0
  fi
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
  export PYTHONNOUSERSITE=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  cd "${REPO}"
  exec "${PY}" -u "${trainer}" \
    --model unet \
    --contract-root "${contract}" \
    --seed "${seed}" \
    --epochs 20 \
    --min-epochs 10 \
    --disable-early-stopping \
    --lr 0.002 \
    --loss-log-every 25 \
    --checkpoint-every 250 \
    --max-runtime-seconds 12600 \
    --validation-reserve-seconds 1200 \
    --run-name "${run_name}" \
    --output-root "${OUT}" \
    --auto-resume
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker "$2" "$3"
  exit $?
fi

all_terminal() {
  local arm seed run_name
  for arm in r2 r3rf; do
    [[ "${arm}" == r2 ]] && run_name=p10_r2_assignment_v1 || run_name=p10_r3_rf_v1
    for seed in 42 43; do
      [[ -f "${OUT}/${run_name}/unet/seed_${seed}/ARM_A_TRAINING_COMPLETE.json" ]] ||
        return 1
    done
  done
}

supervisor_log=${LOG_ROOT}/supervisor.log
echo "$(date -u +%FT%TZ) supervisor_start pid=$$ host=$(hostname)" >> "${supervisor_log}"
attempt=0
while ! all_terminal; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 30 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted attempts=$((attempt - 1))" >> "${supervisor_log}"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${supervisor_log}"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=32 --constraint='gpu&hbm80g' \
    --gpus=4 --qos=interactive --time=04:00:00 --account=desi_g \
    --immediate=600 --job-name=p10resp bash -lc "
      set -uo pipefail
      code=0
      for spec in 'r2 42' 'r3rf 42' 'r2 43' 'r3rf 43'; do
        arm=\${spec% *}
        seed=\${spec#* }
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
          --cpu-bind=cores --export=ALL '${SCRIPT}' --worker \${arm} \${seed} \
          >> '${LOG_ROOT}/'\${arm}'_seed_'\${seed}'.log' 2>&1 &
        pids+=(\$!)
      done
      for pid in \${pids[@]}; do
        wait \${pid} || code=\$?
      done
      exit \${code}
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${supervisor_log}"
  all_terminal && break
  sleep 30
done
echo "$(date -u +%FT%TZ) supervisor_complete attempts=${attempt}" >> "${supervisor_log}"

