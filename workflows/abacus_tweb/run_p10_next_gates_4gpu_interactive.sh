#!/usr/bin/env bash
# Reusable 4-GPU interactive supervisor for the bounded post-Arm-A gates.
#
# GPU 0/1: two independently initialized G-PATCH schedule canaries.
# GPU 2/3: independent P10 CIC phase reconstructions; after CIC, source audit
#          and matched exact-DTFE build/evaluation workers use released slots.
#
# This is intentionally not data-parallel training. A P10 step is one
# variable-size canonical patch; sharding four patches and averaging their
# gradients would change the frozen optimizer/objective/resume contract.
set -uo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
TRAIN=${REPO}/workflows/abacus_tweb/p10_train_arm_a.py
CLASSICAL=${REPO}/workflows/abacus_tweb/p10_classical_fullcap.py
MT_AUDIT=${REPO}/workflows/abacus_tweb/p10_multitracer_source_audit.py
DTFE=${REPO}/workflows/abacus_tweb/p8_dtfe_fullcap.py
RUN_ROOT=${ROOT}/arm_a_training
LOG_ROOT=${ROOT}/next_gates_logs
mkdir -p "${LOG_ROOT}"

run_graph_canary() {
  local name=$1
  local lr=$2
  "${PY}" "${TRAIN}" \
    --model graph --seed 42 --epochs 3 --min-epochs 1 \
    --disable-early-stopping --lr "${lr}" \
    --scheduler-total-updates 400000 --gradient-clip 5.0 \
    --validation-group-cores 4 --loss-log-every 25 --checkpoint-every 250 \
    --max-runtime-seconds 12600 --validation-reserve-seconds 1200 \
    --run-name "${name}" --output-root "${RUN_ROOT}" --auto-resume
}

run_cic_phases() {
  local worker=$1
  shift
  local phase
  for phase in "$@"; do
    "${PY}" "${CLASSICAL}" raw --phase "${phase}"
  done
  echo "$(date -u +%FT%TZ) ${worker} complete"
}

run_dtfe_phases() {
  local worker=$1
  shift
  local phase graph build
  for phase in "$@"; do
    graph="${ROOT}/${phase}/p2_graph"
    build="${ROOT}/classical/dtfe_build_v1/${phase}"
    mkdir -p "${build}"
    if [[ ! -f "${build}/DTFE_FIELD_READY" ]]; then
      "${PY}" "${DTFE}" --mode build \
        --points "${ROOT}/${phase}/p1_canonical/points.npy" \
        --tets "${graph}/${phase}_bgs_bright_full_delaunay_tetrahedra_idx.npy" \
        --volumes "${graph}/${phase}_bgs_bright_full_delaunay_tetrahedra_volumes.npy" \
        --field-adapter "${ROOT}/training_contract/adapters/${phase}/field" \
        --output-root "${build}" --threads 16 --tree-workers 16 \
        --raster-slab 2 || true
    fi
    if [[ ! -f "${build}/DTFE_FIELD_READY" && -f "${build}/vertex_density.npy" ]]; then
      "${PY}" "${DTFE}" --mode walk-retry \
        --points "${ROOT}/${phase}/p1_canonical/points.npy" \
        --tets "${graph}/${phase}_bgs_bright_full_delaunay_tetrahedra_idx.npy" \
        --volumes "${graph}/${phase}_bgs_bright_full_delaunay_tetrahedra_volumes.npy" \
        --field-adapter "${ROOT}/training_contract/adapters/${phase}/field" \
        --output-root "${build}" --threads 16 --tree-workers 16 \
        --raster-slab 2
    fi
    "${PY}" "${CLASSICAL}" raw --phase "${phase}" --estimator dtfe
  done
  echo "$(date -u +%FT%TZ) ${worker} complete"
}
export REPO PY ROOT TRAIN CLASSICAL MT_AUDIT DTFE RUN_ROOT LOG_ROOT
export -f run_graph_canary run_cic_phases run_dtfe_phases

all_terminal() {
  [[ -f "${RUN_ROOT}/g_schedule_canary_lr2e4_v1/graph/seed_42/ARM_A_TRAINING_COMPLETE.json" ]] &&
  [[ -f "${RUN_ROOT}/g_schedule_canary_lr5e4_v1/graph/seed_42/ARM_A_TRAINING_COMPLETE.json" ]] &&
  [[ -f "${ROOT}/classical/cic_fullcap_v1/P10_CIC_PH006_COMPLETE.json" ]] &&
  [[ -f "${ROOT}/multitracer/source_audit_v1.json" ]] &&
  [[ -f "${ROOT}/classical/dtfe_fullcap_v1/P10_DTFE_PH006_COMPLETE.json" ]]
}

echo "$(date -u +%FT%TZ) supervisor_start pid=$$" >> "${LOG_ROOT}/supervisor.log"
attempt=0
while ! all_terminal; do
  attempt=$((attempt + 1))
  echo "$(date -u +%FT%TZ) allocation_request attempt=${attempt}" >> "${LOG_ROOT}/supervisor.log"
  set +e
  salloc --nodes=1 --ntasks=4 --cpus-per-task=16 \
    --constraint='gpu&hbm80g' --gpus=4 --qos=interactive \
    --time=04:00:00 --account=desi_g --immediate=600 \
    --job-name=p10_next4 bash -lc "
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH
      export PYTHONNOUSERSITE=1
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export XLA_PYTHON_CLIENT_ALLOCATOR=platform
      cd '${REPO}'

      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
        --cpu-bind=cores --export=ALL bash -lc \
        'run_graph_canary g_schedule_canary_lr2e4_v1 0.0002' \
        >> '${LOG_ROOT}/g_lr2e4.log' 2>&1 &
      pid0=\$!
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
        --cpu-bind=cores --export=ALL bash -lc \
        'run_graph_canary g_schedule_canary_lr5e4_v1 0.0005' \
        >> '${LOG_ROOT}/g_lr5e4.log' 2>&1 &
      pid1=\$!
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
        --cpu-bind=cores --export=ALL bash -lc \
        'run_cic_phases cic_even ph000 ph003 ph005' \
        >> '${LOG_ROOT}/cic_even.log' 2>&1 &
      pid2=\$!
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
        --cpu-bind=cores --export=ALL bash -lc \
        'run_cic_phases cic_odd ph002 ph004 ph006' \
        >> '${LOG_ROOT}/cic_odd.log' 2>&1 &
      pid3=\$!

      wait \${pid2}; code2=\$?
      wait \${pid3}; code3=\$?
      if [[ \${code2} -eq 0 && \${code3} -eq 0 ]]; then
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL bash -lc \
          \"'${PY}' '${CLASSICAL}' finalize && '${PY}' '${MT_AUDIT}'\" \
          >> '${LOG_ROOT}/cic_finalize_and_multitracer_audit.log' 2>&1
      fi

      # Exact DTFE needs the same five-phase-only affine fit as CIC. Two released
      # GPUs build and sample independent phases; every phase product is resumable.
      if [[ \${code2} -eq 0 && \${code3} -eq 0 ]]; then
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL bash -lc \
          'run_dtfe_phases dtfe_even ph000 ph003 ph005' \
          >> '${LOG_ROOT}/dtfe_even.log' 2>&1 &
        dtfe_pid0=\$!
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL bash -lc \
          'run_dtfe_phases dtfe_odd ph002 ph004 ph006' \
          >> '${LOG_ROOT}/dtfe_odd.log' 2>&1 &
        dtfe_pid1=\$!
      else
        dtfe_pid0=''
        dtfe_pid1=''
      fi

      wait \${pid0}; code0=\$?
      wait \${pid1}; code1=\$?
      if [[ -n \"\${dtfe_pid0}\" ]]; then wait \${dtfe_pid0}; dtfe0=\$?; else dtfe0=0; fi
      if [[ -n \"\${dtfe_pid1}\" ]]; then wait \${dtfe_pid1}; dtfe1=\$?; else dtfe1=0; fi
      if [[ \${dtfe0} -eq 0 && \${dtfe1} -eq 0 ]]; then
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 \
          --cpu-bind=cores --export=ALL '${PY}' '${CLASSICAL}' finalize --estimator dtfe \
          >> '${LOG_ROOT}/dtfe_finalize.log' 2>&1
      fi
      [[ \${code0} -eq 0 || \${code0} -eq 75 ]]
      [[ \${code1} -eq 0 || \${code1} -eq 75 ]]
      [[ \${code2} -eq 0 ]]
      [[ \${code3} -eq 0 ]]
      [[ \${dtfe0} -eq 0 ]]
      [[ \${dtfe1} -eq 0 ]]
    "
  code=$?
  set -e
  echo "$(date -u +%FT%TZ) allocation_exit attempt=${attempt} code=${code}" >> "${LOG_ROOT}/supervisor.log"
  if all_terminal; then
    break
  fi
  if [[ ${attempt} -ge 12 ]]; then
    echo "$(date -u +%FT%TZ) bounded_retry_exhausted" >> "${LOG_ROOT}/supervisor.log"
    exit 1
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) supervisor_complete" >> "${LOG_ROOT}/supervisor.log"
