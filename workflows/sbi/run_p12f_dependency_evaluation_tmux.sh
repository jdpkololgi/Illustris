#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
python=${P12F_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
status_python=/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python
status_helper=/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
config=${repo}/configs/p12f_dependency_rescue_v2.json
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f_dependency_rescue_v2/evaluation_sufficiency_seed42
panel=${root}/panel_1024/P12F_PH006_PANEL_1024.json
archive=${root}/archives/gaussian_correlated_g1/P12F_SAMPLE_ARCHIVE.json
evaluation=${root}/dependency_evaluation
report=${evaluation}/P12F_DEPENDENCY_RESCUE_V2_REPORT.json
evidence_root=${repo}/docs/evidence/p12/p12f_dependency_rescue_v2
durable_report=${evidence_root}/P12F_DEPENDENCY_RESCUE_V2_REPORT.json
plot_marker=${evidence_root}/P12F_DEPENDENCY_RESCUE_PLOTS.json
figure_root=${repo}/docs/figures/p12f_dependency_rescue_20260902
log=${root}/evaluation_supervisor.log
lock=${root}/evaluation_supervisor.lock

mkdir -p "${root}" "${evidence_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12-F dependency evaluator supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

while [[ ! -f "${archive}" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for complete expanded G1 archive"
  sleep 60
done

wait_for_slot() {
  while true; do
    set +e
    timeout 45 "${status_python}" "${status_helper}" --max-interactive 2
    status=$?
    set -e
    if [[ ${status} -ne 0 && ${status} -ne 2 && ${status} -ne 124 ]]; then
      echo "allocation-status helper failed with ${status}"
      return "${status}"
    fi
    # Urgent-reservation interactive jobs can be reported in the helper's
    # `other_jobs` list. Count the standard salloc job name directly as the
    # binding two-allocation guard.
    live_allocations=$(squeue -h -u "${USER}" -o '%j|%T' | awk -F'|' \
      '$1 == "interactive" && ($2 == "RUNNING" || $2 == "PENDING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n += 1} END {print n + 0}')
    if [[ ${live_allocations} -lt 2 ]]; then
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] ${live_allocations} allocations are occupied; waiting 60 s"
    sleep 60
  done
}

attempt=0
while [[ ! -f "${plot_marker}" ]]; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 48 ]]; then
    echo "P12-F dependency evaluation exceeded 48 allocation attempts"
    exit 4
  fi
  wait_for_slot
  worker_started=${root}/evaluation_attempt_$(printf '%03d' "${attempt}")_worker_started.json
  echo "[$(date -u +%FT%TZ)] evaluation allocation attempt ${attempt}"
  set +e
  salloc \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --constraint="gpu&hbm80g" \
    --gpus=1 \
    --qos=interactive \
    --time=02:00:00 \
    --account=desi_g \
    --immediate=600 \
    srun \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=32 \
      --gpus=1 \
      --gpu-bind=none \
      --cpu-bind=cores \
      --export=ALL \
      /bin/bash -lc "
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
        export PYTHONNOUSERSITE=1
        cd '${repo}'
        printf '{\"attempt\":%d,\"started_utc\":\"%s\"}\n' '${attempt}' \"\$(date -u +%FT%TZ)\" > '${worker_started}.tmp'
        mv '${worker_started}.tmp' '${worker_started}'
        timeout 90 '${python}' -c \"import tarp, torch, numpy, h5py, matplotlib; print('P12F_V2_EVAL_RUNTIME_OK', torch.__version__)\"
        '${python}' -m unittest tests.phase4.test_p12f_dependency_rescue_evaluator tests.phase4.test_p12f_production_challengers
        if [[ ! -f '${report}' ]]; then
          '${python}' -u -m workflows.sbi.p12f_dependency_rescue_evaluator \
            --config '${config}' \
            --archive-manifest '${archive}' \
            --panel-marker '${panel}' \
            --output-root '${evaluation}' \
            --device cuda \
            --max-wall-seconds 6300
        fi
        cp '${report}' '${durable_report}.tmp'
        mv '${durable_report}.tmp' '${durable_report}'
        '${python}' -u -m workflows.sbi.plot_p12f_dependency_rescue \
          --report '${durable_report}' \
          --output-dir '${figure_root}' \
          --evidence-output '${plot_marker}'
      "
  code=$?
  set -e
  if [[ ${code} -eq 0 && -f "${plot_marker}" ]]; then
    break
  fi
  if [[ ${code} -eq 75 ]]; then
    echo "[$(date -u +%FT%TZ)] clean compact-evaluation pause; requesting continuation"
    continue
  fi
  if [[ ! -f "${worker_started}" ]]; then
    echo "[$(date -u +%FT%TZ)] evaluation allocation ended before worker start (exit ${code}); retrying after 60 s"
    sleep 60
    continue
  fi
  echo "P12-F v2 evaluation worker failed with exit ${code}; refusing automatic retry"
  exit "${code}"
done

echo "[$(date -u +%FT%TZ)] dependency evaluation and visual audit complete"
