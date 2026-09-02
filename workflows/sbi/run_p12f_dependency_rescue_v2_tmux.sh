#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
python=${P12F_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
status_python=/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python
status_helper=/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
config=${repo}/configs/p12f_dependency_rescue_v2.json
contract=/global/homes/d/dkololgi/p11_contracts/training_contract_r1_random_repair_v2_20260901
phase_root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
root=${phase_root}/p12f_dependency_rescue_v2/evaluation_sufficiency_seed42
panel_root=${root}/panel_1024
panel=${panel_root}/P12F_PH006_PANEL_1024.json
archive_root=${root}/archives
archive=${archive_root}/gaussian_correlated_g1/P12F_SAMPLE_ARCHIVE.json
checkpoint=${phase_root}/p12f_matched_challengers_v1/matched_v1_seed42/gaussian/checkpoint.pt
g1_filter=${phase_root}/p12f_matched_challengers_v1/matched_v1_seed42/gaussian/g1_residual_filter.json
log=${root}/supervisor.log
lock=${root}/supervisor.lock

mkdir -p "${root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12-F v2 supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

wait_for_slot() {
  while true; do
    set +e
    timeout 45 "${status_python}" "${status_helper}" --max-interactive 2
    status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
      return 0
    fi
    if [[ ${status} -ne 2 && ${status} -ne 124 ]]; then
      echo "allocation-status helper failed with ${status}"
      return "${status}"
    fi
    echo "[$(date -u +%FT%TZ)] two interactive allocations are occupied; waiting 60 s"
    sleep 60
  done
}

attempt=0
while [[ ! -f "${archive}" ]]; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 24 ]]; then
    echo "P12-F v2 export exceeded 24 allocation attempts"
    exit 4
  fi
  wait_for_slot
  worker_started=${root}/attempt_$(printf '%03d' "${attempt}")_worker_started.json
  echo "[$(date -u +%FT%TZ)] allocation attempt ${attempt}: panel/export"
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
        timeout 90 '${python}' -c \"import tarp, torch, numpy, h5py; print('P12F_V2_RUNTIME_OK', torch.__version__)\"
        '${python}' -m unittest tests.phase4.test_p12f_challengers tests.phase4.test_p12f_production_challengers
        if [[ ! -f '${panel}' ]]; then
          '${python}' -u -m workflows.sbi.p12f_freeze_selection_panel \
            --config '${config}' \
            --contract-root '${contract}' \
            --output-root '${panel_root}'
        fi
        '${python}' -u -m workflows.sbi.p12f_export_sample_archive \
          --config '${config}' \
          --contract-root '${contract}' \
          --phase-root '${phase_root}' \
          --panel-marker '${panel}' \
          --checkpoint '${checkpoint}' \
          --method gaussian_correlated_g1 \
          --g1-filter '${g1_filter}' \
          --output-root '${archive_root}' \
          --device cuda \
          --resume \
          --max-wall-seconds 6300
      "
  code=$?
  set -e
  if [[ ${code} -eq 0 && -f "${archive}" ]]; then
    break
  fi
  if [[ ${code} -eq 75 ]]; then
    echo "[$(date -u +%FT%TZ)] clean checkpoint pause; requesting continuation"
    continue
  fi
  if [[ ! -f "${worker_started}" ]]; then
    echo "[$(date -u +%FT%TZ)] allocation attempt ended before the compute worker started (exit ${code}); retrying after 60 s"
    sleep 60
    continue
  fi
  echo "P12-F v2 worker failed with exit ${code}; refusing automatic retry"
  exit "${code}"
done

echo "[$(date -u +%FT%TZ)] expanded G1 archive complete: ${archive}"
echo "Evaluation/visualization awaits the separately tested resumable v2 evaluator."
