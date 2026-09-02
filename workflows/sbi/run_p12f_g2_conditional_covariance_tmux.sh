#!/usr/bin/env bash
# Detached, allocation-safe P12-F v2 shell/scale-conditioned G2 rescue.
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
python=${P12F_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
status_python=/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python
status_helper=/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
phase_root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
contract=/global/homes/d/dkololgi/p11_contracts/training_contract_r1_random_repair_v2_20260901
config=${repo}/configs/p12f_conditional_covariance_g2_v2.json
parent=${phase_root}/p12f_dependency_rescue_v2/evaluation_sufficiency_seed42
panel=${parent}/panel_1024/P12F_PH006_PANEL_1024.json
g1_archive=${parent}/archives/gaussian_correlated_g1/P12F_SAMPLE_ARCHIVE.json
g1_report=${parent}/dependency_evaluation/P12F_DEPENDENCY_RESCUE_V2_REPORT.json
gaussian_root=${phase_root}/p12f_matched_challengers_v1/matched_v1_seed42/gaussian
checkpoint=${gaussian_root}/checkpoint.pt
training_manifest=${gaussian_root}/run_manifest.json
global_g1_filter=${gaussian_root}/g1_residual_filter.json
root=${phase_root}/p12f_dependency_rescue_v2/g2_shell_covariance_seed42
filter=${root}/g2_shell_residual_filter.json
archive_root=${root}/archives
archive=${archive_root}/gaussian_shell_correlated_g2/P12F_SAMPLE_ARCHIVE.json
evaluation=${root}/dependency_evaluation
report=${evaluation}/P12F_DEPENDENCY_RESCUE_V2_REPORT.json
proper=${root}/P12F_G2_VS_G1_PROPER_SCORES.json
decision=${root}/P12F_G2_DECISION.json
evidence_root=${repo}/docs/evidence/p12/p12f_g2_conditional_covariance_v2
figure_root=${repo}/docs/figures/p12f_g2_conditional_covariance_20260902
plot_marker=${evidence_root}/P12F_G2_VS_G1_PLOT.json
log=${root}/supervisor.log
lock=${root}/supervisor.lock

mkdir -p "${root}" "${evidence_root}" "${figure_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12-F G2 supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

wait_for_slot() {
  while true; do
    set +e
    timeout 45 "${status_python}" "${status_helper}" --max-interactive 2
    helper_status=$?
    set -e
    if [[ ${helper_status} -ne 0 && ${helper_status} -ne 2 && ${helper_status} -ne 124 ]]; then
      echo "allocation-status helper failed with ${helper_status}"
      return "${helper_status}"
    fi
    live=$(squeue -h -u "${USER}" -o '%j|%T' | awk -F'|' \
      '$1 == "interactive" && ($2 == "RUNNING" || $2 == "PENDING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n += 1} END {print n + 0}')
    if [[ ${live} -lt 2 ]]; then
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] ${live} allocations occupied; waiting 60 s"
    sleep 60
  done
}

run_allocation() {
  local stage=$1
  local attempt=$2
  local command=$3
  local started=${root}/${stage}_attempt_$(printf '%03d' "${attempt}")_worker_started.json
  wait_for_slot
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --constraint="gpu&hbm80g" \
    --gpus=1 --qos=interactive --time=04:00:00 --account=desi_g --immediate=600 \
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=none \
      --cpu-bind=cores --export=ALL /bin/bash -lc "
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
        export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32
        cd '${repo}'
        printf '{\"stage\":\"%s\",\"attempt\":%d,\"started_utc\":\"%s\"}\n' \
          '${stage}' '${attempt}' \"\$(date -u +%FT%TZ)\" > '${started}.tmp'
        mv '${started}.tmp' '${started}'
        timeout 120 '${python}' -c \"import tarp,torch,numpy,h5py,matplotlib; assert torch.cuda.is_available(); print('P12F_G2_RUNTIME_OK',torch.cuda.get_device_name(0))\"
        '${python}' -m unittest -v tests.phase4.test_p12f_production_challengers tests.phase4.test_p12f_dependency_rescue_evaluator
        ${command}
      "
  code=$?
  set -e
  if [[ ${code} -ne 0 && ${code} -ne 75 && ! -f "${started}" ]]; then
    echo "[$(date -u +%FT%TZ)] ${stage} allocation ended before worker start; retryable"
    return 75
  fi
  return "${code}"
}

run_cpu_allocation() {
  local stage=$1
  local attempt=$2
  local command=$3
  local started=${root}/${stage}_attempt_$(printf '%03d' "${attempt}")_worker_started.json
  wait_for_slot
  set +e
  salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu \
    --qos=interactive --time=04:00:00 --account=desi --immediate=600 \
    srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
      --export=ALL /bin/bash -lc "
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
        export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=64
        cd '${repo}'
        printf '{\"stage\":\"%s\",\"attempt\":%d,\"started_utc\":\"%s\"}\n' \
          '${stage}' '${attempt}' \"\$(date -u +%FT%TZ)\" > '${started}.tmp'
        mv '${started}.tmp' '${started}'
        timeout 120 '${python}' -c \"import tarp,torch,numpy,h5py,matplotlib; print('P12F_G2_CPU_RUNTIME_OK')\"
        '${python}' -m unittest -v tests.phase4.test_p12f_production_challengers tests.phase4.test_p12f_dependency_rescue_evaluator
        ${command}
      "
  code=$?
  set -e
  if [[ ${code} -ne 0 && ${code} -ne 75 && ! -f "${started}" ]]; then
    echo "[$(date -u +%FT%TZ)] ${stage} allocation ended before worker start; retryable"
    return 75
  fi
  return "${code}"
}

attempt=0
while [[ ! -f "${archive}" ]]; do
  attempt=$((attempt + 1))
  [[ ${attempt} -le 24 ]] || { echo "G2 fit/export exceeded 24 attempts"; exit 4; }
  command="
    if [[ ! -f '${filter}' ]]; then
      '${python}' -u -m workflows.sbi.p12f_fit_g2_shell_filter \\
        --config '${config}' --contract-root '${contract}' --phase-root '${phase_root}' \\
        --gaussian-checkpoint '${checkpoint}' --run-manifest '${training_manifest}' \\
        --global-g1-filter '${global_g1_filter}' --output '${filter}' --device cuda
    fi
    '${python}' -u -m workflows.sbi.p12f_export_sample_archive \\
      --config '${config}' --contract-root '${contract}' --phase-root '${phase_root}' \\
      --panel-marker '${panel}' --checkpoint '${checkpoint}' \\
      --method gaussian_shell_correlated_g2 --g2-filter '${filter}' \\
      --output-root '${archive_root}' --device cuda --resume --max-wall-seconds 12600
  "
  if run_allocation fit_export "${attempt}" "${command}"; then
    [[ -f "${archive}" ]] || { echo "G2 worker exited zero without archive"; exit 5; }
  else
    code=$?
    [[ ${code} -eq 75 ]] || { echo "G2 fit/export failed with ${code}"; exit "${code}"; }
  fi
done

attempt=0
while [[ ! -f "${report}" ]]; do
  attempt=$((attempt + 1))
  [[ ${attempt} -le 24 ]] || { echo "G2 evaluation exceeded 24 attempts"; exit 6; }
  command="'${python}' -u -m workflows.sbi.p12f_dependency_rescue_evaluator \\
      --config '${config}' --archive-manifest '${archive}' --panel-marker '${panel}' \\
      --output-root '${evaluation}' --device cuda --max-wall-seconds 13500"
  if run_allocation evaluation "${attempt}" "${command}"; then
    [[ -f "${report}" ]] || { echo "G2 evaluator exited zero without report"; exit 7; }
  else
    code=$?
    [[ ${code} -eq 75 ]] || { echo "G2 evaluation failed with ${code}"; exit "${code}"; }
  fi
done

attempt=0
while [[ ! -f "${decision}" ]]; do
  attempt=$((attempt + 1))
  [[ ${attempt} -le 24 ]] || { echo "G2 score/decision exceeded 24 attempts"; exit 8; }
  command="
    '${python}' -u -m workflows.sbi.p12f_compare_g2_g1_scores \\
      --config '${config}' --panel-marker '${panel}' --g1-archive '${g1_archive}' \\
      --g2-archive '${archive}' --output '${proper}' --max-wall-seconds 13500
    '${python}' -u -m workflows.sbi.p12f_freeze_g2_decision \\
      --config '${config}' --g2-report '${report}' --proper-scores '${proper}' \\
      --output '${decision}'
    '${python}' -u -m workflows.sbi.plot_p12f_dependency_rescue \\
      --report '${report}' --output-dir '${figure_root}/g2' \\
      --evidence-output '${evidence_root}/P12F_G2_PLOTS.json'
    '${python}' -u -m workflows.sbi.plot_p12f_g2_vs_g1 \\
      --g1-report '${g1_report}' --g2-report '${report}' --proper-scores '${proper}' \\
      --output '${figure_root}/p12f_g2_vs_g1' --evidence-output '${plot_marker}'
  "
  if run_cpu_allocation score_decision "${attempt}" "${command}"; then
    [[ -f "${decision}" && -f "${plot_marker}" ]] || {
      echo "G2 decision worker exited zero without final evidence"; exit 9;
    }
  else
    code=$?
    [[ ${code} -eq 75 ]] || { echo "G2 score/decision failed with ${code}"; exit "${code}"; }
  fi
done

cp "${report}" "${evidence_root}/P12F_G2_REPORT.json.tmp"
mv "${evidence_root}/P12F_G2_REPORT.json.tmp" "${evidence_root}/P12F_G2_REPORT.json"
cp "${proper}" "${evidence_root}/P12F_G2_VS_G1_PROPER_SCORES.json.tmp"
mv "${evidence_root}/P12F_G2_VS_G1_PROPER_SCORES.json.tmp" "${evidence_root}/P12F_G2_VS_G1_PROPER_SCORES.json"
cp "${decision}" "${evidence_root}/P12F_G2_DECISION.json.tmp"
mv "${evidence_root}/P12F_G2_DECISION.json.tmp" "${evidence_root}/P12F_G2_DECISION.json"
echo "[$(date -u +%FT%TZ)] P12-F G2 conditional-covariance programme complete"
