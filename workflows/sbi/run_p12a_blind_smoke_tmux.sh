#!/usr/bin/env bash
# Detached, allocation-safe truth-free P12-A 512-draw throughput smoke.
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
python=${P12A_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
candidate=${repo}/docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json
checkpoint=${root}/p12a_base_response_v1/fmpe_seed42/fmpe_estimator.pt
quality=${repo}/docs/evidence/p12/production_aux_v1/P12A_QUALITY_THRESHOLDS.json
context=${root}/blind_predictions/ph001/p12a/ph001_p12a_base_context.npz
plan=${root}/blind_predictions/ph001/p12a/P12A_BLIND_SHARD_PLAN.json
output_root=${root}/blind_predictions/ph001/p12a/throughput_smoke_v1
output=${output_root}/p12a_512draw_smoke.npz
log=${output_root}/supervisor.log
lock=${output_root}/supervisor.lock

mkdir -p "${output_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12-A blind-smoke supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

if [[ -f "${output}" || -f "${output%.npz}.json" ]]; then
  [[ -f "${output}" && -f "${output%.npz}.json" ]] || {
    echo "partial P12-A smoke output exists; refusing overwrite"; exit 4;
  }
  echo "P12-A truth-free throughput smoke already exists"
  exit 0
fi
while [[ ! -f "${context}" || ! -f "${plan}" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for truth-free P12-A context and shard plan"
  sleep 60
done
while true; do
  live=$(squeue -h -u "${USER}" -o '%j|%T' | awk -F'|' \
    '$1 == "interactive" && ($2 == "RUNNING" || $2 == "PENDING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n += 1} END {print n + 0}')
  [[ ${live} -lt 2 ]] && break
  echo "[$(date -u +%FT%TZ)] ${live} allocations occupied; waiting 60 s"
  sleep 60
done

echo "[$(date -u +%FT%TZ)] requesting one GPU for truth-free P12-A smoke"
salloc --nodes=1 --ntasks=1 --cpus-per-task=16 --constraint="gpu&hbm80g" \
  --gpus=1 --qos=interactive --time=00:30:00 --account=desi_g --immediate=600 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 --cpu-bind=cores \
  --export=ALL /bin/bash -lc "
    set -euo pipefail
    unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16
    cd '${repo}'
    timeout 120 '${python}' -c \"import torch,numpy,sbi; assert torch.cuda.is_available(); assert sbi.__version__ == '0.26.1'; print('P12A_BLIND_SMOKE_RUNTIME_OK', torch.cuda.get_device_name(0))\"
    '${python}' -m unittest -v tests.phase4.test_p12a_blind_throughput_smoke
    '${python}' -u -m workflows.sbi.p12a_blind_throughput_smoke \
      --candidate '${candidate}' --plan '${plan}' --context '${context}' \
      --quality-thresholds '${quality}' --checkpoint '${checkpoint}' \
      --output '${output}' --minimum-rows 2048
  "

"${python}" -c "import json,pathlib; p=pathlib.Path('${output%.npz}.json'); d=json.loads(p.read_text()); assert d['pass'] and d['draws']==512 and d['truth_files_read']==[] and d['open_count']==0; print('P12A_BLIND_SMOKE_COMPLETE', d['rows'], d['projected_four_gpu_hours'])"
