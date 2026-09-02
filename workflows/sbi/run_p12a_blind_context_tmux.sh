#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
python=${P12A_PYTHON:-/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python}
adapter=${root}/training_contract/adapters/ph001/field
assignment=${root}/ph001/p4_patches/active_assignment.npz
points=${root}/ph001/p1_canonical/points.npy
redshift=${root}/training_contract/phases/ph001/parent_redshift.npy
phase_contract=${root}/training_contract/phases/ph001/phase_contract.json
response=${root}/ph001/p3b_random_response_v1/manifest.json
selection=${root}/training_contract/transforms/field/selection_manifest.json
candidate=${repo}/docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json
checkpoint=${root}/arm_a_training/arm_a_r0_v1/unet/seed_42/best_checkpoint.pt
output_root=${root}/blind_predictions/ph001/p12a
context=${output_root}/ph001_p12a_base_context.npz
context_marker=${output_root}/ph001_p12a_base_context.json
shard_plan=${output_root}/P12A_BLIND_SHARD_PLAN.json
supervisor_root=${output_root}/context_supervisor
log=${supervisor_root}/supervisor.log
lock=${supervisor_root}/supervisor.lock

mkdir -p "${supervisor_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12-A blind-context supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

if [[ -f "${context}" || -f "${context_marker}" || -f "${shard_plan}" ]]; then
  if [[ ! -f "${context}" || ! -f "${context_marker}" || ! -f "${shard_plan}" ]]; then
    echo "partial blind-context output exists; refusing overwrite"
    exit 4
  fi
  echo "truth-free P12-A context and shard plan already exist"
  exit 0
fi

while [[ ! -f "${response}" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for complete ph001 random-response manifest"
  sleep 60
done

attempt=0
while true; do
  # Count all live salloc jobs by their standard name.  The shared status helper
  # can classify urgent-reservation interactive jobs as 'other', so this explicit
  # guard prevents a race to a third allocation.
  live_allocations=$(squeue -h -u "${USER}" -o '%j|%T' | awk -F'|' \
    '$1 == "interactive" && ($2 == "RUNNING" || $2 == "PENDING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n += 1} END {print n + 0}')
  if [[ ${live_allocations} -lt 2 ]]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] ${live_allocations} allocations occupied; waiting 60 s"
  sleep 60
done

attempt=$((attempt + 1))
worker_started=${supervisor_root}/attempt_$(printf '%03d' "${attempt}")_worker_started.json
echo "[$(date -u +%FT%TZ)] requesting one GPU for truth-free P12-A context"
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
    --cpu-bind=cores \
    --export=ALL \
    /bin/bash -lc "
      set -euo pipefail
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
      export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32
      cd '${repo}'
      printf '{\"attempt\":%d,\"started_utc\":\"%s\"}\n' '${attempt}' \"\$(date -u +%FT%TZ)\" > '${worker_started}.tmp'
      mv '${worker_started}.tmp' '${worker_started}'
      timeout 90 '${python}' -c \"import torch,numpy,h5py,fitsio; assert torch.cuda.is_available(); print('P12A_BLIND_CONTEXT_RUNTIME_OK', torch.cuda.get_device_name(0))\"
      '${python}' -m unittest -v \
        tests.phase4.test_p12a_blind_inference \
        tests.phase4.test_p12a_blind_shards
      '${python}' -u -m workflows.sbi.p12a_blind_inference context \
        --adapter-root '${adapter}' \
        --assignment '${assignment}' \
        --points '${points}' \
        --redshift '${redshift}' \
        --phase-contract '${phase_contract}' \
        --response-field-manifest '${response}' \
        --selection-manifest '${selection}' \
        --candidate '${candidate}' \
        --checkpoint '${checkpoint}' \
        --output '${context}' \
        --device cuda
      '${python}' -u -m workflows.sbi.p12a_blind_shards plan \
        --context '${context}' \
        --output '${shard_plan}' \
        --shards 4
      '${python}' -c \"import json,pathlib; c=json.loads(pathlib.Path('${context_marker}').read_text()); p=json.loads(pathlib.Path('${shard_plan}').read_text()); assert c['pass'] and p['pass'] and c['phase']==p['phase']=='ph001' and not c['truth_files_read'] and not p['truth_files_read'] and c['open_count']==p['open_count']==0; print('P12A_BLIND_CONTEXT_COMPLETE', c['rows'])\"
    "
code=$?
set -e
if [[ ${code} -eq 0 && -f "${context}" && -f "${context_marker}" && -f "${shard_plan}" ]]; then
  echo "[$(date -u +%FT%TZ)] truth-free P12-A base context complete"
  exit 0
fi
if [[ ! -f "${worker_started}" ]]; then
  echo "allocation ended before worker start (exit ${code}); rerun supervisor explicitly"
else
  echo "blind-context worker failed after start with exit ${code}; refusing automatic retry"
fi
exit "${code}"
