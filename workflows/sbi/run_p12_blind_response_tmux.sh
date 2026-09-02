#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
# The CFS recovery environment is intentionally minimal and does not contain
# healpy, which the HEALPix random-map stage requires.  Keep the import under a
# bounded compute-node preflight so a node-local Scratch/Lustre stall fails
# quickly rather than burning an allocation.
python=${P12_CPU_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
status_python=/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python
status_helper=/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
authority=${repo}/docs/evidence/p12/P12_BLIND_RESPONSE_BUILD_AUTHORIZED.json
decision=${root}/training_contract/P3BR_RANDOM_DENSITY_DECISION.json
manifest=${root}/ph001/p3b_random_response_v1/manifest.json
supervisor_root=${root}/ph001/p12_blind_response_supervisor
log=${supervisor_root}/supervisor.log
lock=${supervisor_root}/supervisor.lock

mkdir -p "${supervisor_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "ph001 response supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

attempt=0
while [[ ! -f "${manifest}" ]]; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt 12 ]]; then
    echo "ph001 response build exceeded 12 allocation attempts"
    exit 4
  fi
  while true; do
    set +e
    timeout 45 "${status_python}" "${status_helper}" --max-interactive 2
    status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
      break
    fi
    if [[ ${status} -ne 2 && ${status} -ne 124 ]]; then
      echo "allocation-status helper failed with ${status}"
      exit "${status}"
    fi
    echo "[$(date -u +%FT%TZ)] two allocations occupied; waiting 60 s"
    sleep 60
  done
  worker_started=${supervisor_root}/attempt_$(printf '%03d' "${attempt}")_worker_started.json
  echo "[$(date -u +%FT%TZ)] ph001 response allocation attempt ${attempt}"
  set +e
  salloc \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --constraint=cpu \
    --qos=interactive \
    --time=02:00:00 \
    --account=desi \
    --immediate=600 \
    srun \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=64 \
      --cpu-bind=cores \
      --export=ALL \
      /bin/bash -lc "
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
        export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=64
        cd '${repo}'
        printf '{\"attempt\":%d,\"started_utc\":\"%s\"}\n' '${attempt}' \"\$(date -u +%FT%TZ)\" > '${worker_started}.tmp'
        mv '${worker_started}.tmp' '${worker_started}'
        timeout 90 '${python}' -c \"import fitsio,h5py,healpy,numpy,scipy; print('P12_BLIND_RESPONSE_RUNTIME_OK')\"
        '${python}' -m unittest tests.phase4.test_p12_blind_response_authority tests.phase4.test_p3br_random_response
        '${python}' -u -m workflows.abacus_tweb.p3br_build_random_response maps \
          --phase ph001 \
          --snapshots 18 \
          --blind-authority '${authority}' \
          --root '${root}'
        '${python}' -u -m workflows.abacus_tweb.p3br_build_random_response overlay \
          --phase ph001 \
          --blind-authority '${authority}' \
          --decision '${decision}' \
          --root '${root}'
        '${python}' -c \"import json,pathlib; p=pathlib.Path('${manifest}'); d=json.loads(p.read_text()); assert d['phase']=='ph001' and d['pass'] is True and d['ph001_opened'] is False and d['blind_authority']['sha256']; print('P12_BLIND_RESPONSE_COMPLETE', p)\"
      "
  code=$?
  set -e
  if [[ ${code} -eq 0 && -f "${manifest}" ]]; then
    break
  fi
  if [[ ! -f "${worker_started}" ]]; then
    echo "[$(date -u +%FT%TZ)] allocation ended before worker start (exit ${code}); retrying after 60 s"
    sleep 60
    continue
  fi
  echo "ph001 response worker failed after start with exit ${code}; refusing automatic retry"
  exit "${code}"
done

echo "[$(date -u +%FT%TZ)] truth-free ph001 random-response product complete: ${manifest}"
