#!/usr/bin/env bash
# Detached, allocation-safe truth-free ph001 CIC/DTFE prediction chain.
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
python=${P12_CLASSICAL_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
response=${root}/ph001/p3b_random_response_v1/manifest.json
assignment=${root}/ph001/p4_patches/active_assignment.npz
points=${root}/ph001/p1_canonical/points.npy
graph=${root}/ph001/p2_graph
adapter=${root}/training_contract/adapters/ph001/field
dtfe_root=${root}/classical/dtfe_build_v1/ph001
output_root=${root}/blind_predictions/ph001/classical
cic_output=${output_root}/cic_predictions.npz
dtfe_output=${output_root}/dtfe_predictions.npz
cic_affine=${root}/classical/cic_random_response_v1/P10_CIC_PH006_COMPLETE.json
dtfe_affine=${root}/classical/dtfe_random_response_v1/P10_DTFE_PH006_COMPLETE.json
supervisor_root=${output_root}/supervisor
log=${supervisor_root}/supervisor.log
lock=${supervisor_root}/supervisor.lock

mkdir -p "${supervisor_root}" "${dtfe_root}"
cd "${repo}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec 9>"${lock}"
flock -n 9 || { echo "P12 blind-classical supervisor is already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

while [[ ! -f "${response}" ]]; do
  echo "[$(date -u +%FT%TZ)] waiting for complete ph001 random-response manifest"
  sleep 60
done

if [[ -f "${cic_output}" || -f "${cic_output%.npz}.json" ]]; then
  [[ -f "${cic_output}" && -f "${cic_output%.npz}.json" ]] || {
    echo "partial blind CIC output exists; refusing overwrite"; exit 4;
  }
fi
if [[ -f "${dtfe_output}" || -f "${dtfe_output%.npz}.json" ]]; then
  [[ -f "${dtfe_output}" && -f "${dtfe_output%.npz}.json" ]] || {
    echo "partial blind DTFE output exists; refusing overwrite"; exit 4;
  }
fi
if [[ -f "${cic_output}" && -f "${dtfe_output}" ]]; then
  echo "truth-free ph001 CIC and DTFE predictions already exist"
  exit 0
fi

while true; do
  live_allocations=$(squeue -h -u "${USER}" -o '%j|%T' | awk -F'|' \
    '$1 == "interactive" && ($2 == "RUNNING" || $2 == "PENDING" || $2 == "CONFIGURING" || $2 == "COMPLETING") {n += 1} END {print n + 0}')
  if [[ ${live_allocations} -lt 2 ]]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] ${live_allocations} allocations occupied; waiting 60 s"
  sleep 60
done

worker_started=${supervisor_root}/worker_started.json
echo "[$(date -u +%FT%TZ)] requesting one GPU for blind CIC/DTFE reconstruction"
set +e
salloc \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --constraint="gpu&hbm80g" \
  --gpus=1 \
  --qos=interactive \
  --time=04:00:00 \
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
      printf '{\"started_utc\":\"%s\"}\n' \"\$(date -u +%FT%TZ)\" > '${worker_started}.tmp'
      mv '${worker_started}.tmp' '${worker_started}'
      timeout 90 '${python}' -c \"import torch,numpy,h5py,scipy; assert torch.cuda.is_available(); print('P12_BLIND_CLASSICAL_RUNTIME_OK', torch.cuda.get_device_name(0))\"
      '${python}' -m unittest -v \
        tests.phase4.test_p12_blind_classical_predictions \
        tests.phase4.test_p10_classical_fullcap \
        tests.phase4.test_p8_dtfe_fullcap
      if [[ ! -f '${cic_output}' ]]; then
        '${python}' -u -m workflows.sbi.p12_blind_classical_predictions \
          --estimator cic \
          --assignment '${assignment}' \
          --points '${points}' \
          --response-manifest '${response}' \
          --affine-report '${cic_affine}' \
          --output '${cic_output}' \
          --device cuda
      fi
      if [[ ! -f '${dtfe_root}/DTFE_FIELD_READY' ]]; then
        set +e
        '${python}' -u -m workflows.abacus_tweb.p8_dtfe_fullcap \
          --mode build \
          --points '${points}' \
          --tets '${graph}/ph001_bgs_bright_full_delaunay_tetrahedra_idx.npy' \
          --volumes '${graph}/ph001_bgs_bright_full_delaunay_tetrahedra_volumes.npy' \
          --field-adapter '${adapter}' \
          --output-root '${dtfe_root}' \
          --threads 32 --tree-workers 32 --raster-slab 2
        build_code=\$?
        set -e
        if [[ \${build_code} -ne 0 && ! -f '${dtfe_root}/vertex_density.npy' ]]; then
          exit \${build_code}
        fi
      fi
      if [[ ! -f '${dtfe_root}/DTFE_FIELD_READY' ]]; then
        '${python}' -u -m workflows.abacus_tweb.p8_dtfe_fullcap \
          --mode walk-retry \
          --points '${points}' \
          --tets '${graph}/ph001_bgs_bright_full_delaunay_tetrahedra_idx.npy' \
          --volumes '${graph}/ph001_bgs_bright_full_delaunay_tetrahedra_volumes.npy' \
          --field-adapter '${adapter}' \
          --output-root '${dtfe_root}' \
          --threads 32 --tree-workers 32 --raster-slab 2
      fi
      if [[ ! -f '${dtfe_output}' ]]; then
        '${python}' -u -m workflows.sbi.p12_blind_classical_predictions \
          --estimator dtfe \
          --assignment '${assignment}' \
          --points '${points}' \
          --response-manifest '${response}' \
          --affine-report '${dtfe_affine}' \
          --dtfe-root '${dtfe_root}' \
          --output '${dtfe_output}' \
          --device cuda
      fi
      '${python}' -c \"import json,pathlib; paths=[pathlib.Path('${cic_output%.npz}.json'),pathlib.Path('${dtfe_output%.npz}.json')]; rows=[]; [rows.append(json.loads(p.read_text())) for p in paths]; assert all(r['pass'] and r['phase']=='ph001' and r['truth_files_read']==[] and r['open_count']==0 and not r['sealed_phase_opened'] for r in rows); assert rows[0]['rows']==rows[1]['rows']; print('P12_BLIND_CLASSICAL_COMPLETE', rows[0]['rows'])\"
    "
code=$?
set -e
if [[ ${code} -eq 0 && -f "${cic_output}" && -f "${dtfe_output}" ]]; then
  echo "[$(date -u +%FT%TZ)] truth-free ph001 CIC/DTFE predictions complete"
  exit 0
fi
if [[ ! -f "${worker_started}" ]]; then
  echo "allocation ended before worker start (exit ${code}); rerun supervisor explicitly"
else
  echo "blind classical worker failed after start with exit ${code}; inspect resumable DTFE state before retry"
fi
exit "${code}"
