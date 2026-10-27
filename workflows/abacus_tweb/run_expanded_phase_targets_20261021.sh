#!/usr/bin/env bash
set -euo pipefail
phase=${1:?phase required}
case "$phase" in ph008|ph009|ph010|ph011|ph012|ph013|ph020|ph021|ph022|ph023|ph024) ;; *) echo 'Phase not in authorized additional target set' >&2;exit 2;; esac
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/$phase
registry=$repo/configs/p10_phase_registry_v2.json
manifest=$root/targets/density/AbacusSummit_base_c000_${phase}_z0.200_ngrid2048_ab10_tsc_counts.manifest.json
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores "$python" -u workflows/abacus_tweb/p10_expansion_preflight.py --registry "$registry" --adapt-density "$phase" --output "$manifest"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 "$python" -u workflows/abacus_tweb/p10_run_tweb.py --registry "$registry" --phase "$phase" --density /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/matter/$phase/counts_tsc_2048.npy --density-manifest "$manifest"
