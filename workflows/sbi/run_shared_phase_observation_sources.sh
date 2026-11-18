#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Approved allocation required}"
: "${SLURM_STEP_ID:?Compute step required}"
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_20phase_20261027.json
out=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20phase_20261027_v1/source_preparation_${SLURM_JOB_ID}
mkdir -p "$out"
exec > >(tee "$out/observation_sources.log") 2>&1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$py" -u -m workflows.sbi.prepare_shared_phase_observations coordinates
"$py" -u -m workflows.sbi.prepare_shared_phase_observations observations --phases ph004 ph005 ph006
