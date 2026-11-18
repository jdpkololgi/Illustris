#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Approved allocation required}"
: "${SLURM_STEP_ID:?Compute step required}"
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_20phase_20261027.json
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20phase_20261027_v1
for phase in "$@"; do
  case "$phase" in ph008|ph009|ph010|ph011|ph012|ph013|ph020|ph021|ph022|ph023|ph024) ;; *) exit 2;; esac
  while [[ ! -s "$root/input_adoption/$phase/BASE_INPUTS_COMPLETE.json" ]]; do
    if (( SECONDS > 4800 )); then exit 75; fi
    sleep 30
  done
  if (( SECONDS > 4800 )); then exit 75; fi
  "$py" -u -m workflows.sbi.prepare_shared_phase_particles --phases "$phase"
  "$py" -u -m workflows.sbi.adopt_shared_phase_products --phases "$phase"
done
