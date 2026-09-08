#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?allocation required}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$PY" -m unittest tests.phase4.test_p12b_followup_controls -v
"$PY" -m workflows.sbi.p12b_representation_followup --config configs/p12b_representation_followup_v1.json
