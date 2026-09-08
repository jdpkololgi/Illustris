#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
D2=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1
cd "$REPO"
"$PY" -m unittest tests.phase4.test_p12_followup_diagnostics -v
"$PY" -m workflows.sbi.p12f3_d2_support_geometry --root "$D2" \
  --output "$D2/recovery_20260906/SUPPORT_GEOMETRY_${SLURM_JOB_ID}.json"
"$PY" -m workflows.sbi.p12b_representation_diagnostics \
  --config configs/p12b_representation_diagnostics_v1.json
