#!/bin/bash
# Run INSIDE a user-authorized one-GPU shared_interactive allocation.
# No salloc/sbatch, automatic retries, or allocation chaining in this launcher.
set -euo pipefail
: "${SLURM_JOB_ID:?An existing GPU allocation is required}"
REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
OUT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12b_unet_representation_v1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT/logs"
"$PY" -m unittest tests.phase4.test_p12b_unet_representation -v
"$PY" workflows/sbi/p12b_unet_representation.py all --max-runtime-seconds 6600 \
  2>&1 | tee "$OUT/logs/interactive_${SLURM_JOB_ID}.log"
