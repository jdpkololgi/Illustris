#!/usr/bin/env bash
# Build the frozen dense-view response adapter and resume the P11 teacher gate.
# Run only inside a one-GPU NERSC interactive allocation.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1
OUTPUT=${ROOT}/training
LOG=${ROOT}/p11_dense_teacher_interactive_${SLURM_JOB_ID:-manual}.log

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This worker must run inside an interactive allocation." >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != *gpu* && "${SLURM_JOB_CONSTRAINTS:-}" != *gpu* ]]; then
  echo "A GPU interactive allocation is required." >&2
  exit 2
fi

cd "${REPO}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

{
  "${PY}" -u workflows/abacus_tweb/p11_factorial_training.py
  "${PY}" -m unittest     tests.phase4.test_p11_factorial_training     tests.phase4.test_p11_factorial_view_contract     tests.phase4.test_p11_factorial_view_counts
  "${PY}" -u workflows/abacus_tweb/p10_train_arm_a.py     --model unet     --p11-dense-view     --seed 42     --epochs 20     --min-epochs 10     --patience 5     --min-delta 0.002     --lr 0.002     --run-name p11_dense_teacher_v1     --output-root "${OUTPUT}"     --max-runtime-seconds 13200     --validation-reserve-seconds 1200     --checkpoint-every 250     --loss-log-every 25     --auto-resume
} 2>&1 | tee -a "${LOG}"
