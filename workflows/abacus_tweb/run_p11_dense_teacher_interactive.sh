#!/usr/bin/env bash
# Build the frozen dense-view response adapter and resume the P11 teacher gate.
# Run only inside a one-GPU NERSC interactive allocation.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=${P11_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
MAX_RUNTIME_SECONDS=${P11_MAX_RUNTIME_SECONDS:-13200}
VALIDATION_RESERVE_SECONDS=${P11_VALIDATION_RESERVE_SECONDS:-1200}
DESI_PYTHONPATH=${P11_DESI_PYTHONPATH:-/global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/code/desitarget/main/py:/global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/code/desiutil/main/py}
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
export PYTHONPATH="${DESI_PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

{
  echo "P11_PYTHON=${PY}"
  timeout 90 "${PY}" -c 'import h5py, jax, numpy, torch; print("P11_ENV_OK", jax.__version__, torch.__version__)'
  "${PY}" -u -m workflows.abacus_tweb.p11_factorial_training
  # The heavy count products are already frozen by
  # FACTORIAL_VIEW_PRODUCTS_READY.json.  The recovery runtime deliberately
  # tests the trainer/contract path only; builder-only DESI/healpy imports are
  # not required to resume an immutable checkpoint.
  "${PY}" -m unittest     tests.phase4.test_p11_factorial_training     tests.phase4.test_p11_factorial_view_contract
  "${PY}" -u workflows/abacus_tweb/p10_train_arm_a.py     --model unet     --p11-dense-view     --seed 42     --epochs 20     --min-epochs 10     --patience 5     --min-delta 0.002     --lr 0.002     --run-name p11_dense_teacher_v1     --output-root "${OUTPUT}"     --max-runtime-seconds "${MAX_RUNTIME_SECONDS}"     --validation-reserve-seconds "${VALIDATION_RESERVE_SECONDS}"     --checkpoint-every 250     --loss-log-every 25     --auto-resume
} 2>&1 | tee -a "${LOG}"
