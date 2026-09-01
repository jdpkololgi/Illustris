#!/usr/bin/env bash
# Run/resume one frozen P11 matched arm inside an existing GPU allocation.
set -euo pipefail

REPO=/global/homes/d/dkololgi/TNG/Illustris
ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1
CONTRACT_ROOT=${P11_CONTRACT_ROOT:-/global/homes/d/dkololgi/p11_contracts/training_contract_r1_random_repair_v2_20260901}
ARM=${P11_ARM:-jepa}
RUN_NAME=${P11_RUN_NAME:-paired_degrade_jepa_m25_v2}
CONTRACT=${P11_CONTRACT:-${REPO}/configs/p11_paired_degrade_jepa_v2.json}
SEED=${P11_SEED:-42}
LOG=${ROOT}/p11_${ARM}_${SLURM_JOB_ID:-manual}.log

# Prefer an explicit, compute-node-local interpreter.  In particular, never
# silently force the pscratch cosmic_env: imports from that path can block in
# cl_sync_io_wait while Scratch is unhealthy.  Both names are supported so the
# existing P11 and wider project launch contracts remain compatible.
PY=${P11_PYTHON:-${COSMIC_ENV_PYTHON:-}}
if [[ -z "${PY}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY=${CONDA_PREFIX}/bin/python
fi
if [[ -z "${PY}" ]]; then
  PY=$(command -v python3 || command -v python || true)
fi
if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  echo "Set P11_PYTHON or COSMIC_ENV_PYTHON to a working compute-node interpreter." >&2
  exit 2
fi

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

extra=()
# This launcher is a bounded technical canary by default.  Set the variable to
# an empty string only after the step-500 marker and registered 0/250/500 latent
# gate have passed and the matched-arm science run has been explicitly opened.
STOP_AFTER_UPDATES=${P11_STOP_AFTER_UPDATES-500}
if [[ -n "${STOP_AFTER_UPDATES}" ]]; then
  extra+=(--stop-after-updates "${STOP_AFTER_UPDATES}")
fi

{
  echo "P11_PYTHON=${PY}"
  timeout 90 "${PY}" -c 'import fitsio, h5py, jax, numpy, torch; print("P11_ENV_OK", jax.__version__, torch.__version__)'
  "${PY}" -m unittest \
    tests.phase4.test_p3br_prepare_r1_contract \
    tests.phase4.test_p11_jepa_canary \
    tests.phase4.test_p11_jepa_latent_diagnostics \
    tests.phase4.test_p11_factorial_training \
    tests.phase4.test_p11_factorial_view_contract
  "${PY}" -u -m workflows.abacus_tweb.p11_jepa_canary \
    --arm "${ARM}" \
    --seed "${SEED}" \
    --contract "${CONTRACT}" \
    --contract-root "${CONTRACT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --checkpoint-every 250 \
    --loss-log-every 25 \
    --latent-export-every 250 \
    --max-runtime-seconds 12600 \
    --validation-reserve-seconds 1200 \
    --auto-resume \
    "${extra[@]}"
} 2>&1 | tee -a "${LOG}"
