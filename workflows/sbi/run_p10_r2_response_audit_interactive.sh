#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/homes/d/dkololgi/TNG/Illustris}"
PYTHON="${COSMIC_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}"
OUTPUT_ROOT="${P10_R2_AUDIT_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_response_audit_v1}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" -m py_compile \
  workflows/abacus_tweb/p10_audit_r2_response_ladder.py

exec "${PYTHON}" workflows/abacus_tweb/p10_audit_r2_response_ladder.py \
  --registry configs/p10_response_sources_v1.json \
  --output-root "${OUTPUT_ROOT}" \
  --workers 3
