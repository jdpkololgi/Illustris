#!/bin/bash
set -euo pipefail

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

METHOD="${1:?usage: run_p12f3_conditional_eval_pipeline.sh METHOD}"
case "${METHOD}" in
  conditional_gaussian_base3|conditional_gaussian_proxy7|conditional_gaussian_proxy7_shuffled|conditional_flow_proxy7) ;;
  *) echo "unsupported conditional rescue method: ${METHOD}" >&2; exit 2 ;;
esac

ROOT="${P12F3_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_conditional_calibration_v1}"
PY="${P12F3_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}"
ARCHIVE="${ROOT}/evaluation/${METHOD}/P12F_SAMPLE_ARCHIVE.json"
REPORT="${ROOT}/evaluation/reports/${METHOD}.json"
SHEAR="${ROOT}/evaluation/shear/${METHOD}.json"
VISUAL="${ROOT}/evaluation/visual/${METHOD}.json"

mkdir -p "${ROOT}/evaluation/reports" "${ROOT}/evaluation/shear" "${ROOT}/evaluation/visual"
"${PY}" -u -m workflows.sbi.p12f3_evaluate_conditional_archive \
  --archive "${ARCHIVE}" --output "${REPORT}"
"${PY}" -u -m workflows.sbi.p12f3l2_shear_audit \
  --archive-manifest "${ARCHIVE}" --output "${SHEAR}" --draw-batch 8
"${PY}" -u -m workflows.sbi.p12f3_conditional_visual_analyze \
  --archive "${ARCHIVE}" --output "${VISUAL}"
