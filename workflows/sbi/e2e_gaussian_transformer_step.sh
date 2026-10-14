#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == login* ]]; then
  echo 'Approved compute allocation required.' >&2
  exit 2
fi
module load pytorch/2.8.0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
diagnostic_python=/global/common/software/nersc9/pytorch/2.8.0/bin/python
diagnostic_root=/global/u2/d/dkololgi/diagnostics/gaussian_transformer_20261009_v1
diagnostic_mode="${1:?Choose smoke or run}"
case "$diagnostic_mode" in
  smoke|run) ;;
  *) echo 'Invalid mode' >&2; exit 2 ;;
esac
mkdir -p "$diagnostic_root"
set -o noclobber
exec > "$diagnostic_root/${diagnostic_mode}_${SLURM_JOB_ID}.log" 2>&1
cd /global/u2/d/dkololgi/TNG/Illustris
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
if [[ "$diagnostic_mode" == smoke ]]; then
  "$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_gaussian_transformer.py'
  "$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_gaussian_3d_cfm.py'
  "$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_gaussian_3d_audit.py'
fi
"$diagnostic_python" workflows/sbi/e2e_gaussian_transformer_run.py --mode "$diagnostic_mode" \
  --output "$diagnostic_root/$diagnostic_mode" --smoke "$diagnostic_root/smoke/SMOKE.json"
if [[ "$diagnostic_mode" == run ]]; then
  "$diagnostic_python" workflows/sbi/e2e_gaussian_transformer_validate.py --root "$diagnostic_root/run"
fi
