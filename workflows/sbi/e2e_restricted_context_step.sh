#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == login* ]]; then
  echo 'Approved compute allocation required.' >&2
  exit 2
fi
module load pytorch/2.8.0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
diagnostic_python=/global/common/software/nersc9/pytorch/2.8.0/bin/python
diagnostic_root=/global/u2/d/dkololgi/diagnostics/restricted_context_cfm_20261009_v1
mkdir -p "$diagnostic_root"
set -o noclobber
exec > "$diagnostic_root/oracle_negativecases_v2_${SLURM_JOB_ID}.log" 2>&1
cd /global/u2/d/dkololgi/TNG/Illustris
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
"$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_restricted_context.py'
"$diagnostic_python" workflows/sbi/e2e_restricted_context_diagnostic.py --output "$diagnostic_root/oracle_negativecases_v2"
