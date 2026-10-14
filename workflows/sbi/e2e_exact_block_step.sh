#!/usr/bin/env bash
# One foreground CPU step; existing output receipts are never overwritten.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == login* ]]; then
  echo 'Run inside an approved CPU srun step.' >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ "${E2E_EXACT_BLOCK_TEMP_RUNTIME:-0}" == 1 ]]; then
  module load pytorch/2.8.0
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
  diagnostic_python=/global/common/software/nersc9/pytorch/2.8.0/bin/python
  export PYTHONDONTWRITEBYTECODE=1
else
  source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
  diagnostic_python="$(command -v python)"
fi
cd /global/u2/d/dkololgi/TNG/Illustris
diagnostic_root="${E2E_EXACT_BLOCK_OUTPUT:-/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/exact_block_sampler_20261009_v1}"
mkdir -p "$diagnostic_root"
diagnostic_log="$diagnostic_root/run_${SLURM_JOB_ID}.log"
echo "Diagnostic log: $diagnostic_log"
set -o noclobber
exec > "$diagnostic_log" 2>&1
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
"$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_gaussian_block_sampler.py'
"$diagnostic_python" workflows/sbi/e2e_exact_block_diagnostic.py
