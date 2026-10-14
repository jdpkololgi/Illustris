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
diagnostic_root=/global/u2/d/dkololgi/diagnostics/restricted_context_cfm_20261009_v1
pilot_mode="${1:?Choose smoke or train}"
case "$pilot_mode" in
  smoke|train) ;;
  *) echo 'Invalid pilot mode' >&2; exit 2 ;;
esac
mkdir -p "$diagnostic_root"
set -o noclobber
exec > "$diagnostic_root/cfm_${pilot_mode}_${SLURM_JOB_ID}.log" 2>&1
cd /global/u2/d/dkololgi/TNG/Illustris
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
"$diagnostic_python" -m unittest discover -s tests -p 'test_e2e_exterior_cfm.py'
"$diagnostic_python" workflows/sbi/e2e_exterior_cfm_pilot.py --mode "$pilot_mode" \
  --output "$diagnostic_root/cfm_$pilot_mode" \
  --oracle-receipt "$diagnostic_root/oracle_negativecases_v2/COMPLETE.json" \
  --smoke-receipt "$diagnostic_root/cfm_smoke/SMOKE.json"
