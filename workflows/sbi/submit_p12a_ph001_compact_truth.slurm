#!/bin/bash
#SBATCH --job-name=p12a_p1_join
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --licenses=scratch
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=${P12A_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
TRUTH_ROOT=${P12A_TRUTH_ROOT:-/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1}
cd "$REPO"

timeout 120 "$PY" -c 'import fitsio,numpy; print("P12A_PH001_COMPACT_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage compact --truth-root "$TRUTH_ROOT"
if "$PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage compact --truth-root "$TRUTH_ROOT"; then
  echo "Authorized compact-truth stage is already complete and unchanged."
else
  status=$?
  [[ "$status" -eq 3 ]] || exit "$status"
  "$PY" -u -m workflows.sbi.p12a_authorized_truth build-compact \
    --truth-root "$TRUTH_ROOT"
fi
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-terminal \
  --truth-root "$TRUTH_ROOT"
