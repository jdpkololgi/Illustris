#!/bin/bash
#SBATCH --job-name=p12a_p1_join
#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/bin:/bin

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
TRUTH_ROOT=/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1
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
  attempt="$TRUTH_ROOT/attempts/compact/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
  mkdir -p "$attempt"
  "$PY" -u -m workflows.sbi.p12a_authorized_truth build-compact \
    --truth-root "$TRUTH_ROOT" --output "$attempt/ph001_p12a_truth.npz"
fi
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-terminal \
  --truth-root "$TRUTH_ROOT"
