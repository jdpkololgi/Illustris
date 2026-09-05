#!/bin/bash
#SBATCH --job-name=p12a_p1_join_precision
#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_precision_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_compact_precision_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/bin:/bin

P12_REPO=/global/homes/d/dkololgi/TNG/Illustris
P12_PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
P12_TRUTH_ROOT=/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1
cd "$P12_REPO"

timeout 120 "$P12_PY" -c 'import fitsio,numpy; print("P12A_COMPACT_PRECISION_RUNTIME_OK")'
"$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery validate
"$P12_PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage compact --truth-root "$P12_TRUTH_ROOT"
if "$P12_PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage compact --truth-root "$P12_TRUTH_ROOT"; then
  echo "Authorized compact-truth stage is already complete and unchanged."
else
  p12_status=$?
  [[ "$p12_status" -eq 3 ]] || exit "$p12_status"
  P12_ATTEMPT="$P12_TRUTH_ROOT/attempts/compact/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}-precision"
  mkdir -p "$P12_ATTEMPT"
  "$P12_PY" -u -m workflows.sbi.p12a_compact_precision_recovery recover \
    --output "$P12_ATTEMPT/ph001_p12a_truth.npz"
fi
"$P12_PY" -u -m workflows.sbi.p12a_authorized_truth complete-terminal \
  --truth-root "$P12_TRUTH_ROOT"
