#!/bin/bash
#SBATCH --job-name=p12a_p1_b
#SBATCH --account=desi
#SBATCH --qos=xfer
#SBATCH --constraint=cron
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --licenses=hpss,scratch
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_particle_b_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_particle_b_%j.err

# Authorized, single-opening HPSS restore.  htar is intentionally run in the
# xfer/cron queue: it is not available on Perlmutter compute nodes.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export PATH=/usr/bin:/bin

REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
TRUTH_ROOT=/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1
cd "$REPO"

timeout 120 "$PY" -c 'import asdf,numpy; print("P12A_PH001_PARTICLE_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_authorized_truth guard \
  --stage particle_b --truth-root "$TRUTH_ROOT"
if "$PY" -u -m workflows.sbi.p12a_authorized_truth status \
  --stage particle_b --truth-root "$TRUTH_ROOT"; then
  echo "Authorized Particle-B stage is already complete and unchanged."
  exit 0
else
  status=$?
  [[ "$status" -eq 3 ]] || exit "$status"
fi

attempt="$TRUTH_ROOT/attempts/particle_b/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$attempt"
"$PY" -u workflows/abacus_tweb/p10_stage_particle_b.py \
  --phase ph001 \
  --mode stage \
  --staging-root "$attempt/staging" \
  --out "$attempt/stage_result.json"
"$PY" -u -m workflows.sbi.p12a_authorized_truth complete-particle-b \
  --truth-root "$TRUTH_ROOT" --staging-root "$attempt/staging"
