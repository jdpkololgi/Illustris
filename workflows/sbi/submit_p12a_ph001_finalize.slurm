#!/bin/bash
#SBATCH --job-name=p12a_p1_open
#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_finalize_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_finalize_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PATH=/usr/bin:/bin
REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
BLIND=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/blind_predictions/ph001
CONTRACT=$REPO/docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json
cd "$REPO"

timeout 120 "$PY" -c 'import numpy; print("P12A_PH001_FINALIZE_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_open_blind finalize \
  --frozen-predictions "$BLIND/P12_BLIND_PREDICTIONS_FROZEN.json" \
  --evaluation-contract "$CONTRACT" \
  --authorization-marker "$BLIND/P12_BLIND_OPEN_AUTHORIZED.json" \
  --truth-complete "$BLIND/P12A_PH001_TRUTH_COMPLETE.json" \
  --output "$BLIND/P12_BLIND_OPENED.json"
