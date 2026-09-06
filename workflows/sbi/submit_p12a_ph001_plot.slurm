#!/bin/bash
#SBATCH --job-name=p12a_p1_plot
#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --licenses=scratch
#SBATCH --export=NONE
#SBATCH --output=/pscratch/sd/d/dkololgi/logs/p12a_ph001_plot_%j.out
#SBATCH --error=/pscratch/sd/d/dkololgi/logs/p12a_ph001_plot_%j.err

set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PATH=/usr/bin:/bin MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO=/global/homes/d/dkololgi/TNG/Illustris
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
BLIND=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/blind_predictions/ph001
CONTRACT=$REPO/docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json
cd "$REPO"

timeout 120 "$PY" -c 'import matplotlib,numpy; print("P12A_PH001_PLOT_RUNTIME_OK")'
"$PY" -u -m workflows.sbi.p12a_plot_blind_evaluation \
  --evaluation-report "$BLIND/evaluation/P12A_PH001_BLIND_EVALUATION.json" \
  --evaluation-contract "$CONTRACT" \
  --output "$BLIND/evaluation/P12A_PH001_BLIND_EVALUATION.png" \
  --manifest "$BLIND/evaluation/P12A_PH001_BLIND_PLOTS.json"
