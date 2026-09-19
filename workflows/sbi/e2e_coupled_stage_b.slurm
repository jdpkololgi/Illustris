#!/bin/bash
#SBATCH --qos=xfer
#SBATCH --constraint=cron
#SBATCH --account=desi
#SBATCH --time=04:00:00
#SBATCH --job-name=coupled-b-transfer
#SBATCH --licenses=scratch
#SBATCH --mem=4G
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/b_transfer_%j.log
set -euo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$1"
exec python -u -m workflows.sbi.e2e_coupled_stage_b \
  --wait-for-pairing-seconds 13800 \
  --phases ph008 ph009 ph010 ph011 ph020 ph021 ph022 ph023 ph024 \
  ph012 ph013 ph014 ph015 ph016 ph017 ph018 ph019
