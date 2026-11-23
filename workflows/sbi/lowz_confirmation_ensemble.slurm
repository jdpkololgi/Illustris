#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-blind-ensemble-%A-%a.log
set -euo pipefail
case "$1" in NGC|SGC) cap="$1";; *) exit 2;; esac
runner=/global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/lowz_confirmation_ensemble.py
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json CUBLAS_WORKSPACE_CONFIG=:4096:8
for stage in density tides; do
  echo "$2  $runner" | sha256sum -c -
  srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u "$runner" "$stage" "$cap"
done
