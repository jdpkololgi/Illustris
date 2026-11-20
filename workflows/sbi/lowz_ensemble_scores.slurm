#!/bin/bash
#SBATCH --job-name=cfm-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-calibration-%j.log
set -euo pipefail
cd "${PH006_HELPERS:?}"
sha256sum -c SHA256SUMS
cd "${READOUT_HELPERS:?}"
sha256sum -c SHA256SUMS
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -c 'import os,sys,runpy;sys.path.extend(os.environ[k] for k in ("PH006_HELPERS","READOUT_HELPERS"));runpy.run_module("lowz_ensemble_scores",run_name="__main__")'
