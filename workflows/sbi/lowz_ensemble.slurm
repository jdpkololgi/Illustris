#!/bin/bash
#SBATCH --job-name=cfm-cap-draw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-cap-ensemble-%A-%a.log
set -euo pipefail
cd "${PH006_HELPERS:?}"
sha256sum -c SHA256SUMS
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json CUBLAS_WORKSPACE_CONFIG=:4096:8
phases=(ph006 ph012 ph013)
phase=${phases[SLURM_ARRAY_TASK_ID/32]}
draw=$((SLURM_ARRAY_TASK_ID%32))
case "$1" in NGC|SGC) cap=$1;; *) exit 2;; esac
for stage in density tides score; do
  srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -c 'import os,sys,runpy;sys.path.append(os.environ["PH006_HELPERS"]);runpy.run_module("lowz_ph006_cap_draw",run_name="__main__")' "$stage" "$cap" --phase "$phase" --draw "$draw"
done
