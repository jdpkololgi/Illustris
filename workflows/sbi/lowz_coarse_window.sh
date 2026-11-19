#!/bin/bash
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
case "$1" in
  preflight|production) mode="$1" ;;
  *) exit 2 ;;
esac
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.lowz_coarse_fit --mode "$mode" >"/pscratch/sd/d/dkololgi/abacus/lowz-coarse-$mode-$SLURM_JOB_ID.log" 2>&1
