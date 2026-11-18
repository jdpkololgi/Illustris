#!/bin/bash
set -eu
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
export PREFLIGHT_DIR=/pscratch/sd/d/dkololgi/abacus/lowz_gpu_preflight_${SLURM_JOB_ID}
mkdir "$PREFLIGHT_DIR"
exec >"$PREFLIGHT_DIR/window.log" 2>&1
gpu_step() {
  srun -N1 -n1 -c32 --gpus=1 --exact --cpu-bind=cores timeout 1500 /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.lowz_gpu_preflight_20261118 "$1"
}
gpu_step cfm
gpu_step p12
