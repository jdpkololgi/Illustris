#!/usr/bin/env bash
# Invoke as salloc ... srun ... bash <this committed launch script>.
# salloc owns the foreground srun lifetime; never infer completion from squeue.
set -eo pipefail
source /global/homes/d/dkololgi/miniforge3/bin/activate cosmic_env
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -u
cd /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3/source
python -m workflows.sbi.e2e_cfm_conditioning_sample --phase ph012 --cap SGC --replay
python -m workflows.sbi.e2e_cfm_conditioning_train train --arm control --seconds 1500
python -m workflows.sbi.e2e_cfm_conditioning_train train --arm augmented --seconds 1500
