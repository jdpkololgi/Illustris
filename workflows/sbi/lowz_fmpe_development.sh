#!/bin/bash
set -euo pipefail
cd "$EVAL_OUT/src"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun -N4 -n16 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 -c16 --kill-on-bad-exit=1 "$py" -um workflows.sbi.lowz_fmpe_development draw "$1"
srun -N1 -n1 -c16 --gpus=1 "$py" -um workflows.sbi.lowz_fmpe_development score "$1"
