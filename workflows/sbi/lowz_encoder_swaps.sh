#!/bin/bash
set -euo pipefail
cd "$SWAP_OUT/src"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun --kill-on-bad-exit=1 -N4 -n16 --ntasks-per-node=4 -c16 --gpus-per-task=1 --gpu-bind=single:1 "$py" -um workflows.sbi.lowz_encoder_swaps run "$1"
srun -N1 -n1 -c16 --gpus=1 "$py" -um workflows.sbi.lowz_encoder_swaps score "$1"
