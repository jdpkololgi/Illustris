#!/bin/bash
set -euo pipefail
cd "$PARITY_OUT/src"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR="$PARITY_OUT/mpl_$1"
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
srun -N4 -n16 --ntasks-per-node=4 -c16 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 "$py" -um workflows.sbi.lowz_epoch_parity infer "$1"
srun -N1 -n1 -c16 --gpus=1 "$py" -um workflows.sbi.lowz_epoch_parity score "$1"
if [[ "$1" == ph012 ]]; then
    srun -N1 -n1 -c16 --gpus=1 "$py" -um workflows.sbi.lowz_epoch_parity historical
fi
