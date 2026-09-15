#!/bin/bash
# Resource options and immutable paths are supplied by e2e_clean_limit_launch.py.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export SLURM_EXPORT_ENV=ALL
run_root=$1
stage=$2
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
if [[ "$stage" == optimization ]]; then
    exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
        "$python_bin" -m workflows.sbi.e2e_clean_limit train --root "$run_root" \
        --replica "$SLURM_ARRAY_TASK_ID" --arm optimization
elif [[ "$stage" == coverage ]]; then
    replica=$((SLURM_ARRAY_TASK_ID / 2))
    arm=control
    if (( SLURM_ARRAY_TASK_ID % 2 )); then arm=near_zero; fi
    exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
        "$python_bin" -m workflows.sbi.e2e_clean_limit train --root "$run_root" \
        --replica "$replica" --arm "$arm"
elif [[ "$stage" == analysis ]]; then
    exec srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
        "$python_bin" -m workflows.sbi.e2e_clean_limit analyze --root "$run_root"
else
    echo 'Unknown frozen stage' >&2
    exit 2
fi
