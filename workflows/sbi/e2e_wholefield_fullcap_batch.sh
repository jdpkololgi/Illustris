#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --account=desi_g
#SBATCH --time=03:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1/logs/batch-%j.log

# Production owner is slurmstepd, never an app-owned salloc/srun terminal.
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
segment=${1:?segment required}
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec srun --mpi=none --resv-ports=0 --kill-on-bad-exit=1 \
    -N1 -n1 -c32 --gpus=1 --cpu-bind=cores \
    /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u \
    /global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/e2e_wholefield_fullcap_batch.py \
    run --segment "$segment"
