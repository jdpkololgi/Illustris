#!/bin/bash
#SBATCH --job-name=lowz-coarse17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&a100
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-coarse17-%j.log
set -euo pipefail
: "${CFM_ARCHIVE:?}" "${CFM_ARCHIVE_SHA:?}" "${CFM_BASE_REV:?}"
test "$(sha256sum "$CFM_ARCHIVE" | cut -d' ' -f1)" = "$CFM_ARCHIVE_SHA"
run=/pscratch/sd/d/dkololgi/abacus/coarse_source_$SLURM_JOB_ID
mkdir "$run"
cd "$run"
tar -xf "$CFM_ARCHIVE"
printf '%s %s %s\n' "$CFM_BASE_REV" "$CFM_ARCHIVE_SHA" "$CFM_ARCHIVE" > SOURCE.txt
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.lowz_coarse_fit --mode production
