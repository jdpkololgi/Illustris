#!/bin/bash
#SBATCH --job-name=lowz-unet17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --time=03:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-unet17-%j.log
set -eu
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
test "$(git rev-parse HEAD)" = "$FIT_REVISION"
test "$(sha256sum -c "$FIT_SOURCE_LIST" --status && sha256sum "$FIT_SOURCE_LIST" | cut -d' ' -f1)" = "$FIT_DIGEST"
exec srun -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -m workflows.sbi.lowz_unet_fit
