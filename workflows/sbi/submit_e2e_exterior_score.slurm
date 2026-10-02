#!/bin/bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --licenses=scratch
#SBATCH --time=02:00:00
#SBATCH --job-name=e2e-ext-scores
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo 'Expected frozen source directory and freeze manifest' >&2
    exit 2
fi
EXTERIOR_SOURCE="$1"
EXTERIOR_FREEZE="$2"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
cd "$EXTERIOR_SOURCE"
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
    python -u -m workflows.sbi.e2e_exterior_score_batch --freeze "$EXTERIOR_FREEZE"
