#!/bin/bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --licenses=scratch
#SBATCH --time=02:00:00
#SBATCH --job-name=e2e-ext-summary
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo 'Expected immutable postprocessor directory, science freeze, and postprocessor SHA256' >&2
    exit 2
fi
EXTERIOR_SUMMARY_SOURCE="$1"
EXTERIOR_SCIENCE_FREEZE="$2"
EXTERIOR_SUMMARY_SHA="$3"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
    python -u "$EXTERIOR_SUMMARY_SOURCE/e2e_exterior_field_summary.py" \
    --freeze "$EXTERIOR_SCIENCE_FREEZE" --source-sha "$EXTERIOR_SUMMARY_SHA"
