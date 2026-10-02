#!/bin/bash
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --licenses=scratch
#SBATCH --time=04:00:00
#SBATCH --job-name=e2e-exterior
# Array indices and durable log paths are supplied by the reviewed submission.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo 'Expected frozen source directory, freeze manifest, task manifest, task-manifest SHA256' >&2
    exit 2
fi
EXTERIOR_SOURCE="$1"
EXTERIOR_FREEZE="$2"
EXTERIOR_TASKS="$3"
EXTERIOR_TASK_SHA="$4"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=16
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
cd "$EXTERIOR_SOURCE"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores --export=ALL \
    python -u -m workflows.sbi.e2e_exterior_task \
    --freeze "$EXTERIOR_FREEZE" --tasks "$EXTERIOR_TASKS" --tasks-sha "$EXTERIOR_TASK_SHA" \
    --index "${SLURM_ARRAY_TASK_ID:?Missing array task index}"
