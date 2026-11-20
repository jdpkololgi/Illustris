#!/bin/bash
#SBATCH --job-name=lowz-graph-adapters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --time=06:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-graph-adapters-%j.log
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
r=/pscratch/sd/d/dkololgi/abacus
d="$r/lowz_graph_adapters_${SLURM_JOB_ID:?}"
mkdir "$d" "$d/src"
git -C /global/u2/d/dkololgi/TNG/Illustris archive "${1:?pinned source}" shared workflows configs | tar -x -C "$d/src"
cd "$d/src"
export NUMBA_CACHE_DIR="$d/cache"
mkdir "$NUMBA_CACHE_DIR"
echo "source=$1 job=$SLURM_JOB_ID"
srun --exact -N1 -n1 -c64 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u \
    -m workflows.abacus_tweb.lowz_graph_campaign "$d" "$r/lowz_graph_unions_61664731/ALL_UNIONS_READY.json" \
    "$r/lowz_patch_diagnostic_61666488/ph002"
