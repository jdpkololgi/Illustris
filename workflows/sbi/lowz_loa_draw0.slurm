#!/bin/bash
# One cap only; submit only after approval of that cap's budget and wall time.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-loa-draw0-%j.log
set -euo pipefail
[[ $# == 4 ]] || exit 2
case "$1" in NGC|SGC) cap="$1";; *) exit 2;; esac
draw=0
if [[ -v SLURM_ARRAY_TASK_ID ]]; then
  draw="$SLURM_ARRAY_TASK_ID"
  [[ "$draw" =~ ^([1-9]|[12][0-9]|3[01])$ ]] || exit 2
fi
runner=/global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/lowz_loa_draw.py
manifest="$2"
[[ "$3" =~ ^[a-f0-9]{64}$ && "$4" =~ ^[a-f0-9]{64}$ ]] || exit 2
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json CUBLAS_WORKSPACE_CONFIG=:4096:8
for stage in density tides; do
  echo "$3  $manifest" | sha256sum -c -
  echo "$4  $runner" | sha256sum -c -
  srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u "$runner" "$stage" "$cap" "$draw" "$manifest" "$3"
done
