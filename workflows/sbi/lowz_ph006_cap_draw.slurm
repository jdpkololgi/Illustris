#!/bin/bash
#SBATCH --job-name=lowz-ph006-caps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-ph006-caps-%j.log
set -euo pipefail
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${PH006_HELPERS:?}"
sha256sum -c SHA256SUMS
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
run() {
  srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -c 'import os,sys,runpy;sys.path.append(os.environ["PH006_HELPERS"]);name=sys.argv.pop(1);runpy.run_module(name,run_name="__main__")' "$@"
}
date -u
run lowz_exterior_freeze --fit
for cap in SGC NGC; do
  for stage in density tides score; do
    date -u
    echo "$cap $stage"
    run lowz_ph006_cap_draw "$stage" "$cap"
  done
done
date -u
