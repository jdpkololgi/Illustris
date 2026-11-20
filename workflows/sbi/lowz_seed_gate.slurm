#!/bin/bash
#SBATCH --job-name=cfm-seed-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --licenses=scratch
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --array=0-1
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/lowz-seed-gate-%A-%a.log
set -euo pipefail
cd "${PH006_HELPERS:?}"
sha256sum -c SHA256SUMS
cd /pscratch/sd/d/dkololgi/abacus/coarse_source_61551798
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=16
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_lowz_20261029.json CUBLAS_WORKSPACE_CONFIG=:4096:8
run() {
  srun --exact -N1 -n1 -c32 --gpus=1 --cpu-bind=cores /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u -c 'import os,sys,runpy;sys.path.append(os.environ["PH006_HELPERS"]);name=sys.argv.pop(1);runpy.run_module(name,run_name="__main__")' "$@"
}
for stage in density tides score; do
  run lowz_ph006_cap_draw "$stage" SGC --phase ph006 --draw "$SLURM_ARRAY_TASK_ID"
done
run lowz_seed_gate "$SLURM_ARRAY_TASK_ID" "/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_cap_draw_$SLURM_JOB_ID/SGC"
