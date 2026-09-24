#!/bin/bash
#SBATCH --job-name=e2e_wiener
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --licenses=scratch

set -euo pipefail
# Submit with explicit --chdir frozen-source and --output persistent-log.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source /global/homes/d/dkololgi/miniforge3/bin/activate /pscratch/sd/d/dkololgi/conda/envs/cosmic_env
python -m unittest tests.test_e2e_abacus_wiener
python -u -m workflows.sbi.e2e_abacus_wiener \
  --root /pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1 \
  --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/abacus_wiener_20260924_v2/results \
  --workers 8
