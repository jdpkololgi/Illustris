#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
run_root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f_matched_challengers_v1
python=${P12F_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
log=${run_root}/p12f_calibration_plots_20260902.log

mkdir -p "${run_root}"
cd "${repo}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1

exec > >(tee -a "${log}") 2>&1

echo "[$(date -u +%FT%TZ)] requesting one interactive GPU"
salloc \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --constraint=gpu \
  --gpus=1 \
  --qos=interactive \
  --time=02:00:00 \
  --account=desi_g \
  --immediate=600 \
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gpus=1 \
    --gpu-bind=none \
    --cpu-bind=cores \
    --export=ALL \
    /bin/bash -lc "
      set -euo pipefail
      unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
      export PYTHONNOUSERSITE=1
      cd '${repo}'
      timeout 90 '${python}' -c \"import tarp, torch, numpy; print('P12F_RUNTIME_OK', torch.__version__)\"
      '${python}' -m unittest tests.phase4.test_plot_p12f_calibration_comparison
      '${python}' -u -m workflows.sbi.plot_p12f_calibration_comparison
    "

echo "[$(date -u +%FT%TZ)] calibration figures complete"
