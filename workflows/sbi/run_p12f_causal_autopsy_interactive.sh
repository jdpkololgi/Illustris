#!/usr/bin/env bash
set -euo pipefail

repo=${P12F_REPO:-/global/homes/d/dkololgi/TNG/Illustris}
py=${P12F_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
config=${P12F_CAUSAL_CONFIG:-${repo}/configs/p12f_causal_autopsy_v1.json}
output=${P12F_CAUSAL_OUTPUT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f_causal_autopsy_v1}
stage=${P12F_CAUSAL_STAGE:-all}
device=${P12F_CAUSAL_DEVICE:-cuda}
max_wall=${P12F_CAUSAL_MAX_WALL_SECONDS:-13500}

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

cd "${repo}"
timeout 90 "${py}" -c 'import numpy, torch, tarp, h5py, astropy; print("P12F_CAUSAL_ENV_OK", torch.__version__, torch.cuda.is_available(), flush=True)'
"${py}" -m unittest -v tests.phase4.test_p12f_causal_autopsy
"${py}" -u -m workflows.sbi.p12f_causal_autopsy \
  --config "${config}" \
  --output-root "${output}" \
  --stage "${stage}" \
  --device "${device}" \
  --max-wall-seconds "${max_wall}"
