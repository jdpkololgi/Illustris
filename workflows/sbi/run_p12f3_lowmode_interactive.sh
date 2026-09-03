#!/bin/bash
set -euo pipefail

REPO=${P12F3_REPO:-/global/homes/d/dkololgi/TNG/Illustris}
PY=${P12F3_PYTHON:-/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python}
ARM=${P12F3_ARM:?set P12F3_ARM to local_h8 or wide_h24}
RUN_NAME=${P12F3_RUN_NAME:-seed42_v1}
STOP_AFTER=${P12F3_STOP_AFTER_UPDATES:-500}
MAX_WALL=${P12F3_MAX_WALL_SECONDS:-13200}
OUTPUT=${P12F3_OUTPUT_ROOT:-/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_hierarchical_lowmode_v1}

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
timeout 120 "$PY" -c 'import h5py,numpy,torch; assert torch.cuda.is_available(); print("P12F3_ENV_OK", numpy.__version__, torch.__version__, flush=True)'
"$PY" -m unittest -v tests.phase4.test_p12f3_hierarchical_lowmode

ARGS=(
  -u -m workflows.sbi.p12f3_train_lowmode_flow
  --arm "$ARM"
  --run-name "$RUN_NAME"
  --output-root "$OUTPUT"
  --device cuda
  --stop-after-updates "$STOP_AFTER"
  --max-wall-seconds "$MAX_WALL"
)
RUN_DIR="$OUTPUT/$RUN_NAME/$ARM"
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  ARGS+=(--resume)
fi
exec "$PY" "${ARGS[@]}"
