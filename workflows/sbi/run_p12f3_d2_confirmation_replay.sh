#!/usr/bin/env bash
# One explicitly supervised replay after allocation 57928395 timed out with no
# confirmation artifact. Same immutable source, selected weights, cores and seed.
# No automatic retries and no training occur in this wrapper.
set -euo pipefail
export D2_SOURCE_ROOT=/global/u2/d/dkololgi/TNG/Illustris_d2_467f442
export D2_OUTPUT_ROOT=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
if [[ "${1:-}" == worker ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]]
  bash "$D2_SOURCE_ROOT/workflows/sbi/run_p12f3_d2_in_allocation.sh" confirm
  "$PY" -c 'import json,os,sys; from pathlib import Path; p=Path(os.environ["D2_OUTPUT_ROOT"])/"D2_INTERNAL_CONFIRMATION.json"; d=json.loads(p.read_text()); print("D2 confirmation pass:",d.get("pass")); sys.exit(0 if d.get("pass") is True else 1)'
  exit
fi
[[ "$(git -C "$D2_SOURCE_ROOT" rev-parse HEAD)" == 467f442c5c54864658fdfaf948335d6e11a647fe ]]
[[ -z "$(git -C "$D2_SOURCE_ROOT" status --porcelain)" ]]
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
exec salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --constraint=gpu \
  --gpus=1 --qos=shared_interactive --time=02:00:00 --account=desi_g \
  --job-name=d2_confirm_replay --licenses=scratch --immediate=600 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
  bash "${BASH_SOURCE[0]}" worker
