#!/usr/bin/env bash
set -euo pipefail

repo=/global/homes/d/dkololgi/TNG/Illustris
blind=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/blind_predictions/ph001
export_root=${blind}/p12a/production_export_v1
frozen=${blind}/P12_BLIND_PREDICTIONS_FROZEN.json
evaluation_contract=${repo}/docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json
log=${export_root}/logs/p12a_freeze_deep_supervisor_20260904.log
lock=${export_root}/logs/p12a_freeze_deep_supervisor_20260904.lock
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python

worker() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "Refusing P12-A freeze/deep replay outside Slurm" >&2
    exit 2
  }
  unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
  export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
  cd "${repo}"

  timeout 90 "${python}" -c \
    'import numpy, torch, sbi; print("P12A_CPU_RUNTIME_OK", numpy.__version__, torch.__version__, sbi.__version__)'
  "${python}" -m unittest discover -s tests/phase4 -p 'test_p12a*.py'
  "${python}" -m unittest \
    tests.phase4.test_p12_production_contract \
    tests.phase4.test_p12a_blind_shards

  if [[ ! -f "${frozen}" ]]; then
    "${python}" -u -m workflows.sbi.p12_freeze_blind_predictions \
      --candidate "${repo}/docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json" \
      --method-selection "${repo}/docs/evidence/p12/p12f_matched_v1/P12F_NO_FIELD_FINALIST.json" \
      --prediction-manifest "${blind}/p12a/ph001_p12a_base_context.json" \
      --prediction-manifest "${export_root}/P12A_BLIND_EXPORT_COMPLETE.json" \
      --prediction-manifest "${blind}/classical/cic_predictions.json" \
      --prediction-manifest "${blind}/classical/dtfe_predictions.json" \
      --deterministic-contract "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_BLIND_EVALUATION_FROZEN.json" \
      --output "${frozen}"
  else
    echo "P12-A frozen prediction marker already exists; validating rather than overwriting"
  fi

  if [[ ! -f "${evaluation_contract}" ]]; then
    "${python}" -u -m workflows.sbi.p12a_blind_evaluation_contract \
      --candidate "${repo}/docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json" \
      --gaussian-baseline "${repo}/docs/evidence/p12/production_aux_v1/P12A_GAUSSIAN_BASELINE.json" \
      --dataset-marker "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/P12A_DATASET_READY.json" \
      --output "${evaluation_contract}"
  else
    echo "P12-A evaluation contract already exists; validating rather than overwriting"
  fi

  P12A_FROZEN="${frozen}" P12A_EVALUATION_CONTRACT="${evaluation_contract}" \
    "${python}" -c '
import json
import os
from pathlib import Path
from workflows.sbi.p12a_open_blind import validate_evaluation_contract, validate_frozen_predictions

frozen = validate_frozen_predictions(Path(os.environ["P12A_FROZEN"]), deep=True)
contract = validate_evaluation_contract(Path(os.environ["P12A_EVALUATION_CONTRACT"]))
print("P12A_DEEP_TRUTH_FREE_REPLAY_OK")
print(json.dumps(frozen["_authorization_deep_validation"], indent=2, sort_keys=True))
print("P12A_EVALUATION_CONTRACT_OK", contract["schema_version"])
'
  echo "P12-A freeze, evaluation-contract build, and deep truth-free replay complete"
}

if [[ "${1:-}" == worker ]]; then
  worker
  exit 0
fi

mkdir -p "${export_root}/logs"
exec 9>"${lock}"
flock -n 9 || { echo "P12-A freeze/deep supervisor already running"; exit 3; }
exec > >(tee -a "${log}") 2>&1

echo "[$(date -u +%FT%TZ)] requesting one bounded CPU interactive allocation"
salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu \
  --qos=interactive --time=02:00:00 --account=desi \
  --licenses=scratch,u2 --immediate=600 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --export=ALL \
  "${BASH_SOURCE[0]}" worker
