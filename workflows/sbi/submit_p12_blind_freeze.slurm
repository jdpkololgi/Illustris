#!/bin/bash
#SBATCH --job-name=p12_blind_freeze
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

set -euo pipefail
: "${P12A_CANDIDATE:?set the frozen P12-A candidate marker}"
: "${P12F_SELECTION:?set the frozen P12-F selection marker}"
: "${P12A_CONTEXT_MANIFEST:?set the truth-free base-context manifest}"
: "${P12A_EXPORT_COMPLETE:?set the complete P12-A export marker}"
: "${P12_CIC_MANIFEST:?set the truth-free CIC prediction manifest}"
: "${P12_DTFE_MANIFEST:?set the truth-free DTFE prediction manifest}"
: "${P10_DETERMINISTIC_CONTRACT:?set the frozen P10 deterministic contract}"
: "${P12_BLIND_FROZEN_OUTPUT:?set a new truth-free freeze-marker path}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32
PY=${P12_FREEZE_PYTHON:-/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901/bin/python}
REPO=/global/homes/d/dkololgi/TNG/Illustris
cd "$REPO"

timeout 90 "$PY" -c 'import numpy; print("P12_BLIND_FREEZE_RUNTIME_OK", numpy.__version__)'
"$PY" -m unittest -v \
  tests.phase4.test_p12_production_contract \
  tests.phase4.test_p12a_blind_shards

srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --export=ALL \
  "$PY" -u -m workflows.sbi.p12_freeze_blind_predictions \
  --candidate "$P12A_CANDIDATE" \
  --method-selection "$P12F_SELECTION" \
  --prediction-manifest "$P12A_CONTEXT_MANIFEST" \
  --prediction-manifest "$P12A_EXPORT_COMPLETE" \
  --prediction-manifest "$P12_CIC_MANIFEST" \
  --prediction-manifest "$P12_DTFE_MANIFEST" \
  --deterministic-contract "$P10_DETERMINISTIC_CONTRACT" \
  --output "$P12_BLIND_FROZEN_OUTPUT"

"$PY" -c "import json,pathlib; p=pathlib.Path('${P12_BLIND_FROZEN_OUTPUT}'); d=json.loads(p.read_text()); assert d['schema_version']=='p12-blind-predictions-frozen-v1' and d['pass'] and d['truth_files_read']==[] and d['open_count']==0 and not d['sealed_phase_opened']; print('P12_BLIND_PREDICTIONS_FROZEN', p)"
