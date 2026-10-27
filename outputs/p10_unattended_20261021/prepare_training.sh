#!/bin/bash
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_unattended_20261021
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$py" "$campaign/verify_source.py"
cd "$campaign/source"
root=/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$py" -u - "$root" <<'PY'
import json,sys
from pathlib import Path
from workflows.abacus_tweb.p10_phase_assets import load_registry,sha256_file
registry=load_registry(Path('configs/p10_phase_registry_v2.json'));root=Path(sys.argv[1])
for phase,cfg in registry['phases'].items():
    name='BLIND_INPUT_COMPLETE.json' if cfg['role']=='sealed_blind' else 'PHASE_COMPLETE.json'
    receipt=json.loads((root/phase/name).read_text())
    if receipt['phase']!=phase or receipt.get('pass') is not True:raise ValueError('Incomplete phase '+phase)
    for name,path in receipt['artifacts'].items():
        if sha256_file(Path(path))!=receipt['artifact_sha256'][name]:raise ValueError('Stale phase product '+phase+'/'+name)
print('All 25 phase products are complete and current',flush=True)
PY
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$py" -u workflows/abacus_tweb/p10_prepare_training_contract.py --registry configs/p10_phase_registry_v2.json --root "$root" --contract-root "$root/training_contract_20261021" --stage all
