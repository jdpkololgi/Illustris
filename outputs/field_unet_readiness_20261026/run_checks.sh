#!/bin/bash
set -euo pipefail
cd /global/homes/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
verify() {
  "$py" - <<'PY'
import hashlib,json
from pathlib import Path
for path, expected in json.loads(Path('outputs/field_unet_readiness_20261026/SOURCE_BINDINGS.json').read_text()).items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Source changed: '+path)
print('SOURCE_BINDINGS_VERIFIED',flush=True)
PY
}
verify
"$py" -m workflows.sbi.e2e_wholefield_unet_readiness --mode compute --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/field_unet_readiness_20261026_v1
"$py" -m workflows.sbi.p12a_field_training_readiness --mode compute --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/field_unet_readiness_20261026_v1/p12a_encoder
verify
