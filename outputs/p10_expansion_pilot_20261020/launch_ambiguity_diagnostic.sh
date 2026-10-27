#!/usr/bin/env bash
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_expansion_pilot_20261020
exec > >(tee -a "$campaign/ambiguity_diagnostic.log") 2>&1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$python" - "$campaign" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=Path(sys.argv[1]);m=json.loads((c/'AMBIGUITY_DIAGNOSTIC_FREEZE.json').read_text())
for name,digest in m['source_sha256'].items():
 if hashlib.sha256((Path(m['source_root'])/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Diagnostic source changed: '+name)
print('DIAGNOSTIC_SOURCE_VERIFIED',flush=True)
PY
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores "$python" -u "$campaign/ambiguity_diagnostic_source/diagnose_ph007_particle_ambiguity.py" --output /pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph007/ambiguity_diagnostic.json
