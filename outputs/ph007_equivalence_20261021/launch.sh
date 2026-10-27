#!/usr/bin/env bash
set -euo pipefail
c=/global/u2/d/dkololgi/TNG/Illustris/outputs/ph007_equivalence_20261021
exec > >(tee -a "$c/allocation.log") 2>&1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python - "$c" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=Path(sys.argv[1]);m=json.loads((c/'SOURCE_FREEZE.json').read_text())
for name,h in m['source_sha256'].items():
 if hashlib.sha256((Path(m['source_root'])/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+name)
print('SOURCE_FREEZE_VERIFIED',len(m['source_sha256']),flush=True)
PY
printf 'CATALOGUE_START job=%s UTC=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
set +e
bash "$c/source/workflows/abacus_tweb/run_ph007_qualified_catalogue_20261021.sh"
status=$?
printf 'CATALOGUE_EXIT job=%s code=%s UTC=%s\n' "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)"
exit "$status"
