#!/usr/bin/env bash
set -euo pipefail
campaign=/global/u2/d/dkololgi/TNG/Illustris/outputs/p10_expansion_pilot_20261020
stage=${1:?stage required}
exec > >(tee -a "$campaign/${stage}_attempt2.log") 2>&1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python - "$campaign" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=Path(sys.argv[1]);m=json.loads((c/'SOURCE_FREEZE_ATTEMPT2.json').read_text())
for name,digest in m['source_sha256'].items():
 if hashlib.sha256((Path(m['source_root'])/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed: '+name)
print('SOURCE_FREEZE_VERIFIED',len(m['source_sha256']),flush=True)
PY
printf 'STAGE_START stage=%s job=%s UTC=%s\n' "$stage" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
set +e
bash "$campaign/source_attempt2/workflows/abacus_tweb/run_p10_expansion_pilot.sh" "$stage"
status=$?
printf 'STAGE_EXIT stage=%s job=%s code=%s UTC=%s\n' "$stage" "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)"
exit "$status"
