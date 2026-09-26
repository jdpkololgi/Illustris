#!/usr/bin/env bash
# Submit with explicit batch resources; all stages execute the frozen snapshot.
set -euo pipefail
stage=$1
candidate_root=$2
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd "$candidate_root/source"
"$py" - "$candidate_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/'RUN_MANIFEST.json').read_text())
for name,expected in manifest['source_sha256'].items():
    if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('frozen source changed: '+name)
PY
if [[ "$stage" == export ]]; then
    phases=(ph000 ph002 ph003 ph004 ph005 ph006)
    phase=${phases[$SLURM_ARRAY_TASK_ID]}
    srun --ntasks=1 --cpus-per-task=32 --gpus=a100:1 "$py" -m workflows.sbi.p12a_halo48_pipeline export --root "$candidate_root" --phase "$phase"
else
    srun --ntasks=1 --cpus-per-task=32 --gpus=a100:1 "$py" -m workflows.sbi.p12a_halo48_pipeline "$stage" --root "$candidate_root"
fi
