#!/bin/bash
# One explicitly approved 90-minute GPU pilot; Slurm persists beyond tmux/SSH.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 SLURM_EXPORT_ENV=ALL
run_root=$1
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$python_bin" -c 'import json,sys; from pathlib import Path; from workflows.sbi.e2e_direct_experiment import verify; from workflows.sbi import e2e_wide_pipeline as p; r=Path(sys.argv[1]); verify(r); s=json.loads((r/"SMOKE.json").read_text()); assert s["passed"] and s["exact_replay"] and s["manifest_sha256"]==p.sha256(r/"MANIFEST.json")' "$run_root"
exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
    "$python_bin" -u -m workflows.sbi.e2e_direct_experiment matrix --root "$run_root"
