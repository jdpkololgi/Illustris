#!/bin/bash
# Four independent frozen model branches; resource request supplied explicitly.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 SLURM_EXPORT_ENV=ALL
run_root=$1
mode=${2:-run}
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$python_bin" -c 'import json,sys; from pathlib import Path; from workflows.sbi.e2e_vdm_assessment import verify; from workflows.sbi import e2e_wide_pipeline as p; r=Path(sys.argv[1]); verify(r); s=json.loads((r/"SMOKE.json").read_text()); assert s["passed"] and s["committed_chunk_replay"] and s["manifest_sha256"]==p.sha256(r/"MANIFEST.json")' "$run_root"
if [[ "$mode" == report ]]; then
    exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores "$python_bin" -u -m workflows.sbi.e2e_vdm_assessment_report --root "$run_root"
fi
branches=(fixed_vlb_seed0 learned_vlb_seed0 fixed_vlb_seed1 learned_vlb_seed1)
index=${SLURM_ARRAY_TASK_ID:?array task required}
if ((index<0 || index>3)); then exit 2; fi
exec srun --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores "$python_bin" -u \
    -m workflows.sbi.e2e_vdm_assessment run --root "$run_root" --branch "${branches[$index]}"
