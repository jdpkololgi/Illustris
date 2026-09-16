#!/bin/bash
# Submit explicitly only after the matching frozen SMOKE.json passes.
# Training: --array=0-7%2 -A desi_g -C gpu --qos=shared --gpus=1 -c32 -t02:00:00
# Report: -A desi -C cpu --qos=debug -c8 -t00:10:00 --dependency=afterok:<array>
# Both: -N1 -n1 --licenses=scratch --no-requeue --signal=USR1@180
# Use persistent --output/--error paths under the registered run's logs/.
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 SLURM_EXPORT_ENV=ALL
run_root=$1
mode=$2
cd "$run_root/source"
python_bin=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
"$python_bin" -c 'import json,sys; from pathlib import Path; from workflows.sbi.e2e_preservation_experiment import verify; from workflows.sbi import e2e_wide_pipeline as p; r=Path(sys.argv[1]); verify(r); s=json.loads((r/"SMOKE.json").read_text()); assert s["passed"] and s["manifest_sha256"]==p.sha256(r/"MANIFEST.json")' "$run_root"
if [[ "$mode" == train ]]; then
    arms=(control identity_weak identity_strong identity_response)
    task_index=${SLURM_ARRAY_TASK_ID:?array index required}
    if (( task_index < 0 || task_index > 7 )); then exit 2; fi
    exec srun --ntasks=1 --cpus-per-task=32 --gpus=1 --cpu-bind=cores \
        "$python_bin" -m workflows.sbi.e2e_preservation_experiment train --root "$run_root" \
        --replica "$((task_index / 4))" --arm "${arms[$((task_index % 4))]}"
elif [[ "$mode" == report ]]; then
    exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
        "$python_bin" -m workflows.sbi.e2e_preservation_experiment report --root "$run_root"
else
    exit 2
fi
