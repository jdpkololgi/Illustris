#!/bin/bash
# Authorized eight-fit launch, 2026-09-16. Run once from a persistent tmux shell.
# No automatic retry: inspect the intent receipt and Slurm after any failure.
set -euo pipefail
run_root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/preservation_objective_v1_20260916_58397904
batch_script="$run_root/source/workflows/sbi/submit_e2e_preservation.slurm"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
cd "$run_root/source"
/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -c 'import json; from pathlib import Path; from workflows.sbi.e2e_preservation_experiment import verify; from workflows.sbi import e2e_wide_pipeline as p; r=Path.cwd().parent; verify(r); s=json.loads((r/"SMOKE.json").read_text()); assert s["passed"] and s["manifest_sha256"]==p.sha256(r/"MANIFEST.json")'
# Exclusive output protects against accidental repeated launch, including after
# an ambiguous scheduler response. Never automatically remove this marker.
(set -o noclobber; date -u --iso-8601=seconds > "$run_root/LAUNCH_INTENT.txt")
train_id=$(sbatch --parsable --job-name=preservation-v1 --array=0-7%2 \
    --account=desi_g --constraint=gpu --qos=shared --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=32 --time=02:00:00 \
    --licenses=scratch --no-requeue --signal=USR1@180 \
    --chdir="$run_root/source" \
    --output="$run_root/logs/train_%A_%a.out" \
    --error="$run_root/logs/train_%A_%a.err" "$batch_script" "$run_root" train)
[[ "$train_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $train_id"; exit 1; }
printf '%s\n' "$train_id" | tee "$run_root/TRAIN_JOB_ID.txt"
report_id=$(sbatch --parsable --job-name=preservation-report \
    --account=desi --constraint=cpu --qos=debug \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:10:00 \
    --licenses=scratch --no-requeue --signal=USR1@180 \
    --dependency="afterok:$train_id" --chdir="$run_root/source" \
    --output="$run_root/logs/report_%j.out" \
    --error="$run_root/logs/report_%j.err" "$batch_script" "$run_root" report)
[[ "$report_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $report_id"; exit 1; }
printf '%s\n' "$report_id" | tee "$run_root/REPORT_JOB_ID.txt"
squeue -j "$train_id,$report_id" -o '%.20i %.25j %.10T %.10q %.10M %R'
printf 'Submitted eight fits and dependent analysis. Slurm survives loss of SSH or tmux.\n'
