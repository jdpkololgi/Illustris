#!/bin/bash
# First production segments only; no requeue, no auto-retry.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --account=desi_g
#SBATCH --time=24:00:00
#SBATCH --licenses=scratch
#SBATCH --no-requeue
#SBATCH --output=/pscratch/sd/d/dkololgi/abacus/encoder-segment-%j.log
set -euo pipefail
arm=${1:?}
case "$arm" in graph|large_kernel|window_attention) ;; *) exit 2;; esac
repo=/global/u2/d/dkololgi/TNG/Illustris
r=/pscratch/sd/d/dkololgi/abacus
out="$r/encoder_prod_${arm}_${SLURM_JOB_ID:?}"
mkdir "$out"
git clone --quiet --no-checkout --shared "$repo" "$out/src"
cd "$out/src"
git sparse-checkout init --cone
git sparse-checkout set configs shared workflows
rev=71307f51349fd71263d5db8daa01e1d40ce7c010
if [[ $arm == graph ]]; then rev=37bef3a1c972948863d0e632bc07f8beb32092fd; fi
git -c advice.detachedHead=false checkout --quiet --detach "$rev"
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
if [[ $arm == graph ]]; then
    old="$r/gpatch_restart_61659392/probe/graph/seed_42"
    name=historical_extension
    model=graph
    module=workflows.abacus_tweb.p10_train_arm_a
    args=(--lr 0.0002 --scheduler-total-updates 400000 --epochs 5 --min-epochs 1
          --validation-group-cores 4 --stop-after-updates 400000)
else
    old="$r/p12a_grid_probe_61659884/sweep_${arm}_preflight/unet/seed_42"
    name="sweep_${arm}_production"
    model=unet
    module=workflows.abacus_tweb.p12a_encoder_sweep
    args=("$arm" --ordered-prefetch --epochs 4 --min-epochs 4
          --contract-root "$r/p12a_lowz_20261029_v1/training_contract_v1")
fi
d="$out/$name/$model/seed_42"
mkdir -p "$d"
jq -r '.source_contract|to_entries[]|.value+"  "+.key' "$old/run_manifest.json" | sha256sum --check --status
"$py" - "$old" "$d" "$out" "$name" "$arm" <<'PY'
import json,sys,torch
from pathlib import Path
old,d,out=map(Path,sys.argv[1:4])
name,arm=sys.argv[4:]
manifest=json.loads((old/'run_manifest.json').read_text())
ck=torch.load(old/'arm_a_checkpoint.pt',map_location='cpu',weights_only=False)
gate=json.loads((old/'TECHNICAL_CANARY_COMPLETE.json').read_text())
step,epoch,cursor=(253838,4,500) if arm=='graph' else (64,1,64)
assert gate['pass'] and gate['global_step']==ck['global_step']==ck['scheduler_state']['last_epoch']==step
assert gate['epoch']==ck['resume']['epoch']==epoch and gate['cursor']==ck['resume']['cursor']==cursor
assert ck['frozen_arguments']==manifest['frozen_arguments'] and ck['source_contract']==manifest['source_contract']
assert ck['scheduler_state']['T_max']==(400000 if arm=='graph' else 4*346364)
ck['frozen_arguments'].update(output_root=str(out),run_name=name)
if arm=='graph':
    assert abs(ck['history'][-1]['primary_macro_r2_lambda1']-.46985837696817145)<1e-5
    ck['frozen_arguments']['stop_after_updates']=400000
torch.save(ck,d/'arm_a_checkpoint.pt')
print("continuation",old,"step",step,"epoch",epoch,"cursor",cursor,flush=True)
PY
cp "$old/loss_trace.jsonl" "$d/"
if [[ $arm == graph ]]; then cp "$old"/best_* "$d/"; fi
# Budget pauses are preserved as exit 75, never interpreted as completion.
srun -N1 -n1 -c32 --exact --gpus=1 --cpu-bind=cores "$py" -u -m "$module" "${args[@]}" \
    --model "$model" --seed 42 --disable-early-stopping --auto-resume --run-name "$name" \
    --output-root "$out" --checkpoint-every 1000 --max-runtime-seconds 84600 --validation-reserve-seconds 6500
if [[ $arm == graph ]]; then
    jq -e '.pass and .epoch==5 and .global_step==400000' "$d/TECHNICAL_CANARY_COMPLETE.json"
else
    jq -e '.epochs_completed==4' "$d/ARM_A_TRAINING_COMPLETE.json"
fi
