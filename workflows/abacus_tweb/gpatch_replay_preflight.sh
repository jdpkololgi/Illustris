#!/bin/bash
# Run inside the approved single-GPU step. Never mutate the historical run.
set -euo pipefail
: "${SLURM_JOB_ID:?requires allocation}" "${SLURM_STEP_ID:?requires compute step}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
base=/pscratch/sd/d/dkololgi/abacus
old="$base/p10_multiphase/arm_a_training/g_schedule_canary_lr2e4_v1/graph/seed_42"
dest="$base/gpatch_restart_$SLURM_JOB_ID"
mkdir "$dest"
exec >"$dest/preflight.log" 2>&1
repo=/global/u2/d/dkololgi/TNG/Illustris
rev=37bef3a1c972948863d0e632bc07f8beb32092fd
git clone -q --shared --no-checkout "$repo" "$dest/src"
cd "$dest/src"
git sparse-checkout init --cone
git sparse-checkout set shared workflows
git -c advice.detachedHead=false checkout -q --detach "$rev"
jq -r '.source_contract|to_entries[]|.value+"  "+.key' "$old/run_manifest.json" | sha256sum -c -
export OLD="$old" DEST="$dest"
"$py" - <<'PY'
import os,json,torch
from pathlib import Path
old,dest=map(Path,(os.environ['OLD'],os.environ['DEST']))
ck=torch.load(old/'arm_a_checkpoint.pt',map_location='cpu',weights_only=False)
best=torch.load(old/'best_checkpoint.pt',map_location='cpu',weights_only=False)
assert ck['global_step']==best['global_step']==253338 and ck['resume']['cursor']==84446
assert ck['scheduler_state']['T_max']==400000 and ck['scheduler_state']['last_epoch']==253338
assert all(torch.equal(v,best['state_dict'][k]) for k,v in ck['model_state'].items())
p=dest/'probe/graph/seed_42';p.mkdir(parents=True)
f=ck['frozen_arguments']
assert f==json.loads((old/'arm_a_summary.json').read_text())['frozen_arguments']
f.update(output_root=str(dest),run_name='probe',epochs=5,stop_after_updates=253838)
torch.save(ck,p/'arm_a_checkpoint.pt')
PY
"$py" -u -m workflows.abacus_tweb.p10_train_arm_a --model graph --seed 42 --lr 0.0002 --scheduler-total-updates 400000 --epochs 5 --min-epochs 1 --disable-early-stopping --validation-group-cores 4 --auto-resume --run-name probe --output-root "$dest" --stop-after-updates 253838 --max-runtime-seconds 3300 --validation-reserve-seconds 1200
d="$dest/probe/graph/seed_42"
jq -e '.pass and .epoch==4 and .cursor==500 and .global_step==253838' "$d/TECHNICAL_CANARY_COMPLETE.json"
jq -se 'last|.epoch==3 and (.primary_macro_r2_lambda1-0.46985837696817145|fabs)<0.00001' "$d/epoch_history.jsonl"
