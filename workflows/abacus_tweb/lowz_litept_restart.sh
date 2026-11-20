#!/bin/bash
# One interactive GPU step; 32 uninterrupted updates vs 16 + 16 (64 total).
set -euo pipefail
: "${SLURM_STEP_ID:?}" "${SLURM_JOB_ID:?}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 CUMM_CUDA_ARCH_LIST=8.0
r=/pscratch/sd/d/dkololgi/abacus
out="$r/lowz_litept_matched_$SLURM_JOB_ID"
mkdir "$out"
exec >"$out/check.log" 2>&1
git clone -q --shared --no-checkout /global/u2/d/dkololgi/TNG/Illustris "$out/src"
cd "$out/src"
git sparse-checkout init --cone
git sparse-checkout set shared workflows configs
git -c advice.detachedHead=false checkout -q --detach "${1:?pinned source}"
py="$r/lowz_litept_preflight_61662989/venv/bin/python"
export XDG_CACHE_HOME="$out/cache" CUDA_CACHE_PATH="$out/cache/cuda"
# Old factorial module imports astropy/fitsio at import time. Look in cosmic_env
# only after the full private LitePT site-packages, without altering either.
boot='import runpy,sys;sys.path.append("/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/lib/python3.11/site-packages");runpy.run_module("workflows.abacus_tweb.lowz_litept_train",run_name="__main__")'
common=(--model unet --seed 42 --epochs 4 --min-epochs 4 --ordered-prefetch
        --disable-early-stopping --checkpoint-every 16 --loss-log-every 1
        --contract-root "$r/p12a_lowz_20261029_v1/training_contract_v1" --output-root "$out")
"$py" -u -c "$boot" "${common[@]}" --run-name litept_s_uninterrupted --stop-after-updates 32
"$py" -u -c "$boot" "${common[@]}" --run-name litept_s_restart --stop-after-updates 16
"$py" -u -c "$boot" "${common[@]}" --run-name litept_s_restart --stop-after-updates 32 --auto-resume
"$py" - "$out" <<'PY'
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1])
def load(name):
    d=root/("litept_s_"+name)/"unet/seed_42"
    m=json.loads((d/"TECHNICAL_CANARY_COMPLETE.json").read_text())
    assert m["pass"] and m["cursor"]==m["global_step"]==32
    s=torch.load(d/"arm_a_checkpoint.pt",map_location="cpu",weights_only=False)
    assert s["resume"]["cursor"]==s["global_step"]==32
    traces=[json.loads(x) for x in (d/"loss_trace.jsonl").read_text().splitlines()]
    assert [v["global_step"] for v in traces]==list(range(1,33))
    return s,traces
a,ta=load("uninterrupted")
b,tb=load("restart")
assert a["scheduler_state"]==b["scheduler_state"]
assert a["source_contract"]==b["source_contract"]
assert a["resume"]["epoch"]==b["resume"]["epoch"]==1
def compare(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        for k,v in x.items(): compare(v,y[k])
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y)
        for a,b in zip(x,y): compare(a,b)
    elif isinstance(x,torch.Tensor):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
    else: assert x==y
for key in ("model_state","optimizer_state","torch_rng_state","cuda_rng_state_all"):
    compare(a[key],b[key])
for field in ("optimizer_objective_window_mean","preclip_gradient_norm_window_mean"):
    torch.testing.assert_close(torch.tensor([r[field] for r in ta]),
                               torch.tensor([r[field] for r in tb]),rtol=1e-5,atol=1e-7)
receipt=dict(passed=True,matched_updates=32,total_updates=64,split=16,
             rtol=1e-5,atol=1e-7,max_allocated_bytes=max(s["maximum_memory"] for s in (a,b)))
(root/"MATCHED_RESTART_PASS.json").write_text(json.dumps(receipt))
PY
