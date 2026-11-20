"""Controlled restart diagnostic: compare continuations of the same update-40 state."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(p):
    return torch.load(p,map_location="cpu",weights_only=False)


def worker(out, stop, every):
    # Run imports against the pinned FC0B407 checkout staged by the launcher.
    sys.path.insert(0,str(Path.cwd()))
    from workflows.abacus_tweb import p10_train_arm_a as trainer, lowz_litept_train as pt
    from workflows.abacus_tweb.lowz_point_loader import ROOT
    save = trainer.atomic_torch_save
    def capture(payload,path):
        save(payload,path)
        if path.name!="arm_a_checkpoint.pt":
            return
        n=payload["global_step"]
        if n in (40,48) and not (out/f"warm_{n}.pt").exists():
            shutil.copyfile(path,out/f"warm_{n}.pt")
        if n in (128,512):
            with (out/"throughput.jsonl").open("a") as f:
                print(json.dumps(dict(step=n,t=time.monotonic())),file=f)
    trainer.atomic_torch_save=capture
    sys.argv=["litept", "--model","unet","--seed","42","--epochs","4","--min-epochs","4","--ordered-prefetch",
              "--disable-early-stopping","--auto-resume","--checkpoint-every",str(every),"--loss-log-every","1",
              "--contract-root",str(ROOT),"--output-root",str(out),"--run-name","litept_s_fork",
              "--stop-after-updates",str(stop)]
    pt.main()


def metrics(x,y):
    import numpy as np
    if isinstance(x,dict):
        if x.keys()!=y.keys(): raise ValueError("different state keys")
        children=[metrics(v,y[k]) for k,v in x.items()]
    elif isinstance(x,(tuple,list)):
        if len(x)!=len(y): raise ValueError("different state lengths")
        children=[metrics(a,b) for a,b in zip(x,y)]
    elif isinstance(x,(torch.Tensor,np.ndarray)):
        a,b=torch.as_tensor(x),torch.as_tensor(y)
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("invalid compared tensor")
        if not a.is_floating_point():
            return int((a!=b).sum()),float((a!=b).any())
        return int((~torch.isclose(a,b,rtol=1e-5,atol=1e-7)).sum()),float((a-b).abs().max()) if a.numel() else 0.
    elif isinstance(x,float):
        if x==y: return 0,0.
        return int(not np.isclose(x,y,rtol=1e-5,atol=1e-7)),float(abs(x-y))
    else:
        return int(x!=y),float(x!=y)
    return sum(c[0] for c in children),max((c[1] for c in children),default=0.)


def run(out):
    torch.set_num_threads(1)
    old=Path("/pscratch/sd/d/dkololgi/abacus/lowz_litept_matched_61665541/litept_s_uninterrupted/unet/seed_42")
    d=out/"litept_s_fork/unet/seed_42"
    cp="arm_a_checkpoint.pt"
    trace="loss_trace.jsonl"
    d.mkdir(parents=True)
    state=load(old/cp)
    assert state["global_step"]==state["resume"]["cursor"]==32
    state["frozen_arguments"].update(run_name="litept_s_fork",output_root=str(out))
    torch.save(state,d/cp); del state
    shutil.copyfile(old/trace,d/trace)
    def advance(n,every):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"worker",str(out),str(n),str(every)],check=True)
        proof=json.loads((d/"TECHNICAL_CANARY_COMPLETE.json").read_text())
        assert proof["pass"] and proof["global_step"]==proof["cursor"]==n
    advance(48,8)
    shutil.copyfile(d/trace,out/"warm_trace.jsonl")
    assert (out/"warm_40.pt").is_file() and (out/"warm_48.pt").is_file()
    # Measure a sustained interval, excluding environment/loader startup.
    advance(512,128)
    marks=[json.loads(v) for v in (out/"throughput.jsonl").read_text().splitlines()]
    elapsed=next(x["t"] for x in marks if x["step"]==512)-next(x["t"] for x in marks if x["step"]==128)
    assert elapsed>0
    d.rename(out/"warm_throughput")
    source_hash=digest(out/"warm_40.pt")
    for branch in ("B","C"):
        d.mkdir()
        shutil.copyfile(out/"warm_40.pt",d/cp)
        assert digest(d/cp)==source_hash
        rows=[v for v in (out/"warm_trace.jsonl").read_text().splitlines() if json.loads(v)["global_step"]<=40]
        assert len(rows)==40
        (d/trace).write_text("\n".join(rows)+"\n")
        advance(48,8)
        d.rename(out/branch)
    a,b,c=(load(p) for p in (out/"warm_48.pt",out/"B"/cp,out/"C"/cp))
    pairs={}
    for name,s,t in (("warm_vs_B",a,b),("warm_vs_C",a,c),("cold_B_vs_C",b,c)):
        assert s["global_step"]==t["global_step"]==48
        pairs[name]={k:metrics(s[k],t[k]) for k in ("model_state","optimizer_state","scheduler_state",
            "resume","torch_rng_state","cuda_rng_state_all","shell_rows","objective_sum")}
    result=dict(source_40_sha256=source_hash,steps_total=496,timed_updates=384,timed_seconds=elapsed,
                observed_updates_per_second=384/elapsed,rtol=1e-5,atol=1e-7,
                comparisons=pairs,strict_pass=all(v[0]==0 for p in pairs.values() for v in p.values()))
    (out/"FORK_DIAGNOSTIC_COMPLETE.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    if "SLURM_STEP_ID" not in os.environ: raise SystemExit("GPU compute step required")
    sys.path.append("/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/lib/python3.11/site-packages")
    if sys.argv[1]=="worker": worker(Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
    elif sys.argv[1]=="run": run(Path(sys.argv[2]))
    else: raise SystemExit("unknown mode")
