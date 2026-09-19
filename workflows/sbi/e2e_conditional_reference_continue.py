"""Continue frozen Gaussian reference fits without changing their training law."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from workflows.sbi.e2e_conditional_reference import (
    Data, atomic_checkpoint, atomic_json, evaluate, train_step,
)
from workflows.sbi.e2e_conditional_reference_math import null_thresholds
from workflows.sbi.e2e_direct_vdm import ConditionalVDM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_items(base):
    return [dict(seed=s,objective=o,fixed=f,
                 name=f"{o}_seed{s}_"+("amortised" if f is None else f"fixed{f}"))
            for s in base["seeds"] for o in base["objectives"]
            for f in [None]+base["fixed_cases"]]


def validate_extension(extension, base):
    if extension["schema"] != "conditional-reference-continuation-v1":
        raise ValueError("unknown continuation schema")
    if extension["start_update"] != base["updates"]:
        raise ValueError("continuation must start at parent final update")
    points=extension["checkpoints"]
    if points!=sorted(set(points)) or points[0]!=extension["start_update"] or points[-1]!=extension["updates"]:
        raise ValueError("invalid learning curve endpoints/order")
    if any(not isinstance(x,int) or isinstance(x,bool) or x%1024 for x in points):
        raise ValueError("checkpoint updates must be positive multiples of1024")
    if extension["updates"]<=extension["start_update"]:
        raise ValueError("not a continuation")
    if extension["workers"]!=4 or not 0<extension["deadline_seconds"]<=5100 or extension["scratch_cap_gib"]>20:
        raise ValueError("outside registered resource limits")
    if extension["precision_draws"]<base["draws"]:
        raise ValueError("precision ensemble must not shrink")
    for objective in base["objectives"]:
        nfe=extension["nfe_by_objective"][objective]
        if not nfe or nfe!=sorted(set(nfe)) or any(n<2 or n%2 for n in nfe):
            raise ValueError("invalid NFE ladder")
        if objective=="vdm" and min(nfe)<512:
            raise ValueError("VDM numerical floor requires at least512NFE")


def validate_state(saved, sources, update):
    if saved["sources"]!=sources or saved["update"]!=update:
        raise ValueError("checkpoint source/update mismatch")
    for key in ("model","optimizer","generator","cpu_rng","cuda_rng"):
        if key not in saved:
            raise ValueError(f"missing checkpoint state {key}")


def restore_training_state(model, optimizer, generator, saved):
    model.load_state_dict(saved["model"])
    # load_state_dict may otherwise alias same-device optimizer tensors and mutate
    # the in-memory checkpoint during the replay control.
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    generator.set_state(saved["generator"].cpu())
    torch.set_rng_state(saved["cpu_rng"].cpu())
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())


def assert_state_equal(actual, expected):
    """Exact value comparison independent of the checkpoint's storage device."""
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or not torch.equal(actual.cpu(),expected.cpu()):
            raise AssertionError("restored state tensor differs")
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():
            raise AssertionError("restored state keys differ")
        for key in expected:
            assert_state_equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):
            raise AssertionError("restored state sequence differs")
        for a,b in zip(actual,expected):
            assert_state_equal(a,b)
    elif actual!=expected:
        raise AssertionError("restored state value differs")


def prepare(args):
    parent=Path(args.parent).resolve()
    output=Path(args.output).resolve()
    if output==parent or parent in output.parents or output in parent.parents:
        raise ValueError("continuation must be a separate sibling run tree")
    extension=json.loads(Path(args.config).read_text())
    parent_manifest=json.loads((parent/"results/manifest.json").read_text())
    parent_done=json.loads((parent/"results/COMPLETE.json").read_text())
    if parent_done["sources"]!=parent_manifest["sources"]:
        raise ValueError("parent completion provenance differs")
    base=parent_manifest["config"]
    validate_extension(extension,base)
    repo=Path(__file__).resolve().parents[2]
    base_path=repo/"configs/e2e_conditional_reference_v1.json"
    if digest(base_path)!=extension["parent_config_sha256"] or json.loads(base_path.read_text())!=base:
        raise ValueError("immutable parent training configuration changed")
    files=[repo/"workflows/sbi"/name for name in
           ("e2e_conditional_reference.py","e2e_conditional_reference_math.py","e2e_direct_vdm.py")]+[base_path]
    for path in files:
        if digest(path)!=parent_done["sources"][path.name]:
            raise ValueError(f"parent scientific kernel changed: {path.name}")
    files += [Path(__file__),Path(args.config).resolve(),repo/"workflows/sbi/e2e_conditional_reference_continue_step.sh"]
    sources={str(p.relative_to(repo)):digest(p) for p in files}
    if (output/"manifest.json").exists():
        existing=json.loads((output/"manifest.json").read_text())
        if existing["sources"]!=sources or existing["extension"]!=extension or existing["parent"]!=str(parent):
            raise ValueError("refuse to overwrite a different continuation")
        verify_snapshot(output,existing)
        print("Existing frozen continuation verified",flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    for path in files:
        destination=output/"source"/path.relative_to(repo)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    items=fit_items(base)
    for item in items:
        path=parent/"results"/item["name"]/f'checkpoint_{extension["start_update"]}.pt'
        saved=torch.load(path,map_location="cpu",weights_only=False)
        validate_state(saved,parent_done["sources"],extension["start_update"])
        for group in saved["optimizer"]["param_groups"]:
            if group["lr"]!=base["learning_rate"]:
                raise ValueError("parent optimizer LR differs from registered value")
        dest=output/"parents"/item["name"]/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        item["checkpoint_sha256"]=digest(path)
        if digest(dest)!=item["checkpoint_sha256"]:
            raise ValueError("parent checkpoint copy failed")
    manifest=dict(extension=extension,base=base,parent=str(parent),parent_sources=parent_done["sources"],
                  parent_complete_sha256=digest(parent/"results/COMPLETE.json"),sources=sources,
                  items=items,prepared=time.time())
    atomic_json(output/"manifest.json",manifest)
    verify_snapshot(output,manifest)
    print("Prepared",len(items),"source-bound continuations",flush=True)


def verify_snapshot(output, manifest):
    for relative,expected in manifest["sources"].items():
        if digest(output/"source"/relative)!=expected:
            raise ValueError(f"frozen source changed: {relative}")
    for item in manifest["items"]:
        path=output/"parents"/item["name"]/f'checkpoint_{manifest["extension"]["start_update"]}.pt'
        if digest(path)!=item["checkpoint_sha256"]:
            raise ValueError("frozen parent changed")


def worker(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("approved GPU allocation required")
    if torch.cuda.device_count()!=1:
        raise RuntimeError("each independent task must see exactly one assigned GPU")
    if args.deterministic_smoke:
        if args.mode!="smoke":
            raise ValueError("deterministic override is a technical smoke control ONLY")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)
    output=Path(args.output).resolve()
    manifest=json.loads((output/"manifest.json").read_text())
    verify_snapshot(output,manifest)
    cfg,ext=manifest["base"],manifest["extension"]
    rank=int(os.environ.get("SLURM_PROCID","0"))
    if not 0<=rank<ext["workers"]:
        raise ValueError("worker rank outside frozen assignment")
    owned=manifest["items"][rank::ext["workers"]]
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>ext["deadline_seconds"]:
            raise TimeoutError("continuation deadline; no auto extension")
    data=Data(cfg,torch.device("cuda"))
    precise=None
    smoke=[]
    for item in owned:
        seed,objective,fixed=item["seed"],item["objective"],item["fixed"]
        outdir=output/"results"/item["name"]
        outdir.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(seed)
        model=ConditionalVDM(3,cfg["base"],cfg["levels"],False).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
        generator=torch.Generator(device="cuda").manual_seed(seed+3000)
        parent_path=output/"parents"/item["name"]/f'checkpoint_{ext["start_update"]}.pt'
        checkpoint=outdir/"latest.pt"
        if args.mode=="smoke" or not checkpoint.exists():
            saved=torch.load(parent_path,map_location="cuda",weights_only=False)
            validate_state(saved,manifest["parent_sources"],ext["start_update"])
        else:
            saved=torch.load(checkpoint,map_location="cuda",weights_only=False)
            validate_state(saved,manifest["sources"],saved["update"])
            if saved["fit"]!=item["name"] or saved["parent_checkpoint_sha256"]!=item["checkpoint_sha256"]:
                raise ValueError("continuation ancestry mismatch")
        restore_training_state(model,optimizer,generator,saved)
        if args.mode=="smoke":
            assert_state_equal(model.state_dict(),saved["model"])
            assert_state_equal(optimizer.state_dict(),saved["optimizer"])
            x,c=data.batch(cfg["batch"],fixed,generator)
            loss=train_step(model,optimizer,x,c,objective,generator)
            weights=copy.deepcopy(model.state_dict())
            restore_training_state(model,optimizer,generator,saved)
            xx,cc=data.batch(cfg["batch"],fixed,generator)
            if not torch.equal(x,xx) or not torch.equal(c,cc):
                raise AssertionError("restored data RNG differs")
            replay=train_step(model,optimizer,xx,cc,objective,generator)
            maximum=max(float((weights[k]-v).abs().max()) for k,v in model.state_dict().items())
            diagnostic=dict(fit=item["name"],loss=loss,replay_loss=replay,
                loss_difference=abs(loss-replay),parameter_maximum_difference=maximum,
                restored_state_exact=True,data_rng_exact=True,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            diagnostic_path=output/f'worker_{rank}_{item["name"]}_replay_{args.deterministic_smoke}.json'
            atomic_json(diagnostic_path,diagnostic)
            if maximum>1e-6 or abs(loss-replay)>1e-6:
                raise AssertionError(f"GPU checkpoint replay mismatch: {diagnostic}")
            tic=time.monotonic()
            for _ in range(32):
                x,c=data.batch(cfg["batch"],fixed,generator)
                train_step(model,optimizer,x,c,objective,generator)
            torch.cuda.synchronize()
            smoke.append(dict(fit=item["name"],seconds_per_update=(time.monotonic()-tic)/32,
                              replay_maximum=maximum,loss=loss,replay_loss=replay))
            continue
        first=saved["update"]
        cases=list(range(cfg["cases"])) if fixed is None else [fixed]
        evaluation=cfg|{"nfe":ext["nfe_by_objective"][objective]}
        if first in ext["checkpoints"]:
            evaluate(model,objective,data,cases,evaluation,seed,first,outdir,guard)
        tic=time.monotonic(); losses=[]
        for update in range(first+1,ext["updates"]+1):
            guard()
            x,condition=data.batch(cfg["batch"],fixed,generator)
            losses.append(train_step(model,optimizer,x,condition,objective,generator))
            if update%1024==0:
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                           generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state(),update=update,sources=manifest["sources"],
                           fit=item["name"],parent_checkpoint_sha256=item["checkpoint_sha256"])
                atomic_checkpoint(checkpoint,state)
                with (outdir/"learning.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(update=update,loss=float(np.mean(losses)),
                                                 elapsed_seconds=time.monotonic()-tic))+"\n")
                print("TRAIN",rank,item["name"],update,float(np.mean(losses)),flush=True)
                losses=[]
            if update in ext["checkpoints"]:
                atomic_checkpoint(outdir/f"checkpoint_{update}.pt",state)
                evaluate(model,objective,data,cases,evaluation,seed,update,outdir,guard)
        if precise is None:
            precise=copy.copy(data)
            precise.nulls=[null_thresholds(c,data.q,ext["precision_draws"],cfg["null_repeats"])
                           for c in data.cases]
        final_dir=outdir/"precision"
        final_dir.mkdir(exist_ok=True)
        precision_cfg=evaluation|{"draws":ext["precision_draws"],"nfe":[max(evaluation["nfe"])]}
        evaluate(model,objective,precise,cases,precision_cfg,seed,ext["updates"],final_dir,guard)
        atomic_json(final_dir/"nulls.json",precise.nulls)
    receipt=dict(rank=rank,job=os.environ["SLURM_JOB_ID"],node=socket.gethostname(),
                 seconds=time.monotonic()-started,mode=args.mode,owned=[x["name"] for x in owned],
                 sources=manifest["sources"],peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if args.mode=="smoke":
        receipt["smoke"]=smoke
    atomic_json(output/f'worker_{rank}_{args.mode.upper()}_COMPLETE.json',receipt)


def collect(args):
    output=Path(args.output).resolve()
    manifest=json.loads((output/"manifest.json").read_text())
    verify_snapshot(output,manifest)
    cfg,ext=manifest["base"],manifest["extension"]
    workers=[json.loads((output/f"worker_{r}_RUN_COMPLETE.json").read_text()) for r in range(ext["workers"])]
    if any(w["sources"]!=manifest["sources"] for w in workers):
        raise ValueError("worker sources disagree")
    rows=[]; precision=[]
    for item in manifest["items"]:
        cases=list(range(cfg["cases"])) if item["fixed"] is None else [item["fixed"]]
        root=output/"results"/item["name"]
        for update in ext["checkpoints"]:
            for case in cases:
                for nfe in ext["nfe_by_objective"][item["objective"]]:
                    row=json.loads((root/f"evaluation_{update}_{case}_{nfe}.json").read_text())
                    if (row["case"],row["objective"],row["seed"],row["update"],row["nfe"])!=(case,item["objective"],item["seed"],update,nfe):
                        raise ValueError("evaluation identity mismatch")
                    rows.append(row|{"fit":item["name"],"draws":cfg["draws"]})
        for case in cases:
            nfe=max(ext["nfe_by_objective"][item["objective"]])
            row=json.loads((root/"precision"/f'evaluation_{ext["updates"]}_{case}_{nfe}.json').read_text())
            if (row["case"],row["objective"],row["seed"],row["update"],row["nfe"])!=(case,item["objective"],item["seed"],ext["updates"],nfe):
                raise ValueError("precision evaluation identity mismatch")
            precision.append(row|{"fit":item["name"],"draws":ext["precision_draws"]})
    total=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if total>ext["scratch_cap_gib"]*1024**3:
        raise ValueError("continuation Scratch cap exceeded")
    atomic_json(output/"COMPLETE.json",dict(evaluations=rows,precision=precision,
        sources=manifest["sources"],workers=workers,scratch_bytes=total))
    print("COMPLETE",len(rows),"curve ensembles",len(precision),"precision ensembles",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["prepare","worker","collect"])
    p.add_argument("--output",required=True)
    p.add_argument("--parent")
    p.add_argument("--config")
    p.add_argument("--mode",choices=["smoke","run"],default="run")
    p.add_argument("--deterministic-smoke",action="store_true")
    args=p.parse_args()
    {"prepare":prepare,"worker":worker,"collect":collect}[args.command](args)
