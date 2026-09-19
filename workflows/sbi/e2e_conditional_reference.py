"""Bounded learned Gaussian reference. Run ONLY inside an approved allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch import nn

from workflows.sbi.e2e_direct_vdm import ConditionalVDM, LinearSchedule, sample, vlb
from workflows.sbi.e2e_conditional_reference_math import (
    problem, probes, oracle_moments, null_thresholds, metrics, qualify,
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def source_manifest(config_path):
    files = [Path(__file__), Path(__file__).with_name("e2e_conditional_reference_math.py"),
             Path(__file__).with_name("e2e_direct_vdm.py"), Path(config_path)]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class ExactModel(nn.Module):
    """Only for sampler verification, never used as a fitted neural model."""
    def __init__(self, case, objective):
        super().__init__()
        self.schedule = LinearSchedule(learned=False).double()
        self.objective = objective
        for key in ("values", "vectors", "mu"):
            self.register_buffer(key, torch.as_tensor(case[key], dtype=torch.float64))

    def forward(self, z, gamma, condition):
        flat = z.flatten(1).to(self.mu.dtype)
        if self.objective == "vdm":
            a,s = self.schedule.coefficients(gamma)
            residual = (flat-a[:,None]*self.mu) @ self.vectors
            prediction = s[:,None]*residual/(a[:,None]**2*self.values+s[:,None]**2)
            return (prediction @ self.vectors.T).reshape_as(z)
        t = (gamma-self.schedule.low)/self.schedule.slope.abs()
        residual = (flat-t[:,None]*self.mu) @ self.vectors
        A = (t[:,None]*self.values-(1-t[:,None]))/(t[:,None]**2*self.values+(1-t[:,None])**2)
        return (self.mu + (A*residual) @ self.vectors.T).reshape_as(z)


@torch.no_grad()
def sample_cfm(model, condition, nfe, generator):
    if nfe < 2 or nfe % 2:
        raise ValueError("positive even NFE required")
    mode = model.training
    model.eval()
    z = torch.randn(condition[:, :1].shape, device=condition.device,
                    dtype=condition.dtype, generator=generator)
    steps = nfe//2
    dt = 1/steps
    try:
        for i in range(steps):
            t = z.new_full((len(z),), i/steps)
            tt = z.new_full((len(z),), (i+1)/steps)
            first = model(z, model.schedule(t), condition)
            second = model(z+dt*first, model.schedule(tt), condition)
            z = z+dt/2*(first+second)
        if not torch.isfinite(z).all():
            raise FloatingPointError("nonfinite CFM path")
        return z
    finally:
        model.train(mode)


def train_step(model, optimizer, x, condition, objective, generator):
    optimizer.zero_grad(set_to_none=True)
    if objective == "vdm":
        loss, _ = vlb(model, x, condition, generator)
    elif objective == "cfm":
        t = (torch.arange(len(x), device=x.device)+torch.rand((),device=x.device,generator=generator))/len(x)
        noise = torch.randn(x.shape,device=x.device,generator=generator)
        tt = t[:,None,None,None,None]
        z = (1-tt)*noise+tt*x
        prediction = model(z, model.schedule(t), condition)
        loss = (prediction-(x-noise)).square().mean()
    else:
        raise ValueError(objective)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


class Data:
    def __init__(self, config, device):
        self.n = config["grid"]
        _, chol, mask, std, self.cases, _, self.radius = problem(self.n,config["cases"])
        self.q = probes(self.n)
        tensor = lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        self.chol, self.mask, self.std = map(tensor, (chol,mask,std))
        self.conditions = [tensor(np.stack([c["y"],c["mask"],c["std"]])).reshape(1,3,self.n,self.n,self.n)
                           for c in self.cases]
        self.means = [tensor(c["mu"]) for c in self.cases]
        self.factors = [tensor(c["chol"]) for c in self.cases]
        self.nulls = [null_thresholds(c,self.q,config["draws"],config["null_repeats"])
                      for c in self.cases]

    def batch(self, size, fixed_case, generator):
        eps = torch.randn((size,self.n**3),device=self.chol.device,generator=generator)
        if fixed_case is not None:
            x = eps @ self.factors[fixed_case].T + self.means[fixed_case]
            condition = self.conditions[fixed_case].expand(size,-1,-1,-1,-1)
        else:
            x = eps @ self.chol.T
            which = torch.randint(2,(size,),device=x.device,generator=generator)
            mask, std = self.mask[which], self.std[which]
            y = mask*(x+std*torch.randn(x.shape,device=x.device,generator=generator))
            condition = torch.stack([y,mask,std],dim=1).reshape(size,3,self.n,self.n,self.n)
        return x.reshape(size,1,self.n,self.n,self.n), condition


def controls(data, cfg):
    result = dict(oracle=[], metrics=[], nulls=data.nulls)
    rng = np.random.default_rng(419)
    for index, case in enumerate(data.cases):
        for objective in cfg["objectives"]:
            for nfe in cfg["oracle_nfe"]:
                out = oracle_moments(case,objective,nfe)
                out.pop("mean"); out.pop("variance")
                result["oracle"].append(dict(case=index,objective=objective,nfe=nfe,
                    passed=out["covariance_relative"] <= cfg["oracle_covariance_tolerance"],**out))
        epsilon = rng.normal(size=(cfg["draws"],len(case["mu"])))
        ensembles = dict(exact=case["mu"]+epsilon@case["chol"].T,
                         independent=case["mu"]+epsilon*np.sqrt(np.diag(case["sigma"])),
                         shrunk=case["mu"]+np.sqrt(.7)*(epsilon@case["chol"].T))
        for name, ensemble in ensembles.items():
            out = metrics(ensemble,case,data.q,data.radius)
            result["metrics"].append(dict(case=index,control=name,**out,
                                    **qualify(out,data.nulls[index],cfg)))
    return result


@torch.no_grad()
def evaluate(model, objective, data, cases, cfg, seed, update, outdir, guard):
    for case_index in cases:
        for nfe in cfg["nfe"]:
            path = outdir/f"evaluation_{update}_{case_index}_{nfe}.json"
            if path.exists():
                continue
            guard()
            generator = torch.Generator(device=data.chol.device).manual_seed(200000+seed*100+case_index)
            draws = []
            for first in range(0,cfg["draws"],cfg["evaluation_batch"]):
                guard()
                count = min(cfg["evaluation_batch"],cfg["draws"]-first)
                condition = data.conditions[case_index].expand(count,-1,-1,-1,-1)
                if objective == "vdm":
                    draw = sample(model,condition,nfe,generator)
                else:
                    draw = sample_cfm(model,condition,nfe,generator)
                draws.append(draw.flatten(1).cpu().numpy())
            draws = np.concatenate(draws)
            out = metrics(draws,data.cases[case_index],data.q,data.radius)
            np.save(outdir/f"draws_{update}_{case_index}_{nfe}.npy",draws)
            atomic_json(path,dict(case=case_index,objective=objective,seed=seed,
                                 update=update,nfe=nfe,**out,
                                 **qualify(out,data.nulls[case_index],cfg)))
            print("EVALUATED",outdir.name,update,case_index,nfe,round(out["mean_rms"],4),
                  round(out["covariance_relative"],4),flush=True)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("approved GPU allocation required; no login-node run")
    torch.set_num_threads(4)
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    sources = source_manifest(args.config)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sources"] != sources:
            raise RuntimeError("source/config changed since run started")
    else:
        atomic_json(manifest_path,dict(config=cfg,sources=sources,job=os.environ["SLURM_JOB_ID"],
                                      node=socket.gethostname(),started=time.time(),mode=args.mode))
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > cfg["deadline_seconds"]:
            raise TimeoutError("bounded reference deadline; no auto extension")
    device = torch.device("cuda")
    data = Data(cfg,device)
    if args.mode == "smoke":
        records = []
        for objective in cfg["objectives"]:
            torch.manual_seed(17)
            model = ConditionalVDM(3,cfg["base"],cfg["levels"],False).to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
            generator = torch.Generator(device=device).manual_seed(17)
            tic=time.monotonic()
            for _ in range(16):
                x,c = data.batch(cfg["batch"],None,generator)
                loss=train_step(model,optimizer,x,c,objective,generator)
            torch.cuda.synchronize()
            records.append(dict(objective=objective,seconds_per_update=(time.monotonic()-tic)/16,
                                loss=loss,parameters=sum(p.numel() for p in model.parameters())))
        atomic_json(output/"SMOKE.json",dict(records=records,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        print(records,flush=True)
        return
    control_path = output/"controls.json"
    if not control_path.exists():
        atomic_json(control_path,controls(data,cfg))
    # Complete each seed/objective/mode without model selection; all cases retained.
    for seed in cfg["seeds"]:
        for objective in cfg["objectives"]:
            for fixed in [None]+cfg["fixed_cases"]:
                guard()
                name=f"{objective}_seed{seed}_"+("amortised" if fixed is None else f"fixed{fixed}")
                outdir=output/name
                outdir.mkdir(exist_ok=True)
                torch.manual_seed(seed)
                model=ConditionalVDM(3,cfg["base"],cfg["levels"],False).to(device)
                optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
                generator=torch.Generator(device=device).manual_seed(seed+3000)
                checkpoint=outdir/"latest.pt"
                first=0
                if checkpoint.exists():
                    saved=torch.load(checkpoint,map_location=device,weights_only=False)
                    if saved["sources"] != sources:
                        raise RuntimeError("checkpoint source mismatch")
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    generator.set_state(saved["generator"].cpu())
                    torch.set_rng_state(saved["cpu_rng"].cpu())
                    torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
                    first=saved["update"]
                cases=range(cfg["cases"]) if fixed is None else [fixed]
                if first in cfg["checkpoints"]:
                    evaluate(model,objective,data,cases,cfg,seed,first,outdir,guard)
                tic=time.monotonic()
                losses=[]
                for update in range(first+1,cfg["updates"]+1):
                    guard()
                    x,condition=data.batch(cfg["batch"],fixed,generator)
                    losses.append(train_step(model,optimizer,x,condition,objective,generator))
                    if update % 256 == 0:
                        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                   generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
                                   cuda_rng=torch.cuda.get_rng_state(),update=update,sources=sources)
                        atomic_checkpoint(checkpoint,state)
                        with (outdir/"learning.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(update=update,mean_loss=float(np.mean(losses)),
                                                         elapsed_seconds=time.monotonic()-tic))+"\n")
                        print("TRAIN",name,update,float(np.mean(losses)),flush=True)
                        losses=[]
                    if update in cfg["checkpoints"]:
                        atomic_checkpoint(outdir/f"checkpoint_{update}.pt",state)
                        evaluate(model,objective,data,cases,cfg,seed,update,outdir,guard)
    expected=(len(cfg["fixed_cases"])+cfg["cases"])*len(cfg["seeds"])*len(cfg["objectives"])*len(cfg["checkpoints"])*len(cfg["nfe"])
    results=[json.loads(p.read_text())|{"fit":p.parent.name} for p in sorted(output.glob("*/evaluation_*.json"))]
    if len(results)!=expected:
        raise RuntimeError(f"missing evaluations: {len(results)} != {expected}")
    size=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size>cfg["scratch_cap_gib"]*1024**3:
        raise RuntimeError("Scratch cap exceeded")
    atomic_json(output/"COMPLETE.json",dict(evaluations=results,sources=sources,
        seconds=time.monotonic()-started,scratch_bytes=size,job=os.environ["SLURM_JOB_ID"],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),all_learned_gates_pass=all(r["passed"] for r in results)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=["smoke","run"],default="run")
    run(parser.parse_args())
