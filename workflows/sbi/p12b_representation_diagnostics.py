#!/usr/bin/env python3
"""Read-only internal diagnostics of the immutable P12-B terminal models."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.sbi.p12b_unet_representation import Pilot
from workflows.sbi.p12b_unet_representation_common import (
    safe_path, sha256, read_json, atomic_json, atomic_npz, finite,
    heun_sample, eigen_from_theta, row_scores, clustered_difference,
)


def norm(parameters):
    values = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def permuted_indices(cap, shell, seed):
    rng = np.random.default_rng(seed)
    index = np.arange(len(cap))
    for c in np.unique(cap):
        for s in np.unique(shell):
            at = np.flatnonzero((cap==c) & (shell==s))
            index[at] = rng.permutation(at)
    return index


def describe(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), quantiles=np.quantile(x,[0,.1,.5,.9,1]).tolist())


def readonly_pilot(config_path):
    """Reuse model functions without invoking the original writable runner."""
    p = object.__new__(Pilot)
    p.config_path = safe_path(config_path)
    p.config_sha = sha256(config_path)
    p.c = read_json(config_path)
    p.root = safe_path(p.c["output_root"])
    p.phase_root = safe_path(p.c["phase_root"])
    p.contract = p.phase_root / p.c["encoder_contract"]
    p.encoder_run = p.phase_root / p.c["encoder_run"]
    p.device = "cuda"
    p.cache = OrderedDict()
    p.ready = read_json(p.root / "P12B_DATA_READY.json")
    p.ready_sha = sha256(p.root / "P12B_DATA_READY.json")
    p.sources = p.source_contract()
    if p.ready["source_hashes"] != p.sources or p.ready["config_sha256"] != p.config_sha:
        raise RuntimeError("baseline source/config contract mismatch")
    if not p.ready["pass"] or p.c["ph001_access"]:
        raise RuntimeError("unsafe baseline")
    p.transforms = read_json(p.root / "transforms.json")
    if sha256(p.root/"transforms.json") != p.ready["transforms_sha256"]:
        raise RuntimeError("baseline transforms changed")
    for attr, key in (("x_mean","x_mean"),("x_std","x_std"),("y_mean","theta_mean"),("y_std","theta_std")):
        setattr(p, attr, torch.tensor(p.transforms[key],device=p.device,dtype=torch.float32))
    # Read only training/internal patch payloads. Never open ph006 or ph001 data.
    for a in p.ready["artifacts"]:
        if a["role"] in ("train","internal"):
            if a["phase"] != "ph005" or sha256(a["path"]) != a["sha256"]:
                raise RuntimeError("internal artifact identity changed")
    for path, expected in p.ready["input_hashes"].items():
        if str(p.encoder_run) in path or str(p.contract) in path:
            if sha256(path) != expected:
                raise RuntimeError("encoder provenance changed")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    out = safe_path(cfg["output_root"])
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU allocation required")
    if (out/"DIAGNOSTICS_COMPLETE.json").exists():
        raise FileExistsError("immutable completed diagnostic exists")
    if cfg["ph001_access"] or cfg["ph006_access"] or cfg["training_updates"] != 0:
        raise PermissionError("read-only ph005 diagnostic only")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = readonly_pilot(cfg["baseline_config"])
    frozen = dict(config_sha256=sha256(args.config),source_sha256=sha256(__file__),
        baseline_config_sha256=p.config_sha, baseline_data_sha256=p.ready_sha,
        baseline_source_hashes=p.sources,
        checkpoint_sha256={arm:sha256(p.root/arm/"checkpoint.pt") for arm in p.c["arms"]})
    if frozen["baseline_config_sha256"] != cfg["baseline_config_sha256"]:
        raise RuntimeError("registered baseline config changed")
    out.mkdir(parents=True,exist_ok=True)
    launch=out/"DIAGNOSTICS_FROZEN.json"
    if launch.exists() and read_json(launch)!=frozen:
        raise RuntimeError("diagnostic resume source changed")
    atomic_json(launch,frozen)
    internal = [a for a in p.ready["artifacts"] if a["role"]=="internal"]
    train_all = [a for a in p.ready["artifacts"] if a["role"]=="train"]
    probe = [train_all[i] for i in np.linspace(0,len(train_all)-1,cfg["train_probe_cores"],dtype=int)]
    report = dict(schema_version="p12b-representation-diagnostics-v1", frozen=frozen,
        allocation=os.environ["SLURM_JOB_ID"], ph001_access=False,ph006_access=False,
        training_updates=0,production_promotion_allowed=False,arms={})
    for arm in p.c["arms"]:
        encoder, flow, optimizer = p.models(arm)
        initial = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
        checkpoint = torch.load(p.root/arm/"checkpoint.pt",map_location=p.device,weights_only=False)
        if (checkpoint["arm"]!=arm or checkpoint["data_sha256"]!=p.ready_sha or
            checkpoint["config_sha256"]!=p.config_sha or checkpoint["source_hashes"]!=p.sources or
            checkpoint["update"]!=3000):
            raise RuntimeError("terminal checkpoint contract failed")
        flow.load_state_dict(checkpoint["flow"])
        if arm=="joint":
            encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval(); flow.eval()
        movement={}
        for key, value in encoder.state_dict().items():
            delta=float((value-initial[key]).double().square().sum())
            scale=float(initial[key].double().square().sum())
            movement[key]=dict(delta_l2=delta**.5,initial_l2=scale**.5,
                               relative_l2=(delta/max(scale,1e-30))**.5)
            if (arm!="joint" or key.startswith("head.")) and delta!=0:
                raise RuntimeError("frozen encoder/point head changed")
        trace=[json.loads(line) for line in (p.root/arm/"loss_trace.jsonl").read_text().splitlines()]
        trajectory=dict(logged_updates=len(trace),first25_loss=describe([r["loss"] for r in trace[:25]]),
            last25_loss=describe([r["loss"] for r in trace[-25:]]),
            logged_clipping_fraction=float(np.mean([r["gradient_norm"]>p.c["gradient_clip"] for r in trace])),
            last25_clipping_fraction=float(np.mean([r["gradient_norm"]>p.c["gradient_clip"] for r in trace[-25:]])),
            caveat="Sparse logged minibatches differ; this trace alone does not establish convergence")
        gradients=[]
        for a in probe:
            optimizer.zero_grad(set_to_none=True)
            patch=p.patch(a)
            context=p.condition(arm,encoder,patch)
            theta=(p.tensor(patch["theta"])-p.y_mean)/p.y_std
            torch.manual_seed(cfg["seed"]+10000+a["core"])
            loss=flow.loss(theta,context).mean()
            finite(loss);loss.backward()
            head_norm=norm(flow.parameters());encoder_norm=norm(encoder.unet.parameters())
            total=(head_norm**2+encoder_norm**2)**.5
            gradients.append(dict(core=a["core"],rows=len(theta),loss=float(loss.detach()),
                head_gradient_norm=head_norm,encoder_gradient_norm=encoder_norm,
                global_clip_multiplier=min(1.,p.c["gradient_clip"]/(total+1e-6))))
        optimizer.zero_grad(set_to_none=True)
        entries=[]
        with torch.no_grad():
            for a in internal:
                patch=p.patch(a)
                n=min(cfg["rows_per_core"],len(patch["parent"]))
                chosen=np.linspace(0,len(patch["parent"])-1,n,dtype=int)
                context=p.condition(arm,encoder,patch)[chosen]
                latent_initial=(p.tensor(patch["latent"])[chosen]-p.x_mean[7:])/p.x_std[7:]
                entries.append(dict(artifact=a,chosen=chosen,context=context.detach(),initial=latent_initial,
                    truth=patch["truth"][chosen],theta=patch["theta"][chosen],
                    cap=patch["cap"][chosen],shell=patch["shell"][chosen],
                    superblock=patch["superblock"][chosen],parent=patch["parent"][chosen]))
        context=torch.cat([e["context"] for e in entries])
        initial_latent=torch.cat([e["initial"] for e in entries])
        caps=np.concatenate([e["cap"] for e in entries]);shells=np.concatenate([e["shell"] for e in entries])
        permutation=permuted_indices(caps,shells,cfg["seed"])
        variants={"original":context,"zero":context.clone(),"permuted":context.clone()}
        variants["zero"][:,7:]=0
        variants["permuted"][:,7:]=context[torch.as_tensor(permutation,device=p.device),7:]
        if not torch.equal(variants["zero"][:,:7],context[:,:7]) or not torch.equal(variants["permuted"][:,:7],context[:,:7]):
            raise RuntimeError("feature ablation altered base/response")
        if arm=="point" and any(not torch.equal(c,context) for c in variants.values()):
            raise RuntimeError("point negative control is not exact")
        latent_stats=dict(standardized_rms_shift=float((context[:,7:]-initial_latent).square().mean().sqrt()),
            final_channel_mean=context[:,7:].mean(0).cpu().tolist(),
            final_channel_std=context[:,7:].std(0,unbiased=False).cpu().tolist(),
            initial_channel_std=initial_latent.std(0,unbiased=False).cpu().tolist()) if arm!="point" else {}
        collected={v:[] for v in variants};losses={v:[] for v in variants};checks={v:[] for v in variants}
        at=0;replay_max=0.
        for k,e in enumerate(entries):
            if time.monotonic()-started>cfg["max_seconds"]-120:
                raise TimeoutError("bounded diagnostic time exceeded; no automatic continuation")
            n=len(e["truth"]);a=e["artifact"]
            torch.manual_seed(42+200000+a["core"])
            noise=torch.randn(n*cfg["posterior_draws"],3,device=p.device)
            theta=(p.tensor(e["theta"])-p.y_mean)/p.y_std
            for variant, all_context in variants.items():
                x=all_context[at:at+n]
                condition=x.repeat_interleave(cfg["posterior_draws"],0)
                samples=heun_sample(flow,condition,noise,cfg["sample_steps"]).view(n,cfg["posterior_draws"],3)
                eigen=eigen_from_theta((samples*p.y_std+p.y_mean).cpu().numpy())
                scores=row_scores(eigen,e["truth"])
                if variant=="original":
                    baseline_path=p.root/arm/"evaluation"/f'internal_{a["core"]:06d}.npz'
                    marker=read_json(baseline_path.with_suffix(".json"))
                    if sha256(baseline_path)!=marker["sha256"]:
                        raise RuntimeError("baseline evaluation changed")
                    with np.load(baseline_path,allow_pickle=False) as data:
                        if not np.array_equal(data["parent"],e["parent"]):
                            raise RuntimeError("baseline evaluation rows differ")
                        replay_max=max(replay_max,float(np.max(np.abs(data["samples"]-eigen))))
                if k%16==0:
                    refined=heun_sample(flow,condition,noise,cfg["sample_check_steps"]).view_as(samples)
                    ref_eigen=eigen_from_theta((refined*p.y_std+p.y_mean).cpu().numpy())
                    ref_scores=row_scores(ref_eigen,e["truth"])
                    checks[variant].append(dict(rows=n,scaled_mean_abs=float((refined-samples).abs().mean()),
                        coverage_diff=np.concatenate([scores[f"coverage{q}"].mean(0)-ref_scores[f"coverage{q}"].mean(0) for q in (68,90)]).tolist()))
                with torch.no_grad():
                    loss=0.
                    for repeat in range(cfg["loss_noise_repeats"]):
                        torch.manual_seed(cfg["seed"]+500000+a["core"]*10+repeat)
                        loss+=float(flow.loss(theta,x).mean())/cfg["loss_noise_repeats"]
                    losses[variant].append(dict(core=a["core"],rows=n,loss=loss))
                collected[variant].append(scores)
                atomic_npz(out/arm/variant/f'core_{a["core"]:06d}.npz',parent=e["parent"],cap=e["cap"],
                    shell=e["shell"],superblock=e["superblock"],truth=e["truth"],**scores)
            at+=n
            if (k+1)%16==0:
                print(json.dumps(dict(arm=arm,internal_cores=k+1,total=len(entries))),flush=True)
        if replay_max>1e-6:
            raise RuntimeError(f"baseline posterior replay failed: {replay_max}")
        merged={v:{name:np.concatenate([part[name] for part in pieces]) for name in pieces[0]} for v,pieces in collected.items()}
        clusters=np.column_stack((caps,np.concatenate([e["superblock"] for e in entries])))
        scores_report={v:{name:np.mean(arr,axis=0).tolist() for name,arr in values.items() if name!="mean"} for v,values in merged.items()}
        paired={v:clustered_difference(merged[v]["energy"],merged["original"]["energy"],clusters,
                                      cfg["bootstrap_repetitions"],cfg["seed"]) for v in ("zero","permuted")}
        gate={}
        for v,parts in checks.items():
            weights=np.array([x["rows"] for x in parts])
            cov=np.average([x["coverage_diff"] for x in parts],axis=0,weights=weights)
            maximum=max(x["scaled_mean_abs"] for x in parts)
            gate[v]=dict(max_scaled_mean_abs=maximum,pooled_coverage_difference=cov.tolist(),
                pass_=bool(maximum<=.01 and np.max(np.abs(cov))<=.01))
            gate[v]["pass"]=gate[v].pop("pass_")
        if arm=="point" and any(not np.array_equal(merged[v]["energy"],merged["original"]["energy"]) for v in ("zero","permuted")):
            raise RuntimeError("point ablation negative control failed")
        result=dict(weight_movement=movement,trajectory=trajectory,terminal_gradient_probes=gradients,
            latent_drift=latent_stats,internal_rows=len(context),internal_cores=len(entries),
            scores=scores_report,paired_energy_minus_original=paired,internal_fm_loss=losses,
            sampler_gate=gate,baseline_posterior_replay_max_abs=replay_max)
        report["arms"][arm]=result
        atomic_json(out/arm/"DIAGNOSTICS.json",result)
        del encoder,flow,optimizer,checkpoint,initial,context,variants,entries
        torch.cuda.empty_cache()
    p.check_sources()
    if sha256(p.root/"P12B_DATA_READY.json")!=p.ready_sha:
        raise RuntimeError("baseline changed during diagnostics")
    for arm,expected in frozen["checkpoint_sha256"].items():
        if sha256(p.root/arm/"checkpoint.pt")!=expected:
            raise RuntimeError("immutable baseline checkpoint changed")
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        technical_completion=True,sampler_gate_pass=all(g["pass"] for a in report["arms"].values() for g in a["sampler_gate"].values()),
        interpretation="Ablations measure fitted-model reliance, not conditional information sufficiency; shifted contexts can be off-manifold")
    atomic_json(out/"DIAGNOSTICS_COMPLETE.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("arms","frozen")},indent=2),flush=True)


if __name__ == "__main__":
    main()
