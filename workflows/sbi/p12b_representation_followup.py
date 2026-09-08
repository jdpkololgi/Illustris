#!/usr/bin/env python3
"""Registered matched-budget continuation/warm-start/clipping investigation."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from workflows.sbi.p12b_representation_diagnostics import readonly_pilot, norm
from workflows.sbi.p12b_unet_representation_common import (
    safe_path,read_json,sha256,atomic_json,atomic_npz,atomic_torch,finite,
    heun_sample,heun_log_prob,eigen_from_theta,physical_log_jacobian,row_scores,clustered_difference)


def clip_groups(optimizer, policy, threshold):
    groups=[list(g["params"]) for g in optimizer.param_groups]
    norms=[norm(g) for g in groups]
    if policy=="global":
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g],threshold,error_if_nonfinite=True)
    elif policy=="separate":
        for group in groups:
            torch.nn.utils.clip_grad_norm_(group,threshold,error_if_nonfinite=True)
    else:
        raise ValueError(policy)
    return norms


def initialize(p, arm):
    source=arm["parent_arm"]
    encoder,flow,optimizer=p.models(source)
    state=torch.load(p.root/source/"checkpoint.pt",map_location=p.device,weights_only=False)
    if (state["update"]!=3000 or state["data_sha256"]!=p.ready_sha or
        state["source_hashes"]!=p.sources or state["config_sha256"]!=p.config_sha or state["arm"]!=source):
        raise RuntimeError("parent terminal checkpoint changed")
    flow.load_state_dict(state["flow"])
    if source=="joint":encoder.load_state_dict(state["encoder"])
    optimizer.load_state_dict(state["optimizer"])
    if arm["mode"]=="joint" and source=="frozen":
        encoder.unet.requires_grad_(True)
        optimizer.add_param_group({"params":encoder.unet.parameters(),"lr":p.c["encoder_learning_rate"]})
    encoder.head.requires_grad_(False)
    encoder.train(arm["mode"]=="joint");flow.train()
    return encoder,flow,optimizer,state["row_presentations"]


def step(p,arm,encoder,flow,optimizer,patch,update):
    optimizer.zero_grad(set_to_none=True)
    context=p.condition(arm["mode"],encoder,patch)
    theta=(p.tensor(patch["theta"])-p.y_mean)/p.y_std
    torch.manual_seed(p.c["seed"]+100000+update)
    loss=flow.loss(theta,context).mean();finite(loss);loss.backward()
    norms=clip_groups(optimizer,arm["clipping"],p.c["gradient_clip"])
    optimizer.step()
    finite(*[v.detach() for group in optimizer.param_groups for v in group["params"]])
    return float(loss.detach()),norms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();cfg=read_json(args.config)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required")
    if cfg["ph001_access"] or cfg["ph006_access"] or cfg["total_updates"]!=6000:
        raise PermissionError("registered internal-only budget changed")
    started=time.monotonic();deadline=started+cfg["max_seconds"]
    out=safe_path(cfg["output_root"]);out.mkdir(parents=True,exist_ok=True)
    if (out/"FOLLOWUP_COMPARISON.json").exists():
        raise FileExistsError("completed follow-up is immutable")
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=readonly_pilot(cfg["baseline_config"])
    diagnostics=read_json(cfg["diagnostics_report"])
    if not diagnostics["technical_completion"] or not diagnostics["sampler_gate_pass"]:
        raise RuntimeError("initial diagnostic gate failed")
    source_hashes={str(Path(__file__).resolve()):sha256(__file__),
        str(REPO/"workflows/sbi/p12b_representation_diagnostics.py"):sha256(REPO/"workflows/sbi/p12b_representation_diagnostics.py")}
    frozen=dict(config_sha256=sha256(args.config),baseline_data_sha256=p.ready_sha,
        source_hashes=source_hashes,baseline_source_hashes=p.sources,
        baseline_config_sha256=p.config_sha,diagnostics_sha256=sha256(cfg["diagnostics_report"]),
        parents={a:sha256(p.root/a/"checkpoint.pt") for a in p.c["arms"]})
    if frozen["parents"]!=diagnostics["frozen"]["checkpoint_sha256"] or p.ready_sha!=diagnostics["frozen"]["baseline_data_sha256"]:
        raise RuntimeError("diagnostic baseline parent identities changed")
    frozen_path=out/"FOLLOWUP_FROZEN.json"
    if frozen_path.exists() and read_json(frozen_path)!=frozen:
        raise RuntimeError("follow-up contract changed")
    atomic_json(frozen_path,frozen)
    train=[a for a in p.ready["artifacts"] if a["role"]=="train"]
    internal=[a for a in p.ready["artifacts"] if a["role"]=="internal"]
    rng=np.random.default_rng(p.c["seed"]);schedule=[]
    while len(schedule)<cfg["total_updates"]:schedule.extend(rng.permutation(len(train)).tolist())
    expected_first=sum(train[schedule[i]]["rows"] for i in range(3000))
    expected_total=sum(train[schedule[i]]["rows"] for i in range(cfg["total_updates"]))

    def check_sources():
        p.check_sources()
        if sha256(args.config)!=frozen["config_sha256"]:
            raise RuntimeError("follow-up config changed")
        for path,h in source_hashes.items():
            if sha256(path)!=h:raise RuntimeError("follow-up source changed")

    def save(arm,encoder,flow,optimizer,update,rows):
        check_sources()
        atomic_torch(out/arm["name"]/"checkpoint.pt",dict(schema_version="p12b-followup-checkpoint-v1",
            frozen=frozen,arm=arm,encoder=encoder.state_dict() if arm["mode"]=="joint" else None,
            flow=flow.state_dict(),optimizer=optimizer.state_dict(),update=update,row_presentations=rows))

    smoke=[]
    for arm in cfg["arms"]:
        e,f,o,rows=initialize(p,arm)
        if rows!=expected_first:raise RuntimeError("original training exposure mismatch")
        patch=p.patch(train[schedule[3000]])
        with torch.no_grad():
            before=p.condition(arm["mode"],e,patch)
            if arm["parent_arm"]=="frozen" and arm["mode"]=="joint":
                cached=p.condition("frozen",e,patch)
                if float((before-cached).abs().max())>1e-6:
                    raise RuntimeError("warm-start cached/online feature parity failed")
        initial_head={k:v.clone() for k,v in e.head.state_dict().items()}
        loss,norms=step(p,arm,e,f,o,patch,3000)
        if arm["mode"]=="joint" and norms[1]<=0:raise RuntimeError("joint gradient absent")
        for k,v in e.head.state_dict().items():
            if not torch.equal(v,initial_head[k]):raise RuntimeError("old deterministic head changed")
        e2,f2,o2,_=initialize(p,arm);step(p,arm,e2,f2,o2,patch,3000)
        replay=max(float((v-f2.state_dict()[k]).abs().max()) for k,v in f.state_dict().items())
        replay=max(replay,max(float((v-e2.state_dict()[k]).abs().max()) for k,v in e.state_dict().items()))
        if replay>1e-6:raise RuntimeError("follow-up replay tolerance failed")
        smoke.append(dict(arm=arm["name"],loss=loss,preclip_norms=norms,replay_max_abs=replay))
        del e,f,o,e2,f2,o2
    atomic_json(out/"FOLLOWUP_SMOKE_PASS.json",dict(pass_=True,arms=smoke,frozen=frozen))
    print(json.dumps(dict(event="smoke_pass",arms=smoke)),flush=True)

    for arm in cfg["arms"]:
        directory=out/arm["name"];directory.mkdir(exist_ok=True)
        e,f,o,rows=initialize(p,arm);update=3000
        checkpoint=directory/"checkpoint.pt"
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location=p.device,weights_only=False)
            if state["frozen"]!=frozen or state["arm"]!=arm:raise RuntimeError("follow-up resume changed")
            f.load_state_dict(state["flow"]);o.load_state_dict(state["optimizer"])
            if arm["mode"]=="joint":e.load_state_dict(state["encoder"])
            update=state["update"];rows=state["row_presentations"]
        trace=(directory/"loss_trace.jsonl").open("a")
        while update<cfg["total_updates"]:
            if time.monotonic()>deadline-120:
                save(arm,e,f,o,update,rows);trace.close();return 75
            a=train[schedule[update]];loss,norms=step(p,arm,e,f,o,p.patch(a),update)
            update+=1;rows+=a["rows"]
            if update%100==0:
                row=dict(arm=arm["name"],update=update,loss=loss,preclip_norms=norms,row_presentations=rows)
                trace.write(json.dumps(row)+"\n");trace.flush();print(json.dumps(row),flush=True)
            if update%500==0:save(arm,e,f,o,update,rows)
        trace.close()
        if rows!=expected_total:raise RuntimeError("follow-up matched exposure failed")
        save(arm,e,f,o,update,rows)
        atomic_json(directory/"TRAINING_COMPLETE.json",dict(arm=arm,frozen=frozen,updates=update,
            row_presentations=rows,new_updates=3000,checkpoint_sha256=sha256(checkpoint)))
        del e,f,o
        torch.cuda.empty_cache()

    report=dict(schema_version="p12b-followup-comparison-v1",frozen=frozen,arms={},paired={},
        phase="ph005",ph006_access=False,ph001_access=False,production_promotion_allowed=False,
        conditional_calibration_pass=False,allocation=os.environ["SLURM_JOB_ID"],
        exposure=dict(total_updates=6000,new_updates=3000,row_presentations=expected_total))
    data={};clusters=None
    for arm in cfg["arms"]:
        e,f,o,_=initialize(p,arm)
        path=out/arm["name"]/"checkpoint.pt";state=torch.load(path,map_location=p.device,weights_only=False)
        f.load_state_dict(state["flow"])
        if arm["mode"]=="joint":e.load_state_dict(state["encoder"])
        e.eval();f.eval();parts=[];groups=[];checks=[];base_energy=[]
        for index,a in enumerate(internal):
            if time.monotonic()>deadline-120:return 75
            patch=p.patch(a);n=min(16,len(patch["parent"]));chosen=np.linspace(0,len(patch["parent"])-1,n,dtype=int)
            with torch.no_grad():context=p.condition(arm["mode"],e,patch)[chosen]
            condition=context.repeat_interleave(128,0)
            torch.manual_seed(42+200000+a["core"]);noise=torch.randn(n*128,3,device=p.device)
            samples=heun_sample(f,condition,noise,64).view(n,128,3)
            eigen=eigen_from_theta((samples*p.y_std+p.y_mean).cpu().numpy());truth=patch["truth"][chosen]
            scores=row_scores(eigen,truth)
            m=min(2,n);theta=(p.tensor(patch["theta"][chosen[:m]])-p.y_mean)/p.y_std
            logp=heun_log_prob(f,context[:m],theta,64).cpu().numpy()+physical_log_jacobian(truth[:m],p.transforms["theta_std"])
            check={}
            if index%16==0:
                refined=heun_sample(f,condition,noise,128).view_as(samples)
                ref_scores=row_scores(eigen_from_theta((refined*p.y_std+p.y_mean).cpu().numpy()),truth)
                ref_logp=heun_log_prob(f,context[:m],theta,128).cpu().numpy()+physical_log_jacobian(truth[:m],p.transforms["theta_std"])
                check=dict(rows=n,scaled_mean_abs=float((samples-refined).abs().mean()),
                    coverage_difference=np.concatenate([scores[f"coverage{q}"].mean(0)-ref_scores[f"coverage{q}"].mean(0) for q in (68,90)]).tolist(),
                    log_score_mean_abs=float(np.abs(logp-ref_logp).mean()))
                checks.append(check)
            parent=patch["parent"][chosen]
            arr=dict(parent=parent,cap=patch["cap"][chosen],shell=patch["shell"][chosen],
                superblock=patch["superblock"][chosen],truth=truth,log_prob=logp,**scores)
            artifact=out/arm["name"]/"evaluation"/f'internal_{a["core"]:06d}.npz'
            atomic_npz(artifact,**arr);atomic_json(artifact.with_suffix(".json"),dict(sha256=sha256(artifact),sampler_check=check,
                checkpoint_sha256=sha256(path),frozen=frozen))
            parts.append(arr);groups.append(np.column_stack((arr["cap"],arr["superblock"])))
            baseline=p.root/arm["parent_arm"]/"evaluation"/f'internal_{a["core"]:06d}.npz'
            marker=read_json(baseline.with_suffix(".json"))
            if sha256(baseline)!=marker["sha256"]:raise RuntimeError("baseline evaluation changed")
            with np.load(baseline,allow_pickle=False) as old:
                if not np.array_equal(old["parent"],parent):raise RuntimeError("baseline parent mismatch")
                base_energy.append(old["energy"])
            if (index+1)%32==0:print(json.dumps(dict(event="evaluation",arm=arm["name"],cores=index+1)),flush=True)
        merged={key:np.concatenate([x[key] for x in parts]) for key in parts[0]}
        cluster=np.concatenate(groups)
        if clusters is not None and not np.array_equal(cluster,clusters):raise RuntimeError("arm cluster identity mismatch")
        clusters=cluster;data[arm["name"]]=merged
        cov=np.average([x["coverage_difference"] for x in checks],axis=0,weights=[x["rows"] for x in checks])
        gate=bool(max(x["scaled_mean_abs"] for x in checks)<=.01 and np.max(np.abs(cov))<=.01 and max(x["log_score_mean_abs"] for x in checks)<=.01)
        aggregate={key:np.mean(merged[key],axis=0).tolist() for key in ("energy","crps","coverage68","coverage90","width68","width90","log_prob")}
        shells={str(s):{key:np.mean(merged[key][merged["shell"]==s],axis=0).tolist() for key in ("energy","coverage68","coverage90","width68")}
            for s in np.unique(merged["shell"])}
        report["arms"][arm["name"]]=dict(aggregate=aggregate,shells=shells,rows=len(merged["parent"]),
            sampler_gate_pass=gate,sampler_checks=checks,pooled_refinement_coverage_difference=cov.tolist(),
            energy_minus_parent_3000=clustered_difference(merged["energy"],np.concatenate(base_energy),clusters,2000,12062026))
        del e,f,o,state
        torch.cuda.empty_cache()
    for first,second in cfg["contrasts"]:
        report["paired"][first+"_minus_"+second]=clustered_difference(data[first]["energy"],data[second]["energy"],clusters,2000,12062026)
    check_sources()
    for arm,h in frozen["parents"].items():
        if sha256(p.root/arm/"checkpoint.pt")!=h:raise RuntimeError("baseline checkpoint changed")
    report.update(technical_completion=True,sampler_gate_pass=all(a["sampler_gate_pass"] for a in report["arms"].values()),
        created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        claim="Exploratory matched-budget one-phase/one-seed internal investigation; no independent confirmation")
    atomic_json(out/"FOLLOWUP_COMPARISON.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("frozen","arms")},indent=2),flush=True)
    return 0


if __name__=="__main__":
    sys.exit(main())
