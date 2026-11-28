"""Paired encoder and posterior gap residuals; no model fitting or blind data."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.lowz_fmpe_development import FIT, load, reconstruct_fmpe, sample_posterior, sha256

def measure(tr, enc, post, w):
    mean=lambda x: float(np.average(x,weights=w))
    assert np.isfinite([tr,enc,post,w]).all() and np.all(w>0)
    return dict(n=len(w),truth=mean(tr),encoder_minus_truth=mean(enc-tr),
                posterior_minus_encoder=mean(post-enc),posterior_minus_truth=mean(post-tr))

def compare(d,i,p):
    gap=lambda x: x[:,2]-x[:,1]
    t=gap(d["truth_eigenvalues"][i]); e=gap(d["base_prediction_eigenvalues"][i])
    assert np.array_equal(d["base_prediction_eigenvalues"][i],d["context"][i,:3])
    w=d["natural_weight"][i]; z=d["context"][i,3]
    return {"all":measure(t,e,p,w),"by_z":[measure(t[m],e[m],p[m],w[m]) for m in
            [(z>=l)&(z<h) for l,h in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55])]] if z.min()<.15 else []}

def postgap(a,c,tau):
    s=a[:,:,2]; m=s.mean(1,keepdims=True)
    return np.logaddexp(0,(m+tau*(s-m))*c["theta_std"][2]+c["theta_mean"][2]).mean(1)

def main():
    dataset=FIT/"P12A_DATASET_READY.json"
    info=json.loads(dataset.read_text())["training"]; path=Path(info["path"])
    assert sha256(path)==info["sha256"]
    with np.load(path) as f: train=dict(f)
    n=len(train["context"]); assert n==2000000
    hold=torch.randperm(n,device="cuda",generator=torch.Generator(device="cuda").manual_seed(12142))[:n//10].cpu().numpy()
    rng=np.random.default_rng(27311); i=rng.choice(hold,25000,replace=False)
    results={"holdback_sample_sha256":hashlib.sha256(i.tobytes()).hexdigest()}
    for seed in (42,43):
        ck,src,d,rows=load(seed); posterior,c=reconstruct_fmpe(ck,"cuda")
        assert c["dataset_marker_sha256"]==sha256(dataset)
        tau=json.loads((src/"DEVELOPMENT_RESULT.json").read_text())["tau"]
        torch.manual_seed(27987)
        x=((train["context"][i]-c["context_mean"])/c["context_std"]).astype("f4")
        a=sample_posterior(posterior,x,256,128,"cuda")
        r=results[str(seed)]={}
        for tag,t in (("raw",1.),("calibrated",tau)):
            p=postgap(a,c,t); r[tag]={"oof":compare(train,i,p)}
            r[tag]["oof_by_phase"]={ph:measure(
                np.diff(train["truth_eigenvalues"][i[m],1:],axis=1).ravel(),
                np.diff(train["base_prediction_eigenvalues"][i[m],1:],axis=1).ravel(),p[m],train["natural_weight"][i[m]])
                for ph,v in json.loads(dataset.read_text())["phases"].items()
                for m in [train["phase_index"][i]==v["index"]]}
        allrows=np.concatenate(rows)
        for tag,t in (("raw",1.),("calibrated",tau)):
            pred=np.empty(len(allrows))
            for k in range(16):
                with np.load(src/f"rank{k}.npz") as f:
                    assert np.array_equal(f["idx"],allrows[k::16])
                    pred[k::16]=postgap(f["samples"],c,t)
            for ph,ix,pp in zip(["ph012","ph013"],rows[1:],np.split(pred,np.cumsum(list(map(len,rows)))[:-1])[1:]):
                r[tag][ph]=compare(d,ix,pp)
    h=json.loads(Path("docs/evidence/p12/P12A_PHYSICAL_DEPENDENCE_DIAGNOSTIC.json").read_text())["provenance"]
    for key in ("posterior_samples","checkpoint","evaluation_index","validation_sample"):
        assert sha256(Path(h[key]))==h[key+"_sha256"]
    c=torch.load(h["checkpoint"],map_location="cpu",weights_only=False)
    old=np.load(h["posterior_samples"],mmap_mode="r"); ix=np.load(h["evaluation_index"])
    with np.load(h["validation_sample"]) as f:
        d={k:f[k] for k in ("context","truth_eigenvalues","base_prediction_eigenvalues","natural_weight")}
    results["historical_ph006"]=compare(d,ix,postgap(old,c,1.))
    Path(os.environ["GAP_OUT"],"attribution.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__": main()
