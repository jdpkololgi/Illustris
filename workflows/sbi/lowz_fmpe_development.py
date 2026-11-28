"""Locked whole-phase transfer test. No blind data or observational inputs."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from workflows.sbi.p12a_blind_inference import reconstruct_fmpe
from workflows.sbi.p12_train_base_response_fmpe import (
    sample_posterior, report_samples, tarp_diagnostic, sha256, atomic_json)

BASE = Path("/pscratch/sd/d/dkololgi/abacus")
FIT = BASE/"p12a_oof_fmpe_20261128_f036ce3"
DEV = BASE/"p12a_development_table_62094904/selection_sample.npz"
DIGEST = "9f792dd000cadffdd72e5270ffa3d6bf172e6f3704aa31006b965acb0b86fef9"
KEYS = {42:"14b1537cff4182f0ddee33ee3b2cf75a517e08321415aaaa85c18ae0798b7007",
        43:"6cad873deef026a3cc604458b4ed8b3670e3a5741e3f7b5215cd55998b5f6fd7"}
PH = ["ph006","ph012","ph013"]

def load(seed):
    assert seed in KEYS and sha256(DEV)==DIGEST
    ck = FIT/f"seed_{seed}/fmpe_estimator.pt"
    assert sha256(ck)==KEYS[seed]
    out = Path(os.environ["EVAL_OUT"])/str(seed)
    out.mkdir(exist_ok=True)
    with np.load(DEV) as f:
        data=dict(f)
    rng=np.random.default_rng(20261128)
    rows=[]
    for j,n in enumerate((20000,25000,25000)):
        ix=np.flatnonzero(data["phase_index"]==20+j)
        assert len(ix)==200000
        rows.append(rng.choice(ix,n,replace=False))
    return ck,out,data,rows

def main(mode,seed):
    ck,out,d,rows=load(seed)
    if mode=="draw":
        rank=int(os.environ["SLURM_PROCID"])
        assert int(os.environ["SLURM_NTASKS"])==16 and torch.cuda.device_count()==1
        posterior,c=reconstruct_fmpe(ck,"cuda")
        assert c["dataset_marker_sha256"]==sha256(FIT/"P12A_DATASET_READY.json")
        idx=np.concatenate(rows)[rank::16]
        x=((d["context"][idx]-c["context_mean"])/c["context_std"]).astype("f4")
        torch.manual_seed(4788+rank)
        arr=sample_posterior(posterior,x,256,128,"cuda")
        assert np.isfinite(arr).all()
        np.savez(out/f"rank{rank}.npz",idx=idx,samples=arr)
        return
    c=torch.load(ck,map_location="cpu",weights_only=False)
    allrows=np.concatenate(rows)
    draws=np.empty((len(allrows),256,3),"f4")
    for rank in range(16):
        with np.load(out/f"rank{rank}.npz") as f:
            assert np.array_equal(f["idx"],allrows[rank::16])
            draws[rank::16]=f["samples"]
    # First eigenvalue is affine in scaled coordinate 0. Solve the same
    # weighted 68% interval criterion without repeating 3-D quantiles 40 times.
    a=draws[:len(rows[0]),:,0]; m=a.mean(1)
    lo,hi=np.quantile(a,[.16,.84],axis=1)
    z=(d["truth_eigenvalues"][rows[0],0]-c["theta_mean"][0])/c["theta_std"][0]
    w=d["natural_weight"][rows[0]]
    l,h=.25,8.
    for _ in range(40):
        t=(l+h)/2
        if np.average((z>=m+t*(lo-m))&(z<=m+t*(hi-m)),weights=w)<.68: l=t
        else: h=t
    tau=(l+h)/2
    summary={}
    for j in (1,2):
        i=rows[j]; a=draws[sum(map(len,rows[:j])):sum(map(len,rows[:j+1]))]
        y=(d["theta_softplus"][i]-c["theta_mean"])/c["theta_std"]
        report=report_samples(a,d["truth_eigenvalues"][i],d["base_prediction_eigenvalues"][i],y,
            c["theta_mean"],c["theta_std"],d["shell"][i],d["cap"][i],d["context"][i,-1],d["natural_weight"][i],tau)
        m=a[:10000].mean(1,keepdims=True)
        tarp=tarp_diagnostic(m+tau*(a[:10000]-m),y[:10000],seed)
        gates=dict(marginal=bool(np.all(np.abs(np.array(report["coverage68"])-.68)<=.03)
                    and np.all(np.abs(np.array(report["coverage90"])-.90)<=.03)),
            shell=all(abs(v-.68)<=.06 for row in report["by_shell"].values() for v in row["coverage68"]),
            mean=report["posterior_mean_r2"][0]>=report["base_r2"][0]-.02,
            sbc=all(v>.01 for v in report["sbc_ks_p"]),tarp=tarp["pass_0p05"])
        summary[PH[j]]=dict(report=report,tarp=tarp,gates=gates,passed=all(gates.values()))
    atomic_json(out/"DEVELOPMENT_RESULT.json",dict(seed=seed,checkpoint_sha256=KEYS[seed],table_sha256=DIGEST,
        calibration_phase="ph006",tau=tau,phases=summary,passed=all(p["passed"] for p in summary.values())))

if __name__=="__main__": main(sys.argv[1],int(sys.argv[2]))
