"""Development-only rank and TARP replay. No model fit or parameter selection."""
import os
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tarp import get_tarp_coverage
from workflows.sbi.lowz_fmpe_development import BASE, load
from workflows.sbi.p12_train_base_response_fmpe import theta_to_eigenvalues

def block_mean_ci(r,w,b):
    # Equal-probability resampling of occupied 128-Mpc Cartesian cells.
    mass=np.bincount(b,weights=w); val=np.bincount(b,weights=w*r)
    rng=np.random.default_rng(781)
    j=rng.integers(len(mass),size=(500,len(mass)))
    return np.quantile(val[j].sum(1)/mass[j].sum(1),[.025,.975]).tolist()

def main():
    out=Path(os.environ["DIAG_OUT"]); out.mkdir(exist_ok=True)
    ft,at=plt.subplots(1,2,figsize=(11,4),sharey=True)
    fs,ax=plt.subplots(2,5,figsize=(15,6),sharey=True,sharex=True)
    report={}
    for k,seed in enumerate((42,43)):
        ck,source,d,rows=load(seed)
        c=torch.load(ck,map_location="cpu",weights_only=False)
        locked=json.loads((source/"DEVELOPMENT_RESULT.json").read_text())
        tau=locked["tau"]; report[str(seed)]={}
        idx=np.concatenate(rows)
        draws=np.empty((len(idx),256,3),"f4")
        for rank in range(16):
            with np.load(source/f"rank{rank}.npz") as f:
                assert np.array_equal(f["idx"],idx[rank::16])
                draws[rank::16]=f["samples"]
        for j,ph in enumerate(("ph012","ph013"),1):
            color=["#1864ab","#d46116"][j-1]
            i=rows[j]; a=draws[sum(map(len,rows[:j])):sum(map(len,rows[:j+1]))]
            y=(d["theta_softplus"][i]-c["theta_mean"])/c["theta_std"]
            tr=d["truth_eigenvalues"][i]; w=d["natural_weight"][i].astype("f8")
            pp=BASE/f"p12a_lowz_20261029_v1/{ph}/p1_canonical/points.npy"
            xyz=np.load(pp,mmap_mode="r")[d["parent_node_id"][i]]
            assert np.array_equal(xyz[:,3],d["cap"][i])
            _,b=np.unique(np.floor(xyz[:,:3]/128).astype(int),axis=0,return_inverse=True)
            rec=report[str(seed)][ph]={}
            for name,t in (("raw",1.),("calibrated",tau)):
                m=a.mean(1,keepdims=True); s=m+t*(a-m)
                e=theta_to_eigenvalues(s*c["theta_std"]+c["theta_mean"])
                # Randomized discrete ranks; softplus is monotone in each gap.
                counts=np.column_stack(((e<tr[:,None,:]).sum(1),(s[:,:,1:]<y[:,None,1:]).sum(1)))
                r=(counts+np.random.default_rng(671+j).random(counts.shape))/257
                z=d["context"][i,3]; gap=lambda v: v[...,2]-v[...,1]
                rec[name]=dict(mean_rank=np.average(r,axis=0,weights=w).tolist(),
                    mean_rank_ci=[block_mean_ci(v,w,b) for v in r.T],
                    n_blocks=int(b.max()+1),
                    gap23_bias=float(np.average(gap(e).mean(1)-gap(tr),weights=w)),
                    gap23_truth_mean=float(np.average(gap(tr),weights=w)),
                    gap23_rank_by_z=[float(np.average(r[mask,4],weights=w[mask])) for mask in
                      [(z>=lo)&(z<hi) for lo,hi in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55])]])
                if name=="raw": continue
                for col in range(5):
                    h,edges=np.histogram(r[:,col],np.linspace(0,1,11),weights=w)
                    ax[k,col].stairs(100*h/w.sum(),edges,color=color,label=ph,lw=2)
                ecp,alpha=get_tarp_coverage(np.transpose(s[:10000],(1,0,2)),y[:10000],
                    norm=True,bootstrap=False,seed=seed)
                delta=ecp-alpha
                assert abs(np.max(np.abs(delta))-locked["phases"][ph]["tarp"]["max_abs_ecp_minus_alpha"])<1e-7
                at[k].plot(alpha,delta,color=color,label=ph,lw=1.5)
        for col,title in enumerate(["λ₁","λ₂","λ₃","λ₂ − λ₁","λ₃ − λ₂"]):
            ax[k,col].axhline(10,ls=":",color=".35")
            ax[k,col].set_title(title)
            if k==1: ax[k,col].set_xlabel("Truth rank")
        ax[k,0].set_ylabel(f"Seed {seed}\nGalaxies (%)")
        at[k].axhspan(-.05,.05,color=".94")
        at[k].axhline(0,color=".3",ls=":")
        at[k].set(title=f"FMPE seed {seed}",xlabel="Nominal credibility",ylim=(-.055,.055))
        at[k].legend()
    ax[0,0].legend(fontsize=8); at[0].set_ylabel("Empirical coverage − nominal")
    ft.suptitle("Joint TARP • deviation from ideal coverage (grey: ±0.05 gate)")
    fs.suptitle("Marginal calibration • natural-population weighted ranks (flat 10% is ideal)")
    for fig,name in [(ft,"tarp"),(fs,"sbc")]:
        fig.tight_layout(); fig.savefig(out/f"{name}.png",dpi=160)
    (out/"rank_diagnosis.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__": main()
