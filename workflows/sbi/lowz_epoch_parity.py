"""Replay archived epoch one and plot per-galaxy P12-A truth/prediction parity."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from workflows.sbi.lowz_matched_encoders import (
    BASE,ROOT,PHASES,OUT as MATCH,read,P10PhaseBalancedLoader,torch_load,unet_impl,
    sha256,unscale_increments,increments_to_eigenvalues)
from workflows.sbi.lowz_fmpe_development import load
from workflows.sbi.p12_train_base_response_fmpe import theta_to_eigenvalues
OUT=Path(os.environ["PARITY_OUT"])
REPAIR=BASE/"p12a_lowz_full_seed42_61548250/full/unet/seed_42/pre_validation_repair_61548250"

def infer(ph):
    assert ph in PHASES
    rank=int(os.environ["SLURM_PROCID"]); n=int(os.environ["SLURM_NTASKS"])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ck=REPAIR/"arm_a_checkpoint.pt"; mig=read(REPAIR/"migration.json")
    assert sha256(ck)==mig["original"]=="6ceb15b449e0751757dfecff082af98becb7093f1c96e90d0aaac56e98bd5231"
    st=torch_load(ck,"cpu"); assert st["global_step"]==346364==st["resume"]["cursor"] and st["resume"]["epoch"]==1
    model=unet_impl.UPatch(base=24,latent_channels=32).cuda().eval()
    model.load_state_dict(st["model_state"])
    with np.load(MATCH/(ph+".npz")) as f: d=dict(f)
    rows=[];pred=[]
    with P10PhaseBalancedLoader(ROOT/"training_contract_v1") as loader,torch.inference_mode():
        assert ph in loader.validation_phases
        adapter=loader.field_adapter(ph)
        for core in np.unique(d["core"])[rank::n]:
            ii=np.flatnonzero(d["core"]==core)
            patch=adapter.extract(int(core),48,unet_impl.CHANNELS,alignment_voxels=unet_impl.ALIGNMENT_VOXELS)
            perm=np.argsort(patch.authoritative_parent_id); pos=perm[np.searchsorted(patch.authoritative_parent_id[perm],d["p"][ii])]
            assert np.array_equal(patch.authoritative_parent_id[pos],d["p"][ii])
            v,pt=unet_impl.model_inputs(patch,loader.field_normalization,"cuda")
            scaled=model.head(model.sample_latent(v,pt)).cpu().numpy()[pos]
            pred.append(increments_to_eigenvalues(unscale_increments(scaled,loader.target_scaler)).astype("f4"));rows.append(ii)
    np.savez(OUT/f"{ph}_{rank}.npz",idx=np.concatenate(rows),pred=np.vstack(pred))

def plot(name,title,t,pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    fig,axes=plt.subplots(len(pairs),3,figsize=(11,3.6*len(pairs)),squeeze=False,layout="constrained")
    for row,(label,p) in enumerate(pairs):
        assert t.shape==p.shape and np.isfinite(p).all()
        for k,(lo,hi) in enumerate([(-.62,1.05),(-.3,1.5),(-.15,2.)]):
            ax=axes[row,k];y=t[:,k];v=p[:,k]
            hb=ax.hexbin(y,v,gridsize=72,bins="log",cmap="viridis",extent=(lo,hi,lo,hi),mincnt=1)
            ax.plot([lo,hi],[lo,hi],"r--",lw=1)
            ax.axvline(.2,color="gray",ls=":",lw=.8);ax.axhline(.2,color="gray",ls=":",lw=.8)
            ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect="equal",xlabel=f"True λ{k+1}",ylabel=f"{label} λ{k+1}",
                   title=f"R² {r2_score(y,v):.3f}   slope {np.polyfit(y,v,1)[0]:.2f}   σ ratio {v.std()/y.std():.2f}")
            fig.colorbar(hb,ax=ax,shrink=.65,label="Galaxies / hex")
    fig.suptitle(title+"\nOne galaxy per pair · red: equality · dotted: web threshold 0.2",fontsize=13)
    fig.savefig(OUT/(name+".png"),dpi=160);plt.close(fig)

def score(ph):
    assert ph in PHASES
    with np.load(MATCH/(ph+".npz")) as f: d=dict(f)
    parts=[np.load(OUT/f"{ph}_{r}.npz") for r in range(16)]
    ix=np.concatenate([a["idx"] for a in parts]); order=np.argsort(ix)
    assert np.array_equal(ix[order],np.arange(4096))
    e1=np.vstack([a["pred"] for a in parts])[order];e4=d["reference"];truth=d["truth"]
    ck,src,data,rows=load(42); c=torch_load(ck,"cpu"); allrows=np.concatenate(rows)
    tau=read(src/"DEVELOPMENT_RESULT.json")["tau"]
    assert np.array_equal(data["parent_node_id"][d["idx"]],d["p"])
    assert np.array_equal(data["truth_eigenvalues"][d["idx"]],truth)
    posterior=np.empty((len(truth),3));seen=np.zeros(len(truth),bool)
    for rank in range(16):
        with np.load(src/f"rank{rank}.npz") as f:
            assert np.array_equal(f["idx"],allrows[rank::16])
            _,i,j=np.intersect1d(d["idx"],f["idx"],return_indices=True)
            a=f["samples"][j]; m=a.mean(1,keepdims=True)
            posterior[i]=theta_to_eigenvalues((m+tau*(a-m))*c["theta_std"]+c["theta_mean"]).mean(1);seen[i]=True
    assert seen.all()
    pairs=[("Epoch 1",e1),("Epoch 4",e4),("FMPE mean",posterior)]
    plot(ph,"Corrected P12-A · development "+ph+" · z=0.01–0.55 · 4,096 matched galaxies",truth,pairs)
    gap=lambda a:a[:,2]-a[:,1]
    _,b=np.unique(np.floor(d["xyz"]/128),axis=0,return_inverse=True);w=d["w"].astype("f8")
    bw=np.bincount(b,weights=w); bb=np.random.default_rng(210).integers(len(bw),size=(1000,len(bw)))
    def stat(x):
        s=np.bincount(b,weights=x*w)
        return dict(mean=float(np.average(x,weights=w)),ci=np.quantile(s[bb].sum(1)/bw[bb].sum(1),[.025,.975]).tolist())
    record={k:stat(gap(a)-gap(truth)) for k,a in pairs}
    record["epoch4_minus_epoch1"]=stat(gap(e4)-gap(e1))
    (OUT/(ph+".json")).write_text(json.dumps(record,indent=2))

def historical():
    from workflows.sbi.p12a_evaluate_blind import load_prediction_arrays
    h=BASE/"p10_multiphase/blind_predictions/ph001"
    report=read(h/"evaluation/P12A_PH001_BLIND_EVALUATION.json")
    assert report["pass"] and report["phase"]=="ph001" and report["open_count"]==1
    tm=read(h/"P12A_PH001_TRUTH_COMPLETE.json"); path=Path(tm["array"]["path"])
    assert sha256(path)==tm["array"]["sha256"]
    sm,_=load_prediction_arrays(read(h/"P12_BLIND_PREDICTIONS_FROZEN.json"))
    i=np.sort(np.random.default_rng(20261129).choice(report["rows"],50000,replace=False))
    with np.load(path) as f:
        assert np.array_equal(f["parent_node_id"],sm["parent_node_id"])
        t=f["eigenvalues"][i]
    plot("historical_ph001","Historical P12-A · previously opened test ph001 · z=0.15–0.55 · 50,000 galaxies",t,
         [("Encoder",sm["base_prediction_eigenvalues"][i]),("FMPE mean",sm["eigenvalue_mean"][i])])

if __name__=="__main__": {"infer":infer,"score":score,"historical":historical}[sys.argv[1]](*sys.argv[2:])
