"""Inference-only matched development transfer check, no access to final-test rows."""
import os
import json
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.p12_export_unet_summaries import (
    P10PhaseBalancedLoader, terminal_export_checkpoint, torch_load, unet_impl,
    sha256, increments_to_eigenvalues, unscale_increments)
from workflows.abacus_tweb.p12a_matched_field_adapter import MatchedFieldPatchAdapter
from workflows.sbi.lowz_fmpe_development import BASE,DEV,DIGEST

MODELS=["full","ph000","ph010","ph024"]
PHASES=["ph012","ph013"]
OUT=Path(os.environ["MATCH_OUT"])
ROOT=BASE/"p12a_lowz_20261029_v1"
def read(path): return json.loads(path.read_text())

def prepare(ph):
    assert ph in PHASES and sha256(DEV)==DIGEST
    full=ROOT/"training_contract_v1"
    with np.load(DEV) as f: d=dict(f)
    g=np.random.default_rng(20261128)
    ix=[g.choice(np.flatnonzero(d["phase_index"]==20+j),n,replace=False)
        for j,n in enumerate((20000,25000,25000))][1+PHASES.index(ph)][:4096]
    p=d["parent_node_id"][ix]; assert len(np.unique(p))==4096
    record=read(full/"phases"/ph/"phase_contract.json")
    with np.load(record["inputs"]["assignment"]) as f:
        parents=f["parent_node_id"]
        lookup=np.full(record["parent_rows"],-1,dtype="i8");lookup[parents]=np.arange(len(parents))
        a=lookup[p]; assert (a>=0).all() and f["supervised_eligible"][a].all()
        cores=f["core_id"][a]
    xyz=np.load(read(Path(record["inputs"]["p1_manifest"]))["points"],mmap_mode="r")[p,:3]
    np.savez(OUT/(ph+".npz"),p=p,core=cores,xyz=xyz,idx=ix,w=d["natural_weight"][ix],
        truth=d["truth_eigenvalues"][ix],reference=d["base_prediction_eigenvalues"][ix])

def infer(ph):
    rank=int(os.environ["SLURM_PROCID"]); name=MODELS[rank//4]; shard=rank%4
    assert ph in PHASES and int(os.environ["SLURM_NTASKS"])==16
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    full=ROOT/"training_contract_v1"
    receipt=(BASE/"p12a_lowz_development_summaries_61930825"/ph if name=="full"
             else BASE/"oof_campaign_20261125/summaries"/name)/"OOF_SUMMARY_COMPLETE.json"
    m=read(receipt); binding=full if name=="full" else ROOT/"p12_crossfit_contracts"/("omit_"+name)
    assert m["pass"] and Path(m["contract_root"])==binding and m["terminal_epochs"]==4
    ck=Path(m["checkpoint"]); comp=ck.parent/"ARM_A_TRAINING_COMPLETE.json"
    assert sha256(ck)==m["checkpoint_sha256"] and sha256(comp)==m["completion_sha256"]
    with P10PhaseBalancedLoader(binding) as loader:
        state=terminal_export_checkpoint(torch_load(ck,"cpu"),read(comp),loader,4)
    train=read(full/"TRAINING_LOADER_READY.json")["roles"]["training"]
    assert state["training_phases"]==[p for p in train if p!=name] and ph not in train
    adapter=MatchedFieldPatchAdapter(full/"adapters"/ph/"field",selection_manifest=binding/"transforms/field/selection_manifest.json")
    model=unet_impl.UPatch(base=24,latent_channels=32).cuda().eval()
    model.load_state_dict(state["state_dict"])
    with np.load(OUT/(ph+".npz")) as f: data=dict(f)
    cores=data["core"]; rows=[];pred=[]
    with torch.inference_mode():
        for core in np.unique(cores)[shard::4]:
            ii=np.flatnonzero(cores==core)
            patch=adapter.extract(int(core),48,unet_impl.CHANNELS,alignment_voxels=unet_impl.ALIGNMENT_VOXELS)
            perm=np.argsort(patch.authoritative_parent_id); ids=patch.authoritative_parent_id[perm]
            pos=perm[np.searchsorted(ids,data["p"][ii])]
            assert np.array_equal(patch.authoritative_parent_id[pos],data["p"][ii])
            v,pt=unet_impl.model_inputs(patch,state["normalization"],"cuda")
            scaled=model.head(model.sample_latent(v,pt)).cpu().numpy()[pos]
            pred.append(increments_to_eigenvalues(unscale_increments(scaled,state["scaler"])).astype("f4"));rows.append(ii)
    ix=np.concatenate(rows); result=np.vstack(pred); assert np.isfinite(result).all()
    if name=="full": np.testing.assert_allclose(result,data["reference"][ix],atol=2e-6,rtol=1e-6)
    np.savez(OUT/f"{ph}_{name}_{shard}.npz",idx=ix,pred=result,sha=m["checkpoint_sha256"])

def score():
    report={}
    for ph in PHASES:
        with np.load(OUT/(ph+".npz")) as f: d=dict(f)
        w=d["w"];truth=np.diff(d["truth"][:,1:],axis=1).ravel()
        ref=np.diff(d["reference"][:,1:],axis=1).ravel()
        _,block=np.unique(np.floor(d["xyz"]/128),axis=0,return_inverse=True)
        bw=np.bincount(block,weights=w); g=np.random.default_rng(210)
        bb=g.integers(len(bw),size=(1000,len(bw)))
        report[ph]={}
        for name in MODELS:
            chunks=[np.load(OUT/f"{ph}_{name}_{j}.npz") for j in range(4)]
            ix=np.concatenate([a["idx"] for a in chunks]); order=np.argsort(ix)
            assert np.array_equal(ix[order],np.arange(len(ref)))
            pred=np.vstack([a["pred"] for a in chunks])[order]; gap=pred[:,2]-pred[:,1]
            delta=gap-ref; weighted=lambda a:float(np.average(a,weights=w))
            def ci(a):
                sums=np.bincount(block,weights=w*a)
                return np.quantile(sums[bb].sum(1)/bw[bb].sum(1),[.025,.975]).tolist()
            report[ph][name]=dict(rows=len(ix),cores=len(np.unique(d["core"])),blocks=len(bw),
                checkpoint=str(chunks[0]["sha"]),truth_mean=weighted(truth),bias=weighted(gap-truth),
                bias_ci=ci(gap-truth),difference_to_full=weighted(delta),paired_ci=ci(delta),
                full_replay_max=float(np.max(abs(pred-d["reference"]))) if name=="full" else None)
    (OUT/"MATCHED.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    import sys
    {"prepare":prepare,"infer":infer,"score":score}[sys.argv[1]](*sys.argv[2:])
