"""Cross all four full/fold inference components on the frozen matched row set."""
import math
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.lowz_matched_encoders import (BASE, ROOT, OUT as MATCH, MODELS, PHASES, read,
    P10PhaseBalancedLoader, terminal_export_checkpoint, torch_load, unet_impl, sha256,
    increments_to_eigenvalues, unscale_increments, MatchedFieldPatchAdapter)

OUT=Path(os.environ["SWAP_OUT"])
FACTORS=("weights","normalization","selection","scaler")

def setup(ph,name):
    full=ROOT/"training_contract_v1"
    binding=full if name=="full" else ROOT/"p12_crossfit_contracts"/("omit_"+name)
    m=read((BASE/"p12a_lowz_development_summaries_61930825"/ph if name=="full"
        else BASE/"oof_campaign_20261125/summaries"/name)/"OOF_SUMMARY_COMPLETE.json")
    assert m["pass"] and Path(m["contract_root"])==binding and ph not in m["training_phases"]
    ck=Path(m["checkpoint"]); comp=ck.parent/"ARM_A_TRAINING_COMPLETE.json"
    assert sha256(ck)==m["checkpoint_sha256"] and sha256(comp)==m["completion_sha256"]
    with P10PhaseBalancedLoader(binding) as loader:
        state=terminal_export_checkpoint(torch_load(ck,"cpu"),read(comp),loader,4)
    adapter=MatchedFieldPatchAdapter(full/"adapters"/ph/"field",selection_manifest=binding/"transforms/field/selection_manifest.json")
    model=unet_impl.UPatch(base=24,latent_channels=32).cuda().eval()
    model.load_state_dict(state["state_dict"])
    return model,state["normalization"],adapter,state["scaler"]

def run(ph):
    assert ph in PHASES and int(os.environ["SLURM_NTASKS"])==16
    rank=int(os.environ["SLURM_PROCID"]); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(MATCH/(ph+".npz")) as f: d=dict(f)
    allm={name:setup(ph,name) for name in MODELS}
    cores=d["core"]; chosen=np.unique(cores)[rank::16]
    out=np.full((3,len(cores),16),np.nan); done=[]
    with torch.inference_mode():
        for c in chosen:
            ii=np.flatnonzero(cores==c); done.extend(ii)
            fullpatch=allm["full"][2].extract(int(c),48,unet_impl.CHANNELS,alignment_voxels=unet_impl.ALIGNMENT_VOXELS)
            for j,fold in enumerate(MODELS[1:]):
                models=[allm[fold],allm["full"]]
                for sel in (0,1):
                    patch=fullpatch if sel else models[0][2].extract(int(c),48,unet_impl.CHANNELS,alignment_voxels=unet_impl.ALIGNMENT_VOXELS)
                    perm=np.argsort(patch.authoritative_parent_id)
                    pos=perm[np.searchsorted(patch.authoritative_parent_id[perm],d["p"][ii])]
                    assert np.array_equal(patch.authoritative_parent_id[pos],d["p"][ii])
                    for norm in (0,1):
                        v,pt=unet_impl.model_inputs(patch,models[norm][1],"cuda")
                        for wt in (0,1):
                            model=models[wt][0]
                            scaled=model.head(model.sample_latent(v,pt)).cpu().numpy()[pos]
                            for scale in (0,1):
                                eig=increments_to_eigenvalues(unscale_increments(scaled,models[scale][3])).astype("f4")
                                out[j,ii,wt+2*norm+4*sel+8*scale]=eig[:,2]-eig[:,1]
    done=np.array(done)
    assert np.isfinite(out[:,done,:]).all()
    np.savez(OUT/f"{ph}_{rank}.npz",idx=done,values=out[:,done,:])

def score(ph):
    assert ph in PHASES
    with np.load(MATCH/(ph+".npz")) as f: d=dict(f)
    files=[np.load(OUT/f"{ph}_{i}.npz") for i in range(16)]
    ix=np.concatenate([f["idx"] for f in files]); order=np.argsort(ix)
    assert np.array_equal(ix[order],np.arange(4096))
    z=np.concatenate([f["values"] for f in files],axis=1)[:,order,:]
    w=d["w"].astype("f8"); _,blk=np.unique(np.floor(d["xyz"]/128),axis=0,return_inverse=True)
    mass=np.bincount(blk,weights=w);bb=np.random.default_rng(210).integers(len(mass),size=(1000,len(mass)))
    def summary(a):
        sums=np.bincount(blk,weights=w*a)
        return dict(mean=float(np.average(a,weights=w)),ci=np.quantile(sums[bb].sum(1)/mass[bb].sum(1),[.025,.975]).tolist())
    report={}
    for j,fold in enumerate(MODELS[1:]):
        v=z[j]; ref=d["reference"]
        archived=[np.load(MATCH/f"{ph}_{fold}_{k}.npz") for k in range(4)]
        idx=np.concatenate([a["idx"] for a in archived])
        pred=np.vstack([a["pred"] for a in archived])[np.argsort(idx)]
        assert np.array_equal(v[:,0],pred[:,2]-pred[:,1]) and np.array_equal(v[:,15],ref[:,2]-ref[:,1])
        attrs={}; main=[]
        for i,name in enumerate(FACTORS):
            bit=1<<i
            contrasts=[(v[:,b|bit]-v[:,b],b.bit_count()) for b in range(16) if not b&bit]
            shap=sum(delta/(4*math.comb(3,k)) for delta,k in contrasts)
            main.append(shap)
            attrs[name]=dict(shapley=summary(shap),single_swap=summary(v[:,bit]-v[:,0]),
                range_across_backgrounds=[float(fn([np.average(a,weights=w) for a,_ in contrasts])) for fn in (min,max)])
        np.testing.assert_allclose(sum(main),v[:,15]-v[:,0],atol=1e-10)
        report[fold]=dict(total=summary(v[:,15]-v[:,0]),factors=attrs,
            config_means=np.average(v,axis=0,weights=w).tolist(),endpoints_bitwise_equal=True)
    (OUT/f"{ph}.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__": {"run":run,"score":score}[sys.argv[1]](sys.argv[2])
