"""Matched exposed-ph006 historical checkpoint replay; never reads ph001."""
import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.abacus_tweb import p8_train_unet_patch as U
from workflows.abacus_tweb import p8_train_graph_patch as G
from workflows.abacus_tweb.p8_prepare_graph_features import transform_si, curve_values
from workflows.abacus_tweb.p8_deterministic_common import increments_to_eigenvalues, unscale_increments

ROOT = Path('/pscratch/sd/d/dkololgi/abacus')
CONTRACT = ROOT/'p10_multiphase/training_contract'
CANDIDATE = ROOT/'p10_multiphase/p12a_halo48_candidate_20260924_v1'
OLD = ROOT/'p8_recovery_v1/recovery_v1'
SELECTION = ROOT/'p6_unet_patch_adapter/fullcap_selection_v1/selection_manifest.json'

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def sample():
    f=np.load(CANDIDATE/'dataset/ph006_selection_sample.npz')
    ix=np.load(CANDIDATE/'posterior/calibration_audit/evaluation_index.npy')
    return {k:f[k][ix] for k in f.files}

def replay(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    data=sample(); parent=data['parent_node_id']; assert len(np.unique(parent))==len(parent)
    cp=OLD/a.model/f'rotation_{a.rotation}'/'seed_42/best_checkpoint.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    assert ck['model']==a.model and ck['rotation']==a.rotation
    # The field core index supplies identity only; no labels outside the saved sample.
    field=U.CanonicalFieldPatchAdapter(CONTRACT/'adapters/ph006/field',selection_manifest=SELECTION,rotation=a.rotation)
    lookup=np.full(int(field.core_parent.max())+1,-1,dtype=np.int64)
    lookup[parent]=np.arange(len(parent))
    jobs=[]
    for core in range(len(field.core_offsets)-1):
        ids=field.core_parent[field.core_offsets[core]:field.core_offsets[core+1]]
        chosen=ids[lookup[ids]>=0]
        if len(chosen):jobs.append((core,chosen.copy()))
    assert sum(len(p) for _,p in jobs)==len(parent)
    name=f'{a.model}_r{a.rotation}'
    dest=a.output/f'{name}.npz'
    predictions=np.full((len(parent),3),np.nan,dtype=np.float32)
    done=np.zeros(len(parent),bool)
    binding={'checkpoint':str(cp),'checkpoint_sha256':digest(cp),'epoch':ck['epoch'],
             'selection_sha256':digest(SELECTION),'rotation':a.rotation,
             'sample_sha256':digest(CANDIDATE/'dataset/ph006_selection_sample.npz'),
             'index_sha256':digest(CANDIDATE/'posterior/calibration_audit/evaluation_index.npy'),
             'script_sha256':digest(__file__),'model':a.model,
             'phase':'ph006','evaluation':'exposed validation and selection phase, not blind',
             'frozen_transforms':True,'requested_rows':len(parent),'requested_cores':len(jobs)}
    manifest=a.output/f'{name}.json'
    if dest.exists():
        old=json.loads(manifest.read_text());assert old['binding']==binding
        olddata=np.load(dest);assert np.array_equal(olddata['parent_node_id'],parent)
        predictions[:]=olddata['prediction'];done[:]=olddata['done']
    if a.model=='unet':
        model=U.UPatch().cuda();adapter=field
    else:
        model=G.GraphPatchNet().cuda()
        adapter=G.CanonicalGraphPatchAdapter(CONTRACT/'adapters/ph006/graph')
        spec=ck['feature_manifest'];node=spec['node']
        assert digest(node['power_transformer'])==node['power_transformer_sha256']
        with open(node['power_transformer'],'rb') as f:power=pickle.load(f)
        z=np.load(CONTRACT/'phases/ph006/parent_redshift.npy',mmap_mode='r')
        selection=json.loads(SELECTION.read_text())
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    start=time.monotonic();processed=0
    def save():
        tmp=a.output/f'{name}.tmp.npz'
        np.savez(tmp,parent_node_id=parent,prediction=predictions,done=done)
        tmp.replace(dest)
        manifest.write_text(json.dumps({'binding':binding,'complete':bool(done.all()),'rows_done':int(done.sum()),'elapsed_this_run':time.monotonic()-start,'gpu':torch.cuda.get_device_name(),'max_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')
    with torch.inference_mode():
        for core,ids in jobs:
            wanted=lookup[ids]
            if done[wanted].all():continue
            if a.model=='unet':
                patch=adapter.extract(core,U.HALO_VOXELS,U.CHANNELS,alignment_voxels=U.ALIGNMENT_VOXELS)
                tensors=U.model_inputs(patch,ck['normalization'],'cuda')
                pp=patch.authoritative_parent_id;scaled=model(*tensors)
            else:
                patch=adapter.extract(core,G.NUM_PASSES,core_parent_ids=ids,dependency_hops_per_pass=G.DEPENDENCY_HOPS_PER_PASS)
                cap=np.full(patch.n_node,int(adapter.core_cap[core]),dtype=np.int8)
                si=transform_si(patch.node_features,cap,node['si_medians'])
                nf=np.empty((patch.n_node,8),dtype=np.float32)
                nf[:,:7]=power.transform(si+node['boxcox_epsilon'])
                nt=curve_values(z[patch.parent_node_id],cap,selection,a.rotation)
                nf[:,7]=(np.log(np.maximum(nt,1e-12))-node['ntilde_log_mean'])/node['ntilde_log_std']
                patch=dataclasses.replace(patch,node_features=nf)
                tensors=G.transformed_patch(patch,spec['edge'],int(adapter.core_cap[core]),'cuda')
                pp=patch.parent_node_id[patch.loss_mask];scaled=model(*tensors)[patch.loss_mask]
            values=increments_to_eigenvalues(unscale_increments(scaled.cpu().numpy(),ck['scaler']))
            take=lookup[pp]>=0; rows=lookup[pp[take]]
            assert set(rows)==set(wanted) and np.isfinite(values).all()
            predictions[rows]=values[take];done[rows]=True;processed+=1
            if processed%100==0:
                save();print(name,'cores',processed,'rows',int(done.sum()),'seconds',round(time.monotonic()-start),flush=True)
            if a.limit and processed>=a.limit:break
    save();field.close();print(name,'done',int(done.sum()),'/',len(parent),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['unet','graph'],required=True)
    p.add_argument('--rotation',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    replay(p.parse_args())
