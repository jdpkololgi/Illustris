"""Reconstruct old inputs and reproduce saved ph000 predictions before transfer claims."""
import dataclasses,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.sbi.p12a_historical_knot_comparison import ROOT,OLD,SELECTION,U,G,transform_si,curve_values,unscale_increments,increments_to_eigenvalues

torch.set_num_threads(8)
out={}
for name in ['unet','graph']:
    root=OLD/name/'rotation_0/seed_42'
    ck=torch.load(root/'best_checkpoint.pt',weights_only=False,map_location='cpu')
    saved_ids=np.load(root/'best_validation_parent_node_id.npy',mmap_mode='r')
    saved=np.load(root/'best_validation_eigenvalues.npy',mmap_mode='r')
    field=U.CanonicalFieldPatchAdapter(U.ADAPTER,selection_manifest=SELECTION,rotation=0)
    # Pick a complete old validation core; identity is checked against saved rows.
    core=next(i for i in range(len(field.core_fold)) if field.core_fold[i]==1 and field.core_offsets[i+1]>field.core_offsets[i])
    ids=np.asarray(field.core_parent[field.core_offsets[core]:field.core_offsets[core+1]])
    if name=='unet':
        model=U.UPatch().cuda();patch=field.extract(core,U.HALO_VOXELS,U.CHANNELS,alignment_voxels=U.ALIGNMENT_VOXELS)
        tensors=U.model_inputs(patch,ck['normalization'],'cuda');ids=patch.authoritative_parent_id;mask=None
    else:
        model=G.GraphPatchNet().cuda();adapter=G.CanonicalGraphPatchAdapter(G.P5_ROOT)
        patch=adapter.extract(core,G.NUM_PASSES,core_parent_ids=ids,dependency_hops_per_pass=G.DEPENDENCY_HOPS_PER_PASS)
        spec=ck['feature_manifest'];node=spec['node'];cap=np.full(patch.n_node,int(adapter.core_cap[core]),dtype=np.int8)
        with open(node['power_transformer'],'rb') as f:power=pickle.load(f)
        nf=np.empty((patch.n_node,8),dtype=np.float32);nf[:,:7]=power.transform(transform_si(patch.node_features,cap,node['si_medians'])+node['boxcox_epsilon'])
        z=np.load(ROOT/'p8_deterministic_v1/parent_redshift.npy',mmap_mode='r')
        nt=curve_values(z[patch.parent_node_id],cap,json.loads(SELECTION.read_text()),0)
        nf[:,7]=(np.log(np.maximum(nt,1e-12))-node['ntilde_log_mean'])/node['ntilde_log_std']
        original=np.load(node['transformed_path'],mmap_mode='r')[patch.parent_node_id]
        assert np.allclose(nf,original,atol=1e-5,rtol=1e-5)
        patch=dataclasses.replace(patch,node_features=nf);tensors=G.transformed_patch(patch,spec['edge'],int(adapter.core_cap[core]),'cuda');mask=patch.loss_mask;ids=patch.parent_node_id[mask]
    model.load_state_dict(ck['state_dict']);model.eval()
    with torch.inference_mode():
        scaled=model(*tensors)
        if mask is not None:scaled=scaled[mask]
    pred=increments_to_eigenvalues(unscale_increments(scaled.cpu().numpy(),ck['scaler']))
    order=np.argsort(saved_ids);ix=np.searchsorted(saved_ids[order],ids);assert np.array_equal(saved_ids[order[ix]],ids)
    error=float(np.max(np.abs(pred-saved[order[ix]])))
    assert error<1e-4,(name,error)
    out[name]={'core':core,'rows':len(ids),'max_abs_saved_prediction_error':error,'pass':True}
    field.close()
print(json.dumps(out,indent=2))
