"""Smoke-only latent shard; never emit a production OOF completion marker."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from workflows.sbi import p12_export_unet_summaries as ex

root, out = map(Path, sys.argv[1:])
phase = "ph000"
d = out/"probe/unet/seed_42"
manifest = json.loads((root/"TRAINING_LOADER_READY.json").read_text())
ex.validate_export_phase(manifest, phase)
assert len(manifest["roles"]["training"]) == 19
ck = ex.torch_load(d/"arm_a_checkpoint.pt","cpu")
assert ck["global_step"] == ck["resume"]["cursor"] == 256
assert json.loads((d/"run_manifest.json").read_text())["training_phases"] == manifest["roles"]["training"]
loader = ex.P10PhaseBalancedLoader(root, include_blind=False)
model = ex.unet_impl.UPatch(base=24, latent_channels=32).cuda().eval()
model.load_state_dict(ck["model_state"])
truth = loader.targets_by_parent(phase)
rec = loader.phase_records[phase]
redshift=np.load(root/"phases"/phase/"parent_redshift.npy",mmap_mode="r")
selection=json.loads((root/"transforms/field/selection_manifest.json").read_text())
with np.load(rec["inputs"]["assignment"]) as a, torch.inference_mode():
    lookup=ex.parent_to_assignment_index(a,len(truth))
    eligible=a["supervised_eligible"]; caps=a["cap"]; shells=a["shell"]
    for ref in loader.validation_refs(phase):
        patch=loader.field_adapter(phase).extract(ref.core_id,48,ex.unet_impl.CHANNELS,alignment_voxels=ex.unet_impl.ALIGNMENT_VOXELS)
        ids=patch.authoritative_parent_id
        if not len(ids): continue
        rows=lookup[ids]
        assert np.all(rows>=0) and eligible[rows].all() and len(np.unique(ids))==len(ids)
        latent=model.sample_latent(*ex.unet_impl.model_inputs(patch,loader.field_normalization,"cuda"))
        pred=ex.increments_to_eigenvalues(ex.unscale_increments(model.head(latent).cpu().numpy(),loader.target_scaler))
        x=latent.cpu().numpy(); y=np.asarray(truth[ids]); z=redshift[ids]
        n=ex.ntilde_at_rows(selection,caps[rows],z)
        assert x.shape==(len(ids),32) and pred.shape==y.shape==(len(ids),3)
        assert all(np.isfinite(v).all() for v in (x,y,pred,z,n)) and (np.diff(y,axis=1)>=0).all()
        np.savez(out/"SMOKE_ONLY_ph000.npz",parent_node_id=ids,latent_32d=x,base_prediction_eigenvalues=pred,truth_eigenvalues=y,redshift=z,ntilde_mpc3=n,cap=caps[rows],shell=shells[rows])
        print("SMOKE_ONLY_EXPORT",ref.core_id,len(ids),flush=True)
        break
    else: raise RuntimeError("No scored development core")
