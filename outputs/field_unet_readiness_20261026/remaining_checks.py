"""Independent forward checks and saved-checkpoint inspection; no optimizer updates."""
import json
from pathlib import Path
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_wholefield_pilot_train as train
from workflows.sbi.e2e_wholefield_pilot_models import FineModel, Condition, sample
from workflows.sbi.e2e_vdm_context_models import block_mean, decode_density
from workflows.sbi.p12a_field_training_readiness import CONTRACT, sha, require, metadata
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p10_train_arm_a import validation_core_groups
from workflows.abacus_tweb import p8_train_unet_patch as u
from workflows.abacus_tweb.p8_deterministic_common import unscale_increments, increments_to_eigenvalues

c.require_compute(); torch.set_num_threads(4)
root=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/field_unet_readiness_20261026_v1')
report={'training_ready':False,'no_optimization_or_replay_retry':True,'failures':['conditional endpoint restart tolerance','P12-A encoder restart tolerance']}
checkpoints=[torch.load(root/'p12a_encoder/canary'/name/'unet/seed_42/arm_a_checkpoint.pt',map_location='cpu',weights_only=False) for name in ('restart','uninterrupted')]
diff={k:float((v-checkpoints[1]['model_state'][k]).abs().max()) for k,v in checkpoints[0]['model_state'].items()}
report['p12a_saved_replay']={'maximum_parameter_difference':max(diff.values()),'largest_parameters':sorted(diff.items(),key=lambda x:-x[1])[:8],'global_steps':[x['global_step'] for x in checkpoints],'cursors':[x['resume']['cursor'] for x in checkpoints],'epoch_hashes':[x['resume']['epoch_sha256'] for x in checkpoints],'scheduler_equal':checkpoints[0]['scheduler_state']==checkpoints[1]['scheduler_state'],'objective_sum':[x['objective_sum'] for x in checkpoints]}
print(json.dumps(report['p12a_saved_replay']),flush=True)
(root/'SAVED_REPLAY_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
model=u.UPatch().cuda().eval();model.load_state_dict(checkpoints[0]['model_state']);del checkpoints
panels=[]
with P10PhaseBalancedLoader(CONTRACT) as loader:
 def forbidden(*args,**kwargs):raise AssertionError('Graph access in field audit')
 loader.graph_adapter=forbidden
 groups=validation_core_groups(loader)
 for phase in loader.training_phases+loader.validation_phases:
  name='training_core_id.npy' if phase in loader.training_phases else 'validation_core_id.npy'
  ids=np.load(CONTRACT/'phases'/phase/name,mmap_mode='r');core=int(ids[len(ids)//2])
  patch=loader.field_adapter(phase).extract(core,u.HALO_VOXELS,u.CHANNELS,alignment_voxels=u.ALIGNMENT_VOXELS)
  parent=patch.authoritative_parent_id;record=loader.phase_records[phase]
  require(sha(record['inputs']['assignment'])==record['inputs']['assignment_sha256'],'Assignment hash changed')
  with np.load(record['inputs']['assignment']) as a:
   expected=a['parent_node_id'][(a['core_id']==core)&a['supervised_eligible']]
   require(np.array_equal(np.sort(parent),np.sort(expected)),'Wrong field target row alignment')
  require(sha(record['target']['path'])==record['target']['sha256'],'Target hash changed')
  truth=np.asarray(loader.targets_by_parent(phase)[parent]);require(len(parent)>0 and np.isfinite(truth).all() and (np.diff(truth,axis=1)>=0).all(),'Invalid own-position truth')
  values,points=u.model_inputs(patch,loader.field_normalization,'cuda')
  with torch.no_grad():prediction=model(values,points).cpu().numpy()
  eigen=increments_to_eigenvalues(unscale_increments(prediction,loader.target_scaler))
  require(eigen.shape==truth.shape and np.isfinite(eigen).all(),'Nonfinite interpolated base prediction')
  panels.append({'phase':phase,'core':core,'galaxies':len(parent),'base_ordering_violation_fraction':float(np.mean(np.any(np.diff(eigen,axis=1)<0,axis=1))),'pass':True})
  print('P12A_PHASE_FORWARD_PASS',phase,len(parent),flush=True)
  del values,points,patch
 require(not loader._graph,'Graph opened')
report['p12a_phase_checks']=panels
(root/'P12A_FIELD_PANELS_VERIFIED.json').write_text(json.dumps({'pass':True,'panels':panels,'replay_pass':False,'training_ready':False},indent=2)+'\n')
del model;torch.cuda.empty_cache()
chart=views.load_chart(sha(train.coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'))
state=torch.load(root/'DISPOSABLE_ENDPOINT_RESTART.pt',map_location='cuda',weights_only=False)
model=FineModel('unet','x').cuda().eval();model.load_state_dict(state['ema']);weight=state['weight']
development=train.development(model,chart,weight)
require(development['available'] and np.isfinite(development['fixed_noise_cfm_loss']),'Development loss invalid')
ids=train.panel(['ph000']);x,cond=train.batch([ids[0]],chart)
evaluation=Condition(cond.joint,cond.wide,cond.coarse_wide,'oracle_diagnostic')
with torch.no_grad():
 value=sample(model,evaluation,torch.randn_like(x),nfe=4)
 cs=chart['coarse_logrho'];mass=(evaluation.coarse_joint().double()*cs['std'][0]+cs['mean'][0]).exp()
 density=decode_density(block_mean(mass),value.double()*chart['fine_residual']['std'][0])
 error=float((block_mean(density)/block_mean(mass)-1).abs().max())
 require(torch.isfinite(density).all().item() and density.min().item()>0 and error<1e-9,'Endpoint sampler/decode failed')
report['conditional_forward_checks']={'pass':True,'development':development,'decoder_mass_relative':error,'scope':'saved disposable EMA; numerical forward/decode only; replay remains failed'}
(root/'REMAINING_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print('INDEPENDENT_FORWARD_CHECKS_COMPLETE',flush=True)
