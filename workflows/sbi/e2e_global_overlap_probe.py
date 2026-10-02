"""Engineering probe of shared-state CFM coupling; NOT a qualified posterior.

Two adjacent Loa windows share one addressed latent canvas at EVERY Heun step.
Frozen local velocities are averaged on overlap. This changes the generative law;
it tests feasibility and disagreement, not posterior calibration or a global fit.
"""
from dataclasses import replace
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op

def prepare(root,out):
 import h5py
 from workflows.sbi import e2e_loa_field_conditions as cond
 c.require_compute();out.mkdir(parents=True,exist_ok=False)
 plan=json.loads((root/'PLAN.json').read_text());cases=[]
 for cap in ['NGC','SGC']:
  lookup={tuple(t['key']):t for t in plan['tiles'] if t['cap']==cap}
  detail=json.loads((root/'details'/('detail_'+cap)/'COMPLETE.json').read_text())['case']['midpoint']
  candidates=[(a,lookup[(k[0]+1,k[1],k[2])]) for k,a in lookup.items() if (k[0]+1,k[1],k[2]) in lookup]
  a,b=min(candidates,key=lambda ab:np.linalg.norm(np.asarray(ab[0]['midpoint'])-detail))
  response=cond.load_response(cap)
  with h5py.File(root/(cap+'_observation_cache.h5'),'r') as f:
   for side,t in enumerate([a,b]):
    arrays=cond.build(plan['grids'][cap],t['midpoint'],cap,f['counts_raw'],f['wide'],response)
    file=out/f'{cap}_{side}.npz';np.savez_compressed(file,**arrays)
    print('prepared',t['id'],flush=True)
  assert np.array_equal(np.asarray(b['midpoint'])-a['midpoint'],[64,0,0])
  cases.append({'cap':cap,'tiles':[a,b],'hashes':[c.sha256(out/f'{cap}_{i}.npz') for i in range(2)]})
 c.atomic_json(out/'CONDITIONS.json',{'cases':cases,'plan_sha256':c.sha256(root/'PLAN.json'),'observations_only':True})

def slices(stage,side):
 length,shift=(48,8) if stage=='coarse' else (64,32)
 return (slice(None),slice(None),slice(side*shift,side*shift+length),slice(None),slice(None))

def overlap(a,b,stage):
 shift=8 if stage=='coarse' else 32
 return a[:,:,shift:],b[:,:,:-shift]

def run(out,draws=4,nfe=128):
 import torch
 from workflows.sbi import e2e_loa_field_sampling as sampling
 from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
 from workflows.sbi import e2e_coupled_views as views
 from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
 from workflows.sbi.e2e_vdm_context_models import project,replicate,decode_density,block_mean
 c.require_compute();assert torch.cuda.device_count()==1 and nfe%2==0
 sampling.setup(True);models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
 receipt=json.loads((out/'CONDITIONS.json').read_text());results=[];torch.cuda.reset_peak_memory_stats()
 def noise(stage,cap):
  shape=(56,48,48) if stage=='coarse' else (96,48,48)
  parts=[]
  for d in range(draws):
   g=torch.Generator(device='cuda').manual_seed(int(c.digest(['global-probe-v1',cap,stage,d])[:15],16))
   parts.append(torch.randn((1,1,*shape),device='cuda',generator=g))
  z=torch.cat(parts);return project(z) if stage=='fine' else z
 def disagreement(a,b,stage):
  x,y=overlap(a,b,stage);difference=(x-y).float()
  spread=torch.sqrt(.5*(x.var(0,unbiased=True).mean()+y.var(0,unbiased=True).mean()))
  return {'overlap_rmse':float(difference.square().mean().sqrt()),'overlap_maxabs':float(difference.abs().max()),'rmse_over_draw_sd':float(difference.square().mean().sqrt()/spread.clamp_min(1e-12))}
 @torch.no_grad()
 def integrate(stage,conditions,base,fuse):
  model=models[stage];projector=project if stage=='fine' else lambda x:x
  state=base.clone() if fuse else [base[slices(stage,i)].clone() for i in range(2)]
  trace=[]
  def velocity(z,t):
   chunks=[z[slices(stage,i)] for i in range(2)] if fuse else z
   values=[model(x,model.schedule(x.new_full((len(x),),t)),condition) for x,condition in zip(chunks,conditions)]
   if not fuse:return values
   total=torch.zeros_like(z);weight=torch.zeros_like(z[:1])
   for i,v in enumerate(values):total[slices(stage,i)]+=v;weight[slices(stage,i)]+=1
   return projector(total/weight)
  for step in range(nfe//2):
   t=step/(nfe//2);dt=2/nfe;first=velocity(state,t)
   if fuse:
    proposal=projector(state+dt*first);state=projector(state+.5*dt*(first+velocity(proposal,t+dt)))
   else:
    proposal=[projector(z+dt*v) for z,v in zip(state,first)];second=velocity(proposal,t+dt)
    state=[projector(z+.5*dt*(v+w)) for z,v,w in zip(state,first,second)]
   if step in [0,nfe//4,nfe//2-1]:
    chunks=[state[slices(stage,i)] for i in range(2)] if fuse else state
    trace.append({'step':step+1,**disagreement(*chunks,stage)})
  return state,trace
 for case in receipt['cases']:
  cap=case['cap'];conditions=[];raw=[]
  for i,h in enumerate(case['hashes']):
   path=out/f'{cap}_{i}.npz';assert c.sha256(path)==h
   with np.load(path) as f:arrays={k:f[k] for k in f.files}
   raw.append(arrays);t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(draws,*x.shape)
   conditions.append(FieldCondition(t(views.scale_channels(arrays['joint'],chart['joint'])),t(views.scale_channels(arrays['wide'],chart['wide'])),t(np.zeros(3,'f4')),'wide'))
  result={'cap':cap,'tiles':[t['id'] for t in case['tiles']],'draws':draws,'nfe_per_factor_per_window':nfe,'variants':{}}
  coarse_noise=noise('coarse',cap);fine_noise=noise('fine',cap);crop=tuple(slice(*v) for v in op.layout()['joint_coarse_crop_in_wide'])
  for fuse in [False,True]:
   label='shared_initial_noise_only' if not fuse else 'shared_state_every_step';start=time.monotonic()
   z,zc=integrate('coarse',conditions,coarse_noise,fuse)
   zparts=[z[slices('coarse',i)] for i in range(2)] if fuse else z
   fine=[replace(b,region='joint',coarse_joint=replicate(v[(slice(None),slice(None),*crop)]),coarse_wide=v,coarse_source='sampled') for b,v in zip(conditions,zparts)]
   u,uc=integrate('fine',fine,fine_noise,fuse)
   uparts=[u[slices('fine',i)] for i in range(2)] if fuse else u
   densities=[];mass=[]
   for zz,uu in zip(zparts,uparts):
    w=torch.exp(zz.double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
    coarse=w[(slice(None),slice(None),*crop)];r=decode_density(coarse,uu.double()*chart['fine_residual']['std'][0])
    assert torch.isfinite(r).all() and (r>0).all();mass.append(float(((block_mean(r)-coarse)/coarse).abs().max()));densities.append(r)
   torch.cuda.synchronize();elapsed=time.monotonic()-start
   record={'seconds':elapsed,'coarse_path':zc,'fine_path':uc,'density_overlap':disagreement(*densities,'fine'),'max_relative_mass_error':max(mass),'positive_finite':True}
   if fuse:assert record['density_overlap']['overlap_maxabs']<1e-10
   assert max(mass)<2e-6
   np.savez_compressed(out/f'{cap}_{label}.npz',rho0=densities[0].cpu().numpy().astype('f4'),rho1=densities[1].cpu().numpy().astype('f4'))
   result['variants'][label]=record;print(json.dumps({'cap':cap,'variant':label,**record}),flush=True)
  results.append(result)
 record={'technical_passed':True,'posterior_qualified':False,'retraining_performed':False,'confirmation_opened':False,'scope':'Two overlapping Loa windows per cap, not a full survey posterior or calibration test. Consensus modifies the probability flow; shared-state equality is a construction check, not evidence of correct covariance.','results':results,'checkpoints':hashes,'normalizer':normalizer,'script_sha256':c.sha256(__file__),'job':os.environ['SLURM_JOB_ID'],'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
 c.atomic_json(out/'PROBE.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 prepare(a.root,a.out) if a.action=='prepare' else run(a.out)
