"""Frozen shared-state full-cap density inference; exploratory, not calibrated.

No truth inputs, no retraining, no writes to the source atlas or production VAC.
Window networks use a single synchronized canvas at both Heun evaluations.
"""
import argparse,json,os,time
from pathlib import Path
from dataclasses import replace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import h5py
from workflows.sbi import e2e_coupled_contract as c,e2e_coupled_views as views
from workflows.sbi import e2e_loa_field_conditions as cond,e2e_loa_field_sampling as sampling
from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
from workflows.sbi.e2e_vdm_context_models import project,replicate,decode_density,block_mean
from workflows.sbi.e2e_global_canvas import Geometry,integrate

def run(root,out,cap,smoke,benchmark=False):
    c.require_compute();sampling.setup(True)
    if not json.loads(smoke.read_text())['passed']:raise ValueError('Multi-axis smoke missing')
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text());selected=[t for t in plan['tiles'] if t['cap']==cap]
    mid=[t['midpoint'] for t in selected];cg,fg=(Geometry.make(mid,s) for s in ('coarse','fine'))
    models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    binding={'cap':cap,'draws':4,'nfe':128,'checkpoints':hashes,'normalizer':normalizer,
        'source':{p.name:c.sha256(p) for p in [Path(__file__),Path(__file__).with_name('e2e_global_canvas.py'),Path(cond.__file__),Path(sampling.__file__)]},
        'plan':c.sha256(root/'PLAN.json'),'smoke':c.sha256(smoke),'posterior_qualified':False,'job':os.environ['SLURM_JOB_ID']}
    c.atomic_json(out/'BINDING.json',binding)
    cache=json.loads((root/(cap+'_CACHE.json')).read_text())
    if c.sha256(cache['outputs'][0]['path'])!=cache['outputs'][0]['sha256']:raise ValueError('Observation cache drift')
    response=cond.load_response(cap);base=[];support=np.zeros(fg.shape,dtype='f4');condition_hashes=[];start=time.monotonic()
    tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(4,*x.shape)
    # Full-condition preparation is included in the bound, never run on a login node.
    with h5py.File(root/(cap+'_observation_cache.h5'),'r') as f:
        for i,t in enumerate(selected):
            a=cond.build(plan['grids'][cap],t['midpoint'],cap,f['counts_raw'],f['wide'],response)
            import hashlib
            h=hashlib.sha256()
            for k in ('joint','wide'):h.update(a[k].tobytes())
            condition_hashes.append(h.hexdigest())
            sl=fg.crop(i)[2:];support[sl]=np.maximum(support[sl],a['support'])
            base.append(FieldCondition(tensor(views.scale_channels(a['joint'],chart['joint'])),
                tensor(views.scale_channels(a['wide'],chart['wide'])),tensor(np.zeros(3,'f4')),'wide'))
            if i%100==0:print(json.dumps({'cap':cap,'conditions':i+1,'total':len(selected),'seconds':time.monotonic()-start}),flush=True)
    c.atomic_json(out/'CONDITIONS.json',{'windows':[t['id'] for t in selected],'condition_sha256':condition_hashes})
    np.save(out/'support.npy',support);del a
    preparation=time.monotonic()-start;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    def noise(g,stage):
        # Draw-wise streams are independent of batching; geometry is bound in PLAN.
        rows=[]
        for draw in range(4):
            rng=torch.Generator(device='cuda').manual_seed(int(c.digest(['global-cap-v1',cap,stage,draw])[:15],16))
            rows.append(torch.randn((1,1,*g.shape),device='cuda',generator=rng))
        x=torch.cat(rows);return project(x) if stage=='fine' else x
    def checkpoint(stage):
        def save(step,state):
            if step%8==0:
                torch.save({'stage':stage,'step':step,'state':state.cpu(),'binding':binding},out/(stage+'_step.pt.tmp'))
                (out/(stage+'_step.pt.tmp')).replace(out/(stage+'_step.pt'))
                print(json.dumps({'cap':cap,'stage':stage,'step':step,'seconds':time.monotonic()-start}),flush=True)
        return save
    # Benchmark has the actual full-cap condition residency and reduction canvas,
    # but only one Heun step. It is never exposed as a posterior output.
    nfe=2 if benchmark else 128
    z,cw=integrate(models['coarse'],'coarse',base,noise(cg,'coarse'),cg,nfe,checkpoint('coarse'))
    fine=[]
    for i,b in enumerate(base):
        zw=z[cg.crop(i)]
        fine.append(replace(b,region='joint',coarse_joint=replicate(zw[:,:,16:32,18:30,18:30]),coarse_wide=zw,coarse_source='sampled'))
    u,fw=integrate(models['fine'],'fine',fine,noise(fg,'fine'),fg,nfe,checkpoint('fine'))
    torch.cuda.synchronize();seconds=time.monotonic()-start
    if benchmark:
        result={'passed':True,'cap':cap,'windows':len(selected),'draws':4,'nfe':2,'preparation_seconds':preparation,
            'one_heun_step_both_factors_seconds':seconds,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'fine_shape':list(map(int,fg.shape)),'coarse_shape':list(map(int,cg.shape)),
            'posterior_output':False,'warning':'Startup and conditioning included in stage timing; no production forecast guarantee.'}
        c.atomic_json(out/'BENCHMARK.json',result);print(json.dumps(result),flush=True);return
    # Decode in block-aligned slabs to cap temporary memory; preserve coarse mass.
    mass=0.;offset=np.asarray(fg.origin)//4-np.asarray(cg.origin);origin=np.asarray(plan['grids'][cap]['origin_mpc_h'])+np.asarray(fg.origin)*6.766
    with h5py.File(out/'FIELD.h5','x') as f:
        ds=f.create_dataset('rho',shape=(4,*fg.shape),dtype='f4',chunks=(1,16,32,32),compression='lzf')
        f.create_dataset('support',data=support,compression='lzf');f.create_dataset('coverage',data=fw[0,0].cpu().numpy().astype('u2'),compression='lzf')
        f.create_dataset('coarse_logrho_standardized',data=z[:,0].cpu().numpy(),compression='lzf')
        f.attrs['origin_mpc_h']=origin;f.attrs['cell_mpc_h']=6.766;f.attrs['cap']=cap
        f.attrs['posterior_qualified']=False;f.attrs['shared_state']=True
        f.attrs['coarse_origin_raw_cells']=np.asarray(cg.origin)*8
        for lo in range(0,fg.shape[0],16):
            hi=min(lo+16,fg.shape[0]);startc=offset+np.array([lo//4,0,0]);size=np.array([hi-lo,fg.shape[1],fg.shape[2]])//4
            cut=(slice(None),slice(None),*(slice(int(a),int(a+n)) for a,n in zip(startc,size)))
            coarse=torch.exp(z[cut].double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            rho=decode_density(coarse,u[:,:,lo:hi].double()*chart['fine_residual']['std'][0])
            if not torch.isfinite(rho).all() or not (rho>0).all():raise ValueError('Invalid density')
            mass=max(mass,float(((block_mean(rho)-coarse)/coarse).abs().max()))
            values=rho[:,0].cpu().numpy().astype('f4');values[:,fw[0,0,lo:hi].cpu().numpy()==0]=np.nan
            ds[:,lo:hi]=values
    assert mass<2e-6
    result={'passed':True,'cap':cap,'draws':4,'windows':len(selected),'preparation_seconds':preparation,'sampling_seconds':seconds,
        'max_mass_error':mass,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'posterior_qualified':False,
        'field':c.file_record(out/'FIELD.h5',content_hash=True),'binding':c.sha256(out/'BINDING.json')}
    c.atomic_json(out/'COMPLETE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cap',choices=['NGC','SGC'],required=True);p.add_argument('--smoke',type=Path,required=True);p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();run(a.root,a.out,a.cap,a.smoke,a.benchmark)
