"""Observation-only multi-axis engineering smoke; never opens mock truth."""
import argparse,json,os,time,itertools
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
from workflows.sbi.e2e_global_sampler import integrate as pair_integrate

def main(root,out):
    c.require_compute();out.mkdir(parents=True,exist_ok=False);sampling.setup(True)
    plan=json.loads((root/'PLAN.json').read_text())
    models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    record={'job':os.environ['SLURM_JOB_ID'],'checkpoints':hashes,'normalizer':normalizer,
            'plan_sha256':c.sha256(root/'PLAN.json'),'posterior_qualified':False,'results':[]}
    for cap in ('NGC','SGC'):
        tiles={tuple(t['key']):t for t in plan['tiles'] if t['cap']==cap}
        offsets=list(itertools.product(range(2),repeat=3))
        candidates=[k for k in tiles if all(tuple(np.array(k)+o) in tiles for o in offsets)]
        target=json.loads((root/'details'/('detail_'+cap)/'COMPLETE.json').read_text())['case']['midpoint']
        key=min(candidates,key=lambda k:np.linalg.norm(np.asarray(tiles[k]['midpoint'])-target))
        selected=[tiles[tuple(np.array(key)+o)] for o in offsets];mid=[t['midpoint'] for t in selected]
        response=cond.load_response(cap);arrays=[];begin=time.monotonic()
        with h5py.File(root/(cap+'_observation_cache.h5'),'r') as f:
            for t in selected:
                arrays.append(cond.build(plan['grids'][cap],t['midpoint'],cap,f['counts_raw'],f['wide'],response))
                print(cap,'prepared',t['id'],flush=True)
        preparation=time.monotonic()-begin;count=4
        tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(count,*x.shape)
        base=[FieldCondition(tensor(views.scale_channels(a['joint'],chart['joint'])),
            tensor(views.scale_channels(a['wide'],chart['wide'])),tensor(np.zeros(3,'f4')),'wide') for a in arrays]
        cg,fg=(Geometry.make(mid,s) for s in ('coarse','fine'))
        torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        def noise(g,stage):
            rng=torch.Generator(device='cuda').manual_seed(int(c.digest(['canvas-smoke-v1',cap,stage])[:15],16))
            x=torch.randn((count,1,*g.shape),device='cuda',generator=rng)
            return project(x) if stage=='fine' else x
        z,cw=integrate(models['coarse'],'coarse',base,noise(cg,'coarse'),cg)
        fine=[]
        for i,b in enumerate(base):
            zw=z[cg.crop(i)]
            fine.append(replace(b,region='joint',coarse_joint=replicate(zw[:,:,16:32,18:30,18:30]),coarse_wide=zw,coarse_source='sampled'))
        u,fw=integrate(models['fine'],'fine',fine,noise(fg,'fine'),fg)
        parts=[];mass=[]
        for i in range(8):
            w=torch.exp(z[cg.crop(i)].double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            coarse=w[:,:,16:32,18:30,18:30]
            rho=decode_density(coarse,u[fg.crop(i)].double()*chart['fine_residual']['std'][0])
            assert torch.isfinite(rho).all() and (rho>0).all()
            mass.append(float(((block_mean(rho)-coarse)/coarse).abs().max()));parts.append(rho)
        # Independently decoded views must agree on every 3-D intersection.
        overlap=0.;comparisons=0
        for i in range(8):
            for j in range(i):
                a,b=np.array(fg.starts[i]),np.array(fg.starts[j]);lo=np.maximum(a,b);hi=np.minimum(a+fg.window,b+fg.window)
                if np.any(hi<=lo):continue
                cut=lambda p:(slice(None),slice(None),*(slice(int(l-q),int(h-q)) for l,h,q in zip(lo,hi,p)))
                overlap=max(overlap,float((parts[i][cut(a)]-parts[j][cut(b)]).abs().max()));comparisons+=1
        torch.cuda.synchronize();elapsed=time.monotonic()-begin
        assert overlap<1e-10 and max(mass)<2e-6
        result={'cap':cap,'windows':[t['id'] for t in selected],'draws':count,'nfe':128,
            'preparation_seconds':preparation,'sampling_seconds':elapsed,'overlap_maxabs':overlap,
            'intersection_checks':comparisons,'max_mass_error':max(mass),
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'fine_shape':list(map(int,fg.shape))}
        record['results'].append(result);print(json.dumps(result),flush=True)
        del parts,base,fine,z,u,cw,fw;torch.cuda.empty_cache()
    record['passed']=True;record['source']={p.name:c.sha256(p) for p in [Path(__file__),Path(__file__).with_name('e2e_global_canvas.py')]}
    c.atomic_json(out/'SMOKE.json',record)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
