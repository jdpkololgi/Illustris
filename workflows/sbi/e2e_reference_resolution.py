"""Nested physical Gaussian reference; equal observations, not equal voxel noise."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, eigh

from workflows.sbi.e2e_conditional_reference import Data, atomic_json, atomic_checkpoint, evaluate
from workflows.sbi.e2e_conditional_reference_math import observation_templates, probes, null_thresholds, metrics
from workflows.sbi.e2e_conditional_reference_continue import digest, restore_training_state
from workflows.sbi.e2e_reference_target_controls import Teacher, step, excess_risk
from workflows.sbi.e2e_direct_vdm import ConditionalVDM


def physical_prior(n, cutoff=8):
    """Same continuous periodic Fourier series evaluated at n^3 locations.

    Sums both +/- cutoff modes. Aliases are summed, never dropped or renormalized
    separately per grid; C16 on the nested 8^3 sites therefore equals C8.
    """
    ks = np.arange(-cutoff, cutoff+1)
    modes = np.stack(np.meshgrid(ks,ks,ks,indexing='ij'),axis=-1).reshape(-1,3)
    spectral = (1+(np.linalg.norm(modes,axis=1)/1.5)**2)**-2
    spectral /= spectral.sum()
    power = np.zeros((n,n,n))
    np.add.at(power,tuple((modes % n).T),spectral*n**3)
    eye = np.eye(n**3).reshape(n**3,n,n,n)
    covariance = np.fft.ifftn(np.fft.fftn(eye,axes=(1,2,3))*power,axes=(1,2,3)).real.reshape(n**3,n**3)
    k = np.fft.fftfreq(n)*n
    radius = np.sqrt(sum(a*a for a in np.meshgrid(k,k,k,indexing='ij')))
    return covariance, radius


def nested_indices(n, coarse=8):
    if n % coarse:
        raise ValueError('non-nested grid')
    j = np.indices((coarse,coarse,coarse)).reshape(3,-1)*(n//coarse)
    return np.ravel_multi_index(tuple(j), (n,n,n))


def build_problem(n, common, observations):
    covariance, radius = physical_prior(n)
    nested = nested_indices(n)
    if not np.allclose(covariance[np.ix_(nested,nested)],common,rtol=1e-11,atol=1e-11):
        raise AssertionError('nested physical covariance mismatch')
    masks8, std8 = observation_templates(8)
    masks = np.zeros((2,n**3))
    std = np.ones_like(masks)
    masks[:,nested] = masks8
    std[:,nested] = std8
    templates=[]
    for which in range(2):
        active8=np.flatnonzero(masks8[which])
        active=nested[active8]
        cross=covariance[:,active]
        obs=common[np.ix_(active8,active8)]+np.diag(std8[which,active8]**2)
        gain=cho_solve(cho_factor(obs),cross.T).T
        sigma=covariance-gain@cross.T
        sigma=(sigma+sigma.T)/2
        values,vectors=eigh(sigma,driver='evd',check_finite=False)
        if values.min() <= 0:
            raise AssertionError('posterior not positive definite')
        factor=np.linalg.cholesky(sigma)
        # Kalman identity independently checks the observation-noise relation.
        if not np.allclose(gain,sigma[:,active]/std8[which,active8]**2,rtol=1e-8,atol=1e-9):
            raise AssertionError('posterior gain identity failed')
        templates.append(dict(sigma=sigma,chol=factor,values=values,vectors=vectors,gain=gain,active=active))
    cases=[]
    for index,y8 in enumerate(observations):
        which=index%2
        template=templates[which]
        active8=np.flatnonzero(masks8[which])
        y=np.zeros(n**3); y[nested]=y8
        mu=template['gain']@y8[active8]
        cases.append({k:template[k] for k in ['sigma','chol','values','vectors']} |
                     dict(mu=mu,y=y,mask=masks[which],std=std[which]))
    return dict(n=n,chol=np.linalg.cholesky(covariance),mask=masks,std=std,cases=cases,radius=radius)


class PhysicalData(Data):
    def __init__(self,path,device,draws=512):
        raw=torch.load(path,map_location='cpu',weights_only=False)
        self.n=raw['n']; self.cases=raw['cases']; self.radius=raw['radius']; self.q=probes(self.n)
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        self.chol,self.mask,self.std=map(tensor,(raw['chol'],raw['mask'],raw['std']))
        self.conditions=[tensor(np.stack([c['y'],c['mask'],c['std']])).reshape(1,3,self.n,self.n,self.n) for c in self.cases]
        self.means=[tensor(c['mu']) for c in self.cases]
        self.factors=[tensor(c['chol']) for c in self.cases]
        self.nulls=[null_thresholds(c,self.q,draws,128) for c in self.cases]


def prepare(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('reference preparation requires compute allocation')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    common,_=physical_prior(8)
    masks,std=observation_templates(8)
    rng=np.random.default_rng(90123)
    chol=np.linalg.cholesky(common)
    observations=[masks[i%2]*(chol@rng.normal(size=512)+std[i%2]*rng.normal(size=512)) for i in range(4)]
    prior8=None
    for n in [8,16]:
        tic=time.monotonic()
        raw=build_problem(n,common,observations)
        if n==8:
            prior8=raw
        else:
            ix=nested_indices(16)
            for a,b in zip(prior8['cases'],raw['cases']):
                if not np.allclose(a['mu'],b['mu'][ix],atol=1e-10) or not np.allclose(a['sigma'],b['sigma'][np.ix_(ix,ix)],atol=1e-10):
                    raise AssertionError('nested posterior mismatch')
        torch.save(raw,root/f'problem_{n}.pt')
        print('PREPARED',n,time.monotonic()-tic,flush=True)
    files=[Path(__file__),repo/'workflows/sbi/e2e_reference_target_controls.py',
           repo/'workflows/sbi/e2e_conditional_reference.py',repo/'workflows/sbi/e2e_conditional_reference_math.py',
           repo/'workflows/sbi/e2e_conditional_reference_continue.py',repo/'workflows/sbi/e2e_direct_vdm.py',
           repo/'configs/e2e_conditional_reference_v1.json']
    sources={str(p.relative_to(repo)):digest(p) for p in files}
    for p in files:
        dest=root/'source'/p.relative_to(repo); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    cfg=json.loads((repo/'configs/e2e_conditional_reference_v1.json').read_text())
    tasks=[dict(n=n,seed=s,fixed=f,name=f'n{n}_seed{s}_'+('amortised' if f is None else 'fixed0'))
           for s in [17,29] for n in [8,16] for f in [None,0]]
    atomic_json(root/'manifest.json',dict(sources=sources,config=cfg,items=tasks,updates=16384,
        problems={str(n):digest(root/f'problem_{n}.pt') for n in [8,16]},job=os.environ['SLURM_JOB_ID'],
        nested_prior_and_posterior_verified=True,physical_cutoff=8))


def worker(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    torch.set_num_threads(4)
    root=Path(args.output); m=json.loads((root/'manifest.json').read_text())
    for relative,expected in m['sources'].items():
        if digest(root/'source'/relative)!=expected: raise ValueError('source changed')
    rank=int(os.environ.get('SLURM_PROCID','0'))
    # Each GPU owns one grid/mode for both seeds; avoid loading grids repeatedly.
    owned=m['items'][rank::4]
    if args.smoke: owned=[m['items'][2]]
    cfg=m['config']; data=None
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>6800: raise TimeoutError('resolution run checkpoint stop')
    receipts=[]
    for item in owned:
        n,seed,fixed=item['n'],item['seed'],item['fixed']
        if data is None or data.n!=n:
            if digest(root/f'problem_{n}.pt')!=m['problems'][str(n)]: raise ValueError('reference changed')
            data=PhysicalData(root/f'problem_{n}.pt',torch.device('cuda'))
            teacher=Teacher(data)
            precise=copy.copy(data)
            precise.nulls=[null_thresholds(c,data.q,2048,128) for c in data.cases]
        out=root/'results'/item['name']
        if not args.smoke: out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(seed)
        model=ConditionalVDM(3,8,2,False).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        gen=torch.Generator(device='cuda').manual_seed(seed+3000)
        first=0
        if not args.smoke and (out/'latest.pt').exists():
            saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
            if saved['sources']!=m['sources'] or saved['item']!=item: raise ValueError('resume mismatch')
            restore_training_state(model,optimizer,gen,saved); first=saved['update']
        tic=time.monotonic(); losses=[]
        total=64 if args.smoke else m['updates']
        for update in range(first+1,total+1):
            guard()
            x,c=data.batch(32,fixed,gen)
            losses.append(step(model,optimizer,teacher,x,c,gen,True))
            if not args.smoke and update%1024==0:
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator=gen.get_state(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),update=update,sources=m['sources'],item=item)
                atomic_checkpoint(out/'latest.pt',state)
                with (out/'learning.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=update,loss=float(np.mean(losses))))+'\n')
                print('TRAIN',item['name'],update,float(np.mean(losses)),flush=True);losses=[]
            if not args.smoke and update in [8192,16384]:
                atomic_checkpoint(out/f'checkpoint_{update}.pt',state)
                evaluate(model,'cfm',data,range(4) if fixed is None else [0],cfg|{'nfe':[128]},seed,update,out,guard)
        if args.smoke:
            torch.cuda.synchronize()
            receipts.append(dict(item=item,seconds_per_update=(time.monotonic()-tic)/64,loss=losses[-1]))
        else:
            final=out/'precision';final.mkdir(exist_ok=True)
            evaluate(model,'cfm',precise,range(4) if fixed is None else [0],cfg|{'nfe':[128,256],'draws':2048},seed,total,final,guard)
            atomic_json(final/'risk.json',excess_risk(model,teacher,data,fixed,seed))
            atomic_json(final/'nulls.json',precise.nulls)
    atomic_json(root/('SMOKE.json' if args.smoke else f'worker_{rank}_COMPLETE.json'),
                dict(job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,sources=m['sources'],items=owned,timings=receipts))


def collect(args):
    root=Path(args.output);m=json.loads((root/'manifest.json').read_text())
    workers=[json.loads((root/f'worker_{r}_COMPLETE.json').read_text()) for r in range(4)]
    if any(w['sources']!=m['sources'] for w in workers): raise ValueError('worker provenance mismatch')
    coarse=torch.load(root/'problem_8.pt',map_location='cpu',weights_only=False)
    rows=[]
    for item in m['items']:
        for case in (range(4) if item['fixed'] is None else [0]):
            for nfe in [128,256]:
                d=root/'results'/item['name']/'precision'
                r=json.loads((d/f'evaluation_16384_{case}_{nfe}.json').read_text())
                draws=np.load(d/f'draws_16384_{case}_{nfe}.npy',mmap_mode='r')
                if draws.shape!=(2048,item['n']**3) or not np.isfinite(draws).all(): raise ValueError('invalid samples')
                common=metrics(draws[:,nested_indices(item['n'])],coarse['cases'][case],probes(8),coarse['radius'])
                rows.append(r|item|{'power_pass':all(.9<=v<=1.1 for v in r['power_ratio']),
                                   'common_observable_metrics':common})
    atomic_json(root/'COMPLETE.json',dict(rows=rows,workers=workers,sources=m['sources']))
    for n in [8,16]:
        for fixed in [None,0]:
            rr=[r for r in rows if r['n']==n and r['fixed']==fixed and r['nfe']==256]
            print(n,fixed,{k:float(np.mean([r[k] for r in rr])) for k in ['mean_rms','covariance_relative','octant_coverage']},
                  'power',np.mean([r['power_ratio'] for r in rr],axis=0).tolist(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','collect'])
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();globals()[args.command](args)
