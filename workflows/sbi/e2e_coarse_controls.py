"""Frozen neural train/generalization check and Gaussianized classical control.

Only train plus already-open ph012-015; no fine sampling or retraining.
Classical likelihood uses a single coarse observation per cell, not independent
duplicate wide/local counts. Parameters are fitted on training anchors only.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy import fft
from scipy.sparse.linalg import LinearOperator,cg
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi.e2e_coupled_product_audit import read_target
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone,FieldCondition,sample
from workflows.sbi.e2e_cfm_pilot_evaluate import TRAIN_ROOT,draw_seed,regional
from workflows.sbi.e2e_vdm_context_metrics import calibration

OPEN=('ph012','ph013','ph014','ph015')
BASE=TRAIN_ROOT.parent

def guard(phase,fit=False):
    if phase not in (c.TRAIN if fit else c.TRAIN+OPEN):raise PermissionError('sealed/out-of-panel phase')

def panel():
    result=[]
    for j,phase in enumerate(c.TRAIN+OPEN):
        guard(phase)
        receipt=coord.verify_receipt(coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json',payload=False)
        ids=sorted(Path(x['path']).stem for x in receipt['pair_receipts'])
        # Match the sixteen cap/shell/support strata; no target-based selection.
        first=sorted(x for x in ids if x.endswith('_00'))
        if len(first)!=16:raise ValueError('expected16 matched strata')
        chosen=[first[k] for k in range(j%4,16,4)] if phase in c.TRAIN else first
        result.extend((phase,p) for p in chosen)
    return result

def target(phase,pair):
    guard(phase)
    directory=coord.ROOT/'targets'/phase
    receipt=coord.verify_receipt(directory/f'{pair}.json',payload=False)
    if receipt['phase']!=phase or receipt['pair_id']!=pair or len(receipt['outputs'])!=1:raise ValueError('target identity')
    item=receipt['outputs'][0];path=c.guarded(item['path'],phase)
    if path.parent!=directory.resolve() or c.sha256(path)!=item['sha256']:raise ValueError('target checksum')
    value=read_target(path,phase,pair)
    wide_slice,_=views.crop_slices(phase,(0,0,0))
    return value['coarse_rho_extended'][wide_slice],regional(value['rho_joint']-1)[:10],item['sha256']

def regions(wide,phase):
    _,crop=views.crop_slices(phase,(0,0,0))
    return regional(op.lift(wide[crop],4)-1)[:10]

def scored(draws,truth):
    out={}
    for name,sl in (('core_mass',slice(0,2)),('block_mass',slice(2,10))):
        v=calibration(draws[:,sl],truth[sl]);out[name]={
            'crps':float(v['crps'].mean()),'coverage90':float(v['covered_0.9'].mean()),
            'width90':float(v['width_0.9'].mean()),'attainable90':v['attainable']['0.9']}
    return dict(scores=out,truth=truth.tolist(),mean=draws.mean(0).tolist(),
        variance=draws.var(0,ddof=1).tolist(),rank=calibration(draws,truth)['rank'].tolist())

def covariance(x,power):
    return fft.idctn(fft.dctn(x,type=2,norm='ortho')*power,type=2,norm='ortho')

def gaussian_draws(power,response,noise,data,count,seed,tolerance=1e-7):
    """Constrained realizations; diagonal DCT prior, heteroscedastic likelihood."""
    shape=power.shape;n=power.size
    if any(v.shape!=shape for v in (response,noise,data)) or min(power.min(),noise.min())<=0:
        raise ValueError('shape/positive covariance')
    if not all(np.isfinite(v).all() for v in (power,response,noise,data)):raise ValueError('nonfinite inputs')
    weight=response**2/noise
    operator=LinearOperator((n,n),matvec=lambda v:(covariance(v.reshape(shape),1/power)+weight*v.reshape(shape)).ravel())
    pre=1/(1/power+weight.mean())
    preconditioner=LinearOperator((n,n),matvec=lambda v:covariance(v.reshape(shape),pre).ravel())
    residuals=[];iterations=[]
    def solve(rhs):
        steps=[0]
        def tick(_):steps[0]+=1
        x,info=cg(operator,rhs.ravel(),M=preconditioner,rtol=tolerance,atol=1e-12,maxiter=1500,callback=tick)
        residual=np.linalg.norm(operator@x-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
        if info or residual>max(2*tolerance,1e-10):raise RuntimeError(f'CG failed {info} {residual}')
        residuals.append(float(residual));iterations.append(steps[0]);return x.reshape(shape)
    mean=solve(response*data/noise);rng=np.random.default_rng(seed);out=[]
    for _ in range(count):
        rhs=covariance(rng.normal(size=shape),1/np.sqrt(power))+response*rng.normal(size=shape)/np.sqrt(noise)
        out.append(mean+solve(rhs))
    return np.asarray(out),dict(max_residual=max(residuals),max_iterations=max(iterations))

def observation(ob,chart,phase):
    # Undo global affine chart only; do NOT exponentiate averaged logs into counts.
    def raw(key):
        st=chart[key];return ob[key]*np.asarray(st['std'])[:,None,None,None]+np.asarray(st['mean'])[:,None,None,None]
    wide=raw('wide');joint=raw('joint');_,crop=views.crop_slices(phase,(0,0,0))
    y=wide[7].copy();exposure=wide[4].copy();support=wide[2].copy()
    # Replace, never double-count, overlapping coarse likelihood cells.
    y[crop]=op.mean_pool(joint[5],4);exposure[crop]=op.mean_pool(joint[3],4);support[crop]=op.mean_pool(joint[1],4)
    kind=np.zeros_like(y,dtype=int);kind[crop]=1
    valid=(support>0)&(exposure>0)
    # Fixed exposure bins, no held-out calibration or selection of bins.
    bins=np.digitize(exposure,(.1,.5,.9))
    return y,kind,bins,valid

def fit(root,normalizer,ids):
    chart=views.load_chart(normalizer);train=[v for v in ids if v[0] in c.TRAIN]
    accum=np.zeros((2,3,3));rhs=np.zeros((2,3));cache=[];power=np.zeros((48,)*3);mean=0.;hashes={}
    for phase,pair in train:
        guard(phase,fit=True);ob=views.load_observations(phase,pair,normalizer)
        rho,_,sha=target(phase,pair);x=np.log(rho);mean+=float(x.mean())/len(train)
        y,kind,bins,valid=observation(ob,chart,phase)
        # Free offset and galaxy response separately for local/wide encodings.
        for k in (0,1):
            mask=valid&(kind==k);design=np.stack([np.ones(mask.sum()),x[mask]],axis=1)
            accum[k,:2,:2]+=design.T@design;rhs[k,:2]+=design.T@y[mask]
        cache.append((x,y,kind,bins,valid));hashes[pair]=sha
    coefficients=np.stack([np.linalg.solve(accum[k,:2,:2],rhs[k,:2]) for k in (0,1)])
    if np.any(coefficients[:,1]<=0):raise ValueError('nonpositive fitted galaxy response')
    residual=np.zeros((2,4));counts=np.zeros((2,4))
    for x,y,kind,bins,valid in cache:
        power+=fft.dctn(x-mean,type=2,norm='ortho')**2/len(train)
        err=y-coefficients[kind,0]-coefficients[kind,1]*x
        for k in (0,1):
            for b in range(4):
                mask=valid&(kind==k)&(bins==b);residual[k,b]+=float(np.sum(err[mask]**2));counts[k,b]+=mask.sum()
    if np.min(counts.sum(1))<32:raise ValueError('insufficient training likelihood support')
    pooled=residual.sum(1)/counts.sum(1)
    noise=np.maximum(np.divide(residual,counts,out=np.broadcast_to(pooled[:,None],(2,4)).copy(),where=counts>=32),1e-6)
    axes=np.meshgrid(*([np.arange(48)]*3),indexing='ij',sparse=True)
    shell=np.rint(np.sqrt(sum(v*v for v in axes))).astype(int)
    radial=np.bincount(shell.ravel(),weights=power.ravel())/np.bincount(shell.ravel())
    radial=np.maximum(radial,radial.max()*1e-6)
    record=dict(normalizer=normalizer,fit_ids=train,target_hashes=hashes,mean=mean,coefficients=coefficients.tolist(),
        noise=noise.tolist(),noise_bin_counts=counts.tolist(),radial=radial.tolist(),runner=c.sha256(__file__),padding=8,
        limitations='Gaussianized log-density/log-ratio; diagonal residual noise; DCT reflecting padded boundary; not exact Poisson or full neural information equivalence')
    c.atomic_json(root/'FIT.json',record);print('FIT_COMPLETE',len(train),flush=True)

def classical(ob,phase,record,count,seed,tolerance=1e-7):
    chart=views.load_chart(record['normalizer']);y,kind,bins,valid=observation(ob,chart,phase)
    coef=np.asarray(record['coefficients']);noise=np.asarray(record['noise'])[kind,bins]
    response=np.where(valid,coef[kind,1],0.);data=y-coef[kind,0]-coef[kind,1]*record['mean']
    pad=record['padding'];n=48+2*pad;axes=np.meshgrid(*([np.arange(n)*48/n]*3),indexing='ij',sparse=True)
    radius=np.sqrt(sum(v*v for v in axes));radial=np.asarray(record['radial'])
    power=np.interp(radius,np.arange(len(radial)),radial)
    # Padding contains no observation likelihood; no periodic wrapping.
    values,info=gaussian_draws(power,np.pad(response,pad),np.pad(noise,pad,constant_values=1),np.pad(data,pad),count,seed,tolerance)
    wide=np.exp(values[:,pad:pad+48,pad:pad+48,pad:pad+48]+record['mean'])
    if not np.isfinite(wide).all():raise ValueError('nonfinite lognormal draws')
    return np.stack([regions(w,phase) for w in wide]),info

def neural(a,root,ids,normalizer):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    path=TRAIN_ROOT/f'coarse_{a.seed}'/f'CHECKPOINT_{a.step:06d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=False);bound=state['binding']
    if (state['step']!=a.step or bound['seed']!=a.seed or bound['stage']!='coarse' or bound['normalizer']!=normalizer
        or bound['model']!=c.sha256(inspect.getfile(CoupledBackbone))):raise ValueError('checkpoint identity')
    model=CoupledBackbone('coarse','wide',24,3).cuda().eval();model.load_state_dict(state['ema'])
    chart=views.load_chart(normalizer);out=root/f'neural_{a.seed}_{a.step}';out.mkdir(exist_ok=True)
    binding=dict(checkpoint=c.sha256(path),runner=c.sha256(__file__),normalizer=normalizer,ids=ids,weights='ema',nfe=128)
    c.atomic_json(out/'BINDING.json',binding,replace=True)
    for phase,pair in ids:
        if phase not in c.TRAIN:continue
        dest=out/f'{pair}.json'
        if dest.exists():
            if json.loads(dest.read_text())['binding']!=c.digest(binding):raise ValueError('resume mismatch')
            continue
        ob=views.load_observations(phase,pair,normalizer);t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(2,*x.shape)
        cond=FieldCondition(t(ob['joint']),t(ob['wide']),t(ob['joint_center_from_wide_mpc_h']),'wide')
        draws=[]
        for first in range(0,32,2):
            seeds=[draw_seed(a.seed,phase,pair,i,'coarse') for i in (first,first+1)]
            z=sample(model,cond,'cfm',64,seeds).cpu().numpy()[:,0]
            wide=np.exp(z.astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            if not np.isfinite(wide).all():raise ValueError('nonfinite neural draw')
            draws.extend(regions(w,phase) for w in wide)
        _,truth,sha=target(phase,pair);draws=np.asarray(draws)
        c.atomic_json(dest,dict(phase=phase,pair=pair,binding=c.digest(binding),truth_sha256=sha,draws=draws.tolist(),**scored(draws,truth)))
        print('NEURAL_CASE',a.seed,a.step,pair,flush=True)
    c.atomic_json(out/'COMPLETE.json',dict(cases=52,binding=c.digest(binding)),replace=True)

def run(a):
    c.require_compute();root=Path(a.root);root.mkdir(exist_ok=True,parents=True)
    ids=panel();normalizer=c.sha256(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json')
    if a.mode=='fit':fit(root,normalizer,ids);return
    if a.mode=='neural':neural(a,root,ids,normalizer);return
    record=json.loads((root/'FIT.json').read_text());out=root/'classical';out.mkdir(exist_ok=True)
    completed=0
    for index,(phase,pair) in enumerate(ids):
        if index%4!=a.shard:continue
        dest=out/f'{pair}.json'
        if dest.exists():continue
        started=time.monotonic();ob=views.load_observations(phase,pair,normalizer);seed=draw_seed(901,phase,pair,0,'coarse')
        draws,info=classical(ob,phase,record,32,seed)
        if index==a.shard:
            tighter,_=classical(ob,phase,record,2,seed,1e-8)
            if not np.allclose(tighter,draws[:2],rtol=1e-4,atol=1e-5):raise ValueError('classical tolerance instability')
        _,truth,sha=target(phase,pair)
        c.atomic_json(dest,dict(phase=phase,pair=pair,fit_sha256=c.sha256(root/'FIT.json'),truth_sha256=sha,
            draws=draws.tolist(),solver=info,seconds=time.monotonic()-started,**scored(draws,truth)))
        print('CLASSICAL_CASE',pair,'seconds',time.monotonic()-started,flush=True)
        completed+=1
        if a.limit and completed>=a.limit:break

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=('fit','neural','classical'),required=True)
    p.add_argument('--seed',type=int,choices=(17,29));p.add_argument('--step',type=int,choices=(13312,26624));p.add_argument('--shard',type=int,choices=range(4),default=0)
    p.add_argument('--limit',type=int,default=0)
    run(p.parse_args())
