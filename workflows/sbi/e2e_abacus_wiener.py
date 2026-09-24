"""Frozen training-only Wiener baseline on legacy VDM anchors; batch compute only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator, cg

from workflows.sbi.e2e_field_build_products import require_compute, sha256, tensor_from_delta, eigs
from workflows.sbi.e2e_field_wide_coarse import padded_extract, block_sum
from workflows.sbi.e2e_vdm_context_data import read_json, spec
from workflows.sbi.e2e_vdm_context_dataset import Products
from workflows.sbi.e2e_vdm_context_products import verification_sources
from workflows.sbi.e2e_vdm_context_analysis import ensemble, density_spectra, CORE
from workflows.sbi.e2e_vdm_context_metrics import calibration, fair_energy
from workflows.sbi.e2e_conditional_reference import atomic_json


def covariance(x, spectrum):
    return fft.ifftn(fft.fftn(x, norm='ortho')*spectrum, norm='ortho').real


def draw_wiener(spectrum, response, noise, data, count, seed, tolerance=1e-7):
    """Precision-space constrained realizations, including the DC mode."""
    shape=spectrum.shape; size=spectrum.size
    if (any(v.shape!=shape for v in (response,noise,data)) or
        not all(np.isfinite(v).all() for v in (spectrum,response,noise,data)) or
        np.min(spectrum)<=0 or np.min(noise)<=0):
        raise ValueError('invalid finite positive covariance/noise inputs')
    weight=response**2/noise
    operator=LinearOperator((size,size),matvec=lambda x:
        (covariance(x.reshape(shape),1/spectrum)+weight*x.reshape(shape)).ravel())
    pre=1/(1/spectrum+weight.mean())
    preconditioner=LinearOperator((size,size),matvec=lambda x:covariance(x.reshape(shape),pre).ravel())
    residuals=[];iterations=[]
    def solve(rhs):
        steps=[0]
        def tick(x):steps[0]+=1
        value,info=cg(operator,rhs.ravel(),M=preconditioner,rtol=tolerance,atol=1e-12,maxiter=2000,callback=tick)
        residual=np.linalg.norm(operator@value-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
        if info or residual>max(2*tolerance,1e-10):
            raise RuntimeError(f'CG not converged: {info}, {residual}')
        residuals.append(float(residual));iterations.append(steps[0])
        return value.reshape(shape)
    mean=solve(response*data/noise)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(count):
        # C^-1/2 white + A^T N^-1/2 white has covariance equal to precision.
        rhs=covariance(rng.normal(size=shape),1/np.sqrt(spectrum))
        rhs+=response*rng.normal(size=shape)/np.sqrt(noise)
        draws.append(mean+solve(rhs))
    return np.asarray(draws),dict(max_relative_residual=max(residuals),max_iterations=max(iterations),mean_iterations=float(np.mean(iterations)))


def counts_for(row,sources):
    source=next(s for s in sources if s['phase']==row['phase'] and s['cap']==row['cap'])
    with h5py.File(source['response_file']['path'],'r') as f:
        values=[block_sum(padded_extract(f[key],np.asarray(row['center'])-48,96),2)
                for key in ('counts','expected_counts_random')]
    if any(v.shape!=(48,)*3 or not np.isfinite(v).all() or np.min(v)<0 for v in values):
        raise ValueError('invalid count/exposure products')
    if np.any((values[1]==0)&(values[0]!=0)):
        raise ValueError('nonzero counts outside random exposure')
    return values


def fit(root,output):
    """No held-out targets or counts can be read by this fit."""
    products=Products(root,['ph000','ph002','ph003'],targets=True,verify=True)
    rows=sorted(products.rows.values(),key=lambda r:r['anchor_id'])
    rows=[r for r in rows if not r.get('parent_domain_anchor')]
    if len(rows)!=384 or set(r['phase'] for r in rows)!={'ph000','ph002','ph003'}:
        raise ValueError('require frozen 384-core training panel')
    sources=read_json(spec()['screened_source'])['sources']
    verified={}
    for source in sources:
        if source['phase'] in products.phases:
            verified.update(verification_sources(source,spec()))
    totals={};mean=0.;n=48
    # Use temporary local arrays only; retained products stay unchanged.
    samples=[]
    for row in rows:
        delta=products.raw_targets(row['anchor_id'])['rho'].astype(float)-1
        counts,expected=counts_for(row,sources)
        key=f"{row['cap']}_{row['shell']}"
        pair=totals.setdefault(key,[0.,0.]);pair[0]+=float(counts.sum());pair[1]+=float(expected.sum())
        mean+=float(delta.mean())/len(rows)
        samples.append((key,delta,counts,expected))
    normalization={k:v[0]/v[1] for k,v in totals.items()}
    cross=denom=0.;power=np.zeros((n,)*3)
    for key,delta,counts,expected in samples:
        mu=expected*normalization[key];x=delta-mean
        cross+=float(np.sum(x*(counts-mu)));denom+=float(np.sum(mu*x*x))
        power+=abs(fft.fftn(x,norm='ortho'))**2/len(samples)
    bias=cross/denom
    if not np.isfinite(bias) or bias<=0:raise ValueError('nonpositive fitted response')
    k=np.meshgrid(*([fft.fftfreq(n)*n]*3),indexing='ij',sparse=True)
    shell=np.rint(np.sqrt(sum(v*v for v in k))).astype(int)
    powers=np.bincount(shell.ravel(),weights=power.ravel())/np.bincount(shell.ravel())
    floor=float(powers.max()*1e-6)
    spectrum=np.maximum(powers[shell],floor)
    residual=weight=0.
    for key,delta,counts,expected in samples:
        mu=expected*normalization[key]
        residual+=float(np.sum((counts-mu-bias*mu*(delta-mean))**2-mu))
        weight+=float(np.sum(mu*mu))
    extra=max(0.,residual/weight)
    np.savez(output/'PRIOR.npz',spectrum=spectrum)
    receipt=dict(fit_ids=[r['anchor_id'] for r in rows],fit_phases=list(products.phases),
        mean=mean,bias=bias,extra_variance=extra,exposure_normalization=normalization,
        spectrum_floor=floor,floored_mode_fraction=float(np.mean(powers[shell]<floor)),
        shell_power=powers.tolist(),product_receipts=products.receipts,raw_inputs=verified,
        geometry_sha256=sha256(root/'data/GEOMETRY.json'),prior_sha256=sha256(output/'PRIOR.npz'))
    atomic_json(output/'FIT.json',receipt)
    return receipt


def score(draws,truth,tensor_truth):
    x=draws[(slice(None),*CORE)].reshape(len(draws),-1)
    y=truth[CORE].ravel(); c=calibration(x,y)
    def summary(c):
        return dict(crps=float(c['crps'].mean()),bias=float(c['bias'].mean()),
                    coverage90=float(c['covered_0.9'].mean()),width90=float(c['width_0.9'].mean()),
                    rank_histogram=np.histogram(c['rank'],np.linspace(0,1,17))[0].tolist(),
                    attainable90=c['attainable']['0.9'])
    region=x.reshape(len(x),2,8,2,8,2,8).mean((2,4,6)).reshape(len(x),8)
    target=y.reshape(2,8,2,8,2,8).mean((1,3,5)).ravel()
    indices=np.arange(0,4096,16)
    eigen=np.asarray([eigs(tensor_from_delta(d,6.766)[CORE]).reshape(-1,3)[indices] for d in draws])
    target_eigen=eigs(tensor_truth).reshape(-1,3)[indices]
    labels=(eigen>0).sum(-1);actual=(target_eigen>0).sum(-1)
    probabilities=np.stack([(labels==k).mean(0) for k in range(4)],-1)
    spectra,_=density_spectra(draws,truth)
    result=dict(density=summary(c),regional=summary(calibration(region,target)),
        core_mass=summary(calibration(x.mean(1)[:,None],np.array([y.mean()]))),
        tidal=summary(calibration(eigen,target_eigen)),
        eigengap=summary(calibration(np.diff(eigen,axis=-1),np.diff(target_eigen,axis=-1))),
        tidal_energy=float(fair_energy(eigen,target_eigen).mean()),
        class_brier=float(np.mean(np.sum((probabilities-np.eye(4)[actual])**2,axis=-1))),
        unphysical_fraction=float(np.mean(x < -1)),spectra=spectra)
    return result


def one_case(args):
    root,output,anchor=args;root=Path(root);output=Path(output)
    started=time.monotonic(); fitted=read_json(output/'FIT.json')
    products=Products(root,['ph004','ph005'],targets=True,confirmation_receipt=root/'MODELS_FROZEN.json',verify=False)
    row=products.rows[anchor];counts,expected=counts_for(row,read_json(spec()['screened_source'])['sources'])
    mu=expected*fitted['exposure_normalization'][f"{row['cap']}_{row['shell']}"]
    response=fitted['bias']*mu
    # Zero exposure means zero response; its arbitrary positive N has no effect.
    noise=np.where(mu>0,mu+mu**2*fitted['extra_variance'],1.)
    target=products.raw_targets(anchor);truth=target['rho']-1
    spectrum=np.load(output/'PRIOR.npz')['spectrum']
    draws,solver=draw_wiener(spectrum,response,noise,counts-mu,64,70000+int(row['phase'][-3:])*100+sorted(products.rows).index(anchor))
    draws+=fitted['mean']
    np.savez_compressed(output/f'{anchor}_wiener.npz',delta=draws.astype('f4'))
    results={'Wiener':score(draws,truth,target['tensor'])};bindings={}
    ledger=read_json(root/'DRAW_LEDGER.json')
    tasks=[t for t in ledger['tasks'] if t['anchor']==anchor and t['purpose']=='main' and t['checkpoint']==20480 and t['steps']==250]
    if len(tasks)!=8:raise ValueError('missing matched neural tasks')
    for task in tasks:
        values,receipts=ensemble(root,task,count=64)
        results[f"{task['arm']}_seed{task['replica']}"]=score(values,truth,target['tensor'])
        bindings.update(receipts)
    # Quantify finite-domain operator error using true parent, not as a fitted correction.
    local_truth=eigs(tensor_from_delta(truth,6.766)[CORE])
    closure_error=float(np.sqrt(np.mean((local_truth-eigs(target['tensor']))**2)))
    result=dict(anchor=anchor,phase=row['phase'],cap=row['cap'],shell=row['shell'],support=row['support_stratum'],
        solver=solver,scores=results,true_parent_tidal_closure_rmse=closure_error,
        input_chunks=bindings,seconds=time.monotonic()-started)
    atomic_json(output/f'{anchor}.json',result)
    print('CASE',anchor,'seconds',result['seconds'],flush=True)
    return result


def run(args):
    require_compute();root=Path(args.root);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'MANIFEST.json',dict(source_sha256=sha256(__file__),root=str(root),
        original_manifest_sha256=sha256(root/'MANIFEST.json'),job=os.environ['SLURM_JOB_ID'],
        draws=64,scope='legacy coordinate frame, development comparison; no neural refit'))
    fit(root,output)
    # Freeze nuisance parameters before either held-out phase is read; verify targets once.
    Products(root,['ph004','ph005'],targets=True,confirmation_receipt=root/'MODELS_FROZEN.json',verify=True)
    verified={}
    for source in read_json(spec()['screened_source'])['sources']:
        if source['phase'] in ('ph004','ph005'):
            verified.update(verification_sources(source,spec()))
    atomic_json(output/'EVALUATION_INPUTS.json',verified)
    anchors=read_json(root/'DRAW_LEDGER.json')['panels']['evaluation']
    # Representative case is a fail-closed compute-node smoke before the full panel.
    first=one_case((str(root),str(output),anchors[0]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=[first]+list(pool.map(one_case,[(str(root),str(output),a) for a in anchors[1:]]))
    summaries={}
    for phase in ('ph004','ph005'):
        selected=[r for r in results if r['phase']==phase]
        summaries[phase]={name:{group:{key:float(np.mean([r['scores'][name][group][key] for r in selected]))
            for key in ('crps','coverage90','width90','bias')} for group in ('density','regional','core_mass','tidal','eigengap')}
            for name in selected[0]['scores']}
    atomic_json(output/'COMPLETE.json',dict(cases=len(results),summary=summaries,scientific_promotion=False,
        interpretation='two previously exposed phases; no independent-voxel uncertainty claims'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);run(p.parse_args())
