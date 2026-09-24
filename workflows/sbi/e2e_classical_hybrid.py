"""Gaussian classical competitor and conditional low-mode hybrid, no fitting."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_math import problem,probes,metrics
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_reference_usecase import smooth,regions,linear_summary,class_prob
from workflows.sbi.e2e_partial_whitening import context


def wiener_setup(prior,case):
    observed=np.flatnonzero(case['mask']);noise=case['std'][observed]
    cross=prior[:,observed]
    gain=cho_solve(cho_factor(prior[np.ix_(observed,observed)]+np.diag(noise**2)),cross.T).T
    return observed,noise,gain


def constrained_draws(prior_chol,case,setup,rng,count):
    observed,noise,gain=setup
    x=rng.normal(size=(count,len(prior_chol)))@prior_chol.T
    mock=x[:,observed]+rng.normal(size=(count,len(observed)))*noise
    return x+(case['y'][observed]-mock)@gain.T


def low_basis(radius):
    n=len(radius);size=n**3
    projector=np.fft.ifftn(np.fft.fftn(np.eye(size).reshape(size,n,n,n),axes=(1,2,3))*(radius<=1),axes=(1,2,3)).real.reshape(size,size)
    values,basis=np.linalg.eigh(projector)
    low=basis[:,values>.5];high=basis[:,values<=.5]
    return np.concatenate([low,high],axis=1),low.shape[1]


def hybrid_setup(case,basis,k):
    cov=basis.T@case['sigma']@basis;mu=case['mu']@basis
    a=cho_solve(cho_factor(cov[k:,k:]),cov[k:,:k]).T
    conditional=cov[:k,:k]-a@cov[k:,:k]
    return mu,a,np.linalg.cholesky(conditional),np.linalg.cholesky(cov[:k,:k])


def hybrid(draws,basis,k,setup,rng,independent=False):
    mu,a,chol,marginal=setup;z=draws@basis
    noise=rng.normal(size=(len(z),k))
    z[:,:k]=mu[:k]+(0 if independent else (z[:,k:]-mu[k:])@a.T)+noise@(marginal if independent else chol).T
    return z@basis.T


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('compute allocation required')
    source=Path(args.source);m=context(source)
    if not (source/'COMPLETE.json').exists():raise ValueError('finish original evaluation first')
    selection=json.loads((source/'COMPLETE.json').read_text())['selection']['candidate']
    alpha=selection['alpha'];out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    prior,pchol,_,_,cases,_,radius=problem(8,4);q=probes(8);basis,k=low_basis(radius)
    rows=[];timing=[];inputs={};started=time.monotonic()
    paths=['workflows/sbi/e2e_classical_hybrid.py','workflows/sbi/e2e_reference_usecase.py',
           'workflows/sbi/e2e_conditional_reference_math.py','tests/test_e2e_classical_hybrid.py',
           'docs/e2e_classical_hybrid_20260924.md']
    repo=Path(__file__).resolve().parents[2]
    hashes={p:digest(repo/p) for p in paths}
    atomic_json(out/'manifest.json',dict(sources=hashes,source_run=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        selected_alpha=alpha,scope='development Gaussian only',low_rank=k,cutoff='|k|<=1 including DC',draws=2048))
    for index,case in enumerate(cases):
        rng=np.random.default_rng(31000+index)
        reference=rng.normal(size=(2048,512))@case['chol'].T+case['mu']
        tick=time.monotonic();ws=wiener_setup(prior,case);hs=hybrid_setup(case,basis,k)
        setup_seconds=time.monotonic()-tick
        observed,noise,gain=ws
        np.testing.assert_allclose(gain@case['y'][observed],case['mu'],atol=1e-9)
        np.testing.assert_allclose(prior-gain@prior[observed],case['sigma'],atol=1e-9)
        tick=time.monotonic();classical=constrained_draws(pchol,case,ws,rng,2048);draw_seconds=time.monotonic()-tick
        arms=[('classical',classical),('oracle_hybrid_control',hybrid(reference,basis,k,hs,rng))]
        for seed in [17,29]:
            item=next(i for i in m['items'] if i['alpha']==alpha and i['seed']==seed and not i['exact'])
            path=source/'results'/item['name']/'ema'/'checkpoint_98304'/f'draws_98304_{index}_256.npy'
            x=np.load(path);inputs[str(path)]=digest(path)
            if x.shape!=(2048,512) or not np.isfinite(x).all():raise ValueError('invalid neural draws')
            tick=time.monotonic();h=hybrid(x,basis,k,hs,np.random.default_rng(32000+index+seed))
            timing.append(dict(case=index,seed=seed,hybrid_correction_seconds=time.monotonic()-tick))
            arms.extend([(f'neural_{seed}',x),(f'conditional_hybrid_{seed}',h),
                         (f'independent_splice_{seed}',hybrid(x,basis,k,hs,np.random.default_rng(32000+index+seed),True))])
        timing.append(dict(case=index,classical_setup_including_hybrid_seconds=setup_seconds,classical_2048_draw_seconds=draw_seconds))
        for name,x in arms:
            np.save(out/f'draws_case{index}_{name}.npy',x)
        for r in [0.,1.,2.]:
            pref=class_prob(reference,r);qs=smooth(np.eye(512),r).T
            for name,x in arms:
                diff=class_prob(x,r)-pref
                rows.append(dict(case=index,arm=name,r_cells=r,
                    class_probability_rms=float(np.sqrt(np.mean(diff**2))),class_probability_max_abs=float(np.max(abs(diff))),
                    smoothed=linear_summary(x,case,qs),regional={str(w):linear_summary(x,case,qq) for w,qq in regions().items()},
                    field=metrics(x,case,q,radius)))
        atomic_json(out/f'case{index}.json',dict(rows=[v for v in rows if v['case']==index]))
    if hashes!={p:digest(repo/p) for p in paths}:raise ValueError('source changed during execution')
    atomic_json(out/'COMPLETE.json',dict(rows=rows,timing=timing,seconds=time.monotonic()-started,
        inputs=inputs,sources=hashes,selected_alpha=alpha,selection=selection,
        caveat='Known Gaussian covariance used in hybrid conditional; not an Abacus-ready learned residual model. Hybrid still pays for neural draws.'))
    lines=['# Classical / conditional-hybrid comparison','',
        'Gaussian development only. All models use the same prior/observation law. R in cells; threshold-zero toy tidal classes.',
        'Independent splice is a negative control, not a calibrated construction. Classical draws are a competitor, not merely a label oracle.',
        '', '| Arm | R | Field mean error | Covariance | Smoothed variance | Class RMS | Regional width4 coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,_ in arms:
        for r in [0.,1.,2.]:
            rr=[v for v in rows if v['arm']==name and v['r_cells']==r]
            lines.append(f'| {name} | {r} | {np.mean([v["field"]["mean_rms"] for v in rr]):.5f} | {np.mean([v["field"]["covariance_relative"] for v in rr]):.5f} | {np.mean([v["smoothed"]["variance_ratio_mean"] for v in rr]):.5f} | {np.mean([v["class_probability_rms"] for v in rr]):.5f} | {np.mean([v["regional"]["4"]["exact_reference_interval_coverage_mean"] for v in rr]):.5f} |')
    lines+=['','CPU seconds for 2048 classical draws per case: '+str([v['classical_2048_draw_seconds'] for v in timing if 'classical_2048_draw_seconds' in v]),
        'No neural training required for classical draws; hybrid retains the neural generation cost. No cross-hardware speedup claim.',
        'A Gaussian result does not validate a nonlinear lognormal/Poisson reference or DESI robustness.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);run(p.parse_args())
