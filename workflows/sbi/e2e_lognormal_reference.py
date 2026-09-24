"""Bounded lognormal-Poisson MALA reference versus Laplace, development only."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from workflows.sbi.e2e_conditional_reference_math import prior,observation_templates
from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_reference_usecase import smooth,tidal


class Posterior:
    def __init__(self,cov,exposure,counts):
        self.l=np.linalg.cholesky(cov);self.shift=.5*np.diag(cov)
        self.exposure=np.asarray(exposure);self.counts=np.asarray(counts)
        if np.any((self.exposure==0)&(self.counts!=0)):raise ValueError('counts in masked cell')
    def value_grad(self,z):
        eta=z@self.l.T-self.shift
        invalid=np.any(eta>700,axis=-1)
        rate=self.exposure*np.exp(np.minimum(eta,700))
        value=.5*np.sum(z*z,axis=-1)+np.sum(rate-self.counts*eta,axis=-1)
        grad=z+(rate-self.counts)@self.l
        return np.where(invalid,np.inf,value),np.where(np.asarray(invalid)[...,None],0,grad)
    def hessian(self,z):
        rate=self.exposure*np.exp(z@self.l.T-self.shift)
        return np.eye(len(z))+self.l.T@(rate[:,None]*self.l)
    def density(self,z):return np.expm1(z@self.l.T-self.shift)


def laplace(p):
    start=time.monotonic()
    fit=minimize(lambda z:p.value_grad(z),np.zeros(len(p.shift)),jac=True,hess=p.hessian,method='trust-exact',options={'gtol':1e-8,'maxiter':500})
    grad=np.max(abs(p.value_grad(fit.x)[1]))
    if grad>1e-6:raise ValueError(f'MAP not stationary: {grad}')
    factor=np.linalg.cholesky(np.linalg.inv(p.hessian(fit.x)))
    return fit.x,factor,dict(seconds=time.monotonic()-start,gradient_max=float(grad),iterations=int(fit.nit))


def sample(p,mode,factor,seed,draws=32768,warmup=2048):
    rng=np.random.default_rng(seed);u=rng.normal(size=(8,len(mode)))*2
    def vg(x):
        val,g=p.value_grad(mode+x@factor.T);return val,g@factor
    val,g=vg(u);logh=np.full(8,np.log(.4));accepted=np.zeros(8);saved=[];start=time.monotonic()
    for step in range(warmup+draws):
        if step==warmup:logh[4:]+=np.log(.7)
        h=np.exp(logh)[:,None];forward=u-.5*h*h*g
        proposal=forward+h*rng.normal(size=u.shape);vnew,gnew=vg(proposal)
        reverse=proposal-.5*h*h*gnew
        ratio=val-vnew+(np.sum((proposal-forward)**2,1)-np.sum((u-reverse)**2,1))/(2*h[:,0]**2)
        take=np.log(rng.random(8))<np.minimum(ratio,0)
        u=np.where(take[:,None],proposal,u);val=np.where(take,vnew,val);g=np.where(take[:,None],gnew,g)
        if step<warmup:
            logh=np.clip(logh+.05/(1+step/50)**.6*(take-.574),-5,0)
        else:
            accepted+=take;saved.append((mode+u@factor.T).copy())
    return np.stack(saved,axis=1),dict(seconds=time.monotonic()-start,acceptance=(accepted/draws).tolist(),step_size=np.exp(logh).tolist(),warmup=warmup)


def features(p,z):
    delta=p.density(z);g=z@p.l.T
    # Include every latent and density coordinate plus region, log-density and tail-sensitive summaries.
    regions=delta.reshape(*delta.shape[:-1],4,4,4).reshape(*delta.shape[:-1],2,2,2,2,2,2).mean(axis=(-1,-3,-5)).reshape(*delta.shape[:-1],8)
    return np.concatenate([g,delta,regions,delta.mean(-1,keepdims=True),p.value_grad(z)[0][...,None]],axis=-1)


def convergence(p,z):
    import arviz as az
    x=features(p,z);data=az.from_dict(posterior={'q':x})
    r=az.rhat(data,method='rank')['q'].values
    bulk=az.ess(data,method='bulk')['q'].values;tail=az.ess(data,method='tail')['q'].values
    # Independently stepped chain groups must agree, using autocorrelation-aware mean MCSE.
    groups=[]
    for group in [x[:4],x[4:]]:
        d=az.from_dict(posterior={'q':group})
        groups.append((group.mean((0,1)),az.mcse(d,method='mean')['q'].values))
    agreement=np.abs(groups[0][0]-groups[1][0])/np.sqrt(groups[0][1]**2+groups[1][1]**2)
    finite=all(np.isfinite(a).all() for a in [r,bulk,tail,agreement])
    return dict(rhat_max=float(r.max()),bulk_ess_min=float(bulk.min()),tail_ess_min=float(tail.min()),
        group_mean_zmax=float(agreement.max()),passed=bool(finite and r.max()<1.01 and bulk.min()>400 and tail.min()>400 and agreement.max()<5),
        diagnostic_count=x.shape[-1],arviz_version=az.__version__)


def probs(delta,r):
    count=np.zeros((64,4))
    for first in range(0,len(delta),128):
        labels=(np.linalg.eigvalsh(tidal(delta[first:first+128],r,n=4))>0).sum(-1).reshape(-1,64)
        for j in range(4):count[:,j]+=(labels==j).sum(0)
    return count/len(delta)


def compare(reference,approx):
    rows=[]
    for r in [0.,.5,1.]:
        a=smooth(reference,r,4);b=smooth(approx,r,4);var=a.var(0,ddof=1)
        lo,hi=np.quantile(b,[.05,.95],axis=0)
        rows.append(dict(r_cells=r,mean_error_sd=float(np.sqrt(np.mean((b.mean(0)-a.mean(0))**2/var))),
            variance_ratio=float(np.mean(b.var(0,ddof=1)/var)),
            interval_reference_mass=float(np.mean((a>=lo)&(a<=hi))),
            class_probability_rms=float(np.sqrt(np.mean((probs(reference,r)-probs(approx,r))**2)))))
    a=reference.mean(1);b=approx.mean(1);lo,hi=np.quantile(b,[.05,.95])
    return dict(scales=rows,regional_mass_mean_error_sd=float((b.mean()-a.mean())/a.std(ddof=1)),
        regional_mass_variance_ratio=float(b.var(ddof=1)/a.var(ddof=1)),regional_interval_reference_mass=float(np.mean((a>=lo)&(a<=hi))))


def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('compute allocation required')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    paths=['workflows/sbi/e2e_lognormal_reference.py','workflows/sbi/e2e_reference_usecase.py',
        'workflows/sbi/e2e_conditional_reference_math.py','tests/test_e2e_lognormal_reference.py','docs/e2e_lognormal_reference_20260924.md']
    hashes={p:digest(repo/p) for p in paths}
    atomic_json(root/'manifest.json',dict(sources=hashes,grid=4,cases=4,chains=8,draws_per_chain=32768,warmup=2048,scope='synthetic development; no DESI likelihood'))
    cov,_,_=prior(4);mask,_=observation_templates(4);results=[]
    for index in range(4):
        rng=np.random.default_rng(45000+index);rate=[.5,5.][index//2]
        exposure=rate*mask[index%2]*np.linspace(.5,1.5,64)
        truth=rng.normal(size=64)@np.linalg.cholesky(cov).T
        counts=rng.poisson(exposure*np.exp(truth-.5*np.diag(cov)))
        p=Posterior(cov,exposure,counts);mode,factor,fit=laplace(p)
        z,timing=sample(p,mode,factor,46000+index)
        np.savez(root/f'case{index}.npz',z=z,counts=counts,exposure=exposure,truth=truth,mode=mode,factor=factor)
        diag=convergence(p,z);result=dict(case=index,expected_rate=rate,observed_count=int(counts.sum()),map=fit,sampling=timing,convergence=diag)
        if diag['passed']:
            # Deterministic spaced subset in each group; do not treat it as independent for convergence.
            ref=p.density(z[:4,::8].reshape(-1,64));other=p.density(z[4:,::8].reshape(-1,64))
            tick=time.monotonic();approx=p.density(mode+rng.normal(size=ref.shape)@factor.T)
            result['laplace_draw_seconds']=time.monotonic()-tick
            result['laplace']=compare(ref,approx);result['reference_replication']=compare(ref,other)
        atomic_json(root/f'case{index}.json',result);results.append(result)
        print('CASE',index,json.dumps(diag),flush=True)
    if hashes!={p:digest(repo/p) for p in paths}:raise ValueError('source drift')
    atomic_json(root/'COMPLETE.json',dict(results=results,all_references_pass=all(r['convergence']['passed'] for r in results),sources=hashes))
    lines=['# Lognormal-Poisson: classical reference versus Laplace','',
        '4^3 development toy, known prior and selection, no bias/RSD/cosmology uncertainty. No neural training.',
        'Convergence checks are evidence, not proof; interval mass is conditional reference mass, not survey-wide SBC.',
        '', '| Case | Rate | Rhat max | Bulk ESS min | Tail ESS min | Pass |', '|---|---:|---:|---:|---:|---|']
    for r in results:
        d=r['convergence'];lines.append(f'| {r["case"]} | {r["expected_rate"]} | {d["rhat_max"]:.4f} | {d["bulk_ess_min"]:.0f} | {d["tail_ess_min"]:.0f} | {d["passed"]} |')
    lines+=['','| Case | R/cells | Laplace mean error/sd | Variance ratio | 90% interval ref mass | Tidal probability RMS | Reference replication RMS |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if 'laplace' not in r:continue
        for v,w in zip(r['laplace']['scales'],r['reference_replication']['scales']):
            lines.append(f'| {r["case"]} | {v["r_cells"]} | {v["mean_error_sd"]:.4f} | {v["variance_ratio"]:.4f} | {v["interval_reference_mass"]:.4f} | {v["class_probability_rms"]:.4f} | {w["class_probability_rms"]:.4f} |')
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args())
