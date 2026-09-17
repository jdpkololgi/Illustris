"""Coupled finite-step sampling and estimand-explicit field calibration metrics."""
import math
import numpy as np
import torch
from workflows.sbi.e2e_field_build_products import tensor_from_delta, eigs


@torch.no_grad()
def coupled_sample(model, condition, steps, seeds, noise_grid=1000):
    """Same initial noise and Brownian increments across checkpoints/resolutions.

    Coarse innovations are normalized sums of fine iid innovations, hence each
    resolution retains the exact registered ancestral marginal transition.
    This is a common-Brownian coupling, not an assertion that the finite-step
    ancestral transition is an exact reverse SDE solution.
    Each sample owns its RNG, independent of microbatch partition/resumption.
    """
    if not isinstance(steps,int) or steps<1 or noise_grid%steps:
        raise ValueError('steps must be a positive divisor of noise_grid')
    if len(seeds)!=len(condition) or len(set(seeds))!=len(seeds):
        raise ValueError('one distinct addressed seed per sample required')
    generators=[torch.Generator(device=condition.device).manual_seed(int(s)) for s in seeds]
    shape=(1,1,*condition.shape[2:])
    def noise():
        return torch.cat([torch.randn(shape,device=condition.device,dtype=condition.dtype,generator=g) for g in generators])
    before=model.training;model.eval();z=noise();factor=noise_grid//steps
    try:
        for i in range(steps):
            gt=model.schedule(z.new_full((len(z),),1-i/steps))
            gs=model.schedule(z.new_full((len(z),),1-(i+1)/steps))
            at,st=model.schedule.coefficients(gt);ass,ss=model.schedule.coefficients(gs)
            c=-torch.expm1(gs-gt);b=lambda v:v[:,None,None,None,None]
            mean=b(ass/at)*(z-b(c*st)*model(z,gt,condition))
            eps=noise()
            for _ in range(factor-1):eps=eps+noise()
            z=mean+b(ss*c.sqrt())*eps/math.sqrt(factor)
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite coupled trajectory')
        return z
    finally:model.train(before)


def pool2(x):
    x=np.asarray(x);n=x.shape[0]
    if x.shape[:3]!=(n,n,n) or n%2:raise ValueError('even cubic leading axes required')
    return x.reshape(n//2,2,n//2,2,n//2,2,*x.shape[3:]).mean((1,3,5))


def probes(n=48,core=16,stride=2):
    if not 0<core<=n or (n-core)%2 or core%stride:raise ValueError('invalid core/probe grid')
    start=(n-core)//2;return (slice(start,start+core,stride),)*3


def tensor_features(tensor,delta,core=16,stride=2):
    loc=probes(len(delta),core,stride);e=eigs(tensor[loc]).reshape(-1,3)
    return np.concatenate([delta[loc].reshape(-1,1),e,np.diff(e,axis=-1)],axis=1).astype(np.float32)


def tidal_features(delta,boundary,cell=6.766,core=16,stride=2):
    """No extra R7 smoothing. DC isotropic mean/3 retains trace=delta.

    Pad to2N for zero/reflect; report closure-dependent values separately.
    No method supplies the missing exterior mass or earns physical calibration.
    """
    x=np.asarray(delta,dtype=np.float64);n=len(x)
    if not np.isfinite(x).all():raise ValueError('nonfinite field')
    if boundary=='periodic':t=tensor_from_delta(x,cell)
    elif boundary in ('zero','reflect'):
        pad=n//2
        padded=np.pad(x,pad,mode='constant' if boundary=='zero' else 'reflect')
        t=tensor_from_delta(padded,cell)[pad:pad+n,pad:pad+n,pad:pad+n]
    else:raise ValueError('unknown boundary closure')
    if np.max(np.abs(t[...,[0,3,5]].sum(-1)-x))>1e-8*max(1.,float(np.abs(x).max())):
        raise ValueError('tidal trace closure failed')
    return tensor_features(t,x,core,stride)


def regional_density(delta):
    n=len(delta)
    return (np.asarray(delta)+1).reshape(2,n//2,2,n//2,2,n//2).mean((1,3,5)).flatten()


def calibration(draws,truth,levels=(.5,.68,.9,.95),seed=73):
    """Pointwise empirical coverage/ranks/CRPS, NOT independent field-level SBC.

    Leading draw axis; arbitrary parameter axes thereafter. Retain per-parameter
    values so spatial aggregation cannot manufacture independent sample sizes.
    Randomized tie ranks have uniform jitter across their rank support.
    """
    x=np.asarray(draws,dtype=np.float64);y=np.asarray(truth,dtype=np.float64);m=len(x)
    if m<2 or x.shape[1:]!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite matching posterior ensemble/truth required')
    less=(x<y).sum(0);equal=(x==y).sum(0)
    rank=(less+np.random.default_rng(seed).random(y.shape)*(equal+1))/(m+1)
    ordered=np.sort(x,axis=0);weights=(2*np.arange(1,m+1)-m-1).reshape((m,)+(1,)*y.ndim)
    crps=np.mean(np.abs(x-y),axis=0)-np.sum(weights*ordered,axis=0)/m**2
    result=dict(rank=rank,rank_less=less,ties=equal,mean=x.mean(0),std=x.std(0,ddof=1),
                bias=x.mean(0)-y,crps=crps)
    for level in levels:
        low,high=np.quantile(x,[(1-level)/2,(1+level)/2],axis=0)
        result[f'covered_{level}']=(y>=low)&(y<=high)
        result[f'width_{level}']=high-low
    return result


def summarize_calibration(draws,truth,mask=None,levels=(.5,.68,.9,.95)):
    c=calibration(draws,truth,levels)
    mask=np.ones(np.shape(truth),bool) if mask is None else np.broadcast_to(mask,np.shape(truth))
    if not mask.any():return dict(count=0)
    take=lambda k:np.asarray(c[k])[mask]
    return dict(count=int(mask.sum()),draws=len(draws),bias=float(take('bias').mean()),
        rmse_mean=float(np.sqrt(np.mean(take('bias')**2))),rms_spread=float(np.sqrt(np.mean(take('std')**2))),
        mean_crps=float(take('crps').mean()),rank_histogram=np.histogram(take('rank'),bins=np.linspace(0,1,17))[0].tolist(),
        coverage={str(l):float(take(f'covered_{l}').mean()) for l in levels},
        interval_width={str(l):float(take(f'width_{l}').mean()) for l in levels},
        caveat='spatially correlated grid/region probes, not independent simulation replications or joint-field SBC')
