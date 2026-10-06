"""Paired whole-draw MC calculations for the frozen Tier 2 development screen."""
import numpy as np

M=32
REPLICATES=4096
SEED=6100617
LEVELS=(.5,.68,.9,.95)


def resamples():
    index=np.random.default_rng(SEED).integers(0,M,size=(REPLICATES,M))
    counts=np.stack([np.bincount(row,minlength=M) for row in index])
    return index,counts


def energy_kernel(draws,truth):
    """Equal-case mean sufficient statistics; draws=(case,draw,coordinate)."""
    x,y=np.asarray(draws,float),np.asarray(truth,float)
    if x.ndim!=3 or x.shape[1]!=M or y.shape!=(len(x),x.shape[-1]) or not len(x):
        raise ValueError('Nonempty case/32-draw/vector bank required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite energy bank')
    first=np.linalg.norm(x-y[:,None],axis=-1).mean(0)
    pair=np.linalg.norm(x[:,:,None]-x[:,None,:],axis=-1).mean(0)
    return first,pair


def brier_kernel(labels,truth,weights):
    """Squared-one-hot energy equals fair multiclass Brier; fixed cell weights.

    Arrays have shapes (draw,cell), (cell,), (cell,). Galaxy multiplicities
    implement pooled-galaxy weighting; ones implement pooled-cell weighting.
    """
    x,y,w=np.asarray(labels),np.asarray(truth),np.asarray(weights,float)
    if x.shape!=(M,len(y)) or w.shape!=y.shape or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid Brier addresses/weights')
    if np.any((x<0)|(x>3)) or np.any((y<0)|(y>3)) or not np.isfinite(w).all():
        raise ValueError('Invalid classes/weights')
    w=w/w.sum();first=2*((x!=y)*w).sum(1);pair=np.zeros((M,M))
    for i in range(M-1):
        pair[i,i+1:]=2*((x[i+1:]!=x[i])*w).sum(1)
        pair[i+1:,i]=pair[i,i+1:]
    return first,pair


def score_from_kernel(kernel,counts=None):
    first,pair=map(np.asarray,kernel)
    if first.shape!=(M,) or pair.shape!=(M,M) or not np.isfinite(first).all() or not np.isfinite(pair).all():
        raise ValueError('Finite 32-draw score kernel required')
    if not np.allclose(pair,pair.T,atol=1e-14) or not np.allclose(np.diag(pair),0,atol=1e-14):
        raise ValueError('Pair distances must be symmetric with zero diagonal')
    if counts is None:return float(first.mean()-pair.sum()/(2*M*(M-1)))
    w=np.asarray(counts,float)
    if w.ndim!=2 or w.shape[1]!=M or np.any(w<0) or not np.all(w.sum(1)==M):
        raise ValueError('Whole-draw bootstrap multiplicities required')
    return w@first/M-np.einsum('bi,ij,bj->b',w,pair,w,optimize=True)/(2*M*(M-1))


def deterioration(candidate,control,counts):
    point=score_from_kernel(candidate)-score_from_kernel(control)
    samples=score_from_kernel(candidate,counts)-score_from_kernel(control,counts)
    lo,hi=np.quantile(samples,[.05,.95])
    return dict(point=float(point),lower=float(lo),upper=float(hi),limit=0.,
        decision='fail' if point>0 else ('inconclusive' if hi>0 else 'pass'))


def ratio_samples(draws,truth,counts):
    """Trace Q / trace mean S, including the specified finite-32-draw correction."""
    x,y=np.asarray(draws,float),np.asarray(truth,float)
    if x.ndim!=3 or x.shape[1]!=M or y.shape!=(len(x),x.shape[-1]):
        raise ValueError('Expected case/draw/vector moment bank')
    w=np.asarray(counts,float)
    mean=np.einsum('bm,cmd->bcd',w,x,optimize=True)/M
    raw2=np.einsum('bm,cm->bc',w,(x*x).sum(-1),optimize=True)
    variance=((raw2-M*(mean*mean).sum(-1))/(M-1)).mean(-1)
    mse=((mean-y)**2).sum(-1).mean(-1)
    if np.any(variance<=1e-15):raise ValueError('Degenerate predictive trace')
    return (mse-variance/M)/variance


def cell_curve_samples(draws,truth,counts,chunk=2048):
    """Exact resampled central-order coverage without storing bootstrap fields.

    Count comparisons against truth, not interpolated quantiles. Inclusive ties
    match the existing continuous-marginal scorer. Returns MC sensitivities,
    not an independence claim about spatial cells or a replacement gate.
    """
    x,y=np.asarray(draws),np.asarray(truth)
    if x.shape!=(M,len(y)) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite draw/cell bank required')
    w=np.asarray(counts,dtype='f4');total=np.zeros((len(w),len(LEVELS)))
    orders=[]
    for level in LEVELS:
        k=min((abs((M+1-2*k)/(M+1)-level),-(M+1-2*k)/(M+1),k) for k in range(1,M//2+1))[2]
        orders.append((k,M-k))
    for start in range(0,len(y),chunk):
        cut=slice(start,start+chunk)
        less=w@(x[:,cut]<y[cut]).astype('f4')
        leq=w@(x[:,cut]<=y[cut]).astype('f4')
        for j,(k,hi) in enumerate(orders):
            total[:,j]+=((leq>=k)&(less<=hi)).sum(1)
    return total/len(y)


def combine(decisions):
    values=set(decisions)
    if not values or not values<={'pass','fail','inconclusive'}:raise ValueError('Explicit gate decisions required')
    return 'fail' if 'fail' in values else ('inconclusive' if 'inconclusive' in values else 'pass')
