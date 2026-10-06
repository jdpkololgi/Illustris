"""Stage B class reliability and tie-valid regional fraction diagnostics.

Rank inclusion is analytically averaged over the auxiliary tie/random-rank
uniform. Its exact exchangeable reference is nominal even with finite M/ties.
This is a calibration diagnostic, not an ordinary inclusive credible interval.
"""
import numpy as np

LEVELS = (.5, .68, .9, .95)
EDGES = np.array([0., .2, .4, .6, .8, 1.])


def rank_inclusion(less, ties, size, low, high):
    """Probability that (less + U*ties)/size is in [low,high]."""
    less, ties = np.asarray(less), np.asarray(ties)
    if np.any(ties < 1) or np.any(less < 0) or np.any(less+ties > size):
        raise ValueError('invalid rank multiplicities')
    return np.maximum(0., np.minimum(1., (high*size-less)/ties)
                      - np.maximum(0., (low*size-less)/ties))


def probability_bins(probabilities):
    p = np.asarray(probabilities)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('probabilities outside [0,1]')
    return np.minimum(np.searchsorted(EDGES, p, side='right')-1, 4)


def reliability(probabilities, truth, cores, draws=None):
    """Sufficient statistics aggregate exactly without averaging per-core rates."""
    p, y, core = np.asarray(probabilities,float), np.asarray(truth,int), np.asarray(cores)
    if p.shape != (len(y),4) or core.shape != y.shape or np.any((y < 0) | (y > 3)):
        raise ValueError('class probability, target, core shape mismatch')
    probability_bins(p)
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
    onehot = np.eye(4)[y]
    raw = ((p-onehot)**2).sum(0)
    correction = (p*(1-p)).sum(0)/(draws-1) if draws else np.zeros(4)
    bins=[]
    for k in range(4):
        addresses=probability_bins(p[:,k]); rows=[]
        for j in range(5):
            mask=addresses==j
            rows.append(dict(n=int(mask.sum()),probability_sum=float(p[mask,k].sum()),
                truth_sum=int((y[mask]==k).sum()),cores=np.unique(core[mask]).tolist()))
        bins.append(rows)
    hard=np.argmax(p,axis=-1)
    confusion=np.bincount(y*4+hard,minlength=16).reshape(4,4)
    return dict(n=len(y),cores=np.unique(core).tolist(),raw_brier_sum=raw.tolist(),
        fair_brier_sum=(raw-correction).tolist(),confusion=confusion.tolist(),bins=bins)


def merge_reliability(records):
    n=sum(r['n'] for r in records)
    if n==0:return dict(status='empty_support',galaxies=0)
    confusion=np.sum([r['confusion'] for r in records],axis=0)
    raw=np.sum([r['raw_brier_sum'] for r in records],axis=0)/n
    fair=np.sum([r['fair_brier_sum'] for r in records],axis=0)/n
    safe=lambda a,b: [float(x/y) if y else None for x,y in zip(a,b)]
    bins=[]
    for k in range(4):
        rows=[]
        for j in range(5):
            qq=[r['bins'][k][j] for r in records]
            count=sum(q['n'] for q in qq)
            cores=sorted({v for q in qq for v in q['cores']})
            mean=sum(q['probability_sum'] for q in qq)/count if count else None
            actual=sum(q['truth_sum'] for q in qq)/count if count else None
            rows.append(dict(bin=j,lower=float(EDGES[j]),upper=float(EDGES[j+1]),galaxies=count,
                cores=len(cores),mean_probability=mean,truth_frequency=actual,
                residual=actual-mean if count else None,
                screen='empty' if not count else ('sparse' if len(cores)<4 else
                    ('pass' if abs(actual-mean)<=.05 else 'fail'))))
        bins.append(rows)
    return dict(galaxies=n,cores=len({v for r in records for v in r['cores']}),
        brier=float(raw.sum()),fair_brier=float(fair.sum()),classwise_brier=raw.tolist(),
        classwise_fair_brier=fair.tolist(),confusion=confusion.tolist(),
        purity=safe(np.diag(confusion),confusion.sum(0)),
        completeness=safe(np.diag(confusion),confusion.sum(1)),bins=bins,
        status='descriptive_only_dependent_regions')


def joint_fraction_score(draws, truth):
    x,y=np.asarray(draws,float),np.asarray(truth,float)
    x=x.reshape(len(x),-1);y=y.ravel();z=np.vstack((y,x));n=len(z)
    if x.shape[1]!=len(y) or n<4 or not np.isfinite(z).all():
        raise ValueError('finite compatible joint fractions required')
    # Sorting makes each score invariant to input ordering. Identical discrete
    # vectors share exactly one score, avoiding round-off-induced tie splitting.
    unique,inverse,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    score=[]
    for i,value in enumerate(unique):
        weight=counts.copy();weight[i]-=1
        mean=np.sum(unique*weight[:,None],axis=0)/(n-1)
        centered=unique-mean
        cov=(centered*weight[:,None]).T@centered/(n-2)
        scale=max(float(np.trace(cov))/len(y),1e-12)
        cov=.8*cov+.2*np.diag(np.diag(cov))+np.eye(len(y))*1e-8*scale
        delta=value-mean
        score.append(float(delta@np.linalg.solve(cov,delta)))
    scores=np.asarray(score)[inverse];less=int((scores<scores[0]).sum());ties=int((scores==scores[0]).sum())
    distance=np.linalg.norm(x-y,axis=-1).mean()
    pair=sum(np.linalg.norm(x[i+1:]-x[i],axis=-1).sum() for i in range(len(x)-1))
    return dict(coverage={str(p):float(rank_inclusion(less,ties,n,0,p)) for p in LEVELS},
        reference={str(p):p for p in LEVELS},rank_less=less,rank_ties=ties,
        dimensions=len(y),degenerate=bool(np.max(x.std(0))<1e-12),
        energy=float(distance-pair/(len(x)*(len(x)-1))))


def fraction_score(draws, truth):
    x,y=np.asarray(draws,float),np.asarray(truth,float)
    if x.shape[1:]!=y.shape or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite matching fractions required')
    m=len(x);ordered=np.sort(x,axis=0)
    less=(x<y).sum(0);ties=(x==y).sum(0)+1
    weights=(2*np.arange(1,m+1)-m-1).reshape((m,)+(1,)*y.ndim)
    crps=abs(x-y).mean(0)-(weights*ordered).sum(0)/(m*(m-1))
    result=dict(bias=(x.mean(0)-y).tolist(),sd=x.std(0,ddof=1).tolist(),crps=crps.tolist(),
        truth=y.tolist(),mean=x.mean(0).tolist(),rank_less=less.tolist(),rank_ties=ties.tolist(),
        coverage={},reference={str(p):p for p in LEVELS},inclusive_coverage={},width={})
    for p in LEVELS:
        result['coverage'][str(p)]=rank_inclusion(less,ties,m+1,(1-p)/2,(1+p)/2).tolist()
        candidates=[(abs((m+1-2*k)/(m+1)-p),-(m+1-2*k)/(m+1),k) for k in range(1,m//2+1)]
        k=min(candidates)[2];lo,hi=ordered[k-1],ordered[m-k]
        result['inclusive_coverage'][str(p)]=((y>=lo)&(y<=hi)).tolist()
        result['width'][str(p)]=(hi-lo).tolist()
    return result


def region_fractions(labels, weights):
    """(draw,core,cell) labels; (core,cell) fixed nonnegative weights."""
    labels,weights=np.asarray(labels),np.asarray(weights,float)
    if labels.shape[1:]!=weights.shape or np.any(weights<0) or not np.isfinite(weights).all():
        raise ValueError('invalid fixed regional weights')
    denom=weights.sum(-1)
    result=np.full((len(labels),len(weights),4),np.nan)
    valid=denom>0
    for k in range(4):
        result[:,valid,k]=((labels[:,valid]==k)*weights[None,valid]).sum(-1)/denom[None,valid]
    return result,denom
