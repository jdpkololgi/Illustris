"""Finite-ensemble proper scores and exact attainable central coverage."""
import numpy as np


def central_order_interval(size, level):
    if size < 2 or not 0 < level < 1:
        raise ValueError('ensemble size>=2 and probability in (0,1) required')
    candidates = [(abs((size+1-2*k)/(size+1)-level), -(size+1-2*k)/(size+1), k)
                  for k in range(1, size//2+1)]
    _, _, k = min(candidates)
    # zero-based order indices; finite-M probability from uniform truth rank.
    return k-1, size-k, (size+1-2*k)/(size+1)


def calibration(draws, truth, levels=(.5, .68, .9, .95), seed=73):
    x, y = np.asarray(draws, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    m = len(x)
    if m < 2 or x.shape[1:] != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite compatible draws/truth required')
    ordered = np.sort(x, axis=0)
    weights = (2*np.arange(1, m+1)-m-1).reshape((m,)+(1,)*y.ndim)
    # U-statistic pair correction: unbiased proper score, unlike empirical CRPS.
    crps = np.abs(x-y).mean(0)-(weights*ordered).sum(0)/(m*(m-1))
    less, equal = (x < y).sum(0), (x == y).sum(0)
    rank = (less+np.random.default_rng(seed).random(y.shape)*(equal+1))/(m+1)
    result = dict(mean=x.mean(0), std=x.std(0, ddof=1), bias=x.mean(0)-y,
                  crps=crps, rank=rank, rank_less=less, ties=equal, attainable={})
    for level in levels:
        lo, hi, attainable = central_order_interval(m, level)
        result['attainable'][str(level)] = attainable
        result[f'covered_{level}'] = (y >= ordered[lo]) & (y <= ordered[hi])
        result[f'width_{level}'] = ordered[hi]-ordered[lo]
    return result


def fair_energy(draws, truth):
    """Last axis is a vector; first is draw. O(M^2) without quadratic storage."""
    x, y = np.asarray(draws, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    m = len(x)
    if m < 2 or x.shape[1:] != y.shape or x.ndim < 2:
        raise ValueError('compatible vector ensembles required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite score input')
    result = np.linalg.norm(x-y, axis=-1).mean(0)
    pair = np.zeros_like(result)
    for i in range(m-1):
        pair += np.linalg.norm(x[i+1:]-x[i], axis=-1).sum(0)
    return result-pair/(m*(m-1))


def variogram_score(draws, truth, power=.5):
    x, y = np.asarray(draws, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if x.shape[1:] != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError('draws x vector required')
    i, j = np.triu_indices(len(y), 1)
    moment = np.abs(x[:, i]-x[:, j])**power
    residual = np.abs(y[i]-y[j])**power-moment.mean(0)
    # Fair finite-M correction for squared Monte-Carlo mean error.
    return float(np.mean(residual**2-moment.var(0, ddof=1)/len(x)))
