"""New diagnostic-only moments, paired MC ranges, and physical spectra."""
import numpy as np
from scipy import fft

M = 32
SEED = 6100617
REPLICATES = 4096


def resamples():
    index = np.random.default_rng(SEED).integers(0, M, (REPLICATES, M))
    return index, np.stack([np.bincount(row, minlength=M) for row in index])


def interval(values):
    lo, hi = np.quantile(values, [.05, .95], axis=0)
    return dict(lower=np.asarray(lo).tolist(), upper=np.asarray(hi).tolist())


def moments(draws, truth, counts):
    x, y = np.asarray(draws, float), np.asarray(truth, float)
    if x.ndim != 3 or x.shape[1] != M or y.shape != (len(x), x.shape[-1]):
        raise ValueError('Expected case/32-draw/coordinate bank')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonempty finite bank required')
    mean = x.mean(1); error = mean-y; bias = error.mean(0)
    dx = x-mean[:, None]
    s = np.einsum('nmi,nmj->ij', dx, dx)/(len(x)*(M-1))
    second = error.T@error/len(x)
    q = second-s/M
    centered = (error-bias).T@(error-bias)/len(x)-s/M
    np.testing.assert_allclose(q, centered+np.outer(bias, bias), atol=2e-14)
    trace = float(np.trace(s))
    ratio = float(np.trace(q)/trace) if trace > 1e-15 else None
    centered_ratio = float(np.trace(centered)/trace) if trace > 1e-15 else None
    c = np.asarray(counts, float)
    if c.shape[1] != M or np.any(c < 0) or not np.all(c.sum(1) == M):
        raise ValueError('Whole-draw bootstrap counts required')
    mu = np.einsum('bm,nmd->bnd', c, x, optimize=True)/M
    raw2 = np.einsum('bm,nmd->bnd', c, x*x, optimize=True)
    var = ((raw2-M*mu*mu)/(M-1)).mean(1)
    err = mu-y; boot_bias = err.mean(1)
    adjusted = (err*err).mean(1)-var/M
    centered_diag = ((err-boot_bias[:, None])**2).mean(1)-var/M
    boot_trace = var.sum(-1)
    good = boot_trace > 1e-15
    br = adjusted.sum(-1)[good]/boot_trace[good]
    bc = centered_diag.sum(-1)[good]/boot_trace[good]
    coordinate = []
    for j, v in enumerate(np.diag(s)):
        valid = var[:, j] > 1e-15
        coordinate.append(dict(point=float(q[j, j]/v) if v > 1e-15 else None,
            mc=interval(adjusted[valid, j]/var[valid, j]) if valid.any() else None,
            valid_bootstrap_replicates=int(valid.sum())))
    return dict(cases=len(x), draws=M, dimensions=x.shape[-1], bias=bias.tolist(),
        predictive_covariance=s.tolist(), adjusted_error_second_moment=q.tolist(),
        centered_adjusted_error_second_moment=centered.tolist(),
        trace_predictive_variance=trace, ratio=ratio, centered_ratio=centered_ratio,
        classwise_or_coordinate_ratios=coordinate,
        mc=dict(ratio=interval(br) if len(br) else None,
            centered_ratio=interval(bc) if len(bc) else None,
            bias=interval(boot_bias), trace_predictive_variance=interval(boot_trace),
            valid_ratio_replicates=int(good.sum()), replicates=len(c), seed=SEED,
            scope='conditional whole-draw MC only; no case or phase resampling'))


def fractions_report(draws, truth, counts):
    x, y = np.asarray(draws), np.asarray(truth)
    if x.shape[1:] != (M, 2, 4) or y.shape != (len(x), 2, 4):
        raise ValueError('Expected case/32-draw/two-core/four-class fractions')
    np.testing.assert_allclose(x.sum(-1), 1, atol=1e-12)
    np.testing.assert_allclose(y.sum(-1), 1, atol=1e-12)
    return dict(joint=moments(x[..., :3].reshape(len(x), M, 6), y[..., :3].reshape(len(y), 6), counts),
        classwise=moments(x.transpose(0, 2, 1, 3).reshape(-1, M, 4), y.reshape(-1, 4), counts),
        coordinate_order=['core0_void', 'core0_wall', 'core0_filament', 'core1_void', 'core1_wall', 'core1_filament'],
        class_order=['void', 'wall', 'filament', 'cluster'])


def branch(a, b, eligible=True):
    if not eligible:
        return 'inconclusive_unmatched_estimand'
    phases = ('ph012', 'ph013')
    if all(b[p]['ratio'] <= 1.5 and b[p]['mc']['ratio']['upper'] < a[p]['mc']['ratio']['lower'] for p in phases):
        return 'coarse_dominated'
    for p in phases:
        aa, bb = a[p]['mc']['ratio'], b[p]['mc']['ratio']
        if b[p]['ratio'] >= 2 and max(aa['lower'], bb['lower']) <= min(aa['upper'], bb['upper']):
            return 'coarse_factor_not_implicated'
    return 'inconclusive'


def spectra(draws, truth, cell, edges, legacy=False):
    x, y = np.asarray(draws, float), np.asarray(truth, float)
    if x.shape != (M, *y.shape) or y.ndim != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite case-local 32-draw physical density bank required')
    axes = np.meshgrid(*[2*np.pi*fft.fftfreq(n, d=cell) for n in y.shape], indexing='ij', sparse=True)
    k = np.sqrt(sum(a*a for a in axes))
    f = fft.fftn(x, axes=(-3, -2, -1), norm='ortho', workers=8)
    target = fft.fftn(y, norm='ortho', workers=8)
    mean = f.mean(0); variance = ((abs(f-mean)**2).sum(0)/(M-1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = ((k > lo) & (k <= hi)) if legacy else ((k >= lo) & (k < hi))
        # DC was excluded in the historical zero-start band; retain that exclusion.
        mask &= k > 0
        if not mask.any():
            continue
        per_draw = np.mean(abs(f[:, mask])**2, axis=1)
        rows.append(dict(k_min=lo, k_max=None if np.isinf(hi) else hi,
            interval='(lo,hi]' if legacy else '[lo,hi)', modes=int(mask.sum()),
            truth_power=float(np.mean(abs(target[mask])**2)),
            per_draw_power=per_draw.tolist(), sample_power=float(per_draw.mean()),
            mean_power=float(np.mean(abs(mean[mask])**2)),
            posterior_variance_power=float(variance[mask].mean()),
            mean_error_power=float(np.mean(abs(mean[mask]-target[mask])**2)),
            mean_error_minus_draw_noise=float(np.mean(abs(mean[mask]-target[mask])**2-variance[mask]/M))))
    return rows
