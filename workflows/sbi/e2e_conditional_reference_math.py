"""Independent float64 Gaussian reference and finite-ensemble diagnostics.

No Abacus inputs. Exact posterior objects are evaluation targets and fixed-fit
training distributions, NEVER amortised network conditions.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, expit


def prior(n=8):
    k = np.fft.fftfreq(n) * n
    radius = np.sqrt(sum(a*a for a in np.meshgrid(k, k, k, indexing="ij")))
    power = (1 + (radius / 1.5)**2)**-2
    power /= power.mean()
    eye = np.eye(n**3).reshape(n**3, n, n, n)
    covariance = np.fft.ifftn(np.fft.fftn(eye, axes=(1, 2, 3))*power,
                             axes=(1, 2, 3)).real.reshape(n**3, n**3)
    return covariance, power, radius


def observation_templates(n):
    i, j, k = np.indices((n, n, n))
    masks = np.stack([~((i >= n//4) & (i < n//2)),
                      ~((j >= n//2) & (k >= n//2))]).astype(float)
    std = np.stack([.35 + .55*j/(n-1), .35 + .55*i/(n-1)])
    return masks.reshape(2, -1), std.reshape(2, -1)


def posterior(covariance, mask, std, y):
    """Precision-form reference, checked against independent conditioning form."""
    size = len(mask)
    identity = np.eye(size)
    precision = cho_solve(cho_factor(covariance), identity)
    weight = mask/std**2
    sigma = cho_solve(cho_factor(precision + np.diag(weight)), identity)
    mu = sigma @ (weight*y)
    observed = np.flatnonzero(mask)
    cross = covariance[:, observed]
    obs_cov = covariance[np.ix_(observed, observed)] + np.diag(std[observed]**2)
    alternate = covariance - cross @ cho_solve(cho_factor(obs_cov), cross.T)
    alternate_mu = cross @ cho_solve(cho_factor(obs_cov), y[observed])
    if not (np.allclose(sigma, alternate, atol=1e-10, rtol=1e-9)
            and np.allclose(mu, alternate_mu, atol=1e-10, rtol=1e-9)):
        raise AssertionError("independent Gaussian conditioning identities disagree")
    values, vectors = np.linalg.eigh(sigma)
    return dict(mu=mu, sigma=sigma, chol=np.linalg.cholesky(sigma),
                values=values, vectors=vectors, mask=mask, std=std, y=y)


def problem(n=8, count=4):
    covariance, power, radius = prior(n)
    chol = np.linalg.cholesky(covariance)
    masks, std = observation_templates(n)
    rng = np.random.default_rng(90123)
    cases = []
    for index in range(count):
        which = index % 2
        truth = chol @ rng.normal(size=n**3)
        y = masks[which]*(truth + std[which]*rng.normal(size=n**3))
        case = posterior(covariance, masks[which], std[which], y)
        case["truth"] = truth
        cases.append(case)
    return covariance, chol, masks, std, cases, power, radius


def probes(n):
    coordinates = np.indices((n, n, n))
    columns = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                columns.append(np.all(coordinates//(n//2) == np.array([a,b,c])[:,None,None,None],
                                      axis=0).astype(float).ravel())
    for axis in range(3):
        phase = 2*np.pi*coordinates[axis]/n
        columns.extend([np.sin(phase).ravel(), np.cos(phase).ravel()])
    phase = 2*np.pi*(coordinates[0]+coordinates[1])/n
    columns.extend([np.sin(phase).ravel(), np.cos(phase).ravel()])
    raw = np.stack(columns, axis=1)
    q, r = np.linalg.qr(raw)
    if np.min(np.abs(np.diag(r))) < 1e-8:
        raise ValueError("degenerate probe basis")
    return q


def oracle_moments(case, objective, nfe):
    """Exact propagation of affine sampler transitions in covariance eigenbasis."""
    lam = case["values"]
    mu = case["vectors"].T @ case["mu"]
    m, v = np.zeros_like(mu), np.ones_like(mu)
    if objective == "vdm":
        for i in range(nfe):
            gt = -13.3 + 26.6*(1-i/nfe)
            gs = -13.3 + 26.6*(1-(i+1)/nfe)
            at, st = np.sqrt(expit(-gt)), np.sqrt(expit(gt))
            ass, ss = np.sqrt(expit(-gs)), np.sqrt(expit(gs))
            c = -np.expm1(gs-gt)
            denom = at*at*lam + st*st
            A = ass/at*(1-c*st*st/denom)
            B = ass*c*st*st*mu/denom
            m, v = A*m+B, A*A*v+ss*ss*c
    elif objective == "cfm":
        if nfe % 2:
            raise ValueError("Heun requires even NFE")
        steps = nfe//2
        dt = 1/steps
        def coeff(t):
            a = (t*lam-(1-t))/(t*t*lam+(1-t)**2)
            return a, (1-a*t)*mu
        for i in range(steps):
            a, b = coeff(i/steps)
            aa, bb = coeff((i+1)/steps)
            A = 1 + dt/2*(a+aa*(1+dt*a))
            B = dt/2*(b+aa*dt*b+bb)
            m, v = A*m+B, A*A*v
    else:
        raise ValueError(objective)
    return dict(mean_rms=float(np.linalg.norm(m-mu)/np.sqrt(lam.sum())),
                covariance_relative=float(np.linalg.norm(v-lam)/np.linalg.norm(lam)),
                variance_ratio=float(v.sum()/lam.sum()), mean=m, variance=v)


def null_thresholds(case, q, draws=512, repeats=128, seed=884):
    rng = np.random.default_rng(seed)
    means, covs = [], []
    target = q.T @ case["sigma"] @ q
    factor = np.linalg.cholesky(target)
    # Full-field draws for the full-field mean floor; probe draws for covariance.
    for _ in range(repeats):
        means.append(np.linalg.norm(case["chol"] @ rng.normal(size=len(q))) /
                     np.sqrt(draws*np.trace(case["sigma"])))
        z = rng.normal(size=(draws, q.shape[1])) @ factor.T
        covs.append(np.linalg.norm(np.cov(z, rowvar=False)-target)/np.linalg.norm(target))
    return dict(mean_p99=float(np.quantile(means, .99)),
                covariance_p99=float(np.quantile(covs, .99)))


def metrics(draws, case, q, radius):
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != len(case["mu"]) or not np.isfinite(draws).all():
        raise ValueError("invalid field ensemble")
    mu, sigma = case["mu"], case["sigma"]
    projected = draws @ q
    target = q.T @ sigma @ q
    centre = mu @ q
    lo, hi = np.quantile(projected, [.05, .95], axis=0)
    coverage = ndtr((hi-centre)/np.sqrt(np.diag(target))) - ndtr((lo-centre)/np.sqrt(np.diag(target)))
    n = radius.shape[0]
    residual = (draws-draws.mean(0)).reshape(-1,n,n,n)
    estimated_power = (np.abs(np.fft.fftn(residual, axes=(1,2,3), norm="ortho"))**2).sum(0)/(len(draws)-1)
    # E|F(x-mu)|^2 from covariance factor, independently of Monte Carlo.
    factor_fields = case["chol"].T.reshape(-1,n,n,n)
    exact_power = (np.abs(np.fft.fftn(factor_fields, axes=(1,2,3), norm="ortho"))**2).sum(0)
    shells = np.floor(radius).astype(int)
    power_ratios = [float(estimated_power[shells==i].mean()/exact_power[shells==i].mean())
                    for i in np.unique(shells)]
    voxel_var = draws.var(0, ddof=1)
    return dict(mean_rms=float(np.linalg.norm(draws.mean(0)-mu)/np.sqrt(np.trace(sigma))),
                covariance_relative=float(np.linalg.norm(np.cov(projected,rowvar=False)-target)/np.linalg.norm(target)),
                variance_ratio=float(voxel_var.mean()/np.diag(sigma).mean()),
                voxel_variance_relative=float(np.linalg.norm(voxel_var-np.diag(sigma))/np.linalg.norm(np.diag(sigma))),
                octant_coverage=float(coverage[:8].mean()),
                probe_coverage=coverage.tolist(), power_ratio=power_ratios)


def qualify(result, null, config):
    checks = dict(mean=result["mean_rms"] <= max(config["mean_tolerance"],null["mean_p99"]),
                  covariance=result["covariance_relative"] <= max(config["covariance_tolerance"],null["covariance_p99"]),
                  variance=abs(result["variance_ratio"]-1) <= config["variance_tolerance"],
                  coverage=.85 <= result["octant_coverage"] <= .95)
    return dict(checks=checks, passed=all(checks.values()))
