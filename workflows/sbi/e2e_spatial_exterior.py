"""Coherent positive exterior density via a train-fitted Gaussian log field.

This is a modular posterior approximation, not an exact galaxy likelihood.
Each caller supplies one sampled interior; conditional simulation supplies
one spatially correlated exterior. No truth is accepted by the sampler.
The finite periodic prior domain and the free-space tidal domain are distinct.
"""
import numpy as np
from scipy import fft, optimize

LENGTHS = (1., 4., 16.)  # fine cells; registered before development scoring
LAGS = (0, 1, 2, 4, 8, 12, 16)


def basis_spectra(shape, lengths=LENGTHS):
    """Positive discrete Matérn-like spectra, each with unit point variance.

    Q=(ell^-2 - discrete Laplacian)^2. Full FFT normalization is accounted
    for by measuring covariance[0], including the rFFT conjugate weights.
    """
    axes = [2-2*np.cos(2*np.pi*np.fft.fftfreq(n)) for n in shape[:2]]
    axes.append(2-2*np.cos(2*np.pi*np.fft.rfftfreq(shape[2])))
    lap = axes[0][:, None, None]+axes[1][None, :, None]+axes[2][None, None, :]
    result = []
    for length in lengths:
        power = (length**-2+lap)**-2
        power /= fft.irfftn(power, s=shape)[0, 0, 0]
        result.append(power)
    return np.stack(result)


def training_moments(rho):
    """Unwindowed pair products at registered spatial lags, no per-patch centering."""
    x = np.log(np.asarray(rho, np.float64))
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Finite positive 3D training density required')
    products = []
    for lag in LAGS:
        values = []
        for axis in range(3):
            a, b = [slice(None)]*3, [slice(None)]*3
            if lag:
                a[axis], b[axis] = slice(lag, None), slice(None, -lag)
            values.append(np.mean(x[tuple(a)]*x[tuple(b)]))
        products.append(float(np.mean(values)))
    return dict(mean=float(x.mean()), products=products)


def fit_prior(phase_moments, reference_shape=(192, 192, 192)):
    """Equal-phase NNLS covariance fit; inputs must already be training-only."""
    means = np.array([np.mean([r['mean'] for r in rows]) for rows in phase_moments])
    products = np.array([np.mean([r['products'] for r in rows], axis=0)
                         for rows in phase_moments]).mean(0)
    mean = float(means.mean())
    empirical = products-mean**2
    spectra = basis_spectra(reference_shape)
    cov = np.stack([fft.irfftn(p, s=reference_shape) for p in spectra])
    design = np.array([[np.mean([v[lag, 0, 0], v[0, lag, 0], v[0, 0, lag]])
                        for v in cov] for lag in LAGS])
    weights, residual = optimize.nnls(design, empirical)
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        raise ValueError('Degenerate training spatial covariance')
    # Unit ensemble mean density. Report, rather than hide, lognormal mismatch.
    return dict(schema='e2e-spatial-lognormal-v1', lengths_cells=list(LENGTHS),
                weights=weights.tolist(), log_mean=-float(weights.sum())/2,
                empirical_log_mean=mean, empirical_covariance=empirical.tolist(),
                fitted_covariance=(design@weights).tolist(), nnls_residual=float(residual),
                reference_shape=list(reference_shape), lags_cells=list(LAGS),
                training_phase_log_means=means.tolist(),
                approximation='stationary Gaussian logdensity, three fixed covariance scales')


class SpatialPrior:
    """CPU reference conditional simulator with strict numerical residual checks."""

    def __init__(self, shape, fit, workers=1):
        self.shape = tuple(map(int, shape))
        self.workers = workers
        self.mean = float(fit['log_mean'])
        basis = basis_spectra(self.shape, fit['lengths_cells'])
        self.power = np.einsum('i,ijkl->jkl', fit['weights'], basis)
        if np.min(self.power) <= 0:
            raise ValueError('Strictly positive covariance spectrum required')
        self.variance = float(sum(fit['weights']))

    def covariance(self, value):
        return fft.irfftn(fft.rfftn(value, workers=self.workers)*self.power,
                         s=self.shape, workers=self.workers)

    def sample(self, rng):
        white = rng.normal(size=self.shape)
        return self.mean+fft.irfftn(fft.rfftn(white, workers=self.workers)*np.sqrt(self.power),
                                    s=self.shape, workers=self.workers)

    def conditional(self, log_interior, known, rng, rtol=1e-7, maxiter=2000):
        """Hoffman-Ribak conditional draw; hard constraints, no hidden nugget.

        FFT covariance and CG are exact for the declared finite-domain prior.
        Tolerances are checked before restoring the supplied values exactly.
        """
        from scipy.sparse.linalg import LinearOperator, cg
        known = np.asarray(known, bool)
        interior = np.asarray(log_interior, np.float64)
        if known.shape != self.shape or interior.shape != self.shape:
            raise ValueError('Conditional grid mismatch')
        if not np.isfinite(interior[known]).all():
            raise ValueError('Nonfinite conditioning values')
        prior = self.sample(rng)
        if not known.any():
            return prior, dict(iterations=0, relative_residual=0., constraint_maxabs=0.)
        if known.all():
            return interior.copy(), dict(iterations=0, relative_residual=0., constraint_maxabs=0.)
        n = int(known.sum())

        def action(v):
            work = np.zeros(self.shape)
            work[known] = v
            return self.covariance(work)[known]

        operator = LinearOperator((n, n), matvec=action, dtype=np.float64)
        # Stationary diagonal preconditioner; no covariance regularization.
        preconditioner = LinearOperator((n, n), matvec=lambda v: v/self.variance,
                                       dtype=np.float64)
        rhs = interior[known]-prior[known]
        iterations = [0]

        def callback(_):
            iterations[0] += 1

        alpha, status = cg(operator, rhs, M=preconditioner, rtol=rtol, atol=1e-12,
                           maxiter=maxiter, callback=callback)
        relative = float(np.linalg.norm(action(alpha)-rhs)/max(np.linalg.norm(rhs), 1e-30))
        if status != 0 or relative > max(2*rtol, 1e-10):
            raise RuntimeError(f'Conditional solver failed: status={status}, residual={relative}')
        work = np.zeros(self.shape)
        work[known] = alpha
        result = prior+self.covariance(work)
        error = float(np.max(abs(result[known]-interior[known])))
        result[known] = interior[known]
        return result, dict(iterations=iterations[0], relative_residual=relative,
                            constraint_maxabs=error)


class CudaSpatialPrior(SpatialPrior):
    """Same covariance and conditional law as reference, float64 CUDA solve."""

    def __init__(self, shape, fit, workers=1):
        super().__init__(shape, fit, workers)
        import torch
        self.torch = torch
        self.cuda_power = torch.as_tensor(self.power, device='cuda', dtype=torch.float64)

    def conditional(self, log_interior, known, rng, rtol=1e-7, maxiter=2000):
        t = self.torch
        mask = t.as_tensor(known, device='cuda', dtype=t.bool)
        interior = t.as_tensor(log_interior, device='cuda', dtype=t.float64)
        prior = t.as_tensor(self.sample(rng), device='cuda', dtype=t.float64)
        if tuple(mask.shape) != self.shape or tuple(interior.shape) != self.shape:
            raise ValueError('Conditional grid mismatch')
        if not bool(t.isfinite(interior[mask]).all()):
            raise ValueError('Nonfinite conditioning values')
        if not bool(mask.any()):
            return prior.cpu().numpy(), dict(iterations=0, relative_residual=0., constraint_maxabs=0.)
        if bool(mask.all()):
            return interior.cpu().numpy(), dict(iterations=0, relative_residual=0., constraint_maxabs=0.)

        def cov(value):
            return t.fft.irfftn(t.fft.rfftn(value)*self.cuda_power, s=self.shape)

        rhs = t.where(mask, interior-prior, 0.)
        alpha = t.zeros_like(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        rr = t.sum(residual**2)
        norm = float(t.linalg.vector_norm(rhs))
        tolerance = max(rtol*norm, 1e-12)
        iteration = 0
        if norm > tolerance:
            for iteration in range(1, maxiter+1):
                action = cov(direction)*mask
                denominator = t.sum(direction*action)
                if float(denominator) <= 0:
                    raise RuntimeError('Conditional covariance lost positive definiteness')
                step = rr/denominator
                alpha += step*direction
                residual -= step*action
                rr_new = t.sum(residual**2)
                if float(t.sqrt(rr_new)) <= tolerance:
                    break
                direction = residual+(rr_new/rr)*direction
                rr = rr_new
        correction = cov(alpha)
        relative = float(t.linalg.vector_norm(correction[mask]-rhs[mask]))/max(norm, 1e-30)
        if relative > max(2*rtol, 1e-10):
            raise RuntimeError(f'CUDA conditional solve failed: residual={relative}, iterations={iteration}')
        result = prior+correction
        error = float(t.max(t.abs(result[mask]-interior[mask])))
        result[mask] = interior[mask]
        return result.cpu().numpy(), dict(iterations=iteration, relative_residual=relative,
                                         constraint_maxabs=error)


def complete_density(rho, support, prior, rng, method, padding):
    """Complete a single field while keeping all sampled in-footprint cells.

    Fractionally occupied cells retain the full inferred cell-average density:
    this is an effective-cell posterior, NOT a multiplication by completeness.
    The zero-contrast control alone uses fractional occupancy of its source.
    Learned completion conditions on every finite supplied field cell.
    """
    # Stored CFM fields are float32. Convert BEFORE taking logs; performing a
    # float32 log and only then widening changes supposedly fixed densities.
    rho, support = np.asarray(rho, dtype=np.float64), np.asarray(support)
    if rho.shape != support.shape or any(n+2*padding != m for n, m in zip(rho.shape, prior.shape)):
        raise ValueError('Exterior layout mismatch')
    if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
        raise ValueError('Invalid footprint occupancy')
    crop = tuple(slice(padding, padding+n) for n in rho.shape)
    valid = np.isfinite(rho) & (rho > 0)
    occupied = support > 0
    if np.any(occupied & ~valid):
        raise ValueError('Missing interior density')
    known = np.zeros(prior.shape, bool)
    known[crop] = valid if method == 'learned' else occupied
    log_density = np.zeros(prior.shape)
    local_log = np.zeros(rho.shape)
    local_log[valid] = np.log(rho[valid])
    log_density[crop] = local_log
    if method == 'zero':
        result = np.ones(prior.shape)
        local = np.ones(rho.shape)
        local[occupied] += support[occupied]*(rho[occupied]-1)
        result[crop] = local
        return result, dict(method=method)
    if method == 'unconditional':
        log_result, diagnostics = prior.sample(rng), {}
        log_result[known] = log_density[known]
    elif method in ('conditional', 'learned'):
        log_result, diagnostics = prior.conditional(log_density, known, rng)
    else:
        raise ValueError('Unregistered exterior method')
    with np.errstate(over='raise', invalid='raise'):
        result = np.exp(log_result)
    # Preserve supplied density values exactly, without another exp/log round
    # trip. Only exterior cells are new stochastic values.
    local_known = valid if method == 'learned' else occupied
    result[crop][local_known] = rho[local_known]
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FloatingPointError('Invalid positive density draw')
    return result, dict(method=method, **diagnostics)
