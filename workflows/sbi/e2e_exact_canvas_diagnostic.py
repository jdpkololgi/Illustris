"""Exact affine reference through the production canvas integrator; CPU only."""
import itertools
import os
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.sbi import e2e_global_canvas as canvas
from workflows.sbi.e2e_conditional_reference import ExactModel
import e2e_conditional_reference_math as reference
import e2e_true_coarse_fraction_saved as io

OUT = io.PARENT/'threshold_canvas_diagnostics_20261008_v1'
PLAN = io.REPO/'docs/e2e_threshold_canvas_diagnostics_20261008.md'


class Router:
    def __init__(self, cases):
        self.models = [ExactModel(case, 'cfm') for case in cases]
        self.schedule = self.models[0].schedule

    def __call__(self, x, t, index):
        return self.models[index](x, t, None)


def geometry(n, width):
    starts = tuple(itertools.product(range(n-width+1), repeat=3))
    return canvas.Geometry(starts, (n,)*3, (0,)*3, (width,)*3)


def marginal(case, indices):
    sigma = case['sigma'][np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(sigma)
    return dict(mu=case['mu'][indices], sigma=sigma, values=values, vectors=vectors)


def pushforward(cases, geom, nfe):
    size = int(np.prod(geom.shape))
    state = torch.as_tensor(np.concatenate([np.zeros((1, size)), np.eye(size)]), dtype=torch.float64)
    state = state.reshape(size+1, 1, *geom.shape)
    result, weights = canvas.integrate(Router(cases), 'coarse', list(range(len(cases))), state, geom, nfe)
    assert torch.all(weights > 0)
    result = result.numpy().reshape(size+1, size)
    mean = result[0]; matrix = (result[1:]-mean).T
    return mean, matrix@matrix.T


def measure(mean, sigma, target, n):
    true = target['sigma']; size = n**3
    half = np.indices((n,)*3)[0].ravel() < n//2
    probes = np.stack([half/half.sum(), (~half)/(~half).sum()], axis=1)
    estimated_var = float(sigma.sum()/size**2); true_var = float(true.sum()/size**2)
    delta = float(mean.mean()-target['mu'].mean())
    radius = 1.6448536269514722*np.sqrt(estimated_var)
    return dict(mean_rms=float(np.linalg.norm(mean-target['mu'])/np.sqrt(np.trace(true))),
        covariance_relative=float(np.linalg.norm(sigma-true)/np.linalg.norm(true)),
        voxel_variance_relative=float(np.linalg.norm(np.diag(sigma)-np.diag(true))/np.linalg.norm(np.diag(true))),
        domain_mean_variance_ratio=estimated_var/true_var,
        domain_mean_90_coverage=float(ndtr((delta+radius)/np.sqrt(true_var))-ndtr((delta-radius)/np.sqrt(true_var))),
        half_domain_covariance=(probes.T@sigma@probes).tolist(),
        target_half_domain_covariance=(probes.T@true@probes).tolist())


def run():
    io.compute_guard(); torch.set_num_threads(1); OUT.mkdir(exist_ok=True)
    sources = [io.record(p) for p in (__file__, canvas.__file__, reference.__file__,
        sys.modules[ExactModel.__module__].__file__, io.__file__, PLAN,
        io.REPO/'workflows/sbi/e2e_direct_vdm.py', io.REPO/'workflows/sbi/e2e_vdm_context_models.py')]
    rng = np.random.default_rng(6100817); n = 4; masks, stds = reference.observation_templates(n)
    rows = []; checks = 0
    for kind, covariance in [('independent', np.eye(n**3)), ('correlated', reference.prior(n)[0])]:
        for oi, (mask, std) in enumerate(zip(masks, stds)):
            truth = np.linalg.cholesky(covariance)@rng.normal(size=n**3)
            y = mask*(truth+std*rng.normal(size=n**3))
            target = reference.posterior(covariance, mask, std, y)
            for width in (2, 3, 4):
                geom = geometry(n, width); index = np.arange(n**3).reshape(n,n,n)
                indices = [index[geom.crop(i)[2:]].ravel() for i in range(len(geom.starts))]
                for conditioning in ('all_observations', 'local_observations'):
                    cases = [marginal(target, ix) if conditioning == 'all_observations' else
                        reference.posterior(covariance[np.ix_(ix, ix)], mask[ix], std[ix], y[ix]) for ix in indices]
                    for case in cases:
                        own_mean, own_cov = pushforward([case], geometry(width, width), 512)
                        got = measure(own_mean, own_cov, case, width)
                        expected = reference.oracle_moments(case, 'cfm', 512)
                        np.testing.assert_allclose(got['mean_rms'], expected['mean_rms'], atol=1e-12)
                        np.testing.assert_allclose(got['covariance_relative'], expected['covariance_relative'], atol=1e-12)
                        checks += 2
                    for nfe in (128, 256, 512):
                        mean, sigma = pushforward(cases, geom, nfe)
                        got = measure(mean, sigma, target, n)
                        if nfe == 512 and (kind == 'independent' or width == n):
                            assert got['mean_rms'] < 1e-4 and got['covariance_relative'] < 1e-4
                            checks += 2
                        rows.append(dict(prior=kind, observation=oi, width=width, windows=len(cases),
                            conditioning=conditioning, nfe=nfe, **got))
    for source in sources:
        if io.sha(source['path']) != source['sha256']:
            raise ValueError('Diagnostic source changed')
    io.write(OUT/'GAUSSIAN_COMPLETE.json', dict(passed=True, checks=checks, rows=rows, sources=sources,
        job=os.environ['SLURM_JOB_ID'], scope='coarse-stage exact Gaussian structural diagnostic, not Abacus attribution',
        posterior_qualified=False, seed=6100817, numerical_method='exact affine pushforward of unmodified Heun canvas; no Monte Carlo'))
    print(dict(passed=True, checks=checks, rows=len(rows)), flush=True)


if __name__ == '__main__':
    run()
