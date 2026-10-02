"""Exchangeable joint rank and distance-stratified diagnostics, no iid-cell CI."""
import numpy as np


def joint_rank(draws, truth):
    """Permutation-equivariant leave-one-out outlyingness of the full vector.

    This is a joint posterior diagnostic, not a fitted Gaussian likelihood.
    Fixed 20% diagonal shrinkage makes the metric well-defined for small M.
    Under exchangeability, the truth rank is uniform even for non-Gaussian data.
    """
    draws, truth = np.asarray(draws, float), np.asarray(truth, float)
    if draws.ndim != 2 or draws.shape[1:] != truth.shape or len(draws) < 3:
        raise ValueError('At least three compatible vector draws required')
    z = np.vstack((truth, draws))
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite joint sample')
    score = []
    for i in range(len(z)):
        other = np.delete(z, i, axis=0)
        covariance = np.atleast_2d(np.cov(other.T))
        diagonal = np.diag(np.diag(covariance))
        scale = max(float(np.trace(covariance))/z.shape[1], 1e-12)
        covariance = .8*covariance+.2*diagonal+np.eye(z.shape[1])*1e-8*scale
        delta = z[i]-other.mean(0)
        score.append(float(delta@np.linalg.solve(covariance, delta)))
    score = np.asarray(score)
    less, ties = int(np.sum(score < score[0])), int(np.sum(score == score[0]))
    # Deterministic seed is a reproducible randomized tie rule, never fitted.
    tie_fraction = np.random.default_rng(119).uniform()
    rank = (less+tie_fraction*ties)/len(z)
    return dict(rank=float(rank), outlyingness=float(score[0]), dimensions=z.shape[1],
                covered={str(p): bool(rank <= p) for p in (.5, .68, .9, .95)},
                attainable_no_ties={str(p): float(np.clip(np.floor(p*len(z)-tie_fraction)+1, 0, len(z))/len(z))
                                    for p in (.5, .68, .9, .95)},
                interpretation='exchangeable leave-one-out vector rank; independent units are phases')


def distance_masks(support):
    from scipy.ndimage import distance_transform_edt
    from workflows.sbi.e2e_coupled_target_products import owned_core_arrays
    full = np.asarray(support) >= 1-1e-7
    distance = distance_transform_edt(np.pad(full, 1))[1:-1, 1:-1, 1:-1]*6.766
    d = owned_core_arrays(distance)
    occupancy = owned_core_arrays(support)
    return dict(outside=occupancy == 0, mixed_cell=(occupancy > 0) & (occupancy < 1-1e-7),
                within_13p5=(occupancy >= 1-1e-7) & (d <= 13.532),
                between_13p5_27=(d > 13.532) & (d <= 27.064), beyond_27=d > 27.064)
