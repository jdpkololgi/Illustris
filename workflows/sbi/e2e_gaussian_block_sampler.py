"""Exact Gaussian conditional updates and independent affine accounting.

No neural model or cosmological likelihood is supplied by this module. Dense
posterior precision is an explicit oracle input, including exterior dependence.
"""
from dataclasses import dataclass
import itertools

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, null_space, solve_triangular
from scipy.special import ndtr


def symmetric(x):
    return (x+x.T)/2


@dataclass
class ConditionalUpdate:
    """Resample coordinates along an orthonormal basis, fixing its complement."""
    basis: np.ndarray
    gain: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray
    mean: np.ndarray

    @classmethod
    def make(cls, precision, mean, basis):
        precision, mean, basis = [np.asarray(x, dtype=np.float64) for x in (precision, mean, basis)]
        d = len(mean)
        if precision.shape != (d,d) or basis.ndim != 2 or basis.shape[0] != d or basis.shape[1] == 0:
            raise ValueError('Nonempty update basis and compatible target required')
        if not all(np.isfinite(x).all() for x in (precision,mean,basis)):
            raise ValueError('Finite target and update required')
        np.testing.assert_allclose(precision, precision.T, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-11, rtol=1e-11)
        cross = basis.T@precision
        restricted = symmetric(cross@basis)
        factor = cho_factor(restricted)
        covariance = symmetric(cho_solve(factor, np.eye(len(restricted))))
        gain = cho_solve(factor, cross)
        return cls(basis, gain, covariance, np.linalg.cholesky(covariance), mean)

    def step(self, state, rng=None, noise=None):
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != len(self.mean):
            raise ValueError('Expected draw-by-coordinate state')
        if noise is None:
            if rng is None: raise ValueError('Explicit random generator required')
            noise = rng.normal(size=(len(state), self.basis.shape[1]))
        if noise.shape != (len(state), self.basis.shape[1]):
            raise ValueError('Invalid innovation dimensions')
        delta = (self.mean-state)@self.gain.T + noise@self.factor.T
        return state+delta@self.basis.T

    def compose(self, matrix, offset, covariance):
        """Compose after x=M z+b+eta; accumulate actual independent innovations."""
        u, k = self.basis, self.gain
        left = k@covariance
        embedded = u@left
        new_covariance = covariance-embedded-embedded.T+u@(left@k.T+self.covariance)@u.T
        return (matrix-u@(k@matrix), offset+u@(k@(self.mean-offset)), symmetric(new_covariance))


def transition(updates):
    d = len(updates[0].mean)
    matrix, offset, covariance = np.eye(d), np.zeros(d), np.zeros((d,d))
    for update in updates:
        matrix, offset, covariance = update.compose(matrix, offset, covariance)
    return matrix, offset, covariance


def sweep(updates, state, rng):
    for update in updates: state = update.step(state, rng)
    return state


def invariance(affine, mean, covariance):
    matrix, offset, innovation = affine
    return dict(mean_relative=float(np.linalg.norm(matrix@mean+offset-mean)/max(1.,np.linalg.norm(mean))),
        covariance_relative=float(np.linalg.norm(matrix@covariance@matrix.T+innovation-covariance)/np.linalg.norm(covariance)),
        spectral_radius=float(np.max(abs(np.linalg.eigvals(matrix)))))


def patch_indices(n, width, stride):
    if width > n or (n-width) % stride:
        raise ValueError('Windows must cover the domain end exactly')
    grid = np.arange(n**3).reshape((n,)*3)
    result = [grid[tuple(slice(s,s+width) for s in start)].ravel()
              for start in itertools.product(range(0,n-width+1,stride), repeat=3)]
    if len(np.unique(np.concatenate(result))) != n**3:
        raise ValueError('Incomplete patch coverage')
    return result


def constrained_basis(n, factor=4):
    """Orthonormal independent coordinates for the actual block-zero-mean space."""
    patches = patch_indices(n,factor,factor)
    local = null_space(np.ones((1,factor**3)))
    basis = np.zeros((n**3,len(patches)*(factor**3-1)))
    indices = []
    for j, patch in enumerate(patches):
        columns = np.arange(j*local.shape[1],(j+1)*local.shape[1])
        basis[np.ix_(patch,columns)] = local
        indices.append(columns)
    np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-12)
    return basis, indices


def global_modes(n):
    coordinates = np.indices((n,)*3)
    columns = [np.ones(n**3)]
    for axis in range(3):
        phase = 2*np.pi*coordinates[axis].ravel()/n
        columns.extend((np.sin(phase),np.cos(phase)))
    return np.stack(columns,axis=1)


def orthonormal_span(columns):
    left, singular, _ = np.linalg.svd(columns,full_matrices=False)
    keep = singular > max(singular[0]*1e-12,1e-12)
    return left[:,keep]


def probe_bank(n, basis):
    """Fixed physical linear probes, transformed to independent coordinates."""
    coordinates = np.indices((n,)*3)
    vectors, names = [np.ones(n**3)/n**3], ['domain_mean']
    octants = []
    for corner in itertools.product(range(2), repeat=3):
        mask = np.all(coordinates//(n//2)==np.asarray(corner)[:,None,None,None],axis=0)
        v = mask.ravel()/mask.sum(); octants.append(v)
        vectors.append(v); names.append('octant_'+''.join(map(str,corner)))
        small = np.ones((n,)*3,dtype=bool)
        for axis,side in enumerate(corner):
            small &= coordinates[axis] < n//4 if side == 0 else coordinates[axis] >= n-n//4
        vectors.append(small.ravel()/small.sum()); names.append('corner_'+''.join(map(str,corner)))
    for j in range(1,len(octants)):
        vectors.append(octants[0]-octants[j]); names.append('octant_contrast_'+str(j))
    modes = global_modes(n)
    for j in range(1,modes.shape[1]):
        vectors.append(modes[:,j]/np.linalg.norm(modes[:,j])); names.append('mode_'+str(j))
    projected = basis.T@np.stack(vectors,axis=1)
    keep = np.linalg.norm(projected,axis=0) > 1e-12
    return projected[:,keep], [q for q,k in zip(names,keep) if k], [q for q,k in zip(names,keep) if not k]


def convergence(affine, mean, covariance, probes, max_sweeps=512):
    """Exact affine moments from zero; powers-of-two convergence checkpoints."""
    matrix, offset, innovation = [q.copy() for q in affine]
    chol = np.linalg.cholesky(covariance)
    white = solve_triangular(chol,np.eye(len(mean)),lower=True)
    target_var = np.einsum('ij,ij->j',probes,covariance@probes)
    if np.any(target_var <= 0): raise ValueError('All retained probes need positive variance')
    rows = []; count = 1
    while count <= max_sweeps:
        contracted = white@matrix@chol
        largest = eigh(symmetric(contracted@contracted.T),subset_by_index=[len(mean)-1,len(mean)-1],eigvals_only=True)[0]
        worst_mean = float(np.sqrt(max(largest,0)))
        relative_cov = symmetric(white@innovation@white.T)-np.eye(len(mean))
        extreme = (eigh(relative_cov,subset_by_index=[0,0],eigvals_only=True)[0],
                   eigh(relative_cov,subset_by_index=[len(mean)-1,len(mean)-1],eigvals_only=True)[0])
        worst_cov = float(max(abs(q) for q in extreme))
        variance = np.einsum('ij,ij->j',probes,innovation@probes)
        if np.any(variance < -1e-12): raise ValueError('Negative propagated variance')
        error = (offset-mean)@probes
        radius = 1.6448536269514722*np.sqrt(np.maximum(variance,0))
        coverage = ndtr((error+radius)/np.sqrt(target_var))-ndtr((error-radius)/np.sqrt(target_var))
        row = dict(sweeps=count,mean_rms=float(np.linalg.norm(white@(offset-mean))/np.sqrt(len(mean))),
            worst_mean_contraction=worst_mean,worst_relative_covariance_error=worst_cov,
            variance_ratios=(variance/target_var).tolist(),coverage90=coverage.tolist(),
            maximum_coverage_error=float(np.max(abs(coverage-.9))))
        row['passed'] = worst_mean <= .01 and worst_cov <= .01 and row['maximum_coverage_error'] <= .005
        rows.append(row)
        # Composition of this transition with itself, including independent noise.
        innovation = symmetric(matrix@innovation@matrix.T+innovation)
        offset = matrix@offset+offset
        matrix = matrix@matrix
        count *= 2
    return rows


def autocorrelation(affine, covariance, probes):
    """Exact stationary asymptotic-variance / marginal-variance for each probe."""
    matrix = affine[0]
    summed_lags = np.linalg.solve(np.eye(len(matrix))-matrix,matrix@covariance)
    variance = np.einsum('ij,ij->j',probes,covariance@probes)
    tau = 1+2*np.einsum('ij,ij->j',probes,summed_lags@probes)/variance
    if np.any(tau <= 0): raise ValueError('Nonpositive asymptotic variance')
    return tau
