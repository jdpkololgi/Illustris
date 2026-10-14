"""Exact Gaussian conditionals given restricted exterior summaries.

The observations remain fully conditioned on. Missing exterior coordinates
are marginalized, NOT set to zero or treated as independent. No neural model
or Abacus likelihood is implemented here.
"""
from dataclasses import dataclass
import itertools

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh, solve_discrete_lyapunov, solve_triangular
from scipy.special import ndtr

import e2e_gaussian_block_sampler as block
import e2e_conditional_reference_math as reference


CONTEXTS = ('full', 'none', 'collar1', 'collar2', 'collar1_mean', 'collar1_modes')


def prior(n, kind, length=1.5):
    """Fixed cell spacing and correlation parameter as domain size changes."""
    size = n**3
    if kind == 'independent':
        return np.eye(size)
    if kind == 'open':
        xyz = np.indices((n,)*3).reshape(3,-1).T
        distance = ((xyz[:,None]-xyz[None,:])**2).sum(-1)
        return (np.exp(-distance/(2*length**2))+.05*np.eye(size))/1.05
    if kind not in ('spectral','long_range'):
        raise ValueError(kind)
    k = 2*np.sin(np.pi*np.fft.fftfreq(n))
    laplacian = sum(a*a for a in np.meshgrid(k,k,k,indexing='ij'))
    power = (1+length**2*laplacian)**-2
    power /= power.mean()
    identity = np.eye(size).reshape(size,n,n,n)
    covariance = np.fft.ifftn(np.fft.fftn(identity,axes=(1,2,3))*power,
                              axes=(1,2,3)).real.reshape(size,size)
    if kind == 'long_range':
        covariance = .2*covariance+.8*np.ones_like(covariance)
    return block.symmetric(covariance)


def observations(n, regime):
    masks,stds = reference.observation_templates(n)
    mask,std = masks[regime].copy(),stds[regime].copy()
    if regime == 1:
        i,j,k = np.indices((n,)*3)
        mask *= ((i+2*j+3*k)%8==0).ravel()
        std *= 2
    return mask,std


def problem(n, kind, regime, seed):
    covariance = prior(n,kind)
    mask,std = observations(n,regime)
    rng = np.random.default_rng(seed)
    truth = np.linalg.cholesky(covariance)@rng.normal(size=n**3)
    y = mask*(truth+std*rng.normal(size=n**3))
    case = reference.posterior(covariance,mask,std,y)
    case.update(prior=covariance,truth=truth,n=n,kind=kind,regime=regime,
                precision=block.symmetric(cho_solve(cho_factor(case['sigma']),np.eye(n**3))))
    return case


def patches(n, width=2, shift=0):
    if n % width:
        raise ValueError('Aligned blocks must tile the domain')
    grid = np.arange(n**3).reshape((n,)*3)
    offsets = np.asarray(list(itertools.product(range(width),repeat=3)))
    result = []
    for start in itertools.product(range(0,n,width),repeat=3):
        xyz = (np.asarray(start)+shift+offsets)%n
        result.append(grid[tuple(xyz.T)])
    np.testing.assert_array_equal(np.sort(np.concatenate(result)),np.arange(n**3))
    return result


def context_basis(n, indices, context, periodic=True):
    """Columns are disjoint collar coordinates plus modes of the far exterior.

    Restricting the mode to far exterior avoids duplicate collar information;
    jointly with collar values it contains the same information as modes of
    the full exterior. No feature contains an updated-block coordinate.
    """
    size = n**3
    outside = np.ones(size,dtype=bool); outside[indices] = False
    if context == 'full':
        return np.eye(size)[:,outside]
    if context == 'none':
        return np.zeros((size,0))
    if context not in CONTEXTS:
        raise ValueError(context)
    radius = 2 if context == 'collar2' else 1
    xyz = np.indices((n,)*3).reshape(3,-1).T
    delta = np.abs(xyz[:,None,:]-xyz[indices][None,:,:])
    if periodic:
        delta = np.minimum(delta,n-delta)
    near = (delta.max(-1).min(-1) <= radius)&outside
    basis = np.eye(size)[:,near]
    far = outside&~near
    if context.endswith('_mean') or context.endswith('_modes'):
        modes = block.global_modes(n)[:,:1 if context.endswith('_mean') else 7].copy()
        modes[~far] = 0
        modes = block.orthonormal_span(modes)
        basis = np.concatenate([basis,modes],axis=1)
    np.testing.assert_allclose(basis[indices],0,atol=1e-14)
    np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-11)
    return basis


@dataclass
class GaussianBlock:
    indices: np.ndarray
    gain: np.ndarray
    offset: np.ndarray
    covariance: np.ndarray
    mean: np.ndarray
    context_rank: int

    def __post_init__(self):
        self.factor = np.linalg.cholesky(self.covariance)

    def step(self,state,rng=None,noise=None):
        if noise is None:
            if rng is None:
                raise ValueError('Explicit random source required')
            noise = rng.normal(size=(len(state),len(self.indices)))
        result = state.copy()
        result[:,self.indices] = state@self.gain.T+self.offset+noise@self.factor.T
        return result

    def compose(self,matrix,offset,covariance):
        new_matrix,new_offset,new_cov = matrix.copy(),offset.copy(),covariance.copy()
        cross = self.gain@covariance
        diagonal = cross@self.gain.T+self.covariance
        new_matrix[self.indices] = self.gain@matrix
        new_offset[self.indices] = self.gain@offset+self.offset
        new_cov[self.indices,:] = cross
        new_cov[:,self.indices] = cross.T
        new_cov[np.ix_(self.indices,self.indices)] = diagonal
        return new_matrix,new_offset,block.symmetric(new_cov)


def conditional(case,indices,context,periodic=True):
    mu,sigma,precision = case['mu'],case['sigma'],case['precision']
    if context == 'full':
        restricted = cho_factor(precision[np.ix_(indices,indices)])
        covariance = block.symmetric(cho_solve(restricted,np.eye(len(indices))))
        gain = -cho_solve(restricted,precision[indices])
        gain[:,indices] = 0
        rank = len(mu)-len(indices)
    else:
        basis = context_basis(case['n'],indices,context,periodic)
        rank = basis.shape[1]
        if rank:
            cross = sigma[indices]@basis
            factor = cho_factor(block.symmetric(basis.T@sigma@basis))
            response = cho_solve(factor,cross.T).T
            gain = response@basis.T
            covariance = block.symmetric(sigma[np.ix_(indices,indices)]-response@cross.T)
        else:
            gain = np.zeros((len(indices),len(mu)))
            covariance = sigma[np.ix_(indices,indices)].copy()
    offset = mu[indices]-gain@mu
    return GaussianBlock(indices,gain,offset,covariance,mu,rank)


def information_loss(update,full):
    # Expected KL[p(B|E,y) || p(B|summary(E),y)] over p(E|y).
    a = np.linalg.slogdet(update.covariance)
    b = np.linalg.slogdet(full.covariance)
    if a[0] <= 0 or b[0] <= 0:
        raise ValueError('Conditional covariance not positive definite')
    loss = .5*(a[1]-b[1])
    if loss < -1e-9:
        raise AssertionError('Restricted context cannot reduce conditional entropy')
    return float(max(loss,0))


def stationary(affine,target,probes):
    matrix,offset,innovation = affine
    radius = float(np.max(abs(np.linalg.eigvals(matrix))))
    if radius >= 1:
        return dict(stable=False,spectral_radius=radius,accuracy_passed=False),None,None
    covariance = block.symmetric(solve_discrete_lyapunov(matrix,innovation,method='bilinear'))
    mean = np.linalg.solve(np.eye(len(matrix))-matrix,offset)
    residual = np.linalg.norm(covariance-matrix@covariance@matrix.T-innovation)/np.linalg.norm(covariance)
    if residual > 1e-8 or eigvalsh(covariance,subset_by_index=[0,0])[0] <= 0:
        raise AssertionError('Stationary moment solver failed')
    chol = np.linalg.cholesky(target['sigma'])
    white = solve_triangular(chol,np.eye(len(matrix)),lower=True)
    difference = block.symmetric(white@(covariance-target['sigma'])@white.T)
    extremes = [eigvalsh(difference,subset_by_index=[i,i])[0] for i in (0,len(matrix)-1)]
    worst = float(max(abs(x) for x in extremes))
    mean_error = float(np.linalg.norm(white@(mean-target['mu'])))
    target_var = np.einsum('ij,ij->j',probes,target['sigma']@probes)
    variance = np.einsum('ij,ij->j',probes,covariance@probes)
    delta = (mean-target['mu'])@probes
    bound = 1.6448536269514722*np.sqrt(variance)
    coverage = ndtr((delta+bound)/np.sqrt(target_var))-ndtr((delta-bound)/np.sqrt(target_var))
    relative = float(np.linalg.norm(covariance-target['sigma'])/np.linalg.norm(target['sigma']))
    coverage_error = float(np.max(abs(coverage-.9)))
    result = dict(stable=True,spectral_radius=radius,lyapunov_relative_residual=float(residual),
        covariance_relative=relative,worst_relative_covariance_error=worst,
        whitened_mean_error=mean_error,probe_variance_ratios=(variance/target_var).tolist(),
        probe_coverage90=coverage.tolist(),maximum_coverage_error=coverage_error,
        accuracy_passed=worst<=.01 and relative<=.01 and mean_error<=.01 and coverage_error<=.005)
    return result,mean,covariance
