"""Descriptive moment accounting; sensitivity transforms are not field posteriors."""
import numpy as np


def moments(draws, truth):
    """Equal-case accounting for (case, draw, coordinate) arrays.

    Conditional iid posterior draws imply E[e e.T] = (1+1/M) E[S]
    for a calibrated posterior averaged over the conditioning population.
    This identity is not an independence assumption about the selected cases.
    Finite, dependent development panels supply descriptive estimates only.
    """
    x, y = np.asarray(draws, float), np.asarray(truth, float)
    if x.ndim != 3 or y.shape != (x.shape[0], x.shape[2]) or x.shape[1] < 2:
        raise ValueError('expected compatible case/draw/coordinate arrays')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonempty finite arrays required')
    n, m, d = x.shape
    mean = x.mean(1); error = mean-y; bias = error.mean(0)
    residual = x-mean[:,None]
    cov = np.einsum('nmi,nmj->nij',residual,residual)/(m-1)
    predictive = cov.mean(0)
    second = error.T@error/n
    adjusted = second-predictive/m
    centered = (error-bias).T@(error-bias)/n-predictive/m
    np.testing.assert_allclose(adjusted, centered+np.outer(bias,bias), atol=2e-14)
    variance = np.diag(predictive); mse = np.diag(second)
    ratios = [float(q/v) if v>1e-15 else None for q,v in zip(np.diag(adjusted),variance)]
    floor = max(float(np.trace(predictive))/d*1e-8,1e-15)
    eigen, vec = np.linalg.eigh(predictive)
    whitening = (vec/np.sqrt(np.maximum(eigen,floor)))@vec.T
    trace = float(np.trace(predictive)); total = float(np.trace(second))
    return dict(cases=n,draws=m,dimensions=d,bias=bias.tolist(),mse=mse.tolist(),
        predictive_variance=variance.tolist(),adjusted_mse=np.diag(adjusted).tolist(),
        adjusted_mse_over_variance=ratios,
        bias_squared_fraction_of_total_mse=float(bias@bias/total) if total else None,
        trace_adjusted_mse_over_variance=float(np.trace(adjusted)/trace) if trace>1e-15 else None,
        trace_centered_adjusted_mse_over_variance=float(np.trace(centered)/trace) if trace>1e-15 else None,
        predictive_covariance=predictive.tolist(),error_second_moment=second.tolist(),
        adjusted_error_second_moment=adjusted.tolist(),centered_adjusted_error_second_moment=centered.tolist(),
        centered_generalized_eigenvalues=np.linalg.eigvalsh(whitening@centered@whitening.T).tolist() if n>=4 else None,
        generalized_eigenvalues=np.linalg.eigvalsh(whitening@adjusted@whitening.T).tolist() if n>=4 else None,
        floored_predictive_modes=int((eigen<floor).sum()),covariance_floor=floor,
        matrix_status='descriptive' if n>=4 else 'unavailable_fewer_than_four_cases')


def core_moments(draws, truth):
    """(case,draw,core,class) -> equal-core marginal accounting."""
    x,y=np.asarray(draws),np.asarray(truth)
    return moments(x.transpose(0,2,1,3).reshape(-1,x.shape[1],x.shape[-1]),
                   y.reshape(-1,y.shape[-1]))


def affine_probe(draws, bias, scale):
    x,b=np.asarray(draws,float),np.asarray(bias,float)
    if x.ndim!=4 or b.shape!=(x.shape[-1],) or not np.isfinite(b).all() or scale<=0:
        raise ValueError('invalid affine sensitivity inputs')
    if abs(b.sum())>1e-10:raise ValueError('bias must preserve simplex sum')
    mean=x.mean(1,keepdims=True)
    # Preserve exact discrete ties in the unchanged arm.
    z=x-b if scale==1 else mean-b+scale*(x-mean)
    np.testing.assert_allclose(z.sum(-1),x.sum(-1),atol=2e-13)
    invalid=np.any((z < -1e-12)|(z > 1+1e-12),axis=-1)
    return z,dict(out_of_simplex_vectors=int(invalid.sum()),vectors=invalid.size,
                  out_of_simplex_rate=float(invalid.mean()),minimum=float(z.min()),maximum=float(z.max()))


def cyclic_pairings(draws):
    x=np.asarray(draws)
    if x.ndim!=3 or x.shape[1]!=2:raise ValueError('draw/core/class array required')
    for offset in range(len(x)):
        z=x.copy();z[:,1]=np.roll(x[:,1],offset,axis=0)
        yield offset,z


def cross_core(draws,truth):
    x,y=np.asarray(draws,float),np.asarray(truth,float)
    left=moments(x[:,:,0],y[:,0]);right=moments(x[:,:,1],y[:,1])
    pair=moments(x.reshape(len(x),x.shape[1],-1),y.reshape(len(y),-1))
    k=x.shape[-1]
    s=np.asarray(pair['predictive_covariance']);q=np.asarray(pair['adjusted_error_second_moment'])
    return dict(left=left,right=right,mean=moments(x.mean(2),y.mean(1)),
        difference=moments(x[:,:,1]-x[:,:,0],y[:,1]-y[:,0]),
        same_class_predictive_cross_covariance=np.diag(s[:k,k:]).tolist(),
        same_class_adjusted_error_cross_moment=np.diag(q[:k,k:]).tolist(),
        same_class_excess_cross_moment=np.diag(q[:k,k:]-s[:k,k:]).tolist())
