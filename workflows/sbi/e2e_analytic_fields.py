"""Known-distribution diagnostics, not a cosmological posterior model.

Periodic synthetic boxes only. FFT norm='ortho' makes P the covariance
eigenvalues and mean(P) the point variance. Never rescale individual draws.
Log-Gaussian means rho=exp(g-var(g)/2), delta=rho-1, not log(delta).
"""
import math
import numpy as np
import torch
from torch import nn
from scipy.integrate import quad


def spectrum(n, variance=.49, device='cpu', dtype=torch.float64):
    if n < 2 or variance <= 0:
        raise ValueError('positive variance and grid >=2 required')
    k = torch.fft.fftfreq(n, device=device, dtype=dtype)*n
    r2 = sum(x*x for x in torch.meshgrid(k, k, k, indexing='ij'))
    power = (1+r2/4).pow(-1.5)
    return power * (variance/power.mean())


def filt(x, multiplier):
    return torch.fft.ifftn(torch.fft.fftn(x, dim=(-3,-2,-1), norm='ortho') * multiplier,
                           dim=(-3,-2,-1), norm='ortho').real


def gaussian_mean(observed, power, sigma, mean=0.):
    """Exact posterior E[g|g+sigma*eps] for a stationary Gaussian prior."""
    if sigma < 0:
        raise ValueError('negative noise')
    return mean + filt(observed-mean, power/(power+sigma*sigma))


def log_density_mean(observed_log, power, sigma, mean=0.):
    """Exact E[delta|noisy g]; exponentiating E[g|.] alone is NOT this mean."""
    posterior_mean = gaussian_mean(observed_log, power, sigma, mean)
    posterior_variance = (power*sigma*sigma/(power+sigma*sigma)).mean()
    return torch.expm1(posterior_mean + .5*posterior_variance - .5*power.mean())


class GaussianVelocity(nn.Module):
    """Exact conditional VP v or straight-path CFM velocity for sample_field."""
    def __init__(self, power, method):
        super().__init__()
        if method not in ('diffusion','cfm'):
            raise ValueError('unknown path')
        self.register_buffer('power', power)
        self.method = method

    def forward(self, x, time, condition, wide_condition=None):
        mean = condition[:, :1]
        t = time[:,None,None,None,None]
        if self.method == 'cfm':
            variance = (1-t).square()+t.square()*self.power
            return mean+filt(x-t*mean, (t*self.power-(1-t))/variance)
        a, b = torch.cos(t*math.pi/2), torch.sin(t*math.pi/2)
        variance = a.square()*self.power+b.square()
        z = x-a*mean
        clean = mean+filt(z,a*self.power/variance)
        noise = filt(z,b/variance)
        return a*noise-b*clean


def iid_lognormal_density_mean(observed_delta, sigma, log_variance=.49, tol=1e-10):
    """Numerical scalar reference for ADDITIVE DENSITY noise, iid lognormal prior.

    This is not the correlated-field posterior. Integration in likelihood-z
    resolves narrow small-sigma peaks; +/-12 Gaussian noise SD truncation.
    Caller checks tolerance convergence. No clipping of observations/densities.
    """
    if sigma <= 0 or log_variance <= 0 or observed_delta <= -1:
        raise ValueError('positive noise/variance and diagnostic input >-1 required')
    lower = max(-12.,(-1-observed_delta)/sigma)
    def integrand(z, moment):
        rho = 1+observed_delta+sigma*z
        if rho <= 0:
            return 0.
        log_prior = -math.log(rho)-.5*(math.log(rho)+log_variance/2)**2/log_variance
        return math.exp(log_prior-.5*z*z)*rho**moment
    denominator = quad(integrand,lower,12.,args=(0,),epsabs=tol,epsrel=tol,limit=300)[0]
    numerator = quad(integrand,lower,12.,args=(1,),epsabs=tol,epsrel=tol,limit=300)[0]
    if denominator <= 0:
        raise FloatingPointError('quadrature underflow')
    return numerator/denominator-1


def min_snr_v_weight(sigma, gamma=5.):
    """min(SNR,gamma)/(SNR+1), stable in the VP v-MSE chart."""
    if gamma <= 0 or bool((sigma < 0).any()):
        raise ValueError('invalid SNR weight')
    q2=sigma.square()
    return torch.minimum(torch.ones_like(q2),gamma*q2)/(1+q2)


def oracle_report(device, n=16, draws=32):
    from workflows.sbi.e2e_wide_models import sample_field
    power=spectrum(n,device=device)
    condition=torch.zeros(draws,1,n,n,n,device=device,dtype=torch.float64)
    condition += .15*torch.sin(2*math.pi*torch.arange(n,device=device)/n)[None,None,:,None,None]
    rng=torch.Generator(device=device).manual_seed(916220)
    white=torch.randn(condition.shape,device=device,dtype=condition.dtype,generator=rng)
    truth=condition+filt(white,power.sqrt())
    noise=torch.randn(condition.shape,device=device,dtype=condition.dtype,generator=rng)
    clean_records=[]
    for q in (0.,.001,.005,.01,.05,.2,1.):
        pred=gaussian_mean(truth+q*noise,power,q,condition)
        clean=gaussian_mean(truth,power,q,condition)
        postvar=(power*q*q/(power+q*q)).mean()
        clean_records.append(dict(sigma=q,gaussian_clean_rms=float((clean-truth).square().mean().sqrt()),
            gaussian_noisy_mse=float((pred-truth).square().mean()),bayes_risk=float(postvar),
            log_gaussian_clean_density_rms=float((log_density_mean(truth,power,q,condition)-torch.expm1(truth-.5*power.mean())).square().mean().sqrt()),
            log_density_mean_vs_exp_mean_rms=float((log_density_mean(truth+q*noise,power,q,condition)-torch.expm1(pred-.5*power.mean())).square().mean().sqrt())))
    sampling=[]
    for method in ('diffusion','cfm'):
        oracle=GaussianVelocity(power,method)
        for steps in (32,128,512):
            draw=sample_field(oracle,condition,method,steps,torch.Generator(device=device).manual_seed(916220))
            # The exact probability-flow/CFM map uses the identical initial noise.
            rel=lambda a,b:float((a-b).square().mean().sqrt()/b.square().mean().sqrt())
            spectrum_ratio=torch.fft.fftn(draw-condition,dim=(-3,-2,-1),norm='ortho').abs().square().mean(0)[0]/power
            radial=[]
            k=torch.fft.fftfreq(n,device=device)*n
            radius=sum(x*x for x in torch.meshgrid(k,k,k,indexing='ij')).sqrt()
            for lo,hi in ((0,2),(2,4),(4,8),(8,100)):
                radial.append(float(spectrum_ratio[(radius>=lo)&(radius<hi)].mean()))
            sampling.append(dict(method=method,steps=steps,gaussian_paired_relative_rmse=rel(draw,truth),
                log_gaussian_paired_relative_rmse=rel(torch.expm1(draw-.5*power.mean()),torch.expm1(truth-.5*power.mean())),
                whitened_power_bands=radial,delta_min=float(torch.expm1(draw-.5*power.mean()).min())))
    quadrature=[]
    for y in (-.9,-.5,0.,1.,3.):
        for q in (.005,.01,.05,.2,1.):
            a=iid_lognormal_density_mean(y,q,tol=1e-9)
            b=iid_lognormal_density_mean(y,q,tol=1e-11)
            quadrature.append(dict(clean_delta=y,sigma=q,posterior_mean=b,clean_shift=b-y,tolerance_difference=abs(a-b)))
    passed=(all(x['gaussian_paired_relative_rmse']<.025 and x['log_gaussian_paired_relative_rmse']<.05
                for x in sampling if x['steps']==512) and
            max(x['tolerance_difference'] for x in quadrature)<1e-7 and
            all(abs(x['gaussian_noisy_mse']/x['bayes_risk']-1)<.05 for x in clean_records if x['sigma']>0))
    return dict(passed=passed,grid=n,draws=draws,seed=916220,periodic_synthetic_only=True,
        clean=clean_records,sampling=sampling,iid_density_quadrature=quadrature,
        caveat='Log-space exact correlated oracle and density-space iid numerical oracle are different likelihoods; neither calibrates the cosmological posterior.')
