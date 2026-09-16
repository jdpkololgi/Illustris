"""Explicit clean-anchor and near-clean even/odd response supervision.

VP chart: x=a*y+b*epsilon; D=a*x-b*v. Auxiliary residuals are expressed
directly in velocity units, analytically equal to physical errors divided by b.
No numerical division by b, per-field normalization, detach/teacher, or loss FFT.
The auxiliary ratio is strictly positive and <=.05; exact zero is evaluation only.
"""
import math
import torch
from workflows.sbi.e2e_multinoise_models import coefficients


def auxiliary_ratio(update, train_seed, spec):
    # Same six-bin/15-field traversal as the existing schedule avoids field/bin
    # alias; RNG is addressed separately, not consumed from the primary stream.
    lo, hi = spec['auxiliary_bins'][(update // 3) % 6]
    gen = torch.Generator().manual_seed(train_seed + spec['auxiliary_seed_offset'] + update)
    u = float(torch.rand((), generator=gen))
    return math.exp(math.log(lo) + u * math.log(hi / lo))


def residual_terms(v0, vp, vm, target, noise, a, b, fraction):
    even = (vp + vm) / 2
    odd = (vp - vm) / 2
    return {
        'identity': (v0 + b * target).square().mean(),
        'even_consistency': (even - v0).square().mean(),
        'odd_noise_response': (odd - a * fraction * noise).square().mean(),
    }


def objective(model, target, noise, primary_time, condition, wide, auxiliary_noise,
              ratio, fraction, identity_weight, response_weight):
    if not 0 < ratio <= .05 or not 0 < fraction < 1:
        raise ValueError('strictly positive low-noise auxiliary and sub-noise perturbation required')
    if min(identity_weight, response_weight) < 0:
        raise ValueError('nonnegative objective weights required')
    t = target.new_full((target.shape[0],), primary_time)
    a, b, _ = coefficients(t, model.tau)
    pred = model(a * target + b * noise, t, condition, wide_condition=wide)
    denoise = (pred - (a * noise - b * target)).square().mean()
    ta = target.new_full(t.shape, 2 * math.atan(ratio) / math.pi)
    aa, bb, _ = coefficients(ta, model.tau)
    x0 = aa * target
    # Every arm executes all four forwards, including the zero-weight control.
    v0 = model(x0, ta, condition, wide_condition=wide)
    vp = model(x0 + bb * fraction * auxiliary_noise, ta, condition, wide_condition=wide)
    vm = model(x0 - bb * fraction * auxiliary_noise, ta, condition, wide_condition=wide)
    terms = residual_terms(v0, vp, vm, target, auxiliary_noise, aa, bb, fraction)
    total = denoise + identity_weight * terms['identity'] + response_weight * (
        terms['even_consistency'] + terms['odd_noise_response'])
    logs = {'denoising': denoise.detach(), **{k: v.detach() for k, v in terms.items()}}
    return total, logs
