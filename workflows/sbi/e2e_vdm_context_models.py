"""Parameter-matched context VDMs and mass-consistent stochastic factorization.

This new module leaves the previous pilot and its frozen evaluators unchanged.
Projected fine likelihoods are measured per independent subspace dimension.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from workflows.sbi.e2e_direct_vdm import ConditionalVDM, vlb as direct_vlb


def blocks(x, factor=4):
    if x.ndim != 5 or any(n % factor for n in x.shape[2:]):
        raise ValueError('B,C,D,H,W with divisible spatial dimensions required')
    b, c, d, h, w = x.shape
    return x.reshape(b, c, d//factor, factor, h//factor, factor, w//factor, factor)


def block_mean(x, factor=4):
    return blocks(x, factor).mean((3, 5, 7))


def replicate(x, factor=4):
    for dim in (2, 3, 4):
        x = x.repeat_interleave(factor, dim)
    return x


def project(x, factor=4):
    return x-replicate(block_mean(x, factor), factor)


def encode_density(rho, factor=4):
    if not torch.isfinite(rho).all() or (rho <= 0).any():
        raise ValueError('finite strictly positive density required; no clipping')
    return block_mean(rho, factor), project(torch.log(rho), factor)


def decode_density(coarse, u, factor=4):
    if coarse.shape != block_mean(u, factor).shape:
        raise ValueError('coarse/fine block alignment mismatch')
    if not torch.isfinite(u).all() or not torch.isfinite(coarse).all() or (coarse <= 0).any():
        raise ValueError('finite latent and positive coarse density required')
    # Stable normalization within each physical block; never clip density.
    v = blocks(u, factor).permute(0, 1, 2, 4, 6, 3, 5, 7)
    weights = torch.softmax(v.flatten(-3), dim=-1).reshape(v.shape)
    values = weights*coarse[..., None, None, None]*factor**3
    return values.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(u.shape)


@dataclass(frozen=True)
class FineCondition:
    """Observation payload is independent of truth; generated fields are explicit."""
    local: torch.Tensor
    wide: torch.Tensor
    offset_mpc_h: torch.Tensor
    coarse_local: torch.Tensor | None = None
    coarse_wide: torch.Tensor | None = None
    coarse_source: str | None = None

    def validate(self, z, arm, training):
        if (self.local.shape != (len(z), 24, *z.shape[2:]) or self.wide.ndim != 5
                or self.wide.shape[:2] != (len(z), 12)
                or self.offset_mpc_h.shape != (len(z), 3)):
            raise ValueError('invalid local/wide/offset observation dimensions')
        if arm == 'D':
            if self.coarse_local is None or self.coarse_local.shape != z.shape:
                raise ValueError('D needs shared coarse on the aligned fine lattice')
            if self.coarse_wide is None or self.coarse_wide.shape != self.wide[:, :1].shape:
                raise ValueError('D needs shared coarse wide field')
            allowed = ('training_truth',) if training else ('sampled', 'fixed_mean', 'oracle_diagnostic')
            if self.coarse_source not in allowed:
                raise PermissionError('true coarse may not silently enter E2E inference')
        elif any(x is not None for x in (self.coarse_local, self.coarse_wide, self.coarse_source)):
            raise PermissionError('A--C cannot receive matter-derived conditioning')


def position_encoding(shape, dim, device, dtype, *, span, domain_span, stride, offset):
    """Physical centres of stride2 kernel3/pad1 features, relative to wide centre."""
    # First feature remains centred on the first input voxel, not half a token.
    axes = [(torch.arange(n, device=device, dtype=dtype)*stride+.5)/(n*stride)-.5 for n in shape]
    points = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)*span
    points = (points[None]+offset[:, None])/domain_span
    count = math.ceil(dim/6)
    frequency = 2*math.pi*2**torch.linspace(0, 4, count, device=device, dtype=dtype)
    phases = points[..., None]*frequency
    return torch.cat([phases.sin(), phases.cos()], dim=-1).flatten(-2)[..., :dim]


class ContextVDM(ConditionalVDM):
    """Same architecture/parameter count for A--D, different information only."""
    def __init__(self, arm, base=24, levels=3):
        if arm not in ('A', 'B', 'C', 'D'):
            raise ValueError('unknown arm')
        super().__init__(condition_channels=25, base=base, levels=levels, learned=False)
        self.arm = arm
        width = base*2**levels
        self.context_encoder = nn.Sequential(nn.Conv3d(13, 8, 3, 2, 1), nn.SiLU(),
            nn.Conv3d(8, 16, 3, 2, 1), nn.SiLU(), nn.Conv3d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv3d(32, width, 1))
        self.context_norm = nn.LayerNorm(width)
        self.context_attention = nn.MultiheadAttention(width, 1, batch_first=True)

    def forward(self, z, gamma, condition):
        condition.validate(z, self.arm, self.training)
        if z.ndim != 5 or any(n % 2**self.levels for n in z.shape[2:]):
            raise ValueError('grid must divide every U-Net level')
        local_coarse = torch.zeros_like(z) if self.arm != 'D' else condition.coarse_local
        wide_coarse = torch.zeros_like(condition.wide[:, :1]) if self.arm != 'D' else condition.coarse_wide
        wide = condition.wide
        if self.arm in ('A', 'B'):
            # Information firewall: A/B can only see spatial means, including
            # when the caller accidentally passes a spatial array.
            wide = wide.mean((2, 3, 4), keepdim=True).expand_as(wide)
        phase = ((gamma+13.3)/26.6*1000)[:, None]*self.frequency[None]
        e = self.time(torch.cat([phase.sin(), phase.cos()], 1))
        h = self.input(torch.cat([z, condition.local, local_coarse], 1))
        skips = []
        for block, down in zip(self.enc, self.down):
            h = block(h, e)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, e)
        shape = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        q = self.attn_norm(tokens)
        tokens = tokens+self.attn(q, q, q, need_weights=False)[0]
        context = self.context_encoder(torch.cat([wide, wide_coarse], 1))
        width = h.shape[1]
        query_position = position_encoding(h.shape[2:], width, h.device, h.dtype,
            span=324.768, domain_span=1299.072, stride=2**self.levels, offset=condition.offset_mpc_h)
        key_position = position_encoding(context.shape[2:], width, h.device, h.dtype,
            span=1299.072, domain_span=1299.072, stride=8, offset=torch.zeros_like(condition.offset_mpc_h))
        q = self.context_norm(tokens)+query_position
        values = context.flatten(2).transpose(1, 2)
        tokens = tokens+self.context_attention(q, values+key_position, values, need_weights=False)[0]
        h = self.mid2(tokens.transpose(1, 2).reshape(shape), e)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            h = block(torch.cat([up(h), skip], 1), e)
        result = z+self.output(F.silu(h))
        return project(result) if self.arm == 'D' else result


def projected_vlb(model, x, condition, generator, decoder_std=.001, factor=4):
    """Gaussian VLB on the orthogonal block-zero-mean subspace, bits/DOF."""
    if model.arm != 'D' or (block_mean(x, factor).abs().max() > 2e-6):
        raise ValueError('D target must inhabit the declared zero-mean subspace')
    batch = len(x)
    t = (torch.arange(batch, device=x.device)+torch.rand((), device=x.device, generator=generator))/batch
    gamma = model.schedule(t)
    a, s = model.schedule.coefficients(gamma)
    eps = project(torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator), factor)
    expand = lambda v: v[:, None, None, None, None]
    z = expand(a)*x+expand(s)*eps
    error = project(model(z, gamma, condition)-eps, factor).square().flatten(1).mean(1)
    dof_fraction = (factor**3-1)/factor**3
    diffusion = .5*model.schedule.slope.abs()*error.mean()/dof_fraction/math.log(2)
    g0 = model.schedule(x.new_zeros(1))[0]
    g1 = model.schedule(x.new_ones(1))[0]
    a1, s1 = model.schedule.coefficients(g1)
    prior = .5*(a1.square()*x.square().mean()/dof_fraction+s1.square()-1-torch.log(s1.square()))/math.log(2)
    decoder = (.5*math.log(2*math.pi*decoder_std**2)+.5*torch.exp(g0)/decoder_std**2)/math.log(2)
    return diffusion+prior+decoder, dict(diffusion=diffusion, prior=prior, decoder=decoder,
        independent_dimensions=x[0].numel()*dof_fraction)


def loss(model, x, condition, generator, decoder_std=.001):
    objective = projected_vlb if isinstance(model, ContextVDM) and model.arm == 'D' else direct_vlb
    return objective(model, x, condition, generator, decoder_std)


@torch.no_grad()
def coupled_sample(model, condition, steps, seeds, noise_grid=1000):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or noise_grid % steps:
        raise ValueError('steps must divide the coupled noise grid')
    local = condition.local if isinstance(condition, FineCondition) else condition
    if len(seeds) != len(local) or len(set(seeds)) != len(seeds):
        raise ValueError('one distinct addressed seed per draw')
    generators = [torch.Generator(device=local.device).manual_seed(int(seed)) for seed in seeds]
    projector = project if isinstance(model, ContextVDM) and model.arm == 'D' else lambda x: x
    def noise():
        return projector(torch.cat([torch.randn((1, 1, *local.shape[2:]), device=local.device,
            dtype=local.dtype, generator=g) for g in generators]))
    before = model.training
    model.eval()
    z = noise()
    factor = noise_grid//steps
    try:
        for i in range(steps):
            gt = model.schedule(z.new_full((len(z),), 1-i/steps))
            gs = model.schedule(z.new_full((len(z),), 1-(i+1)/steps))
            at, st = model.schedule.coefficients(gt)
            ass, ss = model.schedule.coefficients(gs)
            c = -torch.expm1(gs-gt)
            b = lambda v: v[:, None, None, None, None]
            mean = b(ass/at)*(z-b(c*st)*model(z, gt, condition))
            eps = noise()
            for _ in range(factor-1):
                eps = eps+noise()
            z = projector(mean+b(ss*c.sqrt())*eps/math.sqrt(factor))
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite coupled ancestral path')
        return z
    finally:
        model.train(before)
