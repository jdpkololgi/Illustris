"""Reference-led 3D conditional VDM; full log density, observation-only inputs.

Independent implementation of the continuous VLB/ancestral equations used by
Ono et al. (2403.10648), not a reproduction of their 2D CAMELS experiment.
No positive-noise clean identity penalty, spectral penalty, clipping of draws,
or target/coarse teacher forcing. Survey edges are nonperiodic.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LinearSchedule(nn.Module):
    def __init__(self, low=-13.3, high=13.3, learned=True):
        super().__init__()
        if high <= low:
            raise ValueError('increasing log noise-to-signal variance required')
        self.low = nn.Parameter(torch.tensor(float(low)), requires_grad=learned)
        self.slope = nn.Parameter(torch.tensor(float(high-low)), requires_grad=learned)

    def forward(self, t):
        return self.low + self.slope.abs()*t

    @staticmethod
    def coefficients(gamma):
        return torch.sigmoid(-gamma).sqrt(), torch.sigmoid(gamma).sqrt()


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.time = nn.Linear(emb, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.time.weight); nn.init.zeros_(self.time.bias)

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(e))[:, :, None, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class ConditionalVDM(nn.Module):
    def __init__(self, condition_channels=24, base=24, levels=3, learned=True):
        super().__init__()
        self.schedule = LinearSchedule(learned=learned)
        self.levels = levels
        self.time = nn.Sequential(nn.Linear(48, 192), nn.SiLU(), nn.Linear(192, 192))
        self.register_buffer('frequency', torch.exp(torch.linspace(0, -math.log(10000), 24)))
        self.input = nn.Conv3d(condition_channels+1, base, 3, padding=1)
        self.down = nn.ModuleList(); self.enc = nn.ModuleList(); self.dec = nn.ModuleList(); self.up = nn.ModuleList()
        widths = [base*2**i for i in range(levels+1)]
        for a, b in zip(widths[:-1], widths[1:]):
            self.enc.append(ResidualBlock(a, a, 192))
            self.down.append(nn.Conv3d(a, b, 3, stride=2, padding=1))
        self.mid1 = ResidualBlock(widths[-1], widths[-1], 192)
        self.attn_norm = nn.LayerNorm(widths[-1])
        self.attn = nn.MultiheadAttention(widths[-1], 1, batch_first=True)
        self.mid2 = ResidualBlock(widths[-1], widths[-1], 192)
        for a, b in zip(widths[:0:-1], widths[-2::-1]):
            self.up.append(nn.ConvTranspose3d(a, b, 4, stride=2, padding=1))
            self.dec.append(ResidualBlock(2*b, b, 192))
        self.output = nn.Conv3d(base, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, z, gamma, condition):
        if z.ndim != 5 or condition.shape[0] != z.shape[0] or condition.shape[2:] != z.shape[2:]:
            raise ValueError('matching 3D batches required')
        if any(n % 2**self.levels for n in z.shape[2:]):
            raise ValueError('grid must divide all U-Net levels')
        phase = ((gamma+13.3)/26.6*1000)[:, None]*self.frequency[None]
        e = self.time(torch.cat([phase.sin(), phase.cos()], dim=1))
        h = self.input(torch.cat([z, condition], dim=1)); skips = []
        for block, down in zip(self.enc, self.down):
            h = block(h, e); skips.append(h); h = down(h)
        h = self.mid1(h, e); shape = h.shape
        tokens = h.flatten(2).transpose(1, 2); q = self.attn_norm(tokens)
        tokens = tokens + self.attn(q, q, q, need_weights=False)[0]
        h = self.mid2(tokens.transpose(1, 2).reshape(shape), e)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            h = block(torch.cat([up(h), skip], dim=1), e)
        return z + self.output(F.silu(h))


def vlb(model, x, condition, generator, decoder_std=.001):
    """Continuous VLB in bits/voxel, with analytic expected decoder term.

    Decoder E[-log N(x; z0/alpha0, decoder_std)] equals its analytic expectation;
    no denoiser in this endpoint term. This removes endpoint Monte Carlo noise.
    Antithetic batch times reduce variance; fixed arm has identical VLB factors.
    """
    batch = x.shape[0]
    t = (torch.arange(batch, device=x.device) + torch.rand((), device=x.device, generator=generator))/batch
    gamma = model.schedule(t); a, s = model.schedule.coefficients(gamma)
    eps = torch.randn(x.shape, device=x.device, generator=generator)
    z = a[:,None,None,None,None]*x + s[:,None,None,None,None]*eps
    error = (model(z, gamma, condition)-eps).square().flatten(1).mean(1)
    diffusion = .5*model.schedule.slope.abs()*error.mean()/math.log(2)
    g0 = model.schedule(x.new_zeros(1))[0]; g1 = model.schedule(x.new_ones(1))[0]
    a1, s1 = model.schedule.coefficients(g1)
    prior = .5*(a1.square()*x.square()+s1.square()-1-torch.log(s1.square())).mean()/math.log(2)
    decoder = (.5*math.log(2*math.pi*decoder_std**2)+.5*torch.exp(g0)/decoder_std**2)/math.log(2)
    return diffusion+prior+decoder, dict(diffusion=diffusion, prior=prior, decoder=decoder)


@torch.no_grad()
def sample(model, condition, steps, generator):
    """Finite-endpoint ancestral reverse transitions, no true target access."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('positive integer steps required')
    before = model.training; model.eval()
    z = torch.randn(condition[:, :1].shape, device=condition.device, generator=generator)
    try:
        for i in range(steps):
            t = z.new_full((len(z),), 1-i/steps); s = z.new_full((len(z),), 1-(i+1)/steps)
            gt, gs = model.schedule(t), model.schedule(s)
            at, st = model.schedule.coefficients(gt); ass, ss = model.schedule.coefficients(gs)
            c = -torch.expm1(gs-gt)
            broadcast = lambda v: v[:,None,None,None,None]
            mean = broadcast(ass/at)*(z-broadcast(c*st)*model(z, gt, condition))
            eps = torch.randn(z.shape, device=z.device, generator=generator)
            z = mean+broadcast(ss*c.sqrt())*eps
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite ancestral path')
        # Authors return finite endpoint z0; do not silently add another denoiser.
        return z
    finally:
        model.train(before)
