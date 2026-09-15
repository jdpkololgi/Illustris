"""Endpoint-safe residual denoisers for a training-only architecture comparison.

Independent implementations, inspired by DiT (2212.09748) and wavelet image
restoration (1805.07071); neither is a reproduction of the published model.
No cyclic attention, periodic convolution, or lossy patch averaging.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from workflows.sbi.e2e_wide_models import _downsample_mean


def coefficients(time, tau=.05):
    if not 0 < tau <= 1 or bool(((time < 0) | (time > 1)).any()):
        raise ValueError('VP time in [0,1] and positive tau <= 1 required')
    a = torch.cos(time.double()*math.pi/2).clamp(0, 1).to(time)
    b = torch.sin(time.double()*math.pi/2).to(time)
    d = (b.square()+tau*tau*a.square()).sqrt()
    return tuple(v[:, None, None, None, None] for v in (a, b, d))


class NoiseCode(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.register_buffer('frequencies', 2.**torch.arange(8))
        self.net = nn.Sequential(nn.Linear(21, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, time, present):
        code = torch.tan(time.double()*math.pi/2).clamp(.002, 80).log().to(time)/4
        angles = code[:, None]*self.frequencies[None, :]
        return self.net(torch.cat((time[:, None], torch.sin(time[:, None]*math.pi/2),
                                  torch.cos(time[:, None]*math.pi/2), code[:, None],
                                  angles.sin(), angles.cos(), present[:, None]), dim=1))


def zero_head(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class ResidualUNet(nn.Module):
    def __init__(self, base, film):
        super().__init__()
        self.base = base
        widths = [b.layers[0].out_channels for b in base.encoders]
        self.widths = widths+widths[-2::-1]
        self.code = NoiseCode() if film else None
        self.film = nn.Linear(64, 2*sum(self.widths)) if film else None
        if film:
            zero_head(self.film)
        zero_head(base.output)

    def forward(self, state, time, condition, wide, present):
        if self.film is None:
            return self.base(state, time, condition, wide_condition=wide)
        chunks = self.film(self.code(time, present)).split([2*w for w in self.widths], dim=1)
        def adapt(x, i):
            scale, bias = chunks[i].chunk(2, dim=1)
            return x*(1+scale[:, :, None, None, None])+bias[:, :, None, None, None]
        t = time[:, None, None, None, None]
        features = torch.cat((t, torch.sin(math.pi*t), torch.cos(math.pi*t)), dim=1)
        x = torch.cat((state, condition, features.expand(-1, -1, *state.shape[2:])), dim=1)
        skips = []
        for i, block in enumerate(self.base.encoders):
            if i:
                x = _downsample_mean(x)
            x = adapt(block(x), i); skips.append(x)
        x = x+self.base.wide_encoder(wide)[:, :, None, None, None]
        for i, (block, skip) in enumerate(zip(self.base.decoders, reversed(skips[:-1]))):
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = adapt(block(torch.cat((x, skip), dim=1)), len(skips)+i)
        return self.base.output(x)


def pack_voxels(x):
    b, c, d, h, w = x.shape
    if any(n % 2 for n in (d, h, w)):
        raise ValueError('even spatial dimensions required')
    return x.reshape(b, c, d//2, 2, h//2, 2, w//2, 2).permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c, 8, d//2, h//2, w//2)


def unpack_voxels(x):
    b, c, eight, d, h, w = x.shape
    if eight != 8:
        raise ValueError('eight voxel/subband channels required')
    return x.reshape(b, c, 2, 2, 2, d, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c, 2*d, 2*h, 2*w)


def haar_matrix():
    return torch.tensor([[(-1.)**((i & j).bit_count()) for j in range(8)] for i in range(8)])/math.sqrt(8)


def haar(x, matrix):
    z = torch.einsum('ij,bcjdhw->bcidhw', matrix, pack_voxels(x))
    return z.flatten(1, 2)


def inverse_haar(x, matrix):
    b, c, d, h, w = x.shape
    if c % 8:
        raise ValueError('Haar channels must be divisible by eight')
    return unpack_voxels(torch.einsum('ji,bcjdhw->bcidhw', matrix, x.reshape(b, c//8, 8, d, h, w)))


class WaveletNet(nn.Module):
    def __init__(self, base, width=32, blocks=4):
        super().__init__()
        self.register_buffer('matrix', haar_matrix())
        self.stem = nn.Conv3d(8*(1+base.condition_channels), width, 3, padding=1)
        self.wide = base.wide_encoder
        self.projection = nn.Linear(self.wide[-1].out_features, width)
        self.code = NoiseCode()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv3d(width, width, 3, padding=1), nn.SiLU(),
                                                   nn.Conv3d(width, width, 3, padding=1)) for _ in range(blocks)])
        self.films = nn.ModuleList([nn.Linear(64, 2*width) for _ in range(blocks)])
        for layer in self.films:
            zero_head(layer)
        self.output = nn.Conv3d(width, 8, 1); zero_head(self.output)

    def forward(self, state, time, condition, wide, present):
        x = self.stem(haar(torch.cat((state, condition), dim=1), self.matrix))
        x = x+self.projection(self.wide(wide))[:, :, None, None, None]
        code = self.code(time, present)
        for block, film in zip(self.blocks, self.films):
            scale, bias = film(code).chunk(2, dim=1)
            x = x+.1*block(x*(1+scale[:, :, None, None, None])+bias[:, :, None, None, None])
        return inverse_haar(self.output(x), self.matrix)


def partition(x, size=4, offset=0):
    """Nonperiodic, left-padded offset windows; channels-last input."""
    b, d, h, w, c = x.shape
    pads = [(size-(n+offset) % size) % size for n in (d, h, w)]
    x = F.pad(x.permute(0, 4, 1, 2, 3), (offset, pads[2], offset, pads[1], offset, pads[0]))
    mask = F.pad(x.new_ones((b, 1, d, h, w)), (offset, pads[2], offset, pads[1], offset, pads[0]))
    dd, hh, ww = x.shape[2:]
    def pack(z):
        return z.permute(0, 2, 3, 4, 1).reshape(b, dd//size, size, hh//size, size, ww//size, size, -1).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, size**3, z.shape[1])
    return pack(x), pack(mask)[..., 0].bool(), (b, d, h, w, dd, hh, ww, size, offset)


def unpartition(x, metadata):
    b, d, h, w, dd, hh, ww, size, offset = metadata
    z = x.reshape(b, dd//size, hh//size, ww//size, size, size, size, -1).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, dd, hh, ww, -1)
    return z[:, offset:offset+d, offset:offset+h, offset:offset+w]


class WindowBlock(nn.Module):
    def __init__(self, width=64, heads=4, offset=0):
        super().__init__(); self.heads = heads; self.offset = offset
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width); self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.film = nn.Linear(64, 4*width); zero_head(self.film)

    def forward(self, x, code):
        s1, b1, s2, b2 = [z[:, None, None, None, :] for z in self.film(code).chunk(4, dim=1)]
        z, mask, meta = partition(self.norm1(x)*(1+s1)+b1, offset=self.offset)
        n, tokens, width = z.shape
        q, k, v = self.qkv(z).reshape(n, tokens, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2))/math.sqrt(width//self.heads)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        z = (scores.softmax(-1) @ v).transpose(1, 2).reshape(n, tokens, width)
        x = x+unpartition(self.proj(z), meta)
        return x+self.mlp(self.norm2(x)*(1+s2)+b2)


class PatchTransformer(nn.Module):
    def __init__(self, base, width=64):
        super().__init__()
        self.patch = nn.Conv3d(1+base.condition_channels, width, 2, stride=2)
        self.position = nn.Linear(3, width)
        self.code = NoiseCode(); self.wide = base.wide_encoder
        self.projection = nn.Linear(self.wide[-1].out_features, 64)
        self.blocks = nn.ModuleList([WindowBlock(width, offset=i) for i in (0, 2)])
        self.output = nn.Linear(width, 8); zero_head(self.output)

    def forward(self, state, time, condition, wide, present):
        if any(n % 2 for n in state.shape[2:]):
            raise ValueError('even spatial dimensions required')
        x = self.patch(torch.cat((state, condition), dim=1)).permute(0, 2, 3, 4, 1)
        coords = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, n, device=x.device, dtype=x.dtype) for n in x.shape[1:4]], indexing='ij'), dim=-1)
        x = x+self.position(coords)[None]
        code = self.code(time, present)+self.projection(self.wide(wide))
        for block in self.blocks:
            x = block(x, code)
        return unpack_voxels(self.output(x).permute(0, 4, 1, 2, 3)[:, None])


class BoundedResidual(nn.Module):
    """D=a*x+(b/d)*r, v=-r/d; d=sqrt(b²+tau²*a²).

    Bounded gain <=1/tau, valid at BOTH endpoints. Raw-head rescaling changes
    optimization geometry, not the scalar v-MSE objective. Not the earlier
    singular D=x/a+r specialist and not an equivalent warm-start conversion.
    """
    def __init__(self, net, tau=.05):
        super().__init__(); self.net = net; self.tau = tau

    def forward(self, state, time, condition, wide_condition=None, context_present=None):
        if context_present is None:
            context_present = torch.ones_like(time)
        _, _, d = coefficients(time, self.tau)
        return -self.net(state, time, condition, wide_condition, context_present)/d


def build(arm, base, tau=.05):
    if arm.startswith('unet_'):
        net = ResidualUNet(base, film=arm != 'unet_residual')
    elif arm == 'transformer':
        net = PatchTransformer(base)
    elif arm == 'wavelet':
        net = WaveletNet(base)
    else:
        raise ValueError('unknown architecture')
    return BoundedResidual(net, tau)


def loss_for(model, target, noise, time, condition, wide, clean_weight=0., drop=False):
    t = target.new_full((target.shape[0],), time)
    a, b, _ = coefficients(t, model.tau)
    present = torch.zeros_like(t) if drop else torch.ones_like(t)
    if drop:
        condition, wide = torch.zeros_like(condition), torch.zeros_like(wide)
    pred = model(a*target+b*noise, t, condition, wide_condition=wide, context_present=present)
    primary = (pred-(a*noise-b*target)).square().mean()
    regularizer = primary.new_zeros(())
    if clean_weight:
        clean_pred = model(a*target, t, condition, wide_condition=wide, context_present=present)
        # (D(a*y,t)-y)^2 / b^2, evaluated stably; deliberately a changed objective.
        regularizer = (clean_pred+b*target).square().mean()
    return primary+clean_weight*regularizer, primary.detach(), regularizer.detach()
