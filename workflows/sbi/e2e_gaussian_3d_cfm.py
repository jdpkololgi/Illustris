"""Gaussian interface to the actual base24/levels3 coupled CFM architecture.

The inherited layers are unchanged. Gaussian observation/state channels replace
galaxy/coarse channels; the full context uses one native and one pooled view.
No analytic posterior, whitening, or precision enters the neural inputs.
"""
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone
import e2e_restricted_context as context


def starts(n, width=8):
    if n % width: raise ValueError('Non-tiling block size')
    return list(itertools.product(range(0, n, width), repeat=3))


def crop(x, start, width):
    return x[(slice(None), slice(None), *[slice(s, s+width) for s in start])]


def position(shape, start, stride, n, width, device, dtype):
    axes = [(torch.arange(k, device=device, dtype=dtype)*stride+s+.5)/n
            for k, s in zip(shape, start)]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    frequencies = 2*math.pi*2**torch.linspace(0, 4, math.ceil(width/6), device=device, dtype=dtype)
    phase = xyz[..., None]*frequencies
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width][None]


class GaussianBackbone(CoupledBackbone):
    def __init__(self):
        super().__init__('coarse', 'wide', base=24, levels=3)

    def prepare(self, observation, exterior, start, width):
        """Exterior values inside the updated block are explicitly erased."""
        n = observation.shape[-1]
        if observation.shape[1] != 3 or exterior.shape != observation[:, :1].shape:
            raise ValueError('Expected [masked y, mask, std] and aligned exterior')
        mask = torch.ones_like(exterior)
        crop(mask, start, width).zero_()
        xyz = torch.stack(torch.meshgrid(*[torch.arange(n, device=exterior.device,
            dtype=exterior.dtype)/max(n-1, 1) for _ in range(3)], indexing='ij'), 0)
        fields = torch.cat((observation, exterior*mask, mask,
            xyz[None].expand(len(exterior), -1, -1, -1, -1),
            torch.zeros_like(observation[:, :1]).expand(-1, 4, -1, -1, -1)), 1)
        zero = torch.zeros_like(exterior)
        joint = self.joint_encoder(torch.cat((fields, zero), 1))
        wide = self.wide_encoder(F.avg_pool3d(torch.cat((fields, zero), 1), 2))
        feature_width = joint.shape[1]
        values = torch.cat((joint.flatten(2).transpose(1, 2), wide.flatten(2).transpose(1, 2)), 1)
        keys = torch.cat((position(joint.shape[2:], (0,)*3, 8, n, feature_width, exterior.device, exterior.dtype),
                         position(wide.shape[2:], (.5,)*3, 16, n, feature_width, exterior.device, exterior.dtype)), 1)
        local = crop(fields, start, width)
        local_inputs = torch.cat((local, fields.mean((2, 3, 4), keepdim=True).expand_as(local),
                                 crop(zero, start, width)), 1)
        query_positions = position((width//8,)*3, start, 8, n, feature_width, exterior.device, exterior.dtype)
        return local_inputs, values, keys, query_positions

    def velocity(self, z, t, prepared):
        local, values, keys, query_positions = prepared
        phase = (1000*t)[:, None]*self.frequency[None]
        embedded = self.time(torch.cat((phase.sin(), phase.cos()), 1))
        h = self.input(torch.cat((z, local), 1)); skips = []
        for block, down in zip(self.enc, self.down):
            h = block(h, embedded); skips.append(h); h = down(h)
        h = self.mid1(h, embedded); shape = h.shape
        tokens = h.flatten(2).transpose(1, 2); q = self.attn_norm(tokens)
        tokens = tokens+self.attn(q, q, q, need_weights=False)[0]
        tokens = tokens+self.context_attention(self.context_norm(tokens)+query_positions,
            values+keys, values, need_weights=False)[0]
        h = self.mid2(tokens.transpose(1, 2).reshape(shape), embedded)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            h = block(torch.cat((up(h), skip), 1), embedded)
        return z+self.output(F.silu(h))


class JointGenerator:
    """FFT draws from exactly the existing fixed-cell-scale long-range prior."""
    def __init__(self, n, device='cuda'):
        self.n, self.device = n, device
        k = 2*np.sin(np.pi*np.fft.fftfreq(n))
        laplacian = sum(a*a for a in np.meshgrid(k, k, k, indexing='ij'))
        power = (1+1.5**2*laplacian)**-2; power /= power.mean()
        self.factor = torch.tensor(np.sqrt(.2*power), device=device, dtype=torch.float32)
        mask, std = context.observations(n, 1)
        self.mask = torch.tensor(mask.reshape(1, 1, n, n, n), device=device, dtype=torch.float32)
        self.std = torch.tensor(std.reshape(1, 1, n, n, n), device=device, dtype=torch.float32)

    def batch(self, count, generator):
        shape = (count, 1, self.n, self.n, self.n)
        noise = torch.randn(shape, device=self.device, generator=generator)
        x = torch.fft.ifftn(torch.fft.fftn(noise, dim=(-3, -2, -1))*self.factor,
                           dim=(-3, -2, -1)).real
        x = x+math.sqrt(.8)*torch.randn((count, 1, 1, 1, 1), device=self.device, generator=generator)
        y = (x+self.std*torch.randn(shape, device=self.device, generator=generator))*self.mask
        return x, torch.cat((y, self.mask.expand_as(y), self.std.expand_as(y)), 1)


def train_step(model, optimizer, data, generator, arm, update, batch=8, noise_generator=None):
    x, observation = data.batch(batch, generator)
    width = data.n if arm == 'whole' else 8
    locations = starts(data.n, width); start = locations[update % len(locations)]
    endpoint = crop(x, start, width)
    prepared = model.prepare(observation, x, start, width)
    noise_generator = generator if noise_generator is None else noise_generator
    noise = torch.randn(endpoint.shape, device=x.device, generator=noise_generator)
    t = (torch.arange(batch, device=x.device)+torch.rand((), device=x.device, generator=noise_generator))/batch
    tt = t[:, None, None, None, None]
    prediction = model.velocity((1-tt)*noise+tt*endpoint, t, prepared)
    loss = (prediction-(endpoint-noise)).square().mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training')
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def draw(model, observation, exterior, start, width, noise, nfe):
    if nfe < 2 or nfe % 2: raise ValueError('Positive even NFE required')
    prepared = model.prepare(observation, exterior, start, width)
    state = noise.clone(); steps = nfe//2
    for i in range(steps):
        first = model.velocity(state, state.new_full((len(state),), i/steps), prepared)
        second = model.velocity(state+first/steps, state.new_full((len(state),), (i+1)/steps), prepared)
        state = state+(first+second)/(2*steps)
    if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite draw')
    return state


@torch.no_grad()
def sweep(model, observation, state, generator, nfe=32):
    result = state.clone()
    for start in starts(state.shape[-1]):
        noise = torch.randn((len(state), 1, 8, 8, 8), device=state.device, generator=generator)
        crop(result, start, 8).copy_(draw(model, observation, result, start, 8, noise, nfe))
    return result
