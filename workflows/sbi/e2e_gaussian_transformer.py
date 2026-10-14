"""Whole-domain patch transformer for the matched Gaussian CFM diagnostic.

No convolutional feature hierarchy, posterior summaries or target leakage.
The prepare/velocity interface reuses the frozen CFM objective and Heun solver.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def patchify(field, patch=2):
    if field.ndim != 5 or len(set(field.shape[2:])) != 1:
        raise ValueError('Expected a cubic [batch, channel, x, y, z] field')
    b, c, n, _, _ = field.shape
    if n % patch:
        raise ValueError('Patch size must divide domain side')
    k = n // patch
    return field.reshape(b, c, k, patch, k, patch, k, patch).permute(
        0, 2, 4, 6, 1, 3, 5, 7).reshape(b, k**3, c*patch**3)


def unpatchify(tokens, n, patch=2):
    b, length, features = tokens.shape
    k = n // patch
    if n % patch or length != k**3 or features % patch**3:
        raise ValueError('Inconsistent token geometry')
    c = features // patch**3
    return tokens.reshape(b, k, k, k, c, patch, patch, patch).permute(
        0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, n, n, n)


def positions(n, patch, width, device, dtype):
    axis = (torch.arange(n//patch, device=device, dtype=dtype)*patch+patch/2)/n
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    frequencies = 2*math.pi*2**torch.linspace(0, 4, math.ceil(width/6), device=device, dtype=dtype)
    phase = xyz[..., None]*frequencies
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[None, :, :width]


class TransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3*width)
        self.project = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(approximate='tanh'),
                                 nn.Linear(4*width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens, time):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(time)[:, None].chunk(6, -1)
        h = self.norm1(tokens)*(1+scale1)+shift1
        b, length, width = h.shape
        q, k, v = self.qkv(h).reshape(b, length, 3, self.heads, width//self.heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        tokens = tokens+gate1*self.project(attention.transpose(1, 2).reshape(b, length, width))
        return tokens+gate2*self.mlp(self.norm2(tokens)*(1+scale2)+shift2)


class GaussianTransformer(nn.Module):
    def __init__(self, width=240, depth=8, heads=6, patch=2):
        super().__init__()
        if width % heads or min(width, depth, heads, patch) < 1:
            raise ValueError('Invalid transformer dimensions')
        self.width, self.patch = width, patch
        self.architecture = dict(width=width, depth=depth, heads=heads, patch=patch,
                                 mlp_ratio=4, conditioning='observation tokens plus time AdaLN-zero')
        self.state_embed = nn.Linear(patch**3, width)
        self.observation_embed = nn.Linear(3*patch**3, width)
        self.register_buffer('frequency', torch.exp(torch.linspace(0, -math.log(10000), 24)))
        self.time = nn.Sequential(nn.Linear(48, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))
        self.output = nn.Linear(width, patch**3)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def prepare(self, observation, exterior, start, width):
        n = observation.shape[-1]
        if observation.ndim != 5 or observation.shape[1] != 3:
            raise ValueError('Expected [masked y, mask, std]')
        if exterior.shape != observation[:, :1].shape or tuple(start) != (0, 0, 0) or width != n:
            raise ValueError('Whole field only; no exterior or block inference')
        # Deliberately never read exterior values: training passes truth here.
        conditioning = self.observation_embed(patchify(observation, self.patch))
        return conditioning+positions(n, self.patch, self.width, observation.device, observation.dtype)

    def velocity(self, z, t, prepared):
        phase = (1000*t)[:, None]*self.frequency[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), 1))
        tokens = self.state_embed(patchify(z, self.patch))+prepared
        for block in self.blocks:
            tokens = block(tokens, time)
        shift, scale = self.final_modulation(time)[:, None].chunk(2, -1)
        output = self.output(self.final_norm(tokens)*(1+scale)+shift)
        # Match the old backbone's residual velocity and zero-head initial field.
        return z+unpatchify(output, z.shape[-1], self.patch)
