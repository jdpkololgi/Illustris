"""Whole-domain fine CFM candidates with identical observation information."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone, physical_positions, position_encoding
from workflows.sbi.e2e_gaussian_transformer import TransformerBlock
from workflows.sbi.e2e_vdm_context_models import project, replicate

SHAPE = (96, 64, 64)


@dataclass(frozen=True)
class Condition:
    joint: torch.Tensor
    wide: torch.Tensor
    coarse_wide: torch.Tensor
    source: str

    def validate(self, training):
        b = len(self.joint)
        if (self.joint.shape != (b, 12, *SHAPE) or self.wide.shape != (b, 12, 48, 48, 48)
                or self.coarse_wide.shape != (b, 1, 48, 48, 48)):
            raise ValueError('Wrong common multiwindow condition shape')
        if self.source not in (('training_truth',) if training else ('sampled', 'oracle_diagnostic')):
            raise PermissionError('Wrong coarse provenance')

    def coarse_joint(self):
        return replicate(self.coarse_wide[:, :, 12:36, 16:32, 16:32])


def patchify(x, patch=4):
    b, c, nx, ny, nz = x.shape
    if any(n % patch for n in (nx, ny, nz)):
        raise ValueError('Nondivisible token shape')
    return x.reshape(b, c, nx//patch, patch, ny//patch, patch, nz//patch, patch).permute(
        0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, c*patch**3)


def unpatchify(tokens, shape, patch=4):
    b, length, features = tokens.shape
    nx, ny, nz = shape
    if length != nx*ny*nz//patch**3 or features % patch**3:
        raise ValueError('Token geometry mismatch')
    return tokens.reshape(b, nx//patch, ny//patch, nz//patch, features//patch**3, patch, patch, patch).permute(
        0, 4, 1, 5, 2, 6, 3, 7).reshape(b, features//patch**3, nx, ny, nz)


def token_positions(shape, cell, patch, width, reference):
    axes = [(torch.arange(n//patch, device=reference.device, dtype=reference.dtype)*patch+patch/2-n/2)*cell for n in shape]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(1, -1, 3)
    return position_encoding(xyz, width)


class WholeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = CoupledBackbone('fine', 'joint', base=24, levels=3)

    def prepare(self, condition):
        condition.validate(self.training)
        n = self.net; coarse = condition.coarse_joint()
        local = torch.cat((condition.joint, condition.wide.mean((2, 3, 4), keepdim=True).expand_as(condition.joint), coarse), 1)
        joint = n.joint_encoder(torch.cat((condition.joint, coarse), 1))
        wide = n.wide_encoder(torch.cat((condition.wide, condition.coarse_wide), 1))
        offset = coarse.new_zeros((len(coarse), 3)); width = joint.shape[1]
        jp = physical_positions(joint.shape[2:], 6.766, 8, offset)
        wp = physical_positions(wide.shape[2:], 27.064, 8, offset)
        values = torch.cat((joint.flatten(2).transpose(1, 2), wide.flatten(2).transpose(1, 2)), 1)
        return local, values, position_encoding(torch.cat((jp, wp), 1), width), position_encoding(jp, width)

    def prediction(self, z, t, prepared):
        local, values, positions, queries = prepared; n = self.net
        phase = (1000*t)[:, None]*n.frequency[None]
        embedded = n.time(torch.cat((phase.sin(), phase.cos()), 1))
        h = n.input(torch.cat((z, local), 1)); skips = []
        for block, down in zip(n.enc, n.down):
            h = block(h, embedded); skips.append(h); h = down(h)
        h = n.mid1(h, embedded); shape = h.shape
        tokens = h.flatten(2).transpose(1, 2); q = n.attn_norm(tokens)
        tokens = tokens+n.attn(q, q, q, need_weights=False)[0]
        tokens = tokens+n.context_attention(n.context_norm(tokens)+queries, values+positions, values, need_weights=False)[0]
        h = n.mid2(tokens.transpose(1, 2).reshape(shape), embedded)
        for up, block, skip in zip(n.up, n.dec, reversed(skips)):
            h = block(torch.cat((up(h), skip), 1), embedded)
        return z+n.output(F.silu(h))


class WholeTransformer(nn.Module):
    def __init__(self, width=240, depth=8, heads=6, patch=4):
        super().__init__()
        self.width, self.patch = width, patch
        self.state = nn.Linear(patch**3, width)
        self.local = nn.Linear(13*patch**3, width)
        self.wide = nn.Linear(13*patch**3, width)
        self.types = nn.Parameter(torch.zeros(2, 1, width))
        self.register_buffer('frequency', torch.exp(torch.linspace(0, -math.log(10000), 24)))
        self.time = nn.Sequential(nn.Linear(48, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))
        self.output = nn.Linear(width, patch**3)
        nn.init.zeros_(self.modulation[-1].weight); nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def prepare(self, condition):
        condition.validate(self.training)
        coarse = condition.coarse_joint()
        local = self.local(patchify(torch.cat((condition.joint, coarse), 1), self.patch))
        wide = self.wide(patchify(torch.cat((condition.wide, condition.coarse_wide), 1), self.patch))
        local = local+token_positions(SHAPE, 6.766, self.patch, self.width, coarse)+self.types[0]
        wide = wide+token_positions((48,)*3, 27.064, self.patch, self.width, coarse)+self.types[1]
        return local, wide

    def prediction(self, z, t, prepared):
        local, wide = prepared
        phase = (1000*t)[:, None]*self.frequency[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), 1))
        tokens = torch.cat((self.state(patchify(z, self.patch))+local, wide), 1)
        for block in self.blocks:
            tokens = checkpoint(block, tokens, time, use_reentrant=False) if self.training else block(tokens, time)
        shift, scale = self.modulation(time)[:, None].chunk(2, -1)
        output = self.output(self.norm(tokens[:, :local.shape[1]])*(1+scale)+shift)
        return z+unpatchify(output, SHAPE, self.patch)


class FineModel(nn.Module):
    def __init__(self, architecture, prediction='v'):
        super().__init__()
        if architecture not in ('unet', 'transformer') or prediction not in ('v', 'x'):
            raise ValueError('Unknown fine model')
        self.architecture, self.parameterization = architecture, prediction
        self.network = WholeUNet() if architecture == 'unet' else WholeTransformer()

    def prepare(self, condition):
        return self.network.prepare(condition)

    def velocity(self, z, t, prepared):
        value = self.network.prediction(z, t, prepared).float()
        if self.parameterization == 'x':
            value = (value-z)/(1-t).clamp_min(1/1024)[:, None, None, None, None]
        return project(value)

    def forward(self, z, t, condition):
        return self.velocity(z, t, self.prepare(condition))


@torch.no_grad()
def sample(model, condition, noise, nfe=128):
    if nfe < 2 or nfe % 2 or noise.shape != (len(condition.joint), 1, *SHAPE):
        raise ValueError('Wrong sampler shape/NFE')
    model.eval(); state = project(noise.float())
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=state.is_cuda):
        prepared = model.prepare(condition)
        for step in range(nfe//2):
            dt = 2/nfe; t = step*dt
            a = model.velocity(state, state.new_full((len(state),), t), prepared)
            b = model.velocity(project(state+dt*a), state.new_full((len(state),), t+dt), prepared)
            state = project(state+.5*dt*(a+b))
    if not torch.isfinite(state).all():
        raise FloatingPointError('Nonfinite generated fine state')
    return state
