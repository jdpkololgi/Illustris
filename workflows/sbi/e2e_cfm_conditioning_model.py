"""Noise-aware fine factor without changing the frozen parent backbone source."""
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from workflows.sbi.e2e_coupled_benchmark_models import (
    CoupledBackbone, FieldCondition, parent_crop, physical_positions,
    position_encoding, FINE_CELL,
)
from workflows.sbi.e2e_vdm_context_models import project, replicate


@dataclass(frozen=True)
class AugmentedCondition(FieldCondition):
    # coarse_source records paired-truth versus sampled ENDPOINT provenance;
    # strength explicitly records subsequent conditioning-only corruption.
    strength: torch.Tensor | None = None

    def validate(self, z, stage, domain, training):
        super().validate(z, stage, domain, training)
        if stage != 'fine' or self.strength is None:
            raise ValueError('noise-aware fine condition required')
        validate_strength(self.strength, len(z), z.device)


def validate_strength(s, batch, device):
    if (s.shape != (batch,) or s.device != device or not s.is_floating_point()
            or not torch.isfinite(s).all() or (s < 0).any() or (s > .2).any()):
        raise ValueError('one finite strength in [0,.2] per draw required')


def corrupt_endpoint(coarse, s, eta):
    """Independent-base CFM bridge in standardized log-density, s=1-t.

    Input endpoint is never modified and must be retained for the mass decoder.
    At zero, select the original values exactly (also for mixed-strength batches).
    """
    validate_strength(s, len(coarse), coarse.device)
    if (eta.shape != coarse.shape or eta.dtype != coarse.dtype
            or eta.device != coarse.device or not torch.isfinite(eta).all()
            or not torch.isfinite(coarse).all()):
        raise ValueError('finite aligned coarse noise required')
    w = s[:, None, None, None, None]
    return torch.where(w == 0, coarse, (1-w)*coarse+w*eta)


def augment_condition(condition, s, eta):
    """Aligned training/local-window corruption; crop comes from ONE wide field."""
    if condition.coarse_wide is None or condition.region != 'joint':
        raise ValueError('joint fine window with paired wide endpoint required')
    wide = corrupt_endpoint(condition.coarse_wide, s, eta)
    joint = replicate(wide[:, :, 16:32, 18:30, 18:30])
    return AugmentedCondition(condition.joint, condition.wide,
        condition.joint_center_from_wide_mpc_h, condition.region, joint, wide,
        condition.coarse_source, s)


class AugmentedFine(CoupledBackbone):
    """Parent forward arithmetic with one separate, zero-gated s embedding.

    The copied fine-only forward keeps the original file/hash unchanged. Exact
    s=0 parent parity is a hard GPU smoke test, not assumed from inspection.
    """
    def __init__(self, base=24, levels=3):
        super().__init__('fine', 'joint', base, levels)
        self.augmentation_time = nn.Sequential(nn.Linear(1, 192), nn.SiLU(),
                                               nn.Linear(192, 192))
        nn.init.zeros_(self.augmentation_time[-1].weight)
        nn.init.zeros_(self.augmentation_time[-1].bias)

    def load_parent(self, state):
        result = self.load_state_dict(state, strict=False)
        expected = {name for name in self.state_dict() if name.startswith('augmentation_time.')}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError('parent state differs beyond registered s embedding')

    def forward(self, z, clock, condition):
        if not isinstance(condition, AugmentedCondition):
            raise TypeError('explicit augmentation level required')
        condition.validate(z, self.stage, self.domain, self.training)
        if any(n % 2**self.levels for n in z.shape[2:]):
            raise ValueError('nondivisible U-Net shape')
        local = parent_crop(condition.joint, condition.region)
        summary = condition.wide.mean((2, 3, 4), keepdim=True)
        joint_coarse, wide_coarse = condition.coarse_joint, condition.coarse_wide
        local_coarse = parent_crop(joint_coarse, condition.region)
        phase = ((clock+13.3)/26.6*1000)[:, None]*self.frequency[None]
        embedded = self.time(torch.cat((phase.sin(), phase.cos()), 1))
        s = condition.strength[:, None]
        embedded = embedded+s*self.augmentation_time(s)
        local_inputs = torch.cat((local, summary.expand_as(local), local_coarse), 1)
        h = self.input(torch.cat((z, local_inputs), 1)); skips = []
        for block, down in zip(self.enc, self.down):
            h = block(h, embedded); skips.append(h); h = down(h)
        h = self.mid1(h, embedded); shape = h.shape
        tokens = h.flatten(2).transpose(1, 2); q = self.attn_norm(tokens)
        tokens = tokens+self.attn(q, q, q, need_weights=False)[0]
        joint = self.joint_encoder(torch.cat((condition.joint, joint_coarse), 1))
        wide = self.wide_encoder(torch.cat((condition.wide, wide_coarse), 1))
        width = h.shape[1]
        qpos = physical_positions(h.shape[2:], FINE_CELL, 2**self.levels,
                                 condition.query_offset())
        jpos = physical_positions(joint.shape[2:], FINE_CELL, 8,
                                 condition.joint_center_from_wide_mpc_h)
        wpos = physical_positions(wide.shape[2:], 27.064, 8,
                                 torch.zeros_like(condition.query_offset()))
        values = torch.cat((joint.flatten(2).transpose(1, 2),
                            wide.flatten(2).transpose(1, 2)), 1)
        key_positions = position_encoding(torch.cat((jpos, wpos), 1), width)
        queries = self.context_norm(tokens)+position_encoding(qpos, width)
        tokens = tokens+self.context_attention(queries, values+key_positions,
                                               values, need_weights=False)[0]
        h = self.mid2(tokens.transpose(1, 2).reshape(shape), embedded)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            h = block(torch.cat((up(h), skip), 1), embedded)
        return project(z+self.output(F.silu(h)))
