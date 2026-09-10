"""Matched research-only CFM/DIFF backbone and full-parent sampling.

Inputs/targets are already transformed using frozen training-only normalization.
This module neither reads truth at inference nor performs physical reconstruction.
Coarse96 spans 1299.072 Mpc/h; fine96 spans 324.768 Mpc/h. The caller must
physically crop/interpolate the SAME coarse draw before adding it to fine
conditioning. Resizing the entire wide cube onto the local cube is incorrect.
Full-parent evolution gives exact sibling overlap, not full-cap coherence
between independently sampled anchor parents. No WCFM or MIRA is activated.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _Block(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(1, outgoing), nn.SiLU(),
            nn.Conv3d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(1, outgoing), nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


def _downsample_mean(x):
    """Nonoverlapping 2^3 reduction; floor output shape like average pooling.

    Unlike CUDA AvgPool3d backward, reshape/mean supports strict deterministic
    algorithms. Odd trailing cells are deliberately omitted.
    """
    b, c, d, h, w = x.shape
    d, h, w = d // 2, h // 2, w // 2
    x = x[:, :, :2*d, :2*h, :2*w]
    return x.reshape(b, c, d, 2, h, 2, w, 2).mean(dim=(3, 5, 7))


class _WideSpatialMean(nn.Module):
    """Deterministic adaptive 2^3 summary, including odd spatial sizes."""

    def forward(self, x):
        b, c, d, h, w = x.shape
        if all(n % 2 == 0 for n in (d, h, w)):
            return x.reshape(b, c, 2, d//2, 2, h//2, 2, w//2).mean(dim=(3, 5, 7))
        # Adaptive bins match floor(start), ceil(end); boundary bins can overlap.
        blocks = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    bounds = tuple(slice(a*n//2, ((a+1)*n+1)//2)
                                   for a, n in zip((i, j, k), (d, h, w)))
                    blocks.append(x[(slice(None), slice(None), *bounds)].mean(dim=(2, 3, 4)))
        return torch.stack(blocks, dim=-1).reshape(b, c, 2, 2, 2)


class ConditionalFieldNet(nn.Module):
    """One modest time-conditioned 3D U-Net architecture for both objectives.

    Nonperiodic zero-padding is intentional: these are cutout domains. Spatial
    shape is preserved, including odd dimensions. There is no stochastic layer.
    """

    def __init__(self, condition_channels: int, base_channels: int = 8, levels: int = 2,
                 wide_condition_channels: int = 0):
        super().__init__()
        if condition_channels < 1 or base_channels < 1 or levels < 1:
            raise ValueError("condition_channels, base_channels and levels must be positive")
        self.condition_channels = condition_channels
        self.levels = levels
        if wide_condition_channels < 0:
            raise ValueError("wide_condition_channels must be nonnegative")
        self.wide_condition_channels = wide_condition_channels
        widths = [base_channels * 2**i for i in range(levels)]
        # Separate physical domain; this 2^3 summary is a pilot compression.
        self.wide_encoder = (nn.Sequential(
            nn.Conv3d(wide_condition_channels, base_channels, 3, stride=2, padding=1),
            nn.SiLU(), _WideSpatialMean(), nn.Flatten(),
            nn.Linear(base_channels * 8, widths[-1]),
        ) if wide_condition_channels else None)
        self.encoders = nn.ModuleList([
            _Block(1 + condition_channels + 3 if i == 0 else widths[i - 1], w)
            for i, w in enumerate(widths)
        ])
        self.decoders = nn.ModuleList([
            _Block(widths[i + 1] + widths[i], widths[i])
            for i in reversed(range(levels - 1))
        ])
        self.output = nn.Conv3d(widths[0], 1, 1)

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor,
                wide_condition: torch.Tensor | None = None):
        _validate_pair(state, condition)
        if condition.shape[1] != self.condition_channels:
            raise ValueError("incorrect condition channel count")
        if time.shape != (state.shape[0],):
            raise ValueError("time must have shape [batch]")
        if min(state.shape[2:]) < 2 ** (self.levels - 1):
            raise ValueError("spatial shape too small for model levels")
        if self.wide_encoder is not None:
            if (wide_condition is None or wide_condition.ndim != 5
                    or wide_condition.shape[:2] != (state.shape[0], self.wide_condition_channels)
                    or wide_condition.device != state.device or wide_condition.dtype != state.dtype):
                raise ValueError("wide condition must match batch/channels/device/dtype")
        elif wide_condition is not None:
            raise ValueError("model has no wide-condition branch")
        t = time.to(state).view(-1, 1, 1, 1, 1)
        features = torch.cat((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=1)
        x = torch.cat((state, condition, features.expand(-1, -1, *state.shape[2:])), dim=1)
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = _downsample_mean(x)
            x = encoder(x)
            skips.append(x)
        if self.wide_encoder is not None:
            x = x + self.wide_encoder(wide_condition)[:, :, None, None, None]
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)


def _validate_pair(field, condition):
    if field.ndim != 5 or field.shape[1] != 1 or condition.ndim != 5:
        raise ValueError("expected field [B,1,D,H,W] and condition [B,C,D,H,W]")
    if field.shape[0] != condition.shape[0] or field.shape[2:] != condition.shape[2:]:
        raise ValueError("field and condition batch/spatial shapes differ")
    if field.device != condition.device or field.dtype != condition.dtype:
        raise ValueError("field and condition device/dtype must match")
    if not field.is_floating_point() or not condition.is_floating_point():
        raise ValueError("field and condition must be floating point")


def _noise_like(value, generator):
    if not isinstance(generator, torch.Generator):
        raise ValueError("an explicit torch.Generator is required")
    return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)


def _time(target, generator):
    return torch.rand(target.shape[0], device=target.device, dtype=target.dtype, generator=generator)


def _predict(model, state, time, condition, wide_condition):
    if wide_condition is None:
        return model(state, time, condition)
    return model(state, time, condition, wide_condition=wide_condition)


def flow_matching_loss(model, target, condition, generator: torch.Generator, *, wide_condition=None):
    """Straight Gaussian-to-data CFM: x(t)=(1-t)noise+t*target."""
    _validate_pair(target, condition)
    noise = _noise_like(target, generator)
    time = _time(target, generator)
    t = time.view(-1, 1, 1, 1, 1)
    return F.mse_loss(_predict(model, (1 - t) * noise + t * target, time, condition, wide_condition), target - noise)


def diffusion_loss(model, target, condition, generator: torch.Generator, *, wide_condition=None):
    """Cosine VP v-prediction: t=0 is data, t=1 is pure Gaussian noise."""
    _validate_pair(target, condition)
    noise = _noise_like(target, generator)
    time = _time(target, generator)
    angle = time.view(-1, 1, 1, 1, 1) * (math.pi / 2)
    alpha, sigma = torch.cos(angle), torch.sin(angle)
    state = alpha * target + sigma * noise
    velocity = alpha * noise - sigma * target
    return F.mse_loss(_predict(model, state, time, condition, wide_condition), velocity)


@torch.no_grad()
def sample_field(model, condition, method: str, steps: int, generator: torch.Generator,
                 solver: str = "heun", *, wide_condition=None):
    """Evolve ONE entire parent from explicitly seeded Gaussian noise.

    CFM integrates 0->1 with Euler or Heun (N or 2N network evaluations).
    DIFF uses deterministic DDIM, cosine v-prediction, 1->0 (N evaluations).
    Generator state is consumed, so checkpoint it for exact continuation.
    """
    if method not in ("cfm", "diffusion") or solver not in ("euler", "heun"):
        raise ValueError("method must be cfm/diffusion; solver must be euler/heun")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if condition.ndim != 5 or condition.shape[1] < 1 or min(condition.shape) < 1:
        raise ValueError("condition must be nonempty [B,C,D,H,W]")
    state = _noise_like(condition[:, :1], generator)
    _validate_pair(state, condition)
    was_training = model.training
    model.eval()
    try:
        for i in range(steps):
            if method == "cfm":
                t, next_t = i / steps, (i + 1) / steps
                dt = next_t - t
                time = state.new_full((state.shape[0],), t)
                velocity = _predict(model, state, time, condition, wide_condition)
                proposed = state + dt * velocity
                if solver == "heun":
                    end = state.new_full((state.shape[0],), next_t)
                    velocity_end = _predict(model, proposed, end, condition, wide_condition)
                    proposed = state + (dt / 2) * (velocity + velocity_end)
                state = proposed
            else:
                t, next_t = 1 - i / steps, 1 - (i + 1) / steps
                velocity = _predict(model, state, state.new_full((state.shape[0],), t), condition, wide_condition)
                alpha, sigma = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
                clean = alpha * state - sigma * velocity
                noise = sigma * state + alpha * velocity
                state = math.cos(next_t * math.pi / 2) * clean + math.sin(next_t * math.pi / 2) * noise
        return state
    finally:
        model.train(was_training)


def crop_children(parent: torch.Tensor, starts: Sequence[Sequence[int]], size: int):
    """Deterministic views of a shared draw; never sample children independently."""
    if parent.ndim != 5 or not isinstance(size, int) or size < 1:
        raise ValueError("expected a 5D parent and positive integer crop size")
    children = []
    for start in starts:
        if len(start) != 3 or any(not isinstance(s, int) or s < 0 or s + size > n
                                  for s, n in zip(start, parent.shape[2:])):
            raise ValueError("child extends outside shared parent")
        children.append(parent[(slice(None), slice(None), *(slice(s, s + size) for s in start))])
    return children
