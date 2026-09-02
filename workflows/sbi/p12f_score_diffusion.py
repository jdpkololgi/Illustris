#!/usr/bin/env python3
"""Matched conditional VP score-diffusion components for P12-F."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


def cosine_alpha_sigma(
    time_value: torch.Tensor, *, offset: float = 0.008
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.as_tensor(time_value)
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("diffusion time must lie in [0,1]")
    angle = (t + offset) / (1.0 + offset) * (math.pi / 2.0)
    alpha = torch.cos(angle)
    sigma = torch.sin(angle)
    norm = torch.sqrt(alpha.square() + sigma.square())
    return alpha / norm, sigma / norm


def v_parameterization(
    target: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape != noise.shape or target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("v diffusion expects aligned [batch,1,x,y,z] tensors")
    alpha, sigma = cosine_alpha_sigma(time_value)
    view = (-1, 1, 1, 1, 1)
    alpha = alpha.view(view)
    sigma = sigma.view(view)
    noisy = alpha * target + sigma * noise
    velocity = alpha * noise - sigma * target
    return noisy, velocity


def recover_x0_epsilon(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    time_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noisy.shape != velocity.shape:
        raise ValueError("noisy state and v prediction must align")
    alpha, sigma = cosine_alpha_sigma(time_value)
    alpha = alpha.view(-1, 1, 1, 1, 1)
    sigma = sigma.view(-1, 1, 1, 1, 1)
    x0 = alpha * noisy - sigma * velocity
    epsilon = sigma * noisy + alpha * velocity
    return x0, epsilon


def diffusion_training_pair(
    target: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    minimum_time: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= minimum_time < 1.0:
        raise ValueError("minimum diffusion time is invalid")
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    time_value = minimum_time + (1.0 - minimum_time) * torch.rand(
        target.shape[0],
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    noisy, velocity = v_parameterization(target, noise, time_value)
    return noisy, time_value, velocity, noise


class ConditionalVDiffusionUNet(nn.Module):
    """Predict v from noised delta_R7, time and the same three R1 channels."""

    def __init__(self, *, condition_channels: int = 3, base: int = 8):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.net = UNet3D(
            in_channels=1 + self.condition_channels + 1,
            latent_channels=1,
            base=int(base),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        time_value: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 5 or noisy.shape[1] != 1:
            raise ValueError("noisy field must have shape [batch,1,x,y,z]")
        if condition.shape[0] != noisy.shape[0] or condition.shape[2:] != noisy.shape[2:]:
            raise ValueError("diffusion condition geometry mismatch")
        if condition.shape[1] != self.condition_channels:
            raise ValueError("unexpected diffusion condition channel count")
        t = torch.as_tensor(time_value, device=noisy.device, dtype=noisy.dtype)
        if t.ndim == 0:
            t = t.repeat(noisy.shape[0])
        if t.shape != (noisy.shape[0],):
            raise ValueError("diffusion time must be scalar or one value per batch")
        time_channel = t.view(-1, 1, 1, 1, 1).expand(-1, 1, *noisy.shape[2:])
        return self.net(torch.cat((noisy, condition, time_channel), dim=1))


@torch.inference_mode()
def sample_ddim(
    model: ConditionalVDiffusionUNet,
    condition: torch.Tensor,
    *,
    draws: int,
    steps: int,
    generator: torch.Generator,
    minimum_time: float = 1e-4,
) -> torch.Tensor:
    """Deterministic eta=0 DDIM after the initial seeded Gaussian draw."""
    if draws <= 0 or steps <= 0:
        raise ValueError("DDIM draws and steps must be positive")
    if condition.shape[0] != 1:
        raise ValueError("DDIM sampler expects one conditioning patch")
    condition_batch = condition.expand(draws, -1, -1, -1, -1)
    state = torch.randn(
        (draws, 1, *condition.shape[2:]),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    times = torch.linspace(
        1.0,
        minimum_time,
        steps + 1,
        device=condition.device,
        dtype=condition.dtype,
    )
    for index in range(steps):
        current = times[index].repeat(draws)
        following = times[index + 1].repeat(draws)
        predicted_v = model(state, current, condition_batch)
        x0, epsilon = recover_x0_epsilon(state, predicted_v, current)
        alpha_next, sigma_next = cosine_alpha_sigma(following)
        state = (
            alpha_next.view(-1, 1, 1, 1, 1) * x0
            + sigma_next.view(-1, 1, 1, 1, 1) * epsilon
        )
    final_time = torch.full(
        (draws,), minimum_time, device=state.device, dtype=state.dtype
    )
    final_v = model(state, final_time, condition_batch)
    x0, _ = recover_x0_epsilon(state, final_v, final_time)
    return x0[:, 0]


def sampler_comparison_contract(
    *,
    diffusion_primary_steps: int = 50,
    diffusion_matched_steps: int = 24,
    flow_heun_steps: int = 12,
) -> dict:
    """Record network-evaluation counts without claiming universal method ranking."""
    return {
        "diffusion_primary": {
            "sampler": "deterministic DDIM",
            "steps": int(diffusion_primary_steps),
            "network_evaluations": int(diffusion_primary_steps + 1),
        },
        "diffusion_matched": {
            "sampler": "deterministic DDIM",
            "steps": int(diffusion_matched_steps),
            "network_evaluations": int(diffusion_matched_steps + 1),
        },
        "rectified_flow_reference": {
            "sampler": "Heun",
            "steps": int(flow_heun_steps),
            "network_evaluations": int(2 * flow_heun_steps),
        },
        "interpretation": (
            "Matched implementation comparison on one target/condition/budget; "
            "not a universal flow-versus-diffusion conclusion."
        ),
    }
