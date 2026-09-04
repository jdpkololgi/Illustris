"""Models and exact transforms for the P12-F3 conditional-calibration rescue."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p8_train_unet_patch import UNet3D, model_inputs
from workflows.sbi.p12f3_fourier_modes import (
    FourierModeLayout,
    lowpass_exact,
    pack_fourier_components,
    unpack_fourier_components,
)
from workflows.sbi.p12f_field_posterior_diagnostics import fixed_tidal_tensor
from workflows.sbi.p12f_gaussian_controls import correlated_unit_residuals


BASE_CHANNELS = ("counts", "exposure_apodized", "log_count_ratio")
EXTRA_CHANNELS = ("distance_to_support_boundary",)
ALL_PATCH_CHANNELS = BASE_CHANNELS + EXTRA_CHANNELS
ARMS = ("base3", "proxy7", "proxy7_shuffled")


def shear_amplitude(field: torch.Tensor) -> torch.Tensor:
    if field.ndim != 5 or field.shape[1] != 1:
        raise ValueError("shear proxy expects [batch,1,x,y,z]")
    tensor = fixed_tidal_tensor(field[:, 0])
    trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    shear = tensor - trace[..., None, None] * torch.eye(
        3, device=field.device, dtype=field.dtype
    )
    return torch.sqrt(torch.clamp(torch.sum(shear.square(), dim=(-2, -1)), min=0))[..., None].movedim(-1, 1)


@torch.no_grad()
def proxy_condition(
    patch,
    normalization: dict,
    g1_model: nn.Module,
    *,
    device: str,
    arm: str,
    shuffle_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return capacity-matched seven-channel input and frozen G1 state."""
    if arm not in ARMS:
        raise ValueError(f"unknown conditional proxy arm {arm}")
    at = {name: index for index, name in enumerate(patch.channel_names)}
    base_patch = replace(
        patch,
        values=np.stack([patch.values[at[name]] for name in BASE_CHANNELS]),
        channel_names=BASE_CHANNELS,
    )
    base, _ = model_inputs(base_patch, normalization, device)
    mean, log_std = g1_model(base)
    shear = shear_amplitude(mean)
    boundary = torch.from_numpy(
        np.clip(
            np.asarray(patch.values[at["distance_to_support_boundary"]], dtype=np.float32),
            0.0,
            120.0,
        )[None, None]
        / np.float32(120.0)
    ).to(device)
    physical = torch.cat((mean, log_std, shear), dim=1)
    if arm == "base3":
        physical = torch.zeros_like(physical)
        boundary = torch.zeros_like(boundary)
    elif arm == "proxy7_shuffled":
        if shuffle_seed is None:
            raise ValueError("shuffled proxy arm requires a fixed seed")
        rng = np.random.default_rng(int(shuffle_seed))
        shifts = tuple(int(rng.integers(1, max(size, 2))) for size in physical.shape[-3:])
        physical = torch.roll(physical, shifts=shifts, dims=(-3, -2, -1))
    condition = torch.cat((base, physical, boundary), dim=1)
    if condition.shape[1] != 7:
        raise RuntimeError("conditional proxy interface is not seven channels")
    return condition, mean, log_std


class ConditionalLowModeGaussianUNet(nn.Module):
    """Predict local low-mode residual location and scale from seven channels."""

    def __init__(self, *, condition_channels: int = 7, base: int = 4):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.net = UNet3D(in_channels=self.condition_channels, latent_channels=2, base=int(base))

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError("conditional Gaussian input geometry changed")
        values = self.net(condition)
        return values[:, :1], torch.clamp(values[:, 1:2], min=-5.0, max=3.0)


def low_mode_target(residual: torch.Tensor, layout: FourierModeLayout) -> torch.Tensor:
    return lowpass_exact(residual, layout)


def science_mask(support: np.ndarray, core_slice: tuple[slice, slice, slice], device: str) -> torch.Tensor:
    mask = torch.zeros((1, 1, *support.shape), device=device, dtype=torch.bool)
    support_tensor = torch.as_tensor(support, device=device, dtype=torch.bool)[None, None]
    core = (slice(None), slice(None)) + core_slice
    mask[core] = support_tensor[core]
    if not torch.any(mask):
        raise RuntimeError("conditional calibration core has no exact support")
    return mask


def conditional_gaussian_nll(
    location: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if location.shape != log_scale.shape or target.shape != location.shape or mask.shape != target.shape:
        raise ValueError("conditional Gaussian tensors are not aligned")
    inverse_variance = torch.exp(-2.0 * log_scale)
    value = log_scale + 0.5 * torch.square(target - location) * inverse_variance
    return value[mask].mean()


def standardized_low_field(
    target_low: torch.Tensor,
    location: torch.Tensor,
    log_scale: torch.Tensor,
    layout: FourierModeLayout,
) -> torch.Tensor:
    standardized = (target_low - location) * torch.exp(-log_scale)
    return lowpass_exact(standardized, layout)


def unit_whitening() -> dict[str, list[float]]:
    return {"mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0]}


def reconstruct_conditional_low(
    standardized_low: torch.Tensor,
    location: torch.Tensor,
    log_scale: torch.Tensor,
    layout: FourierModeLayout,
) -> torch.Tensor:
    if standardized_low.ndim == 4:
        standardized_low = standardized_low[:, None]
    location = location.expand(standardized_low.shape[0], -1, -1, -1, -1)
    log_scale = log_scale.expand_as(location)
    return lowpass_exact(location + torch.exp(log_scale) * standardized_low, layout)[:, 0]


def sample_conditional_gaussian_low(
    location: torch.Tensor,
    log_scale: torch.Tensor,
    layout: FourierModeLayout,
    filter_contract: Mapping,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    unit = correlated_unit_residuals(
        dict(filter_contract), draws=draws, seed=seed, shape=layout.shape
    )
    device = location.device
    standard = torch.from_numpy(unit).to(device=device, dtype=location.dtype)
    with torch.inference_mode():
        result = reconstruct_conditional_low(standard, location, log_scale, layout)
    return result.cpu().numpy().astype(np.float32)


def cosine_alpha_sigma(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.any(time < 0) or torch.any(time > 1):
        raise ValueError("diffusion time must lie in [0,1]")
    angle = time * (np.pi / 2.0)
    return torch.cos(angle), torch.sin(angle)


def fourier_v_pair(
    target: torch.Tensor, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim != 2:
        raise ValueError("Fourier diffusion target must be [batch,components]")
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype, generator=generator)
    alpha, sigma = cosine_alpha_sigma(time)
    state = alpha[:, None] * target + sigma[:, None] * noise
    velocity = alpha[:, None] * noise - sigma[:, None] * target
    return state, time, velocity


@torch.inference_mode()
def sample_fourier_ddim(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    layout: FourierModeLayout,
    whitening: Mapping[str, list[float]],
    draws: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if condition.shape[0] != 1 or draws <= 0 or steps <= 0:
        raise ValueError("Fourier DDIM sampling contract is invalid")
    condition = condition.expand(draws, -1, -1, -1, -1)
    state = torch.randn((draws, layout.components), device=condition.device, dtype=condition.dtype, generator=generator)
    for index in range(steps, 0, -1):
        time = torch.full((draws,), index / steps, device=state.device, dtype=state.dtype)
        alpha, sigma = cosine_alpha_sigma(time)
        velocity = model(state, time, condition, layout=layout, whitening=whitening)
        estimate = alpha[:, None] * state - sigma[:, None] * velocity
        noise = sigma[:, None] * state + alpha[:, None] * velocity
        next_time = torch.full((draws,), (index - 1) / steps, device=state.device, dtype=state.dtype)
        next_alpha, next_sigma = cosine_alpha_sigma(next_time)
        state = next_alpha[:, None] * estimate + next_sigma[:, None] * noise
    from workflows.sbi.p12f3_fourier_modes import unwhiten_components

    return unpack_fourier_components(unwhiten_components(state, whitening, layout), layout)[:, 0]


__all__ = [
    "ALL_PATCH_CHANNELS", "ARMS", "ConditionalLowModeGaussianUNet",
    "conditional_gaussian_nll", "fourier_v_pair", "low_mode_target",
    "proxy_condition", "reconstruct_conditional_low", "sample_conditional_gaussian_low",
    "sample_fourier_ddim", "science_mask", "standardized_low_field", "unit_whitening",
]
