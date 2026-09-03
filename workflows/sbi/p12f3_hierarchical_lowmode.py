#!/usr/bin/env python3
"""Core operators for the P12-F3 hierarchical low-mode field posterior."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


class ConditionalVelocityUNet(nn.Module):
    """Velocity field for a one-channel state and spatial conditioner."""

    def __init__(self, *, condition_channels: int = 3, base: int = 8):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.net = UNet3D(
            in_channels=1 + self.condition_channels + 1,
            latent_channels=1,
            base=int(base),
        )

    def forward(
        self, state: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != 1:
            raise ValueError("state must have shape [batch,1,x,y,z]")
        if condition.shape[0] != state.shape[0] or condition.shape[2:] != state.shape[2:]:
            raise ValueError("condition/state geometry mismatch")
        if condition.shape[1] != self.condition_channels:
            raise ValueError("unexpected condition channel count")
        time = torch.as_tensor(time_value, device=state.device, dtype=state.dtype)
        if time.ndim == 0:
            time = time.repeat(state.shape[0])
        if time.shape != (state.shape[0],):
            raise ValueError("time must be scalar or one value per batch")
        time_channel = time.view(-1, 1, 1, 1, 1).expand(-1, 1, *state.shape[2:])
        return self.net(torch.cat((state, condition, time_channel), dim=1))


def rectified_flow_training_pair(
    target: torch.Tensor, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must have shape [batch,1,x,y,z]")
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    time = torch.rand(
        target.shape[0], device=target.device, dtype=target.dtype, generator=generator
    )
    blend = time.view(-1, 1, 1, 1, 1)
    state = (1.0 - blend) * noise + blend * target
    return state, time, target - noise, noise


@torch.inference_mode()
def sample_heun(
    model: ConditionalVelocityUNet,
    condition: torch.Tensor,
    *,
    draws: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if draws <= 0 or steps <= 0 or condition.shape[0] != 1:
        raise ValueError("Heun sampling requires positive draws/steps and one condition")
    repeated = condition.expand(draws, -1, -1, -1, -1)
    state = torch.randn(
        (draws, 1, *condition.shape[2:]),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    delta_time = 1.0 / steps
    for index in range(steps):
        current = torch.full(
            (draws,), index / steps, device=state.device, dtype=state.dtype
        )
        velocity = model(state, current, repeated)
        proposal = state + delta_time * velocity
        following = torch.full(
            (draws,), (index + 1) / steps, device=state.device, dtype=state.dtype
        )
        next_velocity = model(proposal, following, repeated)
        state = state + 0.5 * delta_time * (velocity + next_velocity)
    return state[:, 0]


@lru_cache(maxsize=64)
def _physical_low_mode_mask_numpy(
    shape: tuple[int, int, int], voxel_mpc_h: float, maximum_k_h_mpc: float
) -> np.ndarray:
    """Return the registered non-DC physical rFFT low-mode mask."""
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("low-mode split requires a three-dimensional field")
    if voxel_mpc_h <= 0 or maximum_k_h_mpc <= 0:
        raise ValueError("physical scale and cutoff must be positive")
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=voxel_mpc_h)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=voxel_mpc_h)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(shape[2], d=voxel_mpc_h)[None, None, :]
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    mask = (radius > 0.0) & (radius <= maximum_k_h_mpc)
    mask[0, 0, 0] = False
    return mask


def physical_low_mode_mask(
    shape: Iterable[int],
    *,
    voxel_mpc_h: float,
    maximum_k_h_mpc: float,
    device: torch.device | str,
) -> torch.Tensor:
    key = tuple(int(value) for value in shape)
    values = _physical_low_mode_mask_numpy(
        key, float(voxel_mpc_h), float(maximum_k_h_mpc)
    )
    return torch.from_numpy(values.copy()).to(device=device)


def spectral_split(
    field: torch.Tensor,
    *,
    voxel_mpc_h: float,
    maximum_k_h_mpc: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a real field into registered low modes and their exact complement."""
    if field.ndim != 5 or field.shape[1] != 1:
        raise ValueError("spectral split expects [batch,1,x,y,z]")
    mask = physical_low_mode_mask(
        field.shape[-3:],
        voxel_mpc_h=voxel_mpc_h,
        maximum_k_h_mpc=maximum_k_h_mpc,
        device=field.device,
    )
    coefficients = torch.fft.rfftn(field, dim=(-3, -2, -1), norm="ortho")
    low_coefficients = coefficients * mask.view(1, 1, *mask.shape)
    low = torch.fft.irfftn(
        low_coefficients,
        s=field.shape[-3:],
        dim=(-3, -2, -1),
        norm="ortho",
    )
    high = field - low
    return low, high


def crop_tensor_to_patch(
    values: torch.Tensor,
    *,
    source_start: np.ndarray,
    target_start: np.ndarray,
    target_stop: np.ndarray,
) -> torch.Tensor:
    """Crop a common wide-context tensor to an exactly nested patch."""
    source = np.asarray(source_start, dtype=np.int64)
    start = np.asarray(target_start, dtype=np.int64) - source
    stop = np.asarray(target_stop, dtype=np.int64) - source
    if np.any(start < 0) or np.any(stop <= start):
        raise ValueError("target patch is not nested in the common wide context")
    if np.any(stop > np.asarray(values.shape[-3:], dtype=np.int64)):
        raise ValueError("target patch extends beyond the source tensor")
    selection = tuple(slice(int(left), int(right)) for left, right in zip(start, stop))
    return values[(slice(None), slice(None)) + selection]


def pool_low_mode_state(values: torch.Tensor, factor: int) -> torch.Tensor:
    if factor < 1:
        raise ValueError("coarse factor must be positive")
    if factor == 1:
        return values
    return F.avg_pool3d(
        values,
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
        count_include_pad=False,
    )


def coarse_core_mask(
    shape: tuple[int, int, int],
    core_slice: tuple[slice, slice, slice],
    *,
    factor: int,
    device: torch.device | str,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, *shape), dtype=torch.float32, device=device)
    mask[(slice(None), slice(None)) + core_slice] = 1.0
    if factor > 1:
        mask = F.max_pool3d(mask, kernel_size=factor, stride=factor, ceil_mode=True)
    return mask > 0.0


def prepare_low_mode_example(
    *,
    condition: torch.Tensor,
    low_residual: torch.Tensor,
    core_slice: tuple[slice, slice, slice],
    coarse_factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition.shape[0] != 1 or condition.shape[2:] != low_residual.shape[2:]:
        raise ValueError("condition and low residual are not patch-aligned")
    pooled_condition = pool_low_mode_state(condition, coarse_factor)
    pooled_target = pool_low_mode_state(low_residual, coarse_factor)
    science = coarse_core_mask(
        tuple(int(value) for value in low_residual.shape[-3:]),
        core_slice,
        factor=coarse_factor,
        device=low_residual.device,
    )
    if science.shape != pooled_target.shape or not torch.any(science):
        raise RuntimeError("coarse authoritative-core loss mask is invalid")
    return pooled_condition, pooled_target, science


@torch.inference_mode()
def upsample_low_mode_draws(
    coarse_draws: torch.Tensor, output_shape: tuple[int, int, int]
) -> torch.Tensor:
    if coarse_draws.ndim != 4:
        raise ValueError("coarse draws must have shape [draw,x,y,z]")
    return F.interpolate(
        coarse_draws[:, None],
        size=output_shape,
        mode="trilinear",
        align_corners=False,
    )[:, 0]


def build_low_mode_model(*, condition_channels: int, base: int) -> ConditionalVelocityUNet:
    return ConditionalVelocityUNet(condition_channels=condition_channels, base=base)


__all__ = [
    "build_low_mode_model",
    "coarse_core_mask",
    "crop_tensor_to_patch",
    "physical_low_mode_mask",
    "pool_low_mode_state",
    "prepare_low_mode_example",
    "rectified_flow_training_pair",
    "sample_heun",
    "spectral_split",
    "upsample_low_mode_draws",
]
