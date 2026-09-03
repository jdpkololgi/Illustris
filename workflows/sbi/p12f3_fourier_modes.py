#!/usr/bin/env python3
"""Exact Hermitian Fourier coordinates for the P12-F3-L2 low-mode posterior."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


@dataclass(frozen=True)
class FourierModeLayout:
    """Independent real degrees of freedom for a real periodic 3-D field."""

    shape: tuple[int, int, int]
    representative_flat: np.ndarray
    conjugate_flat: np.ndarray
    self_conjugate: np.ndarray
    mode_band: np.ndarray
    component_group: np.ndarray
    component_mode: np.ndarray
    component_is_imaginary: np.ndarray
    k_vector_h_mpc: np.ndarray

    @property
    def modes(self) -> int:
        return int(len(self.representative_flat))

    @property
    def components(self) -> int:
        return int(len(self.component_group))


def _signed_frequency(index: int, size: int) -> int:
    return index if index <= size // 2 else index - size


@lru_cache(maxsize=128)
def _cached_layout(
    shape: tuple[int, int, int],
    voxel_mpc_h: float,
    band_edges_h_mpc: tuple[float, ...],
) -> FourierModeLayout:
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("Fourier layout requires a valid three-dimensional shape")
    if len(band_edges_h_mpc) < 2 or band_edges_h_mpc[0] != 0.0:
        raise ValueError("Fourier bands must begin at zero")
    if any(right <= left for left, right in zip(band_edges_h_mpc, band_edges_h_mpc[1:])):
        raise ValueError("Fourier band edges must be strictly increasing")
    if voxel_mpc_h <= 0:
        raise ValueError("voxel scale must be positive")

    representatives: list[int] = []
    conjugates: list[int] = []
    self_flags: list[bool] = []
    bands: list[int] = []
    k_vectors: list[tuple[float, float, float]] = []
    maximum = band_edges_h_mpc[-1]
    for ix in range(shape[0]):
        sx = _signed_frequency(ix, shape[0])
        kx = 2.0 * np.pi * sx / (shape[0] * voxel_mpc_h)
        for iy in range(shape[1]):
            sy = _signed_frequency(iy, shape[1])
            ky = 2.0 * np.pi * sy / (shape[1] * voxel_mpc_h)
            for iz in range(shape[2]):
                sz = _signed_frequency(iz, shape[2])
                kz = 2.0 * np.pi * sz / (shape[2] * voxel_mpc_h)
                radius = float(np.sqrt(kx * kx + ky * ky + kz * kz))
                if radius <= 0.0 or radius > maximum:
                    continue
                index = (ix, iy, iz)
                conjugate = ((-ix) % shape[0], (-iy) % shape[1], (-iz) % shape[2])
                if index > conjugate:
                    continue
                band = int(np.searchsorted(band_edges_h_mpc, radius, side="left") - 1)
                band = max(0, min(band, len(band_edges_h_mpc) - 2))
                representatives.append(int(np.ravel_multi_index(index, shape)))
                conjugates.append(int(np.ravel_multi_index(conjugate, shape)))
                self_flags.append(index == conjugate)
                bands.append(band)
                k_vectors.append((kx, ky, kz))
    if not representatives or set(bands) != set(range(len(band_edges_h_mpc) - 1)):
        raise RuntimeError("one or more registered Fourier bands contain no modes")

    self_array = np.asarray(self_flags, dtype=bool)
    mode_band = np.asarray(bands, dtype=np.int16)
    nonself = np.flatnonzero(~self_array).astype(np.int64)
    component_mode = np.concatenate(
        (np.arange(len(mode_band), dtype=np.int64), nonself)
    )
    imaginary = np.concatenate(
        (np.zeros(len(mode_band), dtype=bool), np.ones(len(nonself), dtype=bool))
    )
    component_band = mode_band[component_mode].astype(np.int16)
    # Groups are band-major, then real/imaginary.  This is the frozen whitening unit.
    groups = (2 * component_band + imaginary.astype(np.int16)).astype(np.int16)
    return FourierModeLayout(
        shape=shape,
        representative_flat=np.asarray(representatives, dtype=np.int64),
        conjugate_flat=np.asarray(conjugates, dtype=np.int64),
        self_conjugate=self_array,
        mode_band=mode_band,
        component_group=groups,
        component_mode=component_mode,
        component_is_imaginary=imaginary,
        k_vector_h_mpc=np.asarray(k_vectors, dtype=np.float64),
    )


def build_fourier_layout(
    shape: Iterable[int], *, voxel_mpc_h: float, band_edges_h_mpc: Iterable[float]
) -> FourierModeLayout:
    return _cached_layout(
        tuple(int(value) for value in shape),
        float(voxel_mpc_h),
        tuple(float(value) for value in band_edges_h_mpc),
    )


def _indices(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long, device=device)


def pack_fourier_components(field: torch.Tensor, layout: FourierModeLayout) -> torch.Tensor:
    """Pack the independent low-mode coefficients of a real field."""
    if field.ndim != 5 or field.shape[1] != 1 or tuple(field.shape[-3:]) != layout.shape:
        raise ValueError("field and Fourier layout geometry differ")
    coefficients = torch.fft.fftn(field[:, 0], dim=(-3, -2, -1), norm="ortho")
    flat = coefficients.reshape(field.shape[0], -1)
    representatives = flat.index_select(1, _indices(layout.representative_flat, field.device))
    nonself = _indices(np.flatnonzero(~layout.self_conjugate), field.device)
    return torch.cat((representatives.real, representatives.index_select(1, nonself).imag), dim=1)


def unpack_fourier_components(
    vector: torch.Tensor, layout: FourierModeLayout
) -> torch.Tensor:
    """Unpack independent coefficients into an exactly real low-mode field."""
    if vector.ndim != 2 or vector.shape[1] != layout.components:
        raise ValueError("vector and Fourier layout component counts differ")
    modes = layout.modes
    real = vector[:, :modes]
    nonself_numpy = np.flatnonzero(~layout.self_conjugate).astype(np.int64)
    nonself = _indices(nonself_numpy, vector.device)
    imaginary = torch.zeros_like(real)
    imaginary = imaginary.index_copy(1, nonself, vector[:, modes:])
    values = torch.complex(real, imaginary)
    flat = torch.zeros(
        (vector.shape[0], int(np.prod(layout.shape))),
        dtype=values.dtype,
        device=vector.device,
    )
    representatives = _indices(layout.representative_flat, vector.device)
    flat = flat.index_copy(1, representatives, values)
    if len(nonself_numpy):
        conjugates = _indices(layout.conjugate_flat[nonself_numpy], vector.device)
        flat = flat.index_copy(1, conjugates, torch.conj(values.index_select(1, nonself)))
    field = torch.fft.ifftn(
        flat.reshape(vector.shape[0], *layout.shape),
        dim=(-3, -2, -1),
        norm="ortho",
    ).real
    return field[:, None]


def lowpass_exact(field: torch.Tensor, layout: FourierModeLayout) -> torch.Tensor:
    return unpack_fourier_components(pack_fourier_components(field, layout), layout)


def spectral_lowpass_reference(field: torch.Tensor, layout: FourierModeLayout) -> torch.Tensor:
    """Independent full-FFT mask reference used by the representation gate."""
    if field.ndim != 5 or field.shape[1] != 1 or tuple(field.shape[-3:]) != layout.shape:
        raise ValueError("field and Fourier layout geometry differ")
    coefficients = torch.fft.fftn(field[:, 0], dim=(-3, -2, -1), norm="ortho")
    flat = coefficients.reshape(field.shape[0], -1)
    keep = np.unique(
        np.concatenate((layout.representative_flat, layout.conjugate_flat))
    )
    mask = torch.zeros(flat.shape[1], dtype=torch.bool, device=field.device)
    mask[_indices(keep, field.device)] = True
    filtered = torch.where(mask[None], flat, torch.zeros_like(flat))
    return torch.fft.ifftn(
        filtered.reshape(field.shape[0], *layout.shape),
        dim=(-3, -2, -1),
        norm="ortho",
    ).real[:, None]


def hermitian_max_error(vector: torch.Tensor, layout: FourierModeLayout) -> float:
    field = unpack_fourier_components(vector, layout)
    repacked = pack_fourier_components(field, layout)
    return float(torch.max(torch.abs(repacked - vector)).detach().cpu())


def empty_whitening_accumulator(groups: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(groups, dtype=np.int64),
        "sum": np.zeros(groups, dtype=np.float64),
        "sum_square": np.zeros(groups, dtype=np.float64),
    }


def update_whitening_accumulator(
    accumulator: dict[str, np.ndarray], vector: torch.Tensor, layout: FourierModeLayout
) -> None:
    values = vector.detach().cpu().numpy().reshape(-1)
    if vector.shape[0] != 1:
        raise ValueError("whitening fit is registered per patch with batch one")
    groups = layout.component_group
    for group in range(len(accumulator["count"])):
        selected = values[groups == group]
        accumulator["count"][group] += len(selected)
        accumulator["sum"][group] += selected.sum(dtype=np.float64)
        accumulator["sum_square"][group] += np.square(selected, dtype=np.float64).sum(dtype=np.float64)


def finalize_whitening(accumulator: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    count = np.asarray(accumulator["count"], dtype=np.int64)
    total = np.asarray(accumulator["sum"], dtype=np.float64)
    square = np.asarray(accumulator["sum_square"], dtype=np.float64)
    if np.any(count < 2):
        raise RuntimeError("Fourier whitening group has insufficient samples")
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-12)
    return {
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
    }


def _component_parameters(
    whitening: Mapping[str, list[float]], layout: FourierModeLayout, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    group = _indices(layout.component_group, device)
    mean = torch.as_tensor(whitening["mean"], device=device, dtype=dtype).index_select(0, group)
    std = torch.as_tensor(whitening["std"], device=device, dtype=dtype).index_select(0, group)
    if torch.any(std <= 0) or not torch.isfinite(std).all():
        raise RuntimeError("invalid Fourier whitening scale")
    return mean, std


def whiten_components(
    vector: torch.Tensor, whitening: Mapping[str, list[float]], layout: FourierModeLayout
) -> torch.Tensor:
    mean, std = _component_parameters(whitening, layout, vector.device, vector.dtype)
    return (vector - mean) / std


def unwhiten_components(
    vector: torch.Tensor, whitening: Mapping[str, list[float]], layout: FourierModeLayout
) -> torch.Tensor:
    mean, std = _component_parameters(whitening, layout, vector.device, vector.dtype)
    return vector * std + mean


def whiten_velocity(
    vector: torch.Tensor, whitening: Mapping[str, list[float]], layout: FourierModeLayout
) -> torch.Tensor:
    _, std = _component_parameters(whitening, layout, vector.device, vector.dtype)
    return vector / std


def equal_band_flow_loss(
    predicted: torch.Tensor, target: torch.Tensor, layout: FourierModeLayout, bands: int
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("flow prediction and target shapes differ")
    groups = layout.component_group
    component_band = groups // 2
    losses = []
    for band in range(bands):
        mask = torch.as_tensor(component_band == band, device=predicted.device)
        if not torch.any(mask):
            raise RuntimeError("registered band has no flow components")
        losses.append(torch.mean(torch.square(predicted[:, mask] - target[:, mask])))
    return torch.stack(losses).mean()


class ConditionalFourierVelocityUNet(nn.Module):
    """Spatial directional conditioner with a flow state in exact Fourier coordinates."""

    def __init__(self, *, condition_channels: int = 3, base: int = 4):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.net = UNet3D(
            in_channels=1 + self.condition_channels + 1,
            latent_channels=1,
            base=int(base),
        )

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        condition: torch.Tensor,
        *,
        layout: FourierModeLayout,
        whitening: Mapping[str, list[float]],
    ) -> torch.Tensor:
        if condition.ndim != 5 or condition.shape[0] != state.shape[0]:
            raise ValueError("condition and Fourier state batches differ")
        if condition.shape[1] != self.condition_channels or tuple(condition.shape[-3:]) != layout.shape:
            raise ValueError("condition does not match the Fourier layout")
        physical = unwhiten_components(state, whitening, layout)
        state_field = unpack_fourier_components(physical, layout)
        time = torch.as_tensor(time_value, device=state.device, dtype=state.dtype)
        if time.ndim == 0:
            time = time.repeat(state.shape[0])
        if time.shape != (state.shape[0],):
            raise ValueError("time must be scalar or one value per draw")
        time_field = time.view(-1, 1, 1, 1, 1).expand(-1, 1, *layout.shape)
        velocity_field = self.net(torch.cat((state_field, condition, time_field), dim=1))
        return whiten_velocity(pack_fourier_components(velocity_field, layout), whitening, layout)


def rectified_flow_pair(
    target: torch.Tensor, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim != 2:
        raise ValueError("Fourier flow target must have shape [batch,components]")
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype, generator=generator)
    state = (1.0 - time[:, None]) * noise + time[:, None] * target
    return state, time, target - noise


@torch.inference_mode()
def sample_fourier_heun(
    model: ConditionalFourierVelocityUNet,
    condition: torch.Tensor,
    *,
    layout: FourierModeLayout,
    whitening: Mapping[str, list[float]],
    draws: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if condition.shape[0] != 1 or draws <= 0 or steps <= 0:
        raise ValueError("Fourier Heun sampling requires one condition and positive sizes")
    condition = condition.expand(draws, -1, -1, -1, -1)
    state = torch.randn(
        (draws, layout.components), device=condition.device, dtype=condition.dtype, generator=generator
    )
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full((draws,), step / steps, device=state.device, dtype=state.dtype)
        velocity = model(state, time, condition, layout=layout, whitening=whitening)
        proposal = state + dt * velocity
        following = torch.full((draws,), (step + 1) / steps, device=state.device, dtype=state.dtype)
        next_velocity = model(proposal, following, condition, layout=layout, whitening=whitening)
        state = state + 0.5 * dt * (velocity + next_velocity)
    physical = unwhiten_components(state, whitening, layout)
    return unpack_fourier_components(physical, layout)[:, 0]


__all__ = [
    "ConditionalFourierVelocityUNet",
    "FourierModeLayout",
    "build_fourier_layout",
    "empty_whitening_accumulator",
    "equal_band_flow_loss",
    "finalize_whitening",
    "hermitian_max_error",
    "lowpass_exact",
    "pack_fourier_components",
    "rectified_flow_pair",
    "sample_fourier_heun",
    "spectral_lowpass_reference",
    "unpack_fourier_components",
    "unwhiten_components",
    "update_whitening_accumulator",
    "whiten_components",
    "whiten_velocity",
]
