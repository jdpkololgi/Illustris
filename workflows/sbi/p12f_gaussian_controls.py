#!/usr/bin/env python3
"""Matched heteroscedastic Gaussian controls for the P12-F1b field contract."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


class ConditionalGaussianUNet(nn.Module):
    """Predict voxelwise mean and log standard deviation from the three R1 channels."""

    def __init__(self, *, condition_channels: int = 3, base: int = 8):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.net = UNet3D(
            in_channels=self.condition_channels,
            latent_channels=2,
            base=int(base),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError("condition must have shape [batch,channels,x,y,z]")
        output = self.net(condition)
        mean = output[:, :1]
        log_std = torch.clamp(output[:, 1:2], min=-7.0, max=4.0)
        return mean, log_std


def gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_std.shape or mean.shape != target.shape:
        raise ValueError("Gaussian prediction/target geometry mismatch")
    mask = torch.as_tensor(support, device=mean.device, dtype=torch.bool)
    if mask.shape != mean.shape:
        mask = mask.expand_as(mean)
    if not torch.any(mask):
        raise ValueError("Gaussian NLL has no supported voxels")
    inverse_variance = torch.exp(-2.0 * log_std)
    value = log_std + 0.5 * torch.square(target - mean) * inverse_variance
    return torch.mean(value[mask])


@torch.inference_mode()
def sample_independent_gaussian(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    *,
    draws: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if draws <= 0 or mean.shape != log_std.shape or mean.shape[0] != 1:
        raise ValueError("independent Gaussian sampling contract is invalid")
    noise = torch.randn(
        (draws, *mean.shape[1:]),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return mean.expand(draws, -1, -1, -1, -1) + torch.exp(log_std).expand_as(noise) * noise


def radial_frequency_index(
    shape: tuple[int, int, int],
    bins: int,
    *,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("at least two radial bins are required")
    kx = np.fft.fftfreq(shape[0])[:, None, None]
    ky = np.fft.fftfreq(shape[1])[None, :, None]
    kz = np.fft.rfftfreq(shape[2])[None, None, :]
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    if edges is None:
        edges = np.linspace(
            0.0,
            np.sqrt(3.0) / 2.0 + np.finfo(float).eps,
            bins + 1,
        )
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape != (bins + 1,) or np.any(np.diff(edges) <= 0):
        raise ValueError("radial-frequency edges are invalid")
    index = np.minimum(np.searchsorted(edges[1:], radius, side="right"), bins - 1)
    return index.astype(np.int16), edges


def residual_filter_accumulator(bins: int = 32) -> dict:
    _, edges = radial_frequency_index((2, 2, 2), bins)
    return {
        "bins": int(bins),
        "edges": edges,
        "power_sum": np.zeros(bins, dtype=np.float64),
        "mode_count": np.zeros(bins, dtype=np.int64),
        "fields": 0,
        "shapes": set(),
    }


def update_residual_filter_accumulator(accumulator: dict, residual: np.ndarray) -> None:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("residual spectrum update requires one finite 3D field")
    bins = int(accumulator["bins"])
    index, _ = radial_frequency_index(
        tuple(values.shape), bins, edges=accumulator["edges"]
    )
    power = np.square(
        np.abs(np.fft.rfftn(values, axes=(-3, -2, -1), norm="ortho"))
    )
    for value in range(bins):
        selected = index == value
        accumulator["power_sum"][value] += float(np.sum(power[selected]))
        accumulator["mode_count"][value] += int(np.count_nonzero(selected))
    accumulator["fields"] += 1
    accumulator["shapes"].add(tuple(map(int, values.shape)))


def finalize_residual_filter(accumulator: dict) -> dict:
    if int(accumulator["fields"]) < 2:
        raise ValueError("residual filter requires at least two training fields")
    count = np.asarray(accumulator["mode_count"], dtype=np.int64)
    radial = np.divide(
        accumulator["power_sum"],
        count,
        out=np.full_like(accumulator["power_sum"], np.nan, dtype=np.float64),
        where=count > 0,
    )
    finite = np.isfinite(radial) & (radial > 0)
    if not np.any(finite):
        raise RuntimeError("training residual spectrum is degenerate")
    radial = np.interp(np.arange(len(radial)), np.flatnonzero(finite), radial[finite])
    radial = np.maximum(radial, 1e-8)
    shapes = sorted(accumulator["shapes"])
    return {
        "schema_version": "p12f-g1-radial-residual-filter-v2",
        "fit_scope": "training phases and registered training cores only",
        "shape": list(shapes[0]),
        "reference_shapes": [list(value) for value in shapes],
        "supports_variable_shapes": True,
        "fields": int(accumulator["fields"]),
        "bins": int(accumulator["bins"]),
        "edges": np.asarray(accumulator["edges"]).tolist(),
        "radial_power": radial.tolist(),
        "mode_count": count.tolist(),
        "normalization": "orthonormal-rfftn power of residual/predicted_std",
        "unsupported_voxels": "set to zero before spectral accumulation",
        "real_hermitian_sampling": True,
    }


def fit_radial_residual_filter(
    normalized_residuals: np.ndarray,
    *,
    bins: int = 32,
) -> dict:
    """Fit a train-only radial power filter to unit-scaled residual fields."""
    residual = np.asarray(normalized_residuals, dtype=np.float64)
    if residual.ndim != 4 or len(residual) < 2 or not np.all(np.isfinite(residual)):
        raise ValueError("residual filter requires finite [fields,x,y,z] input")
    accumulator = residual_filter_accumulator(bins)
    for field in residual:
        update_residual_filter_accumulator(accumulator, field)
    return finalize_residual_filter(accumulator)


def finalize_shell_residual_filters(
    global_accumulator: dict,
    shell_accumulators: dict[int, dict],
    *,
    pseudo_fields: int = 32,
) -> dict:
    """Shrink shell-specific spectra toward the training-global G1 spectrum."""
    if pseudo_fields <= 0 or set(shell_accumulators) != {0, 1, 2, 3}:
        raise ValueError("G2 requires four shells and positive fixed shrinkage")
    global_filter = finalize_residual_filter(global_accumulator)
    global_power = np.asarray(global_filter["radial_power"], dtype=np.float64)
    global_count = np.asarray(global_accumulator["mode_count"], dtype=np.float64)
    average_modes = global_count / float(global_accumulator["fields"])
    prior_count = float(pseudo_fields) * average_modes
    shell_filters: dict[str, dict] = {}
    for shell, accumulator in sorted(shell_accumulators.items()):
        if int(accumulator["fields"]) < 2:
            raise ValueError(f"G2 shell {shell} has insufficient training fields")
        if not np.array_equal(accumulator["edges"], global_accumulator["edges"]):
            raise ValueError("G2 shell/global radial-frequency edges differ")
        count = np.asarray(accumulator["mode_count"], dtype=np.float64)
        power_sum = np.asarray(accumulator["power_sum"], dtype=np.float64)
        denominator = count + prior_count
        radial = np.divide(
            power_sum + prior_count * global_power,
            denominator,
            out=global_power.copy(),
            where=denominator > 0,
        )
        radial = np.maximum(radial, 1e-8)
        shell_filters[str(shell)] = {
            "schema_version": "p12f-g2-shell-radial-component-v1",
            "shell": int(shell),
            "shape": global_filter["shape"],
            "reference_shapes": [list(value) for value in sorted(accumulator["shapes"])],
            "supports_variable_shapes": True,
            "fields": int(accumulator["fields"]),
            "bins": int(accumulator["bins"]),
            "edges": np.asarray(accumulator["edges"]).tolist(),
            "radial_power": radial.tolist(),
            "mode_count": np.asarray(accumulator["mode_count"], dtype=np.int64).tolist(),
            "global_prior_mode_count": prior_count.tolist(),
        }
    return {
        "schema_version": "p12f-g2-shell-radial-residual-filter-v1",
        "fit_scope": "training phases and registered training cores only",
        "shell_definition": "median radius of exact-supported core voxels",
        "shrinkage": {
            "kind": "mode-count empirical Bayes toward frozen training-global G1",
            "pseudo_fields": int(pseudo_fields),
        },
        "global_filter": global_filter,
        "shell_filters": shell_filters,
        "real_hermitian_sampling": True,
    }


def correlated_unit_residuals(
    filter_contract: dict,
    *,
    draws: int,
    seed: int,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    shape = tuple(map(int, filter_contract["shape"] if shape is None else shape))
    bins = int(filter_contract["bins"])
    index, _ = radial_frequency_index(
        shape,
        bins,
        edges=np.asarray(filter_contract["edges"], dtype=np.float64),
    )
    radial = np.asarray(filter_contract["radial_power"], dtype=np.float64)
    amplitude = np.sqrt(radial[index])
    rng = np.random.default_rng(seed)
    output = np.empty((draws, *shape), dtype=np.float32)
    for draw in range(draws):
        white = rng.normal(size=shape)
        white_k = np.fft.rfftn(white, norm="ortho")
        field = np.fft.irfftn(
            white_k * amplitude,
            s=shape,
            axes=(-3, -2, -1),
            norm="ortho",
        ).real
        standard = max(float(field.std()), 1e-8)
        output[draw] = (field / standard).astype(np.float32)
    if not np.all(np.isfinite(output)) or not np.isrealobj(output):
        raise RuntimeError("correlated Gaussian residuals are invalid")
    return output


def sample_correlated_gaussian(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    filter_contract: dict,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    standard = np.asarray(standard_deviation, dtype=np.float32)
    if mean.shape != standard.shape:
        raise ValueError("G1 mean/std/filter geometry mismatch")
    if np.any(standard <= 0) or not np.all(np.isfinite(mean + standard)):
        raise ValueError("G1 mean/std values are invalid")
    residual = correlated_unit_residuals(
        filter_contract, draws=draws, seed=seed, shape=tuple(mean.shape)
    )
    return mean[None] + standard[None] * residual


def sample_shell_correlated_gaussian(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    filter_contract: dict,
    *,
    shell: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    if filter_contract.get("schema_version") != "p12f-g2-shell-radial-residual-filter-v1":
        raise ValueError("unsupported G2 shell-filter contract")
    component = filter_contract.get("shell_filters", {}).get(str(int(shell)))
    if component is None or int(component.get("shell", -1)) != int(shell):
        raise ValueError(f"G2 has no frozen filter for shell {shell}")
    return sample_correlated_gaussian(
        mean,
        standard_deviation,
        component,
        draws=draws,
        seed=seed,
    )


@dataclass
class GaussianFieldSampler:
    mean: np.ndarray
    standard_deviation: np.ndarray
    filter_contract: dict | None = None

    def sample(self, condition: np.ndarray, draws: int, seed: int) -> np.ndarray:
        del condition
        if self.filter_contract is None:
            rng = np.random.default_rng(seed)
            return self.mean[None] + self.standard_deviation[None] * rng.normal(
                size=(draws, *self.mean.shape)
            )
        return sample_correlated_gaussian(
            self.mean,
            self.standard_deviation,
            self.filter_contract,
            draws=draws,
            seed=seed,
        )
