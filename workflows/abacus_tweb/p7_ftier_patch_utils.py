#!/usr/bin/env python3
"""Pure NumPy P7 graph-to-field scatter and fixed tidal-operator utilities."""
from __future__ import annotations

import numpy as np


def _axis_stencil(frac: np.ndarray, scheme: str):
    frac = np.asarray(frac, dtype=np.float64)
    if scheme == "cic":
        lower = np.floor(frac).astype(np.int64)
        return [
            (lower, 1.0 - (frac - lower)),
            (lower + 1, frac - lower),
        ]
    if scheme != "tsc":
        raise ValueError("scheme must be 'cic' or 'tsc'")
    nearest = np.rint(frac).astype(np.int64)
    output = []
    for offset in (-1, 0, 1):
        index = nearest + offset
        distance = np.abs(frac - index)
        weight = np.where(
            distance <= 0.5,
            0.75 - distance**2,
            np.where(distance <= 1.5, 0.5 * (1.5 - distance) ** 2, 0.0),
        )
        output.append((index, weight))
    return output


def scatter_nodes(
    latents: np.ndarray,
    frac_index: np.ndarray,
    shape: tuple[int, int, int],
    *,
    scheme: str = "tsc",
    require_complete: bool = True,
) -> tuple[np.ndarray, dict]:
    """Scatter N,C node values to C,ix,iy,iz with CIC or TSC."""
    latents = np.asarray(latents, dtype=np.float32)
    frac = np.asarray(frac_index, dtype=np.float64)
    if latents.ndim == 1:
        latents = latents[:, None]
    if latents.ndim != 2 or frac.shape != (len(latents), 3):
        raise ValueError("latents and frac_index must have shapes (N,C) and (N,3)")
    shape_array = np.asarray(shape, dtype=np.int64)
    stencils = [_axis_stencil(frac[:, axis], scheme) for axis in range(3)]
    grid = np.zeros((latents.shape[1], int(np.prod(shape))), dtype=np.float64)
    total_weight = np.zeros(len(latents), dtype=np.float64)
    dropped_weight = np.zeros(len(latents), dtype=np.float64)
    for ix, wx in stencils[0]:
        for iy, wy in stencils[1]:
            for iz, wz in stencils[2]:
                weight = wx * wy * wz
                valid = (
                    (ix >= 0) & (ix < shape_array[0])
                    & (iy >= 0) & (iy < shape_array[1])
                    & (iz >= 0) & (iz < shape_array[2])
                )
                total_weight += np.where(valid, weight, 0.0)
                dropped_weight += np.where(valid, 0.0, weight)
                if np.any(valid):
                    flat = (
                        (ix[valid] * shape_array[1] + iy[valid])
                        * shape_array[2] + iz[valid]
                    )
                    for channel in range(latents.shape[1]):
                        np.add.at(
                            grid[channel], flat,
                            latents[valid, channel].astype(np.float64) * weight[valid],
                        )
    if require_complete and np.any(np.abs(total_weight - 1.0) > 2e-6):
        raise ValueError("scatter stencil crosses the field frame boundary")
    grid = grid.reshape((latents.shape[1],) + tuple(shape_array)).astype(np.float32)
    diagnostics = {
        "scheme": scheme,
        "input_nodes": len(latents),
        "input_sum_by_channel": latents.sum(axis=0, dtype=np.float64),
        "grid_sum_by_channel": grid.sum(axis=(1, 2, 3), dtype=np.float64),
        "maximum_weight_error": float(np.max(np.abs(total_weight - 1.0))) if len(frac) else 0.0,
        "dropped_weight_sum": float(dropped_weight.sum()),
    }
    return grid, diagnostics


def cosine_apodization(shape: tuple[int, int, int], width: int) -> np.ndarray:
    """Separable cosine taper equal to one outside width voxels from a box edge."""
    if width < 0:
        raise ValueError("width must be non-negative")
    window = np.ones(shape, dtype=np.float64)
    if width == 0:
        return window
    for axis, size in enumerate(shape):
        one = np.ones(size, dtype=np.float64)
        n = min(width, size // 2)
        phase = (np.arange(n, dtype=np.float64) + 0.5) / n
        taper = 0.5 - 0.5 * np.cos(np.pi * phase)
        one[:n] = taper
        one[-n:] = taper[::-1]
        reshape = [1, 1, 1]
        reshape[axis] = size
        window *= one.reshape(reshape)
    return window


def fft_tidal_components(
    delta: np.ndarray,
    *,
    cell_mpc: float,
    rsmooth_mpc: float,
    apodization_width_voxels: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply T_ij(k)=(k_i k_j/k^2) W_R(k) delta(k)."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or min(delta.shape) < 2:
        raise ValueError("delta must be a three-dimensional field")
    window = cosine_apodization(delta.shape, apodization_width_voxels)
    input_field = delta * window
    kx = np.fft.fftfreq(delta.shape[0], d=cell_mpc) * 2.0 * np.pi
    ky = np.fft.fftfreq(delta.shape[1], d=cell_mpc) * 2.0 * np.pi
    kz = np.fft.rfftfreq(delta.shape[2], d=cell_mpc) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    smooth = np.exp(-0.5 * k2 * rsmooth_mpc**2)
    safe_k2 = k2.copy()
    safe_k2[0, 0, 0] = 1.0
    kernel = smooth / safe_k2
    kernel[0, 0, 0] = 0.0
    dk = np.fft.rfftn(input_field)
    axes = {"x": KX, "y": KY, "z": KZ}
    components = {}
    for left, right in ("xx", "xy", "xz", "yy", "yz", "zz"):
        components[left + right] = np.fft.irfftn(
            axes[left] * axes[right] * kernel * dk, s=delta.shape, axes=(0, 1, 2)
        ).real
    smoothed = np.fft.irfftn(
        smooth * dk, s=delta.shape, axes=(0, 1, 2)).real
    smoothed -= smoothed.mean()
    return components, smoothed


def tensor_and_eigensystem(
    components: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(components["xx"]).shape
    tensor = np.empty(shape + (3, 3), dtype=np.float64)
    tensor[..., 0, 0] = components["xx"]
    tensor[..., 1, 1] = components["yy"]
    tensor[..., 2, 2] = components["zz"]
    tensor[..., 0, 1] = tensor[..., 1, 0] = components["xy"]
    tensor[..., 0, 2] = tensor[..., 2, 0] = components["xz"]
    tensor[..., 1, 2] = tensor[..., 2, 1] = components["yz"]
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    return tensor, eigenvalues, eigenvectors


def trace_max_abs_error(
    components: dict[str, np.ndarray], smoothed_delta: np.ndarray
) -> float:
    trace = components["xx"] + components["yy"] + components["zz"]
    return float(np.max(np.abs(trace - smoothed_delta)))
