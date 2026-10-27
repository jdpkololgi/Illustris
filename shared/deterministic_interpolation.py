"""Trilinear interpolation using deterministic-capable gather gradients.

CUDA grid_sample and trilinear upsampling backward use unordered atomic sums.
These equivalents use index_select, for which PyTorch's strict deterministic
mode supplies an ordered backward. Callers must enable deterministic algorithms.
"""
from __future__ import annotations

import torch


def resize_trilinear(values: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Equivalent to interpolate(mode='trilinear', align_corners=False)."""
    if values.ndim != 5 or len(size) != 3 or any(n < 1 for n in size):
        raise ValueError('Expected NCDHW and three positive output dimensions')
    result = values
    for axis, length in enumerate(size, start=2):
        source = result.shape[axis]
        if source == length:
            continue
        coordinate = ((torch.arange(length, device=values.device, dtype=values.dtype) + .5)
                      * (source / length) - .5).clamp_min(0)
        low = coordinate.floor().long()
        high = (low + 1).clamp_max(source - 1)
        weight = coordinate - low
        shape = [1] * 5
        shape[axis] = length
        weight = weight.reshape(shape)
        result = result.index_select(axis, low) * (1 - weight) + result.index_select(axis, high) * weight
    return result


def sample_points_trilinear(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """NCDHW -> NCP, matching grid_sample border/align_corners=True.

    Points have shape N,...,3 and normalized XYZ order, with no point-coordinate
    gradients: their positions are fixed catalogue geometry. Repeated indices
    and border points are supported; value gradients use deterministic gathers.
    """
    if values.ndim != 5 or points.shape[0] != values.shape[0] or points.shape[-1] != 3:
        raise ValueError('Expected NCDHW values and N,...,3 normalized coordinates')
    if points.requires_grad:
        raise ValueError('Catalogue coordinates must be fixed')
    xyz = points.reshape(values.shape[0], -1, 3)
    dimensions = values.shape[2:]
    outputs = []
    for batch in range(values.shape[0]):
        coords = [((xyz[batch, :, 2-axis] + 1) * ((length-1) / 2)).clamp(0, length-1)
                  for axis, length in enumerate(dimensions)]
        low = [v.floor().long() for v in coords]
        high = [(v+1).clamp_max(n-1) for v, n in zip(low, dimensions)]
        frac = [v-i for v, i in zip(coords, low)]
        flat = values[batch].reshape(values.shape[1], -1)
        result = None
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    flags = (z, y, x)
                    indices = [high[a] if f else low[a] for a, f in enumerate(flags)]
                    weight = torch.ones_like(frac[0])
                    for a, f in enumerate(flags):
                        weight = weight * (frac[a] if f else 1-frac[a])
                    address = (indices[0]*dimensions[1] + indices[1])*dimensions[2] + indices[2]
                    term = flat.index_select(1, address) * weight
                    result = term if result is None else result + term
        outputs.append(result)
    return torch.stack(outputs)


def max_pool3d_stride2(values: torch.Tensor) -> torch.Tensor:
    """Exact non-overlapping 2x2x2 max pool with ceil_mode=True.

    Flattening each disjoint block and taking its first maximum preserves native
    max-pool tie selection. Backpropagation never needs overlapping atomics.
    """
    if values.ndim != 5:
        raise ValueError('Expected NCDHW')
    padded = values
    for axis in (2, 3, 4):
        if padded.shape[axis] % 2:
            shape = list(padded.shape)
            shape[axis] = 1
            padded = torch.cat((padded, padded.new_full(shape, -float('inf'))), dim=axis)
    b, c, d, h, w = padded.shape
    blocks = padded.reshape(b, c, d//2, 2, h//2, 2, w//2, 2).permute(0, 1, 2, 4, 6, 3, 5, 7)
    return blocks.reshape(b, c, d//2, h//2, w//2, 8).max(-1).values
