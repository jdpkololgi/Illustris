"""Calibration diagnostics for coherent P12-F density-field samples.

These functions deliberately treat posterior-mean reconstruction scores as
secondary diagnostics.  The primary question is whether the held-out truth is
statistically exchangeable with draws from the learned conditional field law.
"""
from __future__ import annotations

import numpy as np
import torch


def randomized_ranks(samples: np.ndarray, truth: np.ndarray, *, seed: int) -> np.ndarray:
    """Return randomized posterior ranks in [0,1].

    Spatial voxels are correlated, so callers must aggregate/bootstrap by patch
    rather than interpreting the flattened rank count as independent trials.
    """
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if draws.ndim < 2 or draws.shape[1:] != target.shape:
        raise ValueError("samples must have shape [draw,...truth_shape]")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(target)):
        raise ValueError("rank inputs must be finite")
    less = np.sum(draws < target[None], axis=0)
    equal = np.sum(draws == target[None], axis=0)
    rng = np.random.default_rng(seed)
    return (less + rng.random(target.shape) * (equal + 1.0)) / (draws.shape[0] + 1.0)


def central_coverage(samples: np.ndarray, truth: np.ndarray, levels=(0.5, 0.68, 0.9)) -> dict:
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if draws.ndim < 2 or draws.shape[1:] != target.shape:
        raise ValueError("samples must have shape [draw,...truth_shape]")
    result = {}
    for level in levels:
        level = float(level)
        if not 0.0 < level < 1.0:
            raise ValueError("coverage levels must be between zero and one")
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(draws, [tail, 1.0 - tail], axis=0)
        inside = (target >= low) & (target <= high)
        result[f"{level:.2f}"] = {
            "nominal": level,
            "empirical": float(np.mean(inside)),
            "absolute_error": float(abs(np.mean(inside) - level)),
        }
    return result


def rank_cdf_maximum_deviation(ranks: np.ndarray) -> float:
    values = np.sort(np.asarray(ranks, dtype=np.float64).ravel())
    if len(values) == 0:
        return float("nan")
    upper = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
    lower = np.arange(0, len(values), dtype=np.float64) / len(values)
    return float(max(np.max(np.abs(upper - values)), np.max(np.abs(values - lower))))


def scalar_posterior_report(samples: np.ndarray, truth: np.ndarray, *, seed: int) -> dict:
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    ranks = randomized_ranks(draws, target, seed=seed)
    mean = draws.mean(axis=0)
    residual = mean - target
    variance = float(np.sum(np.square(target - target.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / max(variance, 1e-30)
    return {
        "n": int(target.size),
        "draws": int(draws.shape[0]),
        "coverage": central_coverage(draws, target),
        "rank_cdf_maximum_deviation": rank_cdf_maximum_deviation(ranks),
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
        "posterior_mean_r2_diagnostic": float(r2),
        "posterior_mean_rmse_diagnostic": float(np.sqrt(np.mean(np.square(residual)))),
        "posterior_width_median": float(np.median(np.std(draws, axis=0, ddof=1))),
    }


def conditional_reports(
    samples: np.ndarray,
    truth: np.ndarray,
    stratum: np.ndarray,
    *,
    seed: int,
    minimum_rows: int = 64,
) -> dict:
    draws = np.asarray(samples)
    target = np.asarray(truth)
    labels = np.asarray(stratum)
    if target.shape != labels.shape or draws.shape[1:] != target.shape:
        raise ValueError("conditional arrays are not aligned")
    result = {}
    for value in np.unique(labels):
        selected = labels == value
        if int(selected.sum()) < minimum_rows:
            continue
        result[str(value)] = scalar_posterior_report(
            draws[:, selected], target[selected], seed=seed + int(np.sum(selected))
        )
    return result


def quantile_labels(values: np.ndarray, bins: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if bins < 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite values and at least two bins are required")
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int8)


def fixed_tidal_tensor(delta_r7: torch.Tensor) -> torch.Tensor:
    """Apply the no-extra-smoothing periodic tidal projector.

    Input is [...,nx,ny,nz].  Output is [...,nx,ny,nz,3,3].  The zero mode is
    removed, so Tr(T) closes to delta_R7 minus its patch mean.  This is the
    exact differentiable physics layer used for the bounded local canary; the
    production branch still needs the registered large-context/global-mode gate.
    """
    if delta_r7.ndim < 3 or not torch.is_floating_point(delta_r7):
        raise ValueError("delta_r7 must be a floating tensor with at least three dimensions")
    nx, ny, nz = delta_r7.shape[-3:]
    device, dtype = delta_r7.device, delta_r7.dtype
    kx = torch.fft.fftfreq(nx, device=device, dtype=dtype).view(nx, 1, 1)
    ky = torch.fft.fftfreq(ny, device=device, dtype=dtype).view(1, ny, 1)
    kz = torch.fft.fftfreq(nz, device=device, dtype=dtype).view(1, 1, nz)
    k = (kx, ky, kz)
    k2 = kx.square() + ky.square() + kz.square()
    safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
    delta_k = torch.fft.fftn(delta_r7, dim=(-3, -2, -1))
    rows = []
    for a in range(3):
        columns = []
        for b in range(3):
            kernel = k[a] * k[b] / safe
            kernel = torch.where(k2 > 0, kernel, torch.zeros_like(kernel))
            columns.append(
                torch.fft.ifftn(delta_k * kernel, dim=(-3, -2, -1)).real
            )
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def fixed_tidal_eigenvalues(
    delta_r7: torch.Tensor, *, matrix_chunk_size: int = 8192
) -> torch.Tensor:
    """Return ordered tidal eigenvalues with bounded eigensolver workspace.

    ``torch.linalg.eigvalsh`` can request a very large vendor-library workspace
    when millions of 3x3 tensors are presented as one batch.  Chunking only the
    matrix batch leaves the physical FFT projector and numerical result unchanged
    while bounding memory.  Concatenation preserves autograd for later uses of the
    fixed physics layer.
    """
    if matrix_chunk_size <= 0:
        raise ValueError("matrix_chunk_size must be positive")
    tensor = fixed_tidal_tensor(delta_r7)
    flat = tensor.reshape(-1, 3, 3)
    pieces = [
        torch.linalg.eigvalsh(flat[start : start + matrix_chunk_size])
        for start in range(0, len(flat), matrix_chunk_size)
    ]
    return torch.cat(pieces, dim=0).reshape(*tensor.shape[:-2], 3)


def physics_closure_report(delta_r7: torch.Tensor) -> dict:
    tensor = fixed_tidal_tensor(delta_r7)
    trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1)
    centered = delta_r7 - delta_r7.mean(dim=(-3, -2, -1), keepdim=True)
    error = trace - centered
    eigen = torch.linalg.eigvalsh(tensor)
    ordering = eigen[..., 1:] - eigen[..., :-1]
    return {
        "trace_max_abs": float(error.abs().max().detach().cpu()),
        "trace_rmse": float(torch.sqrt(torch.mean(error.square())).detach().cpu()),
        "minimum_eigengap": float(ordering.min().detach().cpu()),
        "all_finite": bool(torch.isfinite(tensor).all() and torch.isfinite(eigen).all()),
        "ordered": bool(torch.all(ordering >= -32.0 * torch.finfo(eigen.dtype).eps)),
        "additional_gaussian_smoothing": False,
    }


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean scalar ensemble CRPS; useful but not a joint-field score."""
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    term1 = np.mean(np.abs(draws - target[None]), axis=0)
    pairwise = np.mean(np.abs(draws[:, None] - draws[None, :]), axis=(0, 1))
    return float(np.mean(term1 - 0.5 * pairwise))


def standard_normal_rank_reference(draws: int) -> dict:
    return {
        "rank_mean": 0.5,
        "rank_variance": 1.0 / 12.0,
        "finite_draw_resolution": 1.0 / (int(draws) + 1),
        "spatial_independence_assumed": False,
    }
