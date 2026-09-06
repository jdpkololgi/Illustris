#!/usr/bin/env python3
"""Frozen sample-based score primitives for the P12-A blind evaluation."""
from __future__ import annotations

import numpy as np


def align_positions(reference: np.ndarray, requested: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    if (
        reference.ndim != 1
        or requested.ndim != 1
        or len(np.unique(reference)) != len(reference)
        or len(np.unique(requested)) != len(requested)
    ):
        raise RuntimeError("parent identities must be one-dimensional and unique")
    order = np.argsort(reference)
    position = np.searchsorted(reference[order], requested)
    if np.any(position >= len(reference)) or not np.array_equal(
        reference[order][position], requested
    ):
        raise RuntimeError("requested parent IDs are absent from the frozen reference")
    return order[position]


def clustered_mean_interval(
    values: np.ndarray, groups: np.ndarray, *, repeats: int, seed: int
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if (
        values.ndim != 1
        or groups.shape != values.shape
        or not np.all(np.isfinite(values))
        or repeats <= 0
    ):
        raise ValueError("clustered score values/groups/repetitions are invalid")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("core bootstrap requires at least two authoritative cores")
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        bootstrap[index] = total[chosen].sum() / count[chosen].sum()
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_repetitions": int(repeats),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "authoritative core",
        "spatial_blocks": int(len(unique)),
    }


def joint_energy_score(
    samples: np.ndarray, truth: np.ndarray, *, pairing_offset: int
) -> np.ndarray:
    """Monte-Carlo multivariate energy score with a fixed cyclic draw pairing."""

    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError("energy-score samples must have shape [rows,draws,3]")
    if (
        truth.shape != (samples.shape[0], 3)
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(truth))
    ):
        raise ValueError("energy-score truth/sample shape or finiteness is invalid")
    draws = samples.shape[1]
    if pairing_offset <= 0 or pairing_offset >= draws:
        raise ValueError("energy pairing offset must be within the draw axis")
    first = np.linalg.norm(samples - truth[:, None, :], axis=2).mean(axis=1)
    paired = np.roll(samples, shift=-pairing_offset, axis=1)
    second = 0.5 * np.linalg.norm(samples - paired, axis=2).mean(axis=1)
    score = first - second
    if not np.all(np.isfinite(score)):
        raise RuntimeError("joint energy score is non-finite")
    return score


def gaussian_samples(
    *,
    base: np.ndarray,
    shell: np.ndarray,
    cap: np.ndarray,
    gaussian: dict,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Draw the frozen shell/cap physical-residual Gaussian exactly as fitted.

    The control is intentionally not post-processed by sorting, isotonic
    projection, rejection, or clipping.  Those operations would define a new
    distribution fitted neither in the Gaussian marker nor in the pre-open
    contract.  Unordered control draws are therefore retained and are penalised
    naturally by the proper score against ordered physical truths.
    """

    base = np.asarray(base, dtype=np.float64)
    shell = np.asarray(shell, dtype=np.int8)
    cap = np.asarray(cap, dtype=np.uint8)
    if (
        base.ndim != 2
        or base.shape[1] != 3
        or len(shell) != len(base)
        or len(cap) != len(base)
        or draws <= 1
    ):
        raise ValueError("Gaussian conditioning arrays are invalid")
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((len(base), draws, 3))
    result = np.empty_like(standard)
    assigned = np.zeros(len(base), dtype=bool)
    for shell_value in range(4):
        for cap_value in (0, 1):
            chosen = (shell == shell_value) & (cap == cap_value)
            group = gaussian.get("groups", {}).get(
                f"shell{shell_value}_cap{cap_value}"
            )
            if not np.any(chosen):
                continue
            if group is None:
                raise RuntimeError("Gaussian baseline lacks an occupied shell/cap group")
            mean = np.asarray(group["mean"], dtype=np.float64)
            covariance = np.asarray(group["covariance"], dtype=np.float64)
            if (
                mean.shape != (3,)
                or covariance.shape != (3, 3)
                or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
            ):
                raise RuntimeError("Gaussian baseline group is invalid")
            cholesky = np.linalg.cholesky(covariance)
            result[chosen] = (
                base[chosen, None, :]
                + mean[None, None, :]
                + np.einsum("ndi,ji->ndj", standard[chosen], cholesky)
            )
            assigned[chosen] = True
    if not np.all(assigned) or not np.all(np.isfinite(result)):
        raise RuntimeError("Gaussian baseline did not draw every audit row")
    return result
