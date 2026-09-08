"""Shared, testable contracts for the bounded P12-B per-galaxy FMPE pilot."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def safe_path(path):
    path = Path(path)
    if "ph001" in str(path) or "ph001" in str(path.resolve()):
        raise PermissionError("P12-B never accesses ph001")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with safe_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(safe_path(path).read_text())


def atomic_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, path)


def atomic_torch(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def finite(*arrays):
    for value in arrays:
        okay = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value).all()
        if not okay:
            raise FloatingPointError("P12-B encountered nonfinite values")


def theta_from_eigen(eigen):
    value = np.asarray(eigen, dtype=np.float64)
    finite(value)
    if value.shape[-1] != 3:
        raise ValueError("expected three ordered eigenvalues")
    gap = np.diff(value, axis=-1)
    if np.any(gap <= 0):
        raise ValueError("strictly positive eigengaps required; no hidden clipping")
    result = np.concatenate((value[..., :1], gap + np.log(-np.expm1(-gap))), axis=-1)
    finite(result)
    return result


def eigen_from_theta(theta):
    value = np.asarray(theta, dtype=np.float64)
    return np.cumsum(np.concatenate((value[..., :1], np.logaddexp(0, value[..., 1:])), axis=-1), axis=-1)


def physical_log_jacobian(eigen, theta_std):
    gaps = np.diff(np.asarray(eigen, dtype=np.float64), axis=-1)
    if np.any(gaps <= 0):
        raise ValueError("density undefined at nonpositive gap")
    return -np.log(np.asarray(theta_std)).sum() - np.log(-np.expm1(-gaps)).sum(axis=-1)


def context_boxes(start, stop, halo, alignment):
    """Conservative boxes before cap clipping; disjoint implies actual disjoint."""
    low = np.floor_divide(np.asarray(start) - halo, alignment) * alignment
    high = -np.floor_divide(-(np.asarray(stop) + halo), alignment) * alignment
    return low, high


def disjoint_candidates(ids, cap, low, high, heldout):
    keep = np.ones(len(ids), dtype=bool)
    for other in heldout:
        overlap = (cap[ids] == cap[other]) & np.all(low[ids] < high[other], axis=1) & np.all(high[ids] > low[other], axis=1)
        keep &= ~overlap
    return np.asarray(ids)[keep]


def balanced_cores(candidates, cap, shell, count, rng):
    groups = [rng.permutation(candidates[(cap[candidates] == c) & (shell[candidates] == s)]).tolist()
              for c in (0, 1) for s in range(4)]
    if any(not group for group in groups):
        raise RuntimeError("panel lacks a cap/shell stratum")
    chosen = []
    while len(chosen) < count:
        before = len(chosen)
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop())
        if len(chosen) == before:
            raise RuntimeError("insufficient eligible cores for registered panel")
    return np.asarray(chosen, dtype=np.int64)


def build_flow(config, device):
    from sbi.neural_nets import posterior_flow_nn
    torch.manual_seed(config["seed"])
    builder = posterior_flow_nn(model="mlp", hidden_features=config["head_hidden_features"],
                                num_layers=config["head_layers"], z_score_theta="none", z_score_x="none")
    # External train-only transforms are used; no builder-fitted data transform.
    return builder(torch.zeros(8, 3), torch.zeros(8, config["context_dimensions"])).to(device)


def heun_sample(velocity, condition, noise, steps):
    """Installed SBI convention: t=1 noise -> t=0 data."""
    with torch.no_grad():
        state = noise.clone()
        dt = -1.0 / steps
        for index in range(steps):
            time = state.new_full((len(state),), 1.0 + index * dt)
            first = velocity(state, condition, time)
            second = velocity(state + dt * first, condition, time + dt)
            state = state + (0.5 * dt) * (first + second)
        finite(state)
        return state


def velocity_divergence(velocity, state, condition, time):
    with torch.enable_grad():
        point = state.detach().requires_grad_(True)
        value = velocity(point, condition.detach(), time)
        divergence = torch.zeros_like(point[:, 0])
        for dimension in range(3):
            component = value[:, dimension].sum()
            if component.requires_grad:
                gradient = torch.autograd.grad(component, point, retain_graph=dimension < 2, allow_unused=True)[0]
                if gradient is not None:
                    divergence = divergence + gradient[:, dimension]
    return value.detach(), divergence.detach()


def heun_log_prob(velocity, condition, theta, steps):
    """Exact 3-D divergence, integrate data->noise; log q0 = log q1 + int div."""
    state = theta.detach().clone()
    integral = state.new_zeros(len(state))
    dt = 1.0 / steps
    for index in range(steps):
        time = state.new_full((len(state),), index * dt)
        first, div1 = velocity_divergence(velocity, state, condition, time)
        second, div2 = velocity_divergence(velocity, state + dt * first, condition, time + dt)
        state = state + 0.5 * dt * (first + second)
        integral = integral + 0.5 * dt * (div1 + div2)
    result = -0.5 * (state.square() + math.log(2 * math.pi)).sum(-1) + integral
    finite(result)
    return result


def row_scores(samples, truth):
    """Unbiased finite-draw energy and marginal CRPS; all physical coordinates."""
    sample = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    finite(sample, target)
    draws = sample.shape[1]
    pair = sample[:, :, None, :] - sample[:, None, :, :]
    energy = np.linalg.norm(sample - target[:, None], axis=-1).mean(1)
    energy -= 0.5 * np.linalg.norm(pair, axis=-1).sum((1, 2)) / (draws * (draws - 1))
    crps = np.abs(sample - target[:, None]).mean(1)
    crps -= 0.5 * np.abs(pair).sum((1, 2)) / (draws * (draws - 1))
    result = {"energy": energy, "crps": crps, "mean": sample.mean(1)}
    for level in (68, 90):
        quantile = (1 - level / 100) / 2
        lo, hi = np.quantile(sample, [quantile, 1 - quantile], axis=1)
        result[f"coverage{level}"] = ((target >= lo) & (target <= hi)).astype(float)
        result[f"width{level}"] = hi - lo
    return result


def clustered_difference(first, second, clusters, repetitions, seed):
    """Paired equal-row panel difference; resample cap+superblock clusters."""
    difference = np.asarray(first) - np.asarray(second)
    _, inverse = np.unique(clusters, return_inverse=True, axis=0)
    count = np.bincount(inverse)
    sums = np.bincount(inverse, weights=difference)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        choice = rng.integers(0, len(count), size=len(count))
        values.append(sums[choice].sum() / count[choice].sum())
    return {"mean": float(difference.mean()), "cluster_count": len(count),
            "interval95": np.quantile(values, [0.025, 0.975]).tolist()}
