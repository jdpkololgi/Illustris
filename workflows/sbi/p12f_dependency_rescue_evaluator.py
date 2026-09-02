#!/usr/bin/env python3
"""Resumable expanded-panel calibration audit for the frozen P12-F G1 sampler.

This is a checkpoint-only ph006 evaluation.  It never trains, fits, corrects, or
opens ph001.  Expensive tidal projection/eigenvalue work is cached per authoritative
core so a two-hour interactive allocation can stop with code 75 and resume exactly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import FieldSampleContract, fixed_pair_indices
from workflows.sbi.p12f_common_evaluator import (
    efficient_crps_ensemble,
    load_core_record,
    validate_archive_manifest,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    _chunked_eigvalsh,
    central_coverage,
    fixed_tidal_tensor,
    quantile_labels,
    randomized_ranks,
    rank_cdf_maximum_deviation,
    scalar_posterior_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_dependency_rescue_v2.json"
DRAW_COUNTS = (64, 128, 256)
TARP_SEEDS = tuple(range(42, 62))
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")
GAP_NAMES = ("gap12", "gap23")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def canonical_digest(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--panel-marker", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-seconds", type=float, default=6300.0)
    parser.add_argument("--precompute-only", action="store_true")
    return parser.parse_args()


def _trilinear_corner_contract(
    coordinates: np.ndarray, shape: tuple[int, int, int]
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[np.ndarray]]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("galaxy coordinates must have shape [rows,3]")
    extent = np.asarray(shape, dtype=np.int64)
    clipped = np.clip(coords, 0.0, extent - 1.0)
    lo = np.floor(clipped).astype(np.int64)
    hi = np.minimum(lo + 1, extent - 1)
    fraction = clipped - lo
    indices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    weights: list[np.ndarray] = []
    for dx in (0, 1):
        wx = 1.0 - fraction[:, 0] if dx == 0 else fraction[:, 0]
        ix = lo[:, 0] if dx == 0 else hi[:, 0]
        for dy in (0, 1):
            wy = 1.0 - fraction[:, 1] if dy == 0 else fraction[:, 1]
            iy = lo[:, 1] if dy == 0 else hi[:, 1]
            for dz in (0, 1):
                wz = 1.0 - fraction[:, 2] if dz == 0 else fraction[:, 2]
                iz = lo[:, 2] if dz == 0 else hi[:, 2]
                indices.append((ix, iy, iz))
                weights.append((wx * wy * wz).astype(np.float32))
    return indices, weights


def tidal_eigenvalues_at_galaxies(
    delta_r7: torch.Tensor,
    coordinates: np.ndarray,
    *,
    matrix_chunk_size: int = 8192,
) -> tuple[np.ndarray, dict]:
    """Project density, diagonalize only interpolation corners, then interpolate.

    This is algebraically identical to diagonalizing the complete tensor grid and
    trilinearly interpolating the ordered eigenvalue fields, but avoids eigvalsh on
    the many grid cells that no authoritative galaxy samples.
    """
    if delta_r7.ndim not in (3, 4):
        raise ValueError("density must have shape [x,y,z] or [draw,x,y,z]")
    scalar = delta_r7.ndim == 3
    values = delta_r7[None] if scalar else delta_r7
    tensor = fixed_tidal_tensor(values)
    trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1)
    centered = values - values.mean(dim=(-3, -2, -1), keepdim=True)
    error = trace - centered
    if len(coordinates) == 0:
        empty = np.empty((values.shape[0], 0, 3), dtype=np.float32)
        report = {
            "trace_max_abs": float(error.abs().max().detach().cpu()),
            "trace_rmse": float(torch.sqrt(torch.mean(error.square())).detach().cpu()),
            "all_finite": bool(torch.isfinite(tensor).all()),
            "all_ordered": True,
        }
        return (empty[0] if scalar else empty), report
    indices, weights = _trilinear_corner_contract(
        coordinates, tuple(int(value) for value in values.shape[-3:])
    )
    output = torch.zeros(
        (values.shape[0], len(coordinates), 3),
        dtype=values.dtype,
        device=values.device,
    )
    for (ix, iy, iz), weight in zip(indices, weights, strict=True):
        corner = tensor[:, ix, iy, iz]
        eigen = _chunked_eigvalsh(corner, matrix_chunk_size=matrix_chunk_size)
        output += eigen * torch.as_tensor(weight, device=values.device)[None, :, None]
    ordering = output[..., 1:] - output[..., :-1]
    report = {
        "trace_max_abs": float(error.abs().max().detach().cpu()),
        "trace_rmse": float(torch.sqrt(torch.mean(error.square())).detach().cpu()),
        "all_finite": bool(torch.isfinite(tensor).all() and torch.isfinite(output).all()),
        "all_ordered": bool(torch.all(ordering >= -32.0 * torch.finfo(output.dtype).eps)),
    }
    result = output[0] if scalar else output
    return result.detach().cpu().numpy().astype(np.float32), report


def _core_slice(bounds: np.ndarray) -> tuple[slice, slice, slice]:
    bounds = np.asarray(bounds, dtype=np.int64)
    if bounds.shape != (2, 3):
        raise RuntimeError("core bounds must have shape [2,3]")
    return tuple(
        slice(int(left), int(right))
        for left, right in zip(bounds[0], bounds[1], strict=True)
    )


def _spatial_pair_diagnostics(
    samples: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    *,
    seed: int,
    pairs: int = 4096,
) -> dict[str, np.ndarray]:
    valid = np.flatnonzero(support.ravel())
    if len(valid) < 2:
        empty = np.empty(0, dtype=np.float32)
        return {
            "pair_distance_voxels": empty,
            "truth_residual_product": empty,
            "posterior_residual_covariance": empty,
            "truth_residual_variogram": empty,
            "posterior_residual_variogram": empty,
        }
    pair = fixed_pair_indices(len(valid), min(pairs, max(1, 2 * len(valid))), seed=seed)
    flat_samples = samples.reshape(samples.shape[0], -1)[:, valid]
    flat_truth = truth.ravel()[valid]
    posterior_mean = flat_samples.mean(axis=0)
    residual_draw = flat_samples - posterior_mean[None]
    residual_truth = flat_truth - posterior_mean
    left, right = pair[:, 0], pair[:, 1]
    coordinates = np.column_stack(np.unravel_index(valid, support.shape)).astype(np.float64)
    distance = np.linalg.norm(coordinates[left] - coordinates[right], axis=1)
    return {
        "pair_distance_voxels": distance.astype(np.float32),
        "truth_residual_product": (residual_truth[left] * residual_truth[right]).astype(np.float32),
        "posterior_residual_covariance": np.mean(
            residual_draw[:, left] * residual_draw[:, right], axis=0
        ).astype(np.float32),
        "truth_residual_variogram": np.square(
            residual_truth[left] - residual_truth[right]
        ).astype(np.float32),
        "posterior_residual_variogram": np.mean(
            np.square(residual_draw[:, left] - residual_draw[:, right]), axis=0
        ).astype(np.float32),
    }


def _spectral_diagnostics(
    samples: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    *,
    bins: int = 12,
) -> dict[str, np.ndarray]:
    mean = samples.mean(axis=0)
    innovation = (truth - mean) * support
    residual = (samples - mean[None]) * support[None]
    truth_k = np.fft.rfftn(innovation, axes=(-3, -2, -1), norm="ortho")
    draw_k = np.fft.rfftn(residual, axes=(-3, -2, -1), norm="ortho")
    shape = truth.shape
    kx = np.fft.fftfreq(shape[0])[:, None, None]
    ky = np.fft.fftfreq(shape[1])[None, :, None]
    kz = np.fft.rfftfreq(shape[2])[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    edges = np.linspace(0.0, float(np.sqrt(0.75)), bins + 1)
    label = np.searchsorted(edges[1:-1], kmag.ravel(), side="right")
    truth_power = np.square(np.abs(truth_k)).ravel()
    posterior_power = np.mean(np.square(np.abs(draw_k)), axis=0).ravel()
    return {
        "spectral_edges": edges.astype(np.float32),
        "spectral_count": np.bincount(label, minlength=bins).astype(np.int64),
        "truth_innovation_power_sum": np.bincount(
            label, weights=truth_power, minlength=bins
        ).astype(np.float64),
        "posterior_residual_power_sum": np.bincount(
            label, weights=posterior_power, minlength=bins
        ).astype(np.float64),
    }


def derive_compact_core(
    metadata: dict,
    record: dict,
    *,
    method: str,
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    samples = np.asarray(record["delta_samples"], dtype=np.float32)
    truth = np.asarray(record["delta_truth"], dtype=np.float32)
    support = np.asarray(record["support"], dtype=bool)
    FieldSampleContract(
        method=method,
        core_id=int(metadata["core_id"]),
        samples=samples,
        truth=truth,
        support=support,
    ).validate()
    core = _core_slice(record["core_bounds"])
    sample_core = samples[(slice(None),) + core]
    truth_core = truth[core]
    support_core = support[core]
    valid = np.flatnonzero(support_core.ravel())
    if len(valid) == 0:
        raise RuntimeError("expanded panel contains an unsupported core")
    if len(valid) > 2048:
        valid = valid[np.linspace(0, len(valid) - 1, 2048, dtype=np.int64)]
    voxel_samples = sample_core.reshape(samples.shape[0], -1)[:, valid]
    coordinates = np.asarray(record["galaxy_frac_index_local"], dtype=np.float32)
    with torch.no_grad():
        lambda_samples, sample_physics = tidal_eigenvalues_at_galaxies(
            torch.from_numpy(samples).to(device), coordinates
        )
        lambda_truth, truth_physics = tidal_eigenvalues_at_galaxies(
            torch.from_numpy(truth).to(device), coordinates
        )
    spatial = _spatial_pair_diagnostics(
        sample_core, truth_core, support_core, seed=42_000 + int(metadata["core_id"])
    )
    spectral = _spectral_diagnostics(sample_core, truth_core, support_core)
    arrays: dict[str, np.ndarray] = {
        "voxel_samples": voxel_samples.astype(np.float32),
        "voxel_truth": truth_core.ravel()[valid].astype(np.float32),
        "voxel_response": np.asarray(record["angular_response"])[core].ravel()[valid].astype(np.float32),
        "voxel_boundary": np.asarray(record["boundary_distance_mpc"])[core].ravel()[valid].astype(np.float32),
        "voxel_tracer": np.asarray(record["tracer_density"])[core].ravel()[valid].astype(np.float32),
        "lambda_samples": lambda_samples,
        "lambda_truth": lambda_truth,
        **spatial,
        **spectral,
    }
    summary = {
        "core_id": int(metadata["core_id"]),
        "shell": int(metadata["shell"]),
        "cap": int(metadata["cap"]),
        "voxel_rows": int(len(valid)),
        "galaxy_rows": int(len(lambda_truth)),
        "sample_physics": sample_physics,
        "truth_physics": truth_physics,
    }
    return arrays, summary


def _ordered_curve(ecp: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(alpha)
    return np.asarray(alpha)[order], np.asarray(ecp)[order]


def _compact_curve(alpha: np.ndarray, ecp: np.ndarray, maximum: int = 301) -> dict:
    if len(alpha) > maximum:
        index = np.linspace(0, len(alpha) - 1, maximum, dtype=np.int64)
    else:
        index = np.arange(len(alpha))
    return {
        "alpha": alpha[index].tolist(),
        "expected_coverage_probability": ecp[index].tolist(),
        "ecp_minus_alpha": (ecp[index] - alpha[index]).tolist(),
        "maximum_deviation": float(np.max(np.abs(ecp - alpha))),
        "full_curve_points": int(len(alpha)),
    }


def tarp_curve(samples: np.ndarray, truth: np.ndarray, *, seed: int) -> dict:
    import tarp

    ecp, alpha = tarp.get_tarp_coverage(
        samples,
        truth,
        norm=True,
        bootstrap=False,
        seed=seed,
    )
    alpha, ecp = _ordered_curve(ecp, alpha)
    return _compact_curve(alpha, ecp)


def _clustered_decile_interval(
    ranks: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(ranks).ravel()
    labels = np.asarray(groups).ravel()
    unique, inverse = np.unique(labels, return_inverse=True)
    bins = np.minimum((values * 10).astype(np.int64), 9)
    counts = np.zeros((len(unique), 10), dtype=np.int64)
    np.add.at(counts, (inverse, bins), 1)
    rng = np.random.default_rng(seed)
    result = np.empty((repeats, 10), dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        total = counts[chosen].sum(axis=0)
        result[repeat] = total / total.sum()
    return np.quantile(result, [0.025, 0.5, 0.975], axis=0).T


def _sbc(samples: np.ndarray, truth: np.ndarray, groups: np.ndarray, *, seed: int) -> dict:
    ranks = randomized_ranks(samples, truth, seed=seed).ravel()
    bins = np.histogram(ranks, bins=np.linspace(0.0, 1.0, 11))[0]
    mass = bins / bins.sum()
    return {
        "rows": int(len(ranks)),
        "decile_mass": mass.tolist(),
        "decile_mass_95ci": _clustered_decile_interval(
            ranks, groups, repeats=4000, seed=seed + 1000
        ).tolist(),
        "rank_cdf_maximum_deviation": rank_cdf_maximum_deviation(ranks),
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
    }


def _subpanel_labels(core_ids: np.ndarray, shells: np.ndarray) -> np.ndarray:
    result = np.empty(len(core_ids), dtype=np.int8)
    for shell in range(4):
        index = np.flatnonzero(shells == shell)
        order = index[np.argsort(core_ids[index])]
        result[order] = np.arange(len(order), dtype=np.int64) % 4
    return result


def aggregate_compact(entries: list[dict], config: dict) -> dict:
    voxel_samples = []
    voxel_truth = []
    voxel_shell = []
    voxel_response = []
    voxel_boundary = []
    voxel_tracer = []
    voxel_core = []
    lambda_samples = []
    lambda_truth = []
    lambda_shell = []
    lambda_core = []
    core_ids = np.asarray([row["core_id"] for row in entries], dtype=np.int64)
    core_shells = np.asarray([row["shell"] for row in entries], dtype=np.int8)
    core_subpanel = _subpanel_labels(core_ids, core_shells)
    spatial_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "pair_distance_voxels",
            "truth_residual_product",
            "posterior_residual_covariance",
            "truth_residual_variogram",
            "posterior_residual_variogram",
        )
    }
    spectral_truth = None
    spectral_posterior = None
    spectral_count = None
    spectral_edges = None
    physics = []
    for ordinal, row in enumerate(entries):
        with np.load(row["compact_path"], allow_pickle=False) as values:
            voxel_samples.append(np.asarray(values["voxel_samples"], dtype=np.float32))
            voxel_truth.append(np.asarray(values["voxel_truth"], dtype=np.float32))
            voxel_response.append(np.asarray(values["voxel_response"], dtype=np.float32))
            voxel_boundary.append(np.asarray(values["voxel_boundary"], dtype=np.float32))
            voxel_tracer.append(np.asarray(values["voxel_tracer"], dtype=np.float32))
            lambda_samples.append(np.asarray(values["lambda_samples"], dtype=np.float32))
            lambda_truth.append(np.asarray(values["lambda_truth"], dtype=np.float32))
            for name in spatial_parts:
                spatial_parts[name].append(np.asarray(values[name]))
            current_edges = np.asarray(values["spectral_edges"], dtype=np.float64)
            if spectral_edges is None:
                spectral_edges = current_edges
                spectral_truth = np.zeros(len(current_edges) - 1, dtype=np.float64)
                spectral_posterior = np.zeros(len(current_edges) - 1, dtype=np.float64)
                spectral_count = np.zeros(len(current_edges) - 1, dtype=np.int64)
            elif not np.array_equal(spectral_edges, current_edges):
                raise RuntimeError("spectral-bin contract changed between cores")
            spectral_truth += np.asarray(values["truth_innovation_power_sum"])
            spectral_posterior += np.asarray(values["posterior_residual_power_sum"])
            spectral_count += np.asarray(values["spectral_count"])
        voxel_shell.append(np.full(row["voxel_rows"], row["shell"], dtype=np.int8))
        voxel_core.append(np.full(row["voxel_rows"], row["core_id"], dtype=np.int64))
        lambda_shell.append(np.full(row["galaxy_rows"], row["shell"], dtype=np.int8))
        lambda_core.append(np.full(row["galaxy_rows"], row["core_id"], dtype=np.int64))
        physics.extend((row["sample_physics"], row["truth_physics"]))
    vox = np.concatenate(voxel_samples, axis=1)
    vox_truth = np.concatenate(voxel_truth)
    vox_shell = np.concatenate(voxel_shell)
    vox_response = np.concatenate(voxel_response)
    vox_boundary = np.concatenate(voxel_boundary)
    vox_tracer = np.concatenate(voxel_tracer)
    vox_core = np.concatenate(voxel_core)
    lam = np.concatenate(lambda_samples, axis=1)
    lam_truth = np.concatenate(lambda_truth, axis=0)
    lam_shell = np.concatenate(lambda_shell)
    lam_core = np.concatenate(lambda_core)
    reports = {}
    for draws in DRAW_COUNTS:
        v = vox[:draws]
        l = lam[:draws]
        gap = l[..., 1:] - l[..., :-1]
        gap_truth = lam_truth[..., 1:] - lam_truth[..., :-1]
        eigen_tarp = tarp_curve(l, lam_truth, seed=TARP_SEEDS[0])
        gap_tarp = tarp_curve(gap, gap_truth, seed=TARP_SEEDS[0] + 1)
        seed_index = np.linspace(
            0, len(lam_truth) - 1, min(100_000, len(lam_truth)), dtype=np.int64
        )
        seed_eigen = [
            tarp_curve(l[:, seed_index], lam_truth[seed_index], seed=seed)["maximum_deviation"]
            for seed in TARP_SEEDS
        ]
        seed_gap = [
            tarp_curve(gap[:, seed_index], gap_truth[seed_index], seed=seed + 100)["maximum_deviation"]
            for seed in TARP_SEEDS
        ]
        lambda_report = {
            name: scalar_posterior_report(l[..., index], lam_truth[..., index], seed=62 + index)
            for index, name in enumerate(EIGEN_NAMES)
        }
        gap_report = {
            name: scalar_posterior_report(gap[..., index], gap_truth[..., index], seed=72 + index)
            for index, name in enumerate(GAP_NAMES)
        }
        reports[str(draws)] = {
            "draws": draws,
            "voxel": scalar_posterior_report(v, vox_truth, seed=41),
            "voxel_crps": efficient_crps_ensemble(v, vox_truth),
            "ordered_eigenvalues": lambda_report,
            "eigengaps": gap_report,
            "tarp": {
                "ordered_eigenvalues": eigen_tarp,
                "eigengaps": gap_tarp,
                "reference_seed_maxima": {
                    "ordered_seeds": list(TARP_SEEDS),
                    "eigengap_seeds": [seed + 100 for seed in TARP_SEEDS],
                    "rows": int(len(seed_index)),
                    "ordered_eigenvalues": seed_eigen,
                    "eigengaps": seed_gap,
                    "ordered_p90": float(np.quantile(seed_eigen, 0.9)),
                    "eigengap_p90": float(np.quantile(seed_gap, 0.9)),
                },
            },
        }
    subpanels = {}
    for subpanel in range(4):
        selected_cores = set(core_ids[core_subpanel == subpanel].tolist())
        selected = np.isin(lam_core, list(selected_cores))
        l = lam[:, selected]
        target = lam_truth[selected]
        gap = l[..., 1:] - l[..., :-1]
        gap_target = target[..., 1:] - target[..., :-1]
        subpanels[str(subpanel)] = {
            "cores": len(selected_cores),
            "shell_counts": np.bincount(
                core_shells[core_subpanel == subpanel], minlength=4
            ).tolist(),
            "galaxies": int(np.count_nonzero(selected)),
            "ordered_eigenvalue_tarp": tarp_curve(l, target, seed=42),
            "eigengap_tarp": tarp_curve(gap, gap_target, seed=43),
        }
    conditional = {"shell": {}, "random_response": {}, "boundary_distance": {}, "tracer_density": {}}
    gap = lam[..., 1:] - lam[..., :-1]
    gap_truth = lam_truth[..., 1:] - lam_truth[..., :-1]
    for shell in range(4):
        selected = lam_shell == shell
        conditional["shell"][str(shell)] = {
            "rows": int(np.count_nonzero(selected)),
            "ordered_eigenvalue_tarp": tarp_curve(lam[:, selected], lam_truth[selected], seed=82 + shell),
            "eigengap_tarp": tarp_curve(gap[:, selected], gap_truth[selected], seed=92 + shell),
        }
    voxel_labels = {
        "random_response": quantile_labels(vox_response),
        "boundary_distance": quantile_labels(vox_boundary),
        "tracer_density": quantile_labels(vox_tracer),
    }
    for variable, labels in voxel_labels.items():
        for value in range(4):
            selected = labels == value
            conditional[variable][str(value)] = scalar_posterior_report(
                vox[:, selected], vox_truth[selected], seed=110 + value
            )
    sbc = {
        name: _sbc(lam[..., index], lam_truth[..., index], lam_core, seed=122 + index)
        for index, name in enumerate(EIGEN_NAMES)
    }
    sbc.update(
        {
            name: _sbc(gap[..., index], gap_truth[..., index], lam_core, seed=132 + index)
            for index, name in enumerate(GAP_NAMES)
        }
    )
    spatial_values = {
        name: np.concatenate(parts) for name, parts in spatial_parts.items()
    }
    distance = spatial_values["pair_distance_voxels"]
    distance_edges = np.linspace(0.0, max(1.0, float(np.max(distance))), 13)
    distance_label = np.searchsorted(distance_edges[1:-1], distance, side="right")
    spatial_report = {"distance_edges_voxels": distance_edges.tolist(), "bins": []}
    for index in range(12):
        selected = distance_label == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            spatial_report["bins"].append({"pairs": 0})
            continue
        spatial_report["bins"].append(
            {
                "pairs": count,
                "truth_residual_covariance": float(np.mean(spatial_values["truth_residual_product"][selected])),
                "posterior_residual_covariance": float(np.mean(spatial_values["posterior_residual_covariance"][selected])),
                "truth_residual_variogram": float(np.mean(spatial_values["truth_residual_variogram"][selected])),
                "posterior_residual_variogram": float(np.mean(spatial_values["posterior_residual_variogram"][selected])),
            }
        )
    return {
        "schema_version": "p12f-dependency-rescue-evaluation-v2",
        "created_utc": utc_now(),
        "method": "gaussian_correlated_g1",
        "phase": "ph006",
        "cores": len(entries),
        "galaxies": int(len(lam_truth)),
        "voxel_rows": int(len(vox_truth)),
        "nested_draw_reports": reports,
        "subpanel_reports_256_draws": subpanels,
        "conditional_reports_256_draws": conditional,
        "sbc_256_draws": sbc,
        "spatial_dependence_256_draws": spatial_report,
        "spectral_dependence_256_draws": {
            "k_edges_cycles_per_voxel": spectral_edges.tolist(),
            "mode_count": spectral_count.tolist(),
            "truth_innovation_power": (spectral_truth / spectral_count).tolist(),
            "posterior_residual_power": (spectral_posterior / spectral_count).tolist(),
        },
        "physics_closure": {
            "maximum_trace_max_abs": max(row["trace_max_abs"] for row in physics),
            "maximum_trace_rmse": max(row["trace_rmse"] for row in physics),
            "all_finite": all(row["all_finite"] for row in physics),
            "all_ordered": all(row["all_ordered"] for row in physics),
            "additional_gaussian_smoothing": False,
        },
        "resampling_unit": "authoritative patch core",
        "local_patch_posterior_only": True,
        "full_cap_coherence_established": False,
        "ph001_opened": False,
        "truth_files_read": ["ph006 density/T-web from frozen sample archive"],
        "config_sha256": sha256(DEFAULT_CONFIG),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("expanded P12-F physics evaluation requires a compute GPU")
    config = json.loads(args.config.read_text())
    if config.get("experiment_version") != "P12-F v2 checkpoint-only evaluation sufficiency":
        raise RuntimeError("unexpected P12-F dependency-rescue contract")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F blind phase contract changed")
    evaluation_contract = config["evaluation_sufficiency"]
    matched = config["matched_contract"]
    if (
        tuple(evaluation_contract["nested_draw_counts"]) != DRAW_COUNTS
        or int(evaluation_contract["tarp_reference_seeds"]) != len(TARP_SEEDS)
        or int(evaluation_contract["disjoint_subpanels"]) != 4
        or int(matched["posterior_draws"]) != DRAW_COUNTS[-1]
        or int(matched["selection_panel_cores"]) != 1024
    ):
        raise RuntimeError("dependency-rescue evaluation constants changed")
    panel = json.loads(args.panel_marker.read_text())
    archive = json.loads(args.archive_manifest.read_text())
    if (
        panel.get("schema_version") != "p12f-truth-free-selection-panel-v1"
        or not panel.get("pass")
        or panel.get("selection_uses_truth")
        or panel.get("truth_files_read")
        or panel.get("ph001_opened")
        or len(panel.get("selected_core_id", [])) != 1024
    ):
        raise RuntimeError("expanded ph006 panel is not truth-free and complete")
    if archive.get("method") != "gaussian_correlated_g1":
        raise RuntimeError("dependency rescue stage 1 is frozen to G1")
    entries = validate_archive_manifest(
        archive,
        archive_path=args.archive_manifest,
        panel=panel,
        panel_path=args.panel_marker,
        config=config,
    )
    frozen = {
        "config_sha256": sha256(args.config),
        "archive_manifest_sha256": sha256(args.archive_manifest),
        "panel_marker_sha256": sha256(args.panel_marker),
        "archive_entries": [{"core_id": row["core_id"], "sha256": row["sha256"]} for row in entries],
        "draw_counts": list(DRAW_COUNTS),
        "tarp_seeds": list(TARP_SEEDS),
        "evaluator_sha256": sha256(Path(__file__)),
        "ph001_opened": False,
    }
    digest = canonical_digest(frozen)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "run_manifest.json"
    if run_path.exists():
        old = json.loads(run_path.read_text())
        if old.get("frozen_digest") != digest:
            raise RuntimeError("dependency-rescue evaluation contract changed")
    else:
        atomic_json(
            run_path,
            {
                "schema_version": "p12f-dependency-rescue-run-v2",
                "created_utc": utc_now(),
                "git_revision_at_launch": git_revision(),
                "frozen_digest": digest,
                "frozen": frozen,
                "ph001_opened": False,
            },
        )
    progress_path = args.output_root / "COMPACT_PROGRESS.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": "p12f-dependency-rescue-progress-v2",
            "frozen_digest": digest,
            "entries": [],
            "ph001_opened": False,
        }
    )
    if progress.get("frozen_digest") != digest or progress.get("ph001_opened"):
        raise RuntimeError("unsafe dependency-rescue progress marker")
    done = {int(row["core_id"]): row for row in progress["entries"]}
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    started = time.monotonic()
    for ordinal, entry in enumerate(entries):
        core_id = int(entry["core_id"])
        if core_id in done:
            path = Path(done[core_id]["compact_path"])
            if not path.is_file() or sha256(path) != done[core_id]["compact_sha256"]:
                raise RuntimeError("completed compact core changed before resume")
            continue
        record = load_core_record(entry, int(archive["draws"]))
        arrays, summary = derive_compact_core(
            metadata[core_id], record, method=str(archive["method"]), device=args.device
        )
        path = args.output_root / "compact" / f"core_{core_id:08d}.npz"
        atomic_npz(path, **arrays)
        row = {
            **summary,
            "source_sha256": entry["sha256"],
            "compact_path": str(path.resolve()),
            "compact_sha256": sha256(path),
        }
        progress["entries"].append(row)
        done[core_id] = row
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {"compact_core": ordinal + 1, "total": len(entries), "core_id": core_id},
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started >= args.max_wall_seconds:
            raise SystemExit(75)
    ordered = [done[int(row["core_id"])] for row in entries]
    ready_path = args.output_root / "COMPACT_ARCHIVE_READY.json"
    atomic_json(
        ready_path,
        {
            "schema_version": "p12f-dependency-rescue-compact-ready-v2",
            "created_utc": utc_now(),
            "frozen_digest": digest,
            "entries": ordered,
            "ph001_opened": False,
            "pass": True,
        },
    )
    if args.precompute_only:
        return
    report = aggregate_compact(ordered, config)
    report.update(
        {
            "frozen_digest": digest,
            "archive_manifest": str(args.archive_manifest.resolve()),
            "archive_manifest_sha256": sha256(args.archive_manifest),
            "panel_marker": str(args.panel_marker.resolve()),
            "panel_marker_sha256": sha256(args.panel_marker),
            "compact_archive_ready": str(ready_path.resolve()),
            "compact_archive_ready_sha256": sha256(ready_path),
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
        }
    )
    atomic_json(args.output_root / "P12F_DEPENDENCY_RESCUE_V2_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
