#!/usr/bin/env python3
"""Matched CIC controls for the P8 Bright-target/Faint-context experiment.

This is MT3a, not the complete MT3 gate.  It evaluates two estimators on the
same authoritative BGS_BRIGHT rows as the frozen Bright-only P8 CIC baseline:

``combined_cic``
    Treat Bright and Faint as equal-response tracers and form
    ``(N_B + N_F) / (mu_B + mu_F) - 1``.

``bias_aware_cic``
    Estimate the relative large-scale tracer response from Gaussian-smoothed
    Bright/Faint fields in the registered training folds only, normalize the
    Faint contrast to the Bright response, and combine the two using local
    inverse-Poisson-variance weights.

The target, folds, R=7 Mpc/h tidal operator, sampled Bright galaxies, and
training-only affine calibration are unchanged.  No validation label enters
the tracer-response fit.  TSC, density-matched thinning, and the Faint-position
null are separate MT3b controls and are deliberately not stamped complete by
this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_refit_fullcap_selection import build_cap_lookup
from workflows.abacus_tweb.p8_classical_fullcap import (
    CAP_NAME,
    RSMOOTH_MPC,
    _sample_tidal_eigenvalues,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    evaluate_complete_fold,
    fit_affine_on_training,
    fold_roles,
    sha256,
)


MT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P3_MANIFEST = Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json")
BRIGHT_SELECTION = Path(
    "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
    "fullcap_selection_v1/selection_manifest.json"
)
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
CORES = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/cores.npz")
P4_MANIFEST = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/spatial_manifest.json")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)


def response_valid(expected: torch.Tensor, fraction: float = 0.05) -> tuple[torch.Tensor, float]:
    """Return the frozen classical response-floor mask and threshold."""
    positive = expected > 0
    if not bool(positive.any()):
        raise RuntimeError("expected-count field has no positive support")
    floor = float((fraction * expected[positive].mean()).item())
    return expected > floor, floor


def combined_count_contrast(
    counts_b: torch.Tensor,
    expected_b: torch.Tensor,
    counts_f: torch.Tensor,
    expected_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Equal-response combined-count contrast with the frozen response floor."""
    counts = counts_b + counts_f
    expected = expected_b + expected_f
    valid, floor = response_valid(expected)
    delta = torch.zeros_like(expected)
    delta[valid] = counts[valid] / expected[valid] - 1.0
    return delta, valid, floor


def fit_relative_bias(
    smooth_b: torch.Tensor,
    smooth_f: torch.Tensor,
    training_mask: torch.Tensor,
) -> dict:
    """Symmetric relative-response fit using label-free training voxels only."""
    x = smooth_b[training_mask].to(torch.float64)
    y = smooth_f[training_mask].to(torch.float64)
    if x.numel() < 100:
        raise RuntimeError("fewer than 100 common training voxels for bias fit")
    x = x - x.mean()
    y = y - y.mean()
    var_b = torch.mean(x * x)
    var_f = torch.mean(y * y)
    covariance = torch.mean(x * y)
    if float(var_b) <= 0 or float(var_f) <= 0:
        raise RuntimeError("non-positive tracer variance in bias fit")
    correlation = float((covariance / torch.sqrt(var_b * var_f)).item())
    relative = float(torch.sqrt(var_f / var_b).item())
    if not np.isfinite(relative) or relative <= 0 or correlation <= 0:
        raise RuntimeError(
            f"invalid relative tracer response: q={relative}, corr={correlation}"
        )
    return {
        "relative_bias_faint_over_bright": relative,
        "bright_variance": float(var_b.item()),
        "faint_variance": float(var_f.item()),
        "cross_covariance": float(covariance.item()),
        "correlation": correlation,
        "training_voxels": int(training_mask.sum().item()),
        "fit": "symmetric RMS response ratio after R=7 Mpc/h Gaussian smoothing",
        "labels_used": False,
    }


def bias_aware_contrast(
    delta_b: torch.Tensor,
    expected_b: torch.Tensor,
    valid_b: torch.Tensor,
    delta_f: torch.Tensor,
    expected_f: torch.Tensor,
    valid_f: torch.Tensor,
    relative_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two tracer contrasts after normalizing Faint to Bright response."""
    q = float(relative_bias)
    weight_b = torch.where(valid_b, expected_b, torch.zeros_like(expected_b))
    # Var(delta_F / q) ~= 1 / (q^2 mu_F), so its inverse variance is q^2 mu_F.
    weight_f = torch.where(valid_f, q * q * expected_f, torch.zeros_like(expected_f))
    denominator = weight_b + weight_f
    valid = denominator > 0
    numerator = weight_b * delta_b
    numerator = numerator + torch.where(
        valid_f, q * expected_f * delta_f, torch.zeros_like(delta_f)
    )
    result = torch.zeros_like(delta_b)
    result[valid] = numerator[valid] / denominator[valid]
    return result, valid


def gaussian_smooth(
    delta: torch.Tensor, *, cell_mpc: float, rsmooth_mpc: float, padding_voxels: int
) -> torch.Tensor:
    """Apply the scalar part of the registered padded R=7 Mpc/h FFT operator."""
    original_shape = tuple(int(value) for value in delta.shape)
    padding = int(padding_voxels)
    work = F.pad(delta[None, None], (padding,) * 6)[0, 0] if padding else delta
    shape = tuple(int(value) for value in work.shape)
    kx = torch.fft.fftfreq(shape[0], d=cell_mpc, device=delta.device) * (2.0 * np.pi)
    ky = torch.fft.fftfreq(shape[1], d=cell_mpc, device=delta.device) * (2.0 * np.pi)
    kz = torch.fft.rfftfreq(shape[2], d=cell_mpc, device=delta.device) * (2.0 * np.pi)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    window = torch.exp(-0.5 * k2 * rsmooth_mpc**2)
    smooth = torch.fft.irfftn(torch.fft.rfftn(work) * window, s=shape)
    if padding:
        smooth = smooth[
            padding : padding + original_shape[0],
            padding : padding + original_shape[1],
            padding : padding + original_shape[2],
        ]
    return smooth


def fold_block(
    *, left: int, right: int, shape: tuple[int, int, int], origin: np.ndarray,
    cell_mpc: float, base_mpc: np.ndarray, core_mpc: float, lookup: np.ndarray,
) -> np.ndarray:
    """Map one field slab to immutable P4 folds without materialising XYZ grids."""
    axes = (
        origin[0] + (np.arange(left, right, dtype=np.float64) + 0.5) * cell_mpc,
        origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell_mpc,
        origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell_mpc,
    )
    indices = [np.floor((axis - base_mpc[i]) / core_mpc).astype(np.int64) for i, axis in enumerate(axes)]
    valid_axes = [(index >= 0) & (index < lookup.shape[i]) for i, index in enumerate(indices)]
    valid = valid_axes[0][:, None, None] & valid_axes[1][None, :, None] & valid_axes[2][None, None, :]
    result = np.full((right - left, shape[1], shape[2]), -1, dtype=np.int8)
    ix = np.clip(indices[0], 0, lookup.shape[0] - 1)
    iy = np.clip(indices[1], 0, lookup.shape[1] - 1)
    iz = np.clip(indices[2], 0, lookup.shape[2] - 1)
    mapped = lookup[ix[:, None, None], iy[None, :, None], iz[None, None, :]]
    result[valid] = mapped[valid]
    return result


def load_fields(
    *, bright_path: Path, faint_path: Path, shape: tuple[int, int, int],
    origin: np.ndarray, cell_mpc: float, bright_curve: dict, faint_curve: dict,
    cosmology: dict, lookup_row: dict, core_mpc: float, train_folds: tuple[int, ...],
    device: str, slab: int,
) -> tuple[dict, dict]:
    """Load two response-matched tracer fields and the training-fold voxel mask."""
    tensors = {
        name: torch.empty(shape, dtype=torch.float32, device=device)
        for name in ("counts_b", "expected_b", "counts_f", "expected_f")
    }
    training = torch.empty(shape, dtype=torch.bool, device=device)
    radius_grid = np.asarray(cosmology["radius_grid_mpc"], dtype=np.float64)
    redshift_grid = np.asarray(cosmology["redshift_grid"], dtype=np.float64)
    y = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell_mpc
    z = origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell_mpc
    curves = {
        "b": (
            np.asarray(bright_curve["grid_z"], dtype=np.float64),
            np.asarray(bright_curve["ntilde"], dtype=np.float64),
        ),
        "f": (
            np.asarray(faint_curve["grid_z"], dtype=np.float64),
            np.asarray(faint_curve["ntilde"], dtype=np.float64),
        ),
    }
    sums = {"counts_b": 0.0, "expected_b": 0.0, "counts_f": 0.0, "expected_f": 0.0}
    with h5py.File(bright_path, "r") as bright, h5py.File(faint_path, "r") as faint:
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            x = origin[0] + (np.arange(left, right, dtype=np.float64) + 0.5) * cell_mpc
            radius = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2)
            redshift = np.interp(radius, radius_grid, redshift_grid)
            counts_b = np.asarray(bright["counts"][left:right], dtype=np.float32)
            counts_f = np.asarray(faint["counts"][left:right], dtype=np.float32)
            exposure_b = np.asarray(bright["exposure_apodized"][left:right], dtype=np.float32)
            exposure_f = np.asarray(faint["exposure_apodized"][left:right], dtype=np.float32)
            expected = {}
            for key, exposure in (("b", exposure_b), ("f", exposure_f)):
                grid_z, ntilde = curves[key]
                radial = np.interp(np.clip(redshift, grid_z[0], grid_z[-1]), grid_z, ntilde)
                expected[key] = (radial * cell_mpc**3 * exposure.astype(np.float64)).astype(np.float32)
            fold = fold_block(
                left=left, right=right, shape=shape, origin=origin, cell_mpc=cell_mpc,
                base_mpc=np.asarray(lookup_row["base_mpc"], dtype=np.float64),
                core_mpc=core_mpc, lookup=np.asarray(lookup_row["lookup"], dtype=np.int8),
            )
            arrays = {
                "counts_b": counts_b, "expected_b": expected["b"],
                "counts_f": counts_f, "expected_f": expected["f"],
            }
            for name, value in arrays.items():
                tensors[name][left:right].copy_(torch.from_numpy(value).to(device))
                sums[name] += float(value.sum(dtype=np.float64))
            training[left:right].copy_(torch.from_numpy(np.isin(fold, train_folds)).to(device))
    return {**tensors, "training": training}, sums


def evaluate_estimator(
    *, name: str, raw: np.ndarray, parent: np.ndarray, train: np.ndarray,
    validation: np.ndarray, truth: np.ndarray, assignment, validation_fold: int,
    output: Path, runtime: dict, extra: dict,
) -> dict:
    calibrated, affine = fit_affine_on_training(raw, np.asarray(truth[parent]), train)
    validation_parent = parent[validation]
    raw_report = evaluate_complete_fold(
        parent_node_id=validation_parent, predicted_eigenvalues=raw[validation],
        truth_by_parent=truth, assignment=assignment, validation_fold=validation_fold,
        runtime=runtime,
    )
    calibrated_report = evaluate_complete_fold(
        parent_node_id=validation_parent,
        predicted_eigenvalues=calibrated[validation].astype(np.float32),
        truth_by_parent=truth, assignment=assignment, validation_fold=validation_fold,
        runtime=runtime,
    )
    estimator_dir = output / name
    estimator_dir.mkdir(parents=True, exist_ok=True)
    np.save(estimator_dir / "validation_parent_node_id.npy", validation_parent)
    np.save(estimator_dir / "raw_eigenvalues.npy", raw[validation].astype(np.float32))
    np.save(estimator_dir / "train_affine_eigenvalues.npy", calibrated[validation].astype(np.float32))
    report = {
        "schema_version": "p8-mt3a-estimator-v1", "estimator": name,
        "affine": affine, "raw": raw_report, "train_affine": calibrated_report,
        "runtime": runtime, **extra,
    }
    atomic_json(estimator_dir / "report.json", report)
    return report


def run_rotation(rotation: int, args, manifests: dict) -> dict:
    started = time.time()
    print(f"[MT3a] rotation={rotation} loading benchmark ownership", flush=True)
    assignment = np.load(args.assignment, mmap_mode="r")
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    points = np.load(args.points, mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    train_folds, validation_fold, _ = fold_roles(rotations, rotation)
    auth = authoritative_mask(assignment)
    row_fold = np.asarray(assignment["fold"], dtype=np.int8)
    active_rows = np.flatnonzero(auth & np.isin(row_fold, (*train_folds, validation_fold)))
    parent = np.asarray(assignment["parent_node_id"][active_rows], dtype=np.int64)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("P8 authoritative parent IDs are not unique")
    train = np.isin(row_fold[active_rows], train_folds)
    validation = row_fold[active_rows] == validation_fold
    cap_id = np.asarray(points[parent, 3], dtype=np.int8)
    positions = np.asarray(points[parent, :3], dtype=np.float64)
    raw = {
        "combined_cic": np.empty((len(parent), 3), dtype=np.float32),
        "bias_aware_cic": np.empty((len(parent), 3), dtype=np.float32),
    }
    cap_reports = {}
    cores = np.load(args.cores, mmap_mode="r")
    p4_manifest = json.loads(args.p4_manifest.read_text())
    core_mpc = float(p4_manifest["unit_contract"]["core_mpc"])
    lookups = {cap: build_cap_lookup(cores, cap, core_mpc) for cap in (0, 1)}

    bright_rotation = manifests["bright_selection"]["rotations"][str(rotation)]
    faint_rotation = manifests["mt_selection"]["tracers"]["BGS_FAINT"]["rotations"][str(rotation)]
    for cap in (0, 1):
        selected = cap_id == cap
        cap_name = CAP_NAME[cap]
        print(
            f"[MT3a] rotation={rotation} cap={cap_name} loading response-matched fields",
            flush=True,
        )
        grid = manifests["p3"]["components"][cap_name]["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell_mpc = float(grid["cell_mpc"])
        fields, field_sums = load_fields(
            bright_path=Path(manifests["p3"]["components"][cap_name]["file"]),
            faint_path=Path(manifests["mt_fields"]["components"][cap_name]["file"]),
            shape=shape, origin=origin, cell_mpc=cell_mpc,
            bright_curve=bright_rotation["caps"][cap_name],
            faint_curve=faint_rotation["caps"][cap_name],
            cosmology=manifests["bright_selection"]["cosmology"],
            lookup_row=lookups[cap], core_mpc=core_mpc,
            train_folds=tuple(train_folds), device=args.device, slab=args.slab,
        )
        valid_b, floor_b = response_valid(fields["expected_b"])
        valid_f, floor_f = response_valid(fields["expected_f"])
        delta_b = torch.zeros_like(fields["expected_b"])
        delta_f = torch.zeros_like(fields["expected_f"])
        delta_b[valid_b] = fields["counts_b"][valid_b] / fields["expected_b"][valid_b] - 1.0
        delta_f[valid_f] = fields["counts_f"][valid_f] / fields["expected_f"][valid_f] - 1.0
        combined, valid_combined, floor_combined = combined_count_contrast(
            fields["counts_b"], fields["expected_b"], fields["counts_f"], fields["expected_f"]
        )
        smooth_b = gaussian_smooth(
            delta_b, cell_mpc=cell_mpc, rsmooth_mpc=args.rsmooth_mpc,
            padding_voxels=args.padding_voxels,
        )
        smooth_f = gaussian_smooth(
            delta_f, cell_mpc=cell_mpc, rsmooth_mpc=args.rsmooth_mpc,
            padding_voxels=args.padding_voxels,
        )
        bias = fit_relative_bias(
            smooth_b, smooth_f, fields["training"] & valid_b & valid_f
        )
        print(
            f"[MT3a] rotation={rotation} cap={cap_name} "
            f"q_F_over_B={bias['relative_bias_faint_over_bright']:.6f} "
            f"corr={bias['correlation']:.6f}",
            flush=True,
        )
        del smooth_b, smooth_f
        bias_delta, valid_bias = bias_aware_contrast(
            delta_b, fields["expected_b"], valid_b,
            delta_f, fields["expected_f"], valid_f,
            bias["relative_bias_faint_over_bright"],
        )
        fft_reports = {}
        for estimator, delta in (("combined_cic", combined), ("bias_aware_cic", bias_delta)):
            print(
                f"[MT3a] rotation={rotation} cap={cap_name} estimator={estimator} FFT",
                flush=True,
            )
            prediction, fft = _sample_tidal_eigenvalues(
                delta, positions=positions[selected], origin=origin,
                cell_mpc=cell_mpc, padding_voxels=args.padding_voxels,
                rsmooth_mpc=args.rsmooth_mpc,
            )
            raw[estimator][selected] = prediction
            fft_reports[estimator] = fft
        cap_reports[cap_name] = {
            "n_sampled": int(selected.sum()), "shape": list(shape),
            "response_floor": {"bright": floor_b, "faint": floor_f, "combined": floor_combined},
            "supported_voxels": {
                "bright": int(valid_b.sum().item()), "faint": int(valid_f.sum().item()),
                "combined": int(valid_combined.sum().item()),
                "bias_aware": int(valid_bias.sum().item()),
            },
            "field_sums": field_sums, "relative_bias_fit": bias, "fft": fft_reports,
        }
        del fields, delta_b, delta_f, combined, bias_delta
        torch.cuda.empty_cache()

    output = args.root / "classical/mt3a_cic" / f"rotation_{rotation}"
    output.mkdir(parents=True, exist_ok=True)
    runtime = {
        "elapsed_seconds": time.time() - started, "device": args.device,
        "padding_voxels": int(args.padding_voxels), "rsmooth_mpc": float(args.rsmooth_mpc),
    }
    reports = {
        name: evaluate_estimator(
            name=name, raw=value, parent=parent, train=train, validation=validation,
            truth=truth, assignment=assignment, validation_fold=validation_fold,
            output=output, runtime=runtime,
            extra={
                "rotation": int(rotation), "train_folds": list(train_folds),
                "validation_fold": int(validation_fold), "caps": cap_reports,
                "supervised_and_evaluated_population": "frozen BGS_BRIGHT authoritative rows only",
                "faint_labels_used": False,
            },
        )
        for name, value in raw.items()
    }
    rotation_report = {
        "schema_version": "p8-mt3a-cic-rotation-v1", "rotation": int(rotation),
        "train_folds": list(train_folds), "validation_fold": int(validation_fold),
        "estimators": {
            name: {
                "report": str(output / name / "report.json"),
                "primary_macro_r2_lambda1": report["train_affine"]["primary_macro_r2_lambda1"],
            }
            for name, report in reports.items()
        },
        "inputs": manifests["input_records"], "runtime": runtime,
    }
    atomic_json(output / "rotation_report.json", rotation_report)
    print(
        f"[MT3a] rotation={rotation} complete elapsed={runtime['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return rotation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MT_ROOT)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p3-manifest", type=Path, default=P3_MANIFEST)
    parser.add_argument("--bright-selection", type=Path, default=BRIGHT_SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--cores", type=Path, default=CORES)
    parser.add_argument("--p4-manifest", type=Path, default=P4_MANIFEST)
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument("--screen-rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MT3a full-cap CIC controls require a CUDA allocation")
    paths = {
        "p3_manifest": args.p3_manifest,
        "bright_selection": args.bright_selection,
        "mt_fields": args.root / "fields" / args.product / "manifest.json",
        "mt_selection": args.root / "selection" / args.product / "multitracer_selection_manifest.json",
        "assignment": args.assignment, "rotations": args.rotations,
        "cores": args.cores, "p4_manifest": args.p4_manifest, "points": args.points,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    manifests = {
        "p3": json.loads(args.p3_manifest.read_text()),
        "bright_selection": json.loads(args.bright_selection.read_text()),
        "mt_fields": json.loads(paths["mt_fields"].read_text()),
        "mt_selection": json.loads(paths["mt_selection"].read_text()),
        "input_records": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
    }
    if not manifests["mt_fields"].get("pass") or not manifests["mt_selection"].get("pass"):
        raise RuntimeError("passing multitracer field and selection manifests required")
    reports = [run_rotation(rotation, args, manifests) for rotation in args.screen_rotations]
    bright_summary_path = args.p8_root / "classical/classical_summary.json"
    bright_summary = json.loads(bright_summary_path.read_text())
    summary = {
        "schema_version": "p8-mt3a-cic-summary-v1",
        "stage": "MT3a matched multitracer CIC controls",
        "status": "complete_mt3a_only",
        "product": args.product,
        "screen_rotations": list(args.screen_rotations),
        "bright_only_frozen_reference": {
            "path": str(bright_summary_path), "sha256": sha256(bright_summary_path),
            "primary_score_by_rotation": bright_summary["primary_score_by_rotation"],
        },
        "estimators": {},
        "remaining_mt3b": ["exact TSC", "density-matched thinning (three seeds)", "Faint-position null"],
        "neural_training_unlocked": False,
        "rotation_reports": [
            str(args.root / "classical/mt3a_cic" / f"rotation_{rotation}" / "rotation_report.json")
            for rotation in args.screen_rotations
        ],
    }
    bright_reference_by_rotation = dict(
        zip(
            bright_summary["screen_rotations"],
            bright_summary["primary_score_by_rotation"],
        )
    )
    for estimator in ("combined_cic", "bias_aware_cic"):
        scores = [row["estimators"][estimator]["primary_macro_r2_lambda1"] for row in reports]
        reference = np.asarray(
            [bright_reference_by_rotation[rotation] for rotation in args.screen_rotations],
            dtype=np.float64,
        )
        summary["estimators"][estimator] = {
            "primary_score_by_rotation": scores,
            "primary_score_mean": float(np.mean(scores)),
            "delta_vs_bright_by_rotation": (np.asarray(scores) - reference).tolist(),
        }
    output = args.root / "classical/mt3a_cic/summary.json"
    atomic_json(output, summary)
    (output.parent / "MT3A_CIC_COMPLETE").write_text(
        f"summary_sha256={sha256(output)}\nneural_training_unlocked=false\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
