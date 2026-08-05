#!/usr/bin/env python3
"""Matched full-cap classical rows for the deterministic P8 benchmark.

The estimator sees the same immutable P3 CIC counts and the same rotation-specific,
training-only selection curve as U-PATCH.  It reconstructs a scalar overdensity,
applies the fixed R=7 Mpc/h tidal operator separately to NGC and SGC, samples the
tensor at the same P4 authoritative galaxies, and fits only a three-parameter
per-eigenvalue affine response on the registered training folds.

This script intentionally calls its primary estimator ``cic``.  Retaining Delaunay
tetrahedra and vertex densities is not sufficient to call a gridded approximation
piecewise-linear DTFE; an exact full-cap point-location/rasterisation implementation
must pass a separate parity gate before the name ``dtfe`` is used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    evaluate_complete_fold,
    fit_affine_on_training,
    fold_roles,
    sha256,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
FIELD_ADAPTER = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
SELECTION = FIELD_ADAPTER / "fullcap_selection_v1/selection_manifest.json"
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)
CAP_NAME = {0: "SGC", 1: "NGC"}
RSMOOTH_MPC = 7.0 / 0.6766


def _grid_coords(frac: np.ndarray, shape: tuple[int, int, int], device: str) -> torch.Tensor:
    norm = np.empty_like(frac, dtype=np.float32)
    for axis, size in enumerate(shape):
        norm[:, axis] = 2.0 * frac[:, axis] / max(size - 1, 1) - 1.0
    # grid_sample coordinate order is W,H,D = iz,iy,ix for a C,ix,iy,iz field.
    grid = np.ascontiguousarray(norm[:, (2, 1, 0)])
    return torch.from_numpy(grid).to(device).view(1, 1, 1, -1, 3)


def _load_delta_to_gpu(
    *,
    field_path: Path,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    cell_mpc: float,
    curve: dict,
    cosmology: dict,
    minimum_exposure: float,
    device: str,
    slab: int,
) -> tuple[torch.Tensor, dict]:
    """Derive the frozen-selection CIC contrast slabwise; never materialise it twice."""
    delta = torch.empty(shape, dtype=torch.float32, device=device)
    counts_gpu = torch.empty(shape, dtype=torch.float32, device=device)
    expected_gpu = torch.empty(shape, dtype=torch.float32, device=device)
    radius_grid = np.asarray(cosmology["radius_grid_mpc"], dtype=np.float64)
    redshift_grid = np.asarray(cosmology["redshift_grid"], dtype=np.float64)
    grid_z = np.asarray(curve["grid_z"], dtype=np.float64)
    ntilde = np.asarray(curve["ntilde"], dtype=np.float64)
    y = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell_mpc
    z = origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell_mpc
    supported_voxels = 0
    expected_sum = 0.0
    counts_sum = 0.0
    with h5py.File(field_path, "r") as handle:
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            counts = np.asarray(handle["counts"][left:right], dtype=np.float32)
            exposure = np.asarray(handle["exposure_apodized"][left:right], dtype=np.float32)
            x = origin[0] + (np.arange(left, right, dtype=np.float64) + 0.5) * cell_mpc
            radius = np.sqrt(
                x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2
            )
            redshift = np.interp(radius, radius_grid, redshift_grid)
            radial_density = np.interp(
                np.clip(redshift, grid_z[0], grid_z[-1]), grid_z, ntilde
            )
            expected = radial_density * cell_mpc**3 * exposure.astype(np.float64)
            supported = exposure > minimum_exposure
            counts_gpu[left:right].copy_(torch.from_numpy(counts).to(device))
            expected_gpu[left:right].copy_(torch.from_numpy(expected.astype(np.float32)).to(device))
            supported_voxels += int(supported.sum())
            expected_sum += float(expected[supported].sum(dtype=np.float64))
            counts_sum += float(counts[supported].sum(dtype=np.float64))
    # Established classical convention: define contrast only where the response is
    # greater than 5% of its positive-volume mean. This prevents the apodized survey
    # edge from producing enormous count/expected ratios.
    positive = expected_gpu > 0
    response_floor = 0.05 * expected_gpu[positive].mean()
    valid = expected_gpu > response_floor
    delta.zero_()
    delta[valid] = counts_gpu[valid] / expected_gpu[valid] - 1.0
    supported_voxels = int(valid.sum().item())
    return delta, {
        "supported_voxels": supported_voxels,
        "expected_sum": expected_sum,
        "counts_sum_supported": counts_sum,
        "expected_over_counts": expected_sum / max(counts_sum, 1e-30),
    }


def _sample_tidal_eigenvalues(
    delta: torch.Tensor,
    *,
    positions: np.ndarray,
    origin: np.ndarray,
    cell_mpc: float,
    padding_voxels: int,
    rsmooth_mpc: float,
) -> tuple[np.ndarray, dict]:
    """Zero-pad, solve six tensor components, and retain values only at galaxies."""
    device = str(delta.device)
    original_shape = tuple(int(v) for v in delta.shape)
    padding = int(padding_voxels)
    if padding:
        work = F.pad(delta[None, None], (padding,) * 6)[0, 0]
    else:
        work = delta
    shape = tuple(int(v) for v in work.shape)
    dk = torch.fft.rfftn(work)
    kx = torch.fft.fftfreq(shape[0], d=cell_mpc, device=device) * (2.0 * np.pi)
    ky = torch.fft.fftfreq(shape[1], d=cell_mpc, device=device) * (2.0 * np.pi)
    kz = torch.fft.rfftfreq(shape[2], d=cell_mpc, device=device) * (2.0 * np.pi)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    smooth = torch.exp(-0.5 * k2 * rsmooth_mpc**2)
    kernel = smooth / torch.where(k2 > 0, k2, torch.ones_like(k2))
    kernel[0, 0, 0] = 0.0
    frac = (positions - origin[None, :]) / cell_mpc - 0.5
    grid = _grid_coords(frac, original_shape, device)
    tensor = torch.empty((len(positions), 3, 3), dtype=torch.float32, device=device)
    axes = {"x": kx, "y": ky, "z": kz}
    axis_id = {"x": 0, "y": 1, "z": 2}
    maximum_memory = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    for name in ("xx", "xy", "xz", "yy", "yz", "zz"):
        left, right = name
        left_k = axes[left].reshape((-1, 1, 1) if left == "x" else ((1, -1, 1) if left == "y" else (1, 1, -1)))
        right_k = axes[right].reshape((-1, 1, 1) if right == "x" else ((1, -1, 1) if right == "y" else (1, 1, -1)))
        field = torch.fft.irfftn(dk * kernel * left_k * right_k, s=shape)
        if padding:
            field = field[
                padding : padding + original_shape[0],
                padding : padding + original_shape[1],
                padding : padding + original_shape[2],
            ]
        sampled = F.grid_sample(
            field[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0, 0, 0]
        i, j = axis_id[left], axis_id[right]
        tensor[:, i, j] = sampled
        tensor[:, j, i] = sampled
        maximum_memory = max(
            maximum_memory,
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        )
        del field, sampled
    # cuSOLVER's batched eigensolver requests a pathological workspace for several
    # million 3x3 matrices at once. Chunking is numerically identical and keeps the
    # workspace bounded on an 80-GB A100.
    tensor_cpu = tensor.cpu().numpy()
    eigenvalues = np.empty((len(positions), 3), dtype=np.float32)
    eig_chunk = 250_000
    for start in range(0, len(positions), eig_chunk):
        stop = min(start + eig_chunk, len(positions))
        eigenvalues[start:stop] = np.linalg.eigvalsh(
            tensor_cpu[start:stop]
        ).astype(np.float32)
    trace = tensor[:, 0, 0] + tensor[:, 1, 1] + tensor[:, 2, 2]
    diagnostics = {
        "maximum_cuda_memory_bytes": maximum_memory,
        "sampled_tensor_trace_std": float(trace.std().cpu()),
        "finite": bool(np.all(np.isfinite(eigenvalues))),
    }
    del tensor, trace, dk, kernel, smooth, k2, work, delta, grid
    torch.cuda.empty_cache()
    return eigenvalues, diagnostics


def run_rotation(rotation: int, args, manifests: dict) -> dict:
    started = time.time()
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
    raw = np.empty((len(parent), 3), dtype=np.float32)
    cap_reports = {}
    selection_row = manifests["selection"]["rotations"][str(rotation)]
    for cap in (0, 1):
        selected = cap_id == cap
        cap_name = CAP_NAME[cap]
        field_row = manifests["adapter"]["caps"][cap_name]
        shape = tuple(int(v) for v in field_row["shape"])
        delta, response = _load_delta_to_gpu(
            field_path=Path(field_row["field_path"]),
            shape=shape,
            origin=np.asarray(field_row["origin_mpc"], dtype=np.float64),
            cell_mpc=float(field_row["cell_mpc"]),
            curve=selection_row["caps"][cap_name],
            cosmology=manifests["selection"]["cosmology"],
            minimum_exposure=float(manifests["selection"]["contrast"]["minimum_exposure"]),
            device=args.device,
            slab=args.slab,
        )
        prediction, fft = _sample_tidal_eigenvalues(
            delta,
            positions=positions[selected],
            origin=np.asarray(field_row["origin_mpc"], dtype=np.float64),
            cell_mpc=float(field_row["cell_mpc"]),
            padding_voxels=args.padding_voxels,
            rsmooth_mpc=args.rsmooth_mpc,
        )
        raw[selected] = prediction
        cap_reports[cap_name] = {
            "n_sampled": int(selected.sum()),
            "response": response,
            "fft": fft,
            "grid_shape": list(shape),
        }
    calibrated, affine = fit_affine_on_training(raw, np.asarray(truth[parent]), train)
    validation_parent = parent[validation]
    validation_raw = raw[validation]
    validation_calibrated = calibrated[validation].astype(np.float32)
    runtime = {
        "elapsed_seconds": time.time() - started,
        "device": args.device,
        "padding_voxels": int(args.padding_voxels),
    }
    raw_report = evaluate_complete_fold(
        parent_node_id=validation_parent,
        predicted_eigenvalues=validation_raw,
        truth_by_parent=truth,
        assignment=assignment,
        validation_fold=validation_fold,
        runtime=runtime,
    )
    calibrated_report = evaluate_complete_fold(
        parent_node_id=validation_parent,
        predicted_eigenvalues=validation_calibrated,
        truth_by_parent=truth,
        assignment=assignment,
        validation_fold=validation_fold,
        runtime=runtime,
    )
    output = args.p8_root / "classical" / f"rotation_{rotation}"
    output.mkdir(parents=True, exist_ok=True)
    # The validation-only files remain the immutable comparison row.  The keyed
    # active-fold files are the frozen classical anchor for a later learned
    # residual; they contain no test-fold rows and retain the same training-only
    # affine map used by the comparison.
    np.save(output / "active_parent_node_id.npy", parent)
    np.save(output / "active_is_training.npy", train)
    np.save(output / "active_is_validation.npy", validation)
    np.save(output / "cic_train_affine_active_eigenvalues.npy", calibrated.astype(np.float32))
    np.save(output / "validation_parent_node_id.npy", validation_parent)
    np.save(output / "cic_raw_eigenvalues.npy", validation_raw)
    np.save(output / "cic_train_affine_eigenvalues.npy", validation_calibrated)
    report = {
        "schema_version": 1,
        "estimator": "cic",
        "rotation": int(rotation),
        "validation_fold": int(validation_fold),
        "train_folds": list(train_folds),
        "affine": affine,
        "raw": raw_report,
        "train_affine": calibrated_report,
        "caps": cap_reports,
        "inputs": {
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "points": str(args.points),
            "selection_manifest": str(args.selection),
            "selection_manifest_sha256": sha256(args.selection),
        },
        "residual_anchor": {
            "parent_node_id": str(output / "active_parent_node_id.npy"),
            "train_affine_eigenvalues": str(
                output / "cic_train_affine_active_eigenvalues.npy"
            ),
            "scope": "registered training plus validation folds only",
        },
        "dtfe_status": (
            "not claimed: exact piecewise-linear full-cap point location is not yet "
            "implemented; historical wedge DTFE is not adoption-eligible"
        ),
        "runtime": runtime,
    }
    atomic_json(output / "cic_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--field-adapter", type=Path, default=FIELD_ADAPTER)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--screen-rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P8 full-cap classical FFT requires a CUDA allocation")
    adapter = json.loads((args.field_adapter / "adapter_manifest.json").read_text())
    selection = json.loads(args.selection.read_text())
    reports = [
        run_rotation(rotation, args, {"adapter": adapter, "selection": selection})
        for rotation in args.screen_rotations
    ]
    primary = [row["train_affine"]["primary_macro_r2_lambda1"] for row in reports]
    summary = {
        "schema_version": 1,
        "stage": "P8 matched classical full-cap screen",
        "screen_rotations": list(args.screen_rotations),
        "estimator": "cic",
        "calibration": "three scalar affine maps fit on registered training folds only",
        "primary_score_mean": float(np.mean(primary)),
        "primary_score_by_rotation": primary,
        "reports": [
            str(args.p8_root / "classical" / f"rotation_{r}" / "cic_report.json")
            for r in args.screen_rotations
        ],
        "dtfe_adoption_row_ready": False,
        "dtfe_note": "exact full-cap DTFE remains a separately named implementation gate",
    }
    output = args.p8_root / "classical" / "classical_summary.json"
    atomic_json(output, summary)
    (output.parent / "P8_CLASSICAL_CIC_READY").write_text(
        f"mean_macro_r2_lambda1={summary['primary_score_mean']:.8f}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
