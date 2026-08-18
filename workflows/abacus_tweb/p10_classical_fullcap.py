#!/usr/bin/env python3
"""Matched multi-phase CIC/DTFE baselines for P10 Arm A.

Each visible Abacus phase is reconstructed independently from either its immutable
P3 full-cap count/expected-count fields (CIC) or an exact piecewise-linear DTFE
density raster built from the immutable P2 Delaunay tessellation.  The fixed
R=7 Mpc/h tidal operator is applied separately to NGC and SGC and sampled at exactly
all authoritative P4 galaxies.  Scalar affine response maps are fitted once on
ph000+ph002--ph005 with the frozen P10 phase/shell weights and then applied
unchanged to ph006.  ph001 is rejected by construction.

The workflow is staged so independent phase reconstructions can occupy separate
GPUs in one interactive Perlmutter node::

    p10_classical_fullcap.py raw --phase ph000 --estimator cic
    ...
    p10_classical_fullcap.py raw --phase ph006 --estimator cic
    p10_classical_fullcap.py finalize --estimator cic

The same commands with ``--estimator dtfe`` consume phase-local
``DTFE_FIELD_READY`` rasters.  ph001 is rejected by construction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_classical_fullcap import _sample_tidal_eigenvalues
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    evaluate_complete_phase,
    sha256,
)
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader


CONTRACT_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract"
)
CLASSICAL_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/classical"
)
CAP_NAME = {0: "SGC", 1: "NGC"}
VISIBLE_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
RSMOOTH_MPC = 7.0 / 0.6766


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stored_delta_to_gpu(
    field_path: Path,
    *,
    device: str,
    slab: int,
    response_floor_fraction: float,
) -> tuple[torch.Tensor, dict]:
    """Load the stored P3 response and form the historical CIC contrast."""
    with h5py.File(field_path, "r") as handle:
        shape = tuple(int(value) for value in handle["counts"].shape)
        expected_sum = 0.0
        expected_positive = 0
        counts_sum = 0.0
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            expected = np.asarray(
                handle["expected_counts"][left:right], dtype=np.float32
            )
            positive = expected > 0
            expected_sum += float(expected[positive].sum(dtype=np.float64))
            expected_positive += int(positive.sum())
        if expected_positive == 0:
            raise RuntimeError(f"{field_path} contains no positive expected response")
        response_floor = response_floor_fraction * expected_sum / expected_positive
        delta = torch.zeros(shape, dtype=torch.float32, device=device)
        valid_voxels = 0
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            counts = np.asarray(handle["counts"][left:right], dtype=np.float32)
            expected = np.asarray(
                handle["expected_counts"][left:right], dtype=np.float32
            )
            valid = expected > response_floor
            values = np.zeros_like(counts, dtype=np.float32)
            values[valid] = counts[valid] / expected[valid] - 1.0
            delta[left:right].copy_(torch.from_numpy(values).to(device))
            valid_voxels += int(valid.sum())
            counts_sum += float(counts[valid].sum(dtype=np.float64))
    return delta, {
        "field_path": str(field_path),
        "field_sha256": sha256(field_path),
        "shape": list(shape),
        "response_floor_fraction": float(response_floor_fraction),
        "response_floor": float(response_floor),
        "valid_voxels": valid_voxels,
        "expected_positive_voxels": expected_positive,
        "expected_sum_positive": expected_sum,
        "counts_sum_valid": counts_sum,
    }


def _dtfe_delta_to_gpu(
    density_path: Path,
    field_path: Path,
    *,
    cell_mpc: float,
    device: str,
    slab: int,
) -> tuple[torch.Tensor, dict]:
    """Convert exact DTFE number density to the P8 response-aware contrast.

    The DTFE raster has units of galaxies/Mpc^3.  P3 ``expected_counts`` has
    units of galaxies/voxel and already contains the phase-local radial response
    and apodized angular exposure, hence ``expected_counts / cell_mpc**3`` is the
    matched expected number density.  This is algebraically the same convention
    as the P8 ``ntilde(z) * exposure`` evaluator but avoids a second selection fit.
    """
    density = np.load(density_path, mmap_mode="r")
    with h5py.File(field_path, "r") as handle:
        shape = tuple(int(value) for value in handle["expected_counts"].shape)
    if tuple(density.shape) != shape:
        raise RuntimeError(
            f"DTFE/field shape mismatch: {density_path} {density.shape} != {shape}"
        )
    delta = torch.zeros(shape, dtype=torch.float32, device=device)
    used = 0
    finite_supported = 0
    expected_density_sum = 0.0
    with h5py.File(field_path, "r") as handle:
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            rho = np.asarray(density[left:right], dtype=np.float32)
            expected = np.asarray(
                handle["expected_counts"][left:right], dtype=np.float32
            )
            exposure = np.asarray(
                handle["exposure_apodized"][left:right], dtype=np.float32
            )
            nbar = expected / float(cell_mpc) ** 3
            valid = np.isfinite(rho) & (nbar > 0) & (exposure > 0)
            values = np.zeros_like(rho, dtype=np.float32)
            values[valid] = (
                np.clip(rho[valid] / nbar[valid] - 1.0, -1.0, 200.0)
                * exposure[valid]
            )
            delta[left:right].copy_(torch.from_numpy(values).to(device))
            used += int(valid.sum())
            finite_supported += int(np.sum(np.isfinite(rho) & (exposure > 0)))
            expected_density_sum += float(nbar[valid].sum(dtype=np.float64))
    return delta, {
        "density_path": str(density_path),
        "density_sha256": sha256(density_path),
        "field_path": str(field_path),
        "field_sha256": sha256(field_path),
        "shape": list(shape),
        "used_voxels": used,
        "finite_supported_voxels": finite_supported,
        "expected_density_sum_used": expected_density_sum,
        "contrast": "clip(rho_dtfe/(expected_counts/cell^3)-1,-1,200)*exposure",
    }


def _phase_paths(loader: P10PhaseBalancedLoader, phase: str) -> dict:
    if phase not in VISIBLE_PHASES or phase == loader.blind_phase:
        raise RuntimeError(f"phase {phase!r} is not visible to P10 classical evaluation")
    record = loader.phase_records[phase]
    field_manifest_path = Path(
        json.loads((loader.root / "adapter_inventory.json").read_text())["phases"][phase][
            "field_manifest"
        ]
    )
    field_manifest = json.loads(field_manifest_path.read_text())
    p1_manifest_path = Path(record["inputs"]["p1_manifest"])
    p1_manifest = json.loads(p1_manifest_path.read_text())
    return {
        "assignment": Path(record["inputs"]["assignment"]),
        "truth": Path(record["target"]["path"]),
        "points": Path(p1_manifest["points"]),
        "field_manifest_path": field_manifest_path,
        "field_manifest": field_manifest,
    }


def raw_phase(args: argparse.Namespace) -> dict:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P10 classical full-cap reconstruction requires a CUDA allocation")
    loader = P10PhaseBalancedLoader(args.contract_root, include_blind=False)
    paths = _phase_paths(loader, args.phase)
    output = args.output_root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f"{args.estimator.upper()}_RAW_COMPLETE.json"
    if marker.exists():
        row = json.loads(marker.read_text())
        print(json.dumps(row, indent=2), flush=True)
        return row

    started = time.time()
    assignment = np.load(paths["assignment"], mmap_mode="r")
    auth_rows = np.flatnonzero(authoritative_mask(assignment))
    parent = np.asarray(assignment["parent_node_id"][auth_rows], dtype=np.int64)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError(f"{args.phase} authoritative parent IDs are not unique")
    points = np.load(paths["points"], mmap_mode="r")
    positions = np.asarray(points[parent, :3], dtype=np.float64)
    cap_id = np.asarray(points[parent, 3], dtype=np.int8)
    raw = np.empty((len(parent), 3), dtype=np.float32)
    cap_reports = {}
    torch.cuda.reset_peak_memory_stats()
    for cap in (0, 1):
        selected = cap_id == cap
        name = CAP_NAME[cap]
        cap_row = paths["field_manifest"]["caps"][name]
        if args.estimator == "cic":
            delta, response = _stored_delta_to_gpu(
                Path(cap_row["field_path"]),
                device=args.device,
                slab=args.slab,
                response_floor_fraction=args.response_floor_fraction,
            )
        else:
            dtfe_root = args.dtfe_build_root / args.phase
            ready = dtfe_root / "DTFE_FIELD_READY"
            if not ready.exists():
                raise RuntimeError(f"missing exact DTFE phase marker: {ready}")
            delta, response = _dtfe_delta_to_gpu(
                dtfe_root / f"dtfe_density_{name}.npy",
                Path(cap_row["field_path"]),
                cell_mpc=float(cap_row["cell_mpc"]),
                device=args.device,
                slab=args.slab,
            )
        prediction, fft = _sample_tidal_eigenvalues(
            delta,
            positions=positions[selected],
            origin=np.asarray(cap_row["origin_mpc"], dtype=np.float64),
            cell_mpc=float(cap_row["cell_mpc"]),
            padding_voxels=args.padding_voxels,
            rsmooth_mpc=args.rsmooth_mpc,
        )
        raw[selected] = prediction
        cap_reports[name] = {
            "n_authoritative": int(selected.sum()),
            "response": response,
            "fft": fft,
        }
    assignment.close()
    if not np.all(np.isfinite(raw)):
        raise RuntimeError(
            f"{args.phase} {args.estimator} reconstruction contains non-finite values"
        )
    if np.any(np.diff(raw, axis=1) < -1.0e-6):
        raise RuntimeError(f"{args.phase} raw tensor eigensolve violated ordering")
    np.save(output / "parent_node_id.npy", parent)
    np.save(output / f"{args.estimator}_raw_eigenvalues.npy", raw)
    report = {
        "schema_version": f"p10-{args.estimator}-raw-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "estimator": (
            "CIC stored P3 counts/expected -> fixed FFT tidal solve"
            if args.estimator == "cic"
            else "exact piecewise-linear DTFE -> response contrast -> fixed FFT tidal solve"
        ),
        "n_authoritative": int(len(parent)),
        "complete_authoritative_coverage": True,
        "caps": cap_reports,
        "parameters": {
            "rsmooth_mpc": float(args.rsmooth_mpc),
            "rsmooth_mpc_h": float(args.rsmooth_mpc * 0.6766),
            "padding_voxels": int(args.padding_voxels),
            "response_floor_fraction": float(args.response_floor_fraction),
        },
        "inputs": {
            "assignment": str(paths["assignment"]),
            "assignment_sha256": sha256(paths["assignment"]),
            "points": str(paths["points"]),
            "points_sha256": sha256(paths["points"]),
            "field_manifest": str(paths["field_manifest_path"]),
            "field_manifest_sha256": sha256(paths["field_manifest_path"]),
        },
        "artifacts": {
            "parent_node_id": str(output / "parent_node_id.npy"),
            "raw_eigenvalues": str(
                output / f"{args.estimator}_raw_eigenvalues.npy"
            ),
        },
        "elapsed_seconds": time.time() - started,
        "blind_phase_opened": False,
        "pass": True,
    }
    atomic_json(output / "raw_report.json", report)
    atomic_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _weighted_affine(
    loader: P10PhaseBalancedLoader,
    output_root: Path,
    estimator: str,
) -> dict:
    """Fit phase-balanced WLS sufficient statistics without concatenating phases."""
    sw = np.zeros(3, dtype=np.float64)
    sx = np.zeros(3, dtype=np.float64)
    sy = np.zeros(3, dtype=np.float64)
    sxx = np.zeros(3, dtype=np.float64)
    sxy = np.zeros(3, dtype=np.float64)
    phase_rows = {}
    for phase in loader.training_phases:
        paths = _phase_paths(loader, phase)
        assignment = np.load(paths["assignment"], mmap_mode="r")
        auth_rows = np.flatnonzero(authoritative_mask(assignment))
        expected_parent = np.asarray(
            assignment["parent_node_id"][auth_rows], dtype=np.int64
        )
        parent = np.load(output_root / phase / "parent_node_id.npy", mmap_mode="r")
        raw = np.load(
            output_root / phase / f"{estimator}_raw_eigenvalues.npy", mmap_mode="r"
        )
        if not np.array_equal(parent, expected_parent):
            raise RuntimeError(
                f"{phase} {estimator} parents do not match authoritative order"
            )
        truth = loader.targets_by_parent(phase)[parent]
        row_weight = np.asarray(loader.row_weights(phase), dtype=np.float64)[auth_rows]
        row_weight /= float(loader.phase_records[phase]["phase_weight_denominator"])
        phase_rows[phase] = int(len(parent))
        for column in range(3):
            x = np.asarray(raw[:, column], dtype=np.float64)
            y = np.asarray(truth[:, column], dtype=np.float64)
            sw[column] += np.sum(row_weight)
            sx[column] += np.sum(row_weight * x)
            sy[column] += np.sum(row_weight * y)
            sxx[column] += np.sum(row_weight * x * x)
            sxy[column] += np.sum(row_weight * x * y)
        assignment.close()
    slopes = (sxy - sx * sy / sw) / (sxx - sx * sx / sw)
    intercepts = sy / sw - slopes * sx / sw
    if not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(intercepts)):
        raise RuntimeError(f"non-finite P10 {estimator} affine calibration")
    return {
        "fit_split": list(loader.training_phases),
        "fit_rows_by_phase": phase_rows,
        "weighting": (
            "equal phase status with frozen within-phase sqrt-shell row weights"
        ),
        "coefficients": [
            {"slope": float(slopes[i]), "intercept": float(intercepts[i])}
            for i in range(3)
        ],
    }


def apply_affine(raw: np.ndarray, affine: dict) -> np.ndarray:
    result = np.empty_like(raw, dtype=np.float32)
    for column, row in enumerate(affine["coefficients"]):
        result[:, column] = (
            float(row["slope"]) * np.asarray(raw[:, column], dtype=np.float64)
            + float(row["intercept"])
        ).astype(np.float32)
    return result


def finalize(args: argparse.Namespace) -> dict:
    loader = P10PhaseBalancedLoader(args.contract_root, include_blind=False)
    required = (*loader.training_phases, loader.validation_phase)
    for phase in required:
        marker = args.output_root / phase / f"{args.estimator.upper()}_RAW_COMPLETE.json"
        if not marker.exists():
            raise RuntimeError(f"missing raw {args.estimator} phase artifact: {marker}")
    affine = _weighted_affine(loader, args.output_root, args.estimator)
    phase = loader.validation_phase
    paths = _phase_paths(loader, phase)
    parent = np.load(args.output_root / phase / "parent_node_id.npy", mmap_mode="r")
    raw = np.load(
        args.output_root / phase / f"{args.estimator}_raw_eigenvalues.npy",
        mmap_mode="r",
    )
    calibrated = apply_affine(raw, affine)
    ordered = np.sort(calibrated, axis=1)
    assignment = np.load(paths["assignment"], mmap_mode="r")
    truth = loader.targets_by_parent(phase)
    raw_report = evaluate_complete_phase(
        parent_node_id=parent,
        predicted_eigenvalues=raw,
        truth_by_parent=truth,
        assignment=assignment,
        phase=phase,
    )
    calibrated_report = evaluate_complete_phase(
        parent_node_id=parent,
        predicted_eigenvalues=calibrated,
        truth_by_parent=truth,
        assignment=assignment,
        phase=phase,
    )
    ordered_report = evaluate_complete_phase(
        parent_node_id=parent,
        predicted_eigenvalues=ordered,
        truth_by_parent=truth,
        assignment=assignment,
        phase=phase,
    )
    assignment.close()
    np.save(
        args.output_root / phase / f"{args.estimator}_train_affine_eigenvalues.npy",
        calibrated,
    )
    np.save(
        args.output_root
        / phase
        / f"{args.estimator}_train_affine_ordered_eigenvalues.npy",
        ordered,
    )
    report = {
        "schema_version": f"p10-{args.estimator}-final-v1",
        "created_utc": utc_now(),
        "estimator": f"{args.estimator.upper()} full-cap fixed-physics baseline",
        "training_phases": list(loader.training_phases),
        "validation_phase": phase,
        "sealed_blind_phase": loader.blind_phase,
        "blind_phase_opened": False,
        "affine": affine,
        "raw": raw_report,
        "train_affine_primary": calibrated_report,
        "train_affine_ordered_diagnostic": ordered_report,
        "ordering_policy": (
            "unsorted componentwise train-affine row is primary for continuity with "
            "P8; sorted row is reported as a transparent ordered diagnostic"
        ),
        "pass": True,
    }
    atomic_json(args.output_root / f"{args.estimator}_ph006_report.json", report)
    atomic_json(
        args.output_root / f"P10_{args.estimator.upper()}_PH006_COMPLETE.json", report
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("raw", "finalize"))
    parser.add_argument("--estimator", choices=("cic", "dtfe"), default="cic")
    parser.add_argument("--phase", choices=VISIBLE_PHASES)
    parser.add_argument("--contract-root", type=Path, default=CONTRACT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--dtfe-build-root", type=Path, default=CLASSICAL_ROOT / "dtfe_build_v1"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    parser.add_argument("--response-floor-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.mode == "raw" and args.phase is None:
        parser.error("raw mode requires --phase")
    if args.slab <= 0 or args.padding_voxels < 0 or args.rsmooth_mpc <= 0:
        parser.error("invalid slab, padding, or smoothing scale")
    if not 0 <= args.response_floor_fraction < 1:
        parser.error("response-floor-fraction must lie in [0,1)")
    if args.output_root is None:
        args.output_root = CLASSICAL_ROOT / f"{args.estimator}_fullcap_v1"
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "raw":
        raw_phase(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
