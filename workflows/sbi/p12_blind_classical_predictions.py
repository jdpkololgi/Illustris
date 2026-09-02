#!/usr/bin/env python3
"""Truth-free CIC/DTFE predictions for the shared ph001 blind opening.

This is intentionally separate from ``p10_classical_fullcap.py``: that visible-phase
workflow owns target-bearing evaluation and continues to reject ph001.  Here the
train-only affine map is frozen by its ph006 report, while every ph001 input is an
observed catalogue, response field, patch assignment, or observed-galaxy DTFE raster.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from workflows.abacus_tweb.p8_classical_fullcap import _sample_tidal_eigenvalues
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p10_classical_fullcap import (
    CAP_NAME,
    RSMOOTH_MPC,
    _dtfe_delta_to_gpu,
    _stored_delta_to_gpu,
    apply_affine,
)
from workflows.sbi.p12_prepare_base_response_dataset import sample_random_support_distance
from workflows.sbi.p12a_blind_inference import validate_observed_assignment


TRAINING_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def validate_affine_report(report: dict, estimator: str) -> dict:
    expected_schema = f"p10-{estimator}-final-v1"
    if (
        report.get("schema_version") != expected_schema
        or report.get("pass") is not True
        or tuple(report.get("training_phases", ())) != TRAINING_PHASES
        or report.get("validation_phase") != "ph006"
        or report.get("sealed_blind_phase") != "ph001"
        or report.get("blind_phase_opened") is not False
    ):
        raise PermissionError("classical affine report is not frozen blind-safe")
    affine = report.get("affine", {})
    coefficients = affine.get("coefficients", ())
    if len(coefficients) != 3 or any(
        not np.isfinite(float(row[key]))
        for row in coefficients
        for key in ("slope", "intercept")
    ):
        raise RuntimeError("classical affine coefficients are invalid")
    return affine


def authoritative_rows(assignment: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_observed_assignment(assignment)
    if "core_id" not in assignment.files:
        raise RuntimeError("observed assignment is missing core_id")
    selected = np.flatnonzero(np.asarray(assignment["supervised_eligible"], dtype=bool))
    parent = np.asarray(assignment["parent_node_id"][selected], dtype=np.int64)
    core = np.asarray(assignment["core_id"][selected], dtype=np.int64)
    cap = np.asarray(assignment["cap"][selected], dtype=np.uint8)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("blind authoritative parents are duplicated")
    order = np.lexsort((parent, core))
    return parent[order], core[order], cap[order]


def validate_response_manifest(response: dict) -> None:
    if (
        response.get("schema_version") != "p3br-response-overlay-manifest-v1"
        or response.get("phase") != "ph001"
        or response.get("pass") is not True
        or response.get("ph001_opened") is not False
        or response.get("blind_authority") is None
        or set(response.get("components", {})) != {"NGC", "SGC"}
    ):
        raise PermissionError("ph001 random-response manifest is not frozen and sealed")


def validate_canonical_points(points: np.ndarray, parent: np.ndarray) -> None:
    """Validate the truth-free (x,y,z,cap) catalogue used by CIC/DTFE."""
    if points.ndim != 2 or points.shape[1] < 4 or np.any(parent >= len(points)):
        raise RuntimeError("blind canonical points are invalid")


def predict(
    *,
    estimator: str,
    assignment_path: Path,
    points_path: Path,
    response_manifest_path: Path,
    affine_report_path: Path,
    output_path: Path,
    dtfe_root: Path | None,
    device: str,
    slab: int,
    padding_voxels: int,
    rsmooth_mpc: float,
    response_floor_fraction: float,
) -> dict:
    if estimator not in {"cic", "dtfe"}:
        raise ValueError("estimator must be cic or dtfe")
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite blind prediction {output_path}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("blind classical reconstruction requires a CUDA allocation")
    response = json.loads(response_manifest_path.read_text())
    validate_response_manifest(response)
    affine_report = json.loads(affine_report_path.read_text())
    affine = validate_affine_report(affine_report, estimator)
    assignment = np.load(assignment_path, mmap_mode="r")
    parent, core, assignment_cap = authoritative_rows(assignment)
    points = np.load(points_path, mmap_mode="r")
    # Redshift lives in the separate sealed loader vector and is not an input
    # to the gridded classical estimators.
    validate_canonical_points(points, parent)
    position = np.asarray(points[parent, :3], dtype=np.float64)
    cap = np.asarray(points[parent, 3], dtype=np.uint8)
    if not np.array_equal(cap, assignment_cap):
        raise RuntimeError("blind assignment/point cap identity mismatch")

    raw = np.empty((len(parent), 3), dtype=np.float32)
    cap_reports = {}
    for cap_id in (0, 1):
        cap_name = CAP_NAME[cap_id]
        component = response["components"][cap_name]
        field_path = Path(component["file"])
        selected = cap == cap_id
        if estimator == "cic":
            delta, density_report = _stored_delta_to_gpu(
                field_path,
                device=device,
                slab=slab,
                response_floor_fraction=response_floor_fraction,
            )
        else:
            if dtfe_root is None:
                raise ValueError("DTFE prediction requires --dtfe-root")
            ready = dtfe_root / "DTFE_FIELD_READY"
            density_path = dtfe_root / f"dtfe_density_{cap_name}.npy"
            if not ready.is_file() or not density_path.is_file():
                raise FileNotFoundError(f"missing truth-free ph001 DTFE raster: {density_path}")
            delta, density_report = _dtfe_delta_to_gpu(
                density_path,
                field_path,
                cell_mpc=float(component["grid"]["cell_mpc"]),
                device=device,
                slab=slab,
            )
        prediction, fft_report = _sample_tidal_eigenvalues(
            delta,
            positions=position[selected],
            origin=np.asarray(component["grid"]["origin_mpc"], dtype=np.float64),
            cell_mpc=float(component["grid"]["cell_mpc"]),
            padding_voxels=padding_voxels,
            rsmooth_mpc=rsmooth_mpc,
        )
        raw[selected] = prediction
        cap_reports[cap_name] = {
            "rows": int(selected.sum()),
            "density": density_report,
            "fft": fft_report,
        }
        del delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.all(np.isfinite(raw)) or np.any(np.diff(raw, axis=1) < -1.0e-6):
        raise RuntimeError("blind classical raw prediction is non-finite or unordered")
    calibrated = apply_affine(raw, affine)
    ordered = np.sort(calibrated, axis=1).astype(np.float32)
    boundary, support = sample_random_support_distance(response, points, parent)
    support = np.asarray(support, dtype=bool)
    arrays = {
        "parent_node_id": parent[support],
        "core_id": core[support],
        "raw_eigenvalues": raw[support],
        "train_affine_eigenvalues": calibrated[support],
        "train_affine_ordered_eigenvalues": ordered[support],
        "distance_to_support_boundary_mpc": np.asarray(boundary[support], dtype=np.float32),
        "support_random": np.ones(int(support.sum()), dtype=bool),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    marker = {
        "schema_version": f"p12-blind-{estimator}-prediction-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "phase": "ph001",
        "estimator": estimator,
        "rows": int(support.sum()),
        "unsupported_rows_omitted": int((~support).sum()),
        "prediction": str(output_path),
        "prediction_sha256": sha256(output_path),
        "assignment": {"path": str(assignment_path), "sha256": sha256(assignment_path)},
        "points": {"path": str(points_path), "sha256": sha256(points_path)},
        "response": {"path": str(response_manifest_path), "sha256": sha256(response_manifest_path)},
        "affine_report": {"path": str(affine_report_path), "sha256": sha256(affine_report_path)},
        "dtfe_root": None if dtfe_root is None else str(dtfe_root),
        "parameters": {
            "rsmooth_mpc": float(rsmooth_mpc),
            "padding_voxels": int(padding_voxels),
            "response_floor_fraction": float(response_floor_fraction),
        },
        "caps": cap_reports,
        "ordering_policy": affine_report["ordering_policy"],
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    atomic_json(output_path.with_suffix(".json"), marker)
    assignment.close()
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("cic", "dtfe"), required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--response-manifest", type=Path, required=True)
    parser.add_argument("--affine-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtfe-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    parser.add_argument("--response-floor-fraction", type=float, default=0.05)
    args = parser.parse_args()
    result = predict(
        estimator=args.estimator,
        assignment_path=args.assignment,
        points_path=args.points,
        response_manifest_path=args.response_manifest,
        affine_report_path=args.affine_report,
        output_path=args.output,
        dtfe_root=args.dtfe_root,
        device=args.device,
        slab=args.slab,
        padding_voxels=args.padding_voxels,
        rsmooth_mpc=args.rsmooth_mpc,
        response_floor_fraction=args.response_floor_fraction,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
