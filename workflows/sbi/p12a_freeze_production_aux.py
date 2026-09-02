#!/usr/bin/env python3
"""Freeze train/ph006-only quality thresholds and the P12-A Gaussian control.

This builder is deliberately blind-safe: it accepts only the frozen training OOF
sample, the frozen ph006 selection sample and the already-generated ph006 FMPE
draw audit. It never discovers or opens ph001 products.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_patch_recovery import torch_load
from workflows.sbi.p12_production_contract import fit_shell_cap_gaussian
from workflows.sbi.p12_train_base_response_fmpe import theta_to_eigenvalues


DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")
DEFAULT_TRAINING = DEFAULT_ROOT / "training_oof_sample.npz"
DEFAULT_VALIDATION = DEFAULT_ROOT / "ph006_selection_sample.npz"
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "fmpe_seed42/fmpe_estimator.pt"
DEFAULT_AUDIT = DEFAULT_ROOT / "fmpe_seed42/calibration_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_visible(path: Path, *, role: str) -> np.lib.npyio.NpzFile:
    if "ph001" in str(path).lower():
        raise PermissionError("sealed ph001 is forbidden from production-auxiliary fitting")
    archive = np.load(path, mmap_mode="r")
    required = {"context", "base_prediction_eigenvalues", "truth_eigenvalues", "shell", "cap"}
    missing = required - set(archive.files)
    if missing:
        raise RuntimeError(f"{role} archive is missing {sorted(missing)}")
    context = np.asarray(archive["context"])
    if context.ndim != 2 or context.shape[1] != 7 or not np.all(np.isfinite(context)):
        raise RuntimeError(f"{role} seven-feature context is invalid")
    return archive


def freeze_auxiliary_contracts(
    *,
    training_path: Path,
    validation_path: Path,
    checkpoint_path: Path,
    evaluation_samples_path: Path,
    evaluation_index_path: Path,
    output_root: Path,
    width_tail_quantile: float,
) -> dict:
    if not 0.5 < width_tail_quantile < 1.0:
        raise ValueError("width-tail quantile must lie between 0.5 and 1")
    training = _load_visible(training_path, role="training")
    validation = _load_visible(validation_path, role="validation")
    if "ph001" in str(evaluation_samples_path).lower() or "ph001" in str(evaluation_index_path).lower():
        raise PermissionError("sealed ph001 audit artifacts are forbidden")
    scaled_samples = np.load(evaluation_samples_path, mmap_mode="r")
    evaluation_index = np.asarray(np.load(evaluation_index_path), dtype=np.int64)
    if scaled_samples.ndim != 3 or scaled_samples.shape[-1] != 3:
        raise RuntimeError("frozen FMPE audit draws have invalid shape")
    if len(scaled_samples) != len(evaluation_index):
        raise RuntimeError("FMPE audit draw/index rows are not aligned")
    if np.any(evaluation_index < 0) or np.any(evaluation_index >= len(validation["context"])):
        raise RuntimeError("FMPE audit index lies outside ph006")
    checkpoint = torch_load(checkpoint_path, "cpu")
    if checkpoint.get("schema_version") != "p12a-fmpe-estimator-v1":
        raise RuntimeError("unexpected P12-A checkpoint schema")
    theta = np.asarray(scaled_samples, dtype=np.float32)
    theta = theta * np.asarray(checkpoint["theta_std"], dtype=np.float32) + np.asarray(
        checkpoint["theta_mean"], dtype=np.float32
    )
    eigen = theta_to_eigenvalues(theta)
    q16, q84 = np.quantile(eigen, [0.16, 0.84], axis=1)
    width = q84 - q16
    width_threshold = np.quantile(width, width_tail_quantile, axis=0)
    response = np.asarray(training["context"][:, 4], dtype=np.float64)
    if not np.all(np.isfinite(response)):
        raise RuntimeError("training response covariate is non-finite")

    from astropy.cosmology import Planck18

    hubble_h = float(Planck18.h)
    boundary_r_mpc = 7.0 / hubble_h
    boundary_2r_mpc = 14.0 / hubble_h
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "P12A_QUALITY_THRESHOLDS.json"
    baseline_path = output_root / "P12A_GAUSSIAN_BASELINE.json"
    for destination in (quality_path, baseline_path):
        if destination.exists():
            raise FileExistsError(destination)

    sources = {
        "training_oof": {"path": str(training_path), "sha256": sha256(training_path)},
        "ph006_selection": {"path": str(validation_path), "sha256": sha256(validation_path)},
        "fmpe_checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "ph006_evaluation_samples": {
            "path": str(evaluation_samples_path),
            "sha256": sha256(evaluation_samples_path),
        },
        "ph006_evaluation_index": {
            "path": str(evaluation_index_path),
            "sha256": sha256(evaluation_index_path),
        },
    }
    quality = {
        "schema_version": "p12a-production-quality-thresholds-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "response_covariate": {
            "name": "log_ntilde_mpc3",
            "context_index": 4,
            "training_minimum": float(np.min(response)),
            "training_maximum": float(np.max(response)),
        },
        "boundary_distance": {
            "stored_unit": "Mpc",
            "smoothing_radius_mpc_h": 7.0,
            "hubble_h": hubble_h,
            "threshold_r_mpc": boundary_r_mpc,
            "threshold_2r_mpc": boundary_2r_mpc,
        },
        "prior_dominated_width": {
            "definition": "ph006 posterior q84-q16 component width",
            "quantile": float(width_tail_quantile),
            "threshold_by_ordered_eigenvalue": width_threshold.tolist(),
            "audit_rows": int(len(width)),
        },
        "sources": sources,
        "truth_files_read": [str(training_path), str(validation_path)],
        "sealed_phase_opened": False,
        "ph001_opened": False,
        "pass": True,
    }
    residual = np.asarray(training["truth_eigenvalues"], dtype=np.float64) - np.asarray(
        training["base_prediction_eigenvalues"], dtype=np.float64
    )
    baseline = fit_shell_cap_gaussian(
        residual,
        np.asarray(training["shell"], dtype=np.int8),
        np.asarray(training["cap"], dtype=np.int8),
        weight=(
            np.asarray(training["natural_weight"], dtype=np.float64)
            if "natural_weight" in training.files
            else None
        ),
    )
    baseline.update(
        {
            "schema_version": "p12a-shell-cap-residual-gaussian-v1",
            "created_utc": utc_now(),
            "git_revision": git_revision(),
            "conditioning": "shell and cap only",
            "base_prediction": "frozen five-phase U-PATCH",
            "training_rows": int(len(residual)),
            "sources": {"training_oof": sources["training_oof"]},
            "truth_files_read": [str(training_path)],
            "sealed_phase_opened": False,
            "ph001_opened": False,
            "pass": True,
        }
    )
    atomic_json(quality_path, quality)
    atomic_json(baseline_path, baseline)
    training.close()
    validation.close()
    return {
        "quality_thresholds": {"path": str(quality_path), "sha256": sha256(quality_path)},
        "gaussian_baseline": {"path": str(baseline_path), "sha256": sha256(baseline_path)},
        "ph001_opened": False,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--evaluation-samples",
        type=Path,
        default=DEFAULT_AUDIT / "evaluation_samples_scaled.npy",
    )
    parser.add_argument("--evaluation-index", type=Path, default=DEFAULT_AUDIT / "evaluation_index.npy")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width-tail-quantile", type=float, default=0.95)
    args = parser.parse_args()
    result = freeze_auxiliary_contracts(
        training_path=args.training,
        validation_path=args.validation,
        checkpoint_path=args.checkpoint,
        evaluation_samples_path=args.evaluation_samples,
        evaluation_index_path=args.evaluation_index,
        output_root=args.output_root,
        width_tail_quantile=args.width_tail_quantile,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
