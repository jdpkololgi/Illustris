#!/usr/bin/env python3
"""Rescore frozen D0 on a DarkAI-like equal-volume NGC grid subset.

This is an evaluation-only diagnostic.  It never loads model weights or changes
the stitched prediction.  The primary domain is every P8.9 science-support cell
whose centre lies at 0.15 < z < 0.4 in NGC.  Each 5-Mpc cell has one vote.

The spectral fields are separately mean-subtracted inside that binary domain and
zero outside it.  The reported transfer is exactly P_cross/P_true and the reported
correlation is P_cross/sqrt(P_pred*P_true).  No random-window deconvolution is used.
The same two masked fields are passed through the frozen unsmoothed tidal projector
for grid-cell class recall.  Sign classes (threshold zero) are the DarkAI-like
primary row; the native GraphWeb threshold 0.2 is retained as a secondary row.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.abacus_tweb.p8_evaluate_stitched_density import (
    spectral_sums,
    spectra_report,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
STITCHED = ROOT / "p8_density_phys_v1/d0_stitched/rotation_0/seed_42"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
OUTPUT = ROOT / "p8_density_phys_v1/d0_darkai_like_rescore/rotation_0/seed_42"
TRACKED = REPO_ROOT / "docs/evidence/p8/density_d0_darkai_like_rescore.json"
CLASS_NAMES = ("void", "sheet", "filament", "knot")
COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched", type=Path, default=STITCHED)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tracked-output", type=Path, default=TRACKED)
    parser.add_argument("--z-min", type=float, default=0.15)
    parser.add_argument("--z-max", type=float, default=0.40)
    parser.add_argument("--spectral-bins", type=int, default=30)
    parser.add_argument("--k-min-h-mpc", type=float, default=0.002)
    parser.add_argument("--k-max-h-mpc", type=float, default=1.0)
    parser.add_argument("--padding-voxels", type=int, default=24)
    parser.add_argument("--eig-chunk", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def predicted_path(stitched: Path) -> Path:
    return stitched / "ngc_predicted_delta_r7.h5"


def selection_mask(
    support: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    origin_mpc: np.ndarray,
    cell_mpc: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    axes = [
        torch.as_tensor(
            origin_mpc[axis]
            + (np.arange(shape[axis], dtype=np.float32) + 0.5) * cell_mpc,
            dtype=torch.float32,
            device=support.device,
        )
        for axis in range(3)
    ]
    radius2 = (
        axes[0][:, None, None].square()
        + axes[1][None, :, None].square()
        + axes[2][None, None, :].square()
    )
    lower2 = float(Planck18.comoving_distance(z_min).value) ** 2
    upper2 = float(Planck18.comoving_distance(z_max).value) ** 2
    return support.bool() & (radius2 > lower2) & (radius2 < upper2)


def masked_mean_subtracted(field: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, float]:
    weight = int(torch.count_nonzero(selected).item())
    if weight == 0:
        raise RuntimeError("DarkAI-like subset contains zero selected grid cells")
    mean = torch.sum(field[selected], dtype=torch.float64) / weight
    return (field - mean.to(field.dtype)) * selected, float(mean.item())


def selected_tidal_components(
    field: torch.Tensor,
    selected: torch.Tensor,
    *,
    cell_mpc: float,
    padding_voxels: int,
) -> tuple[np.ndarray, dict]:
    """Apply the fixed projector and retain six components only at selected cells."""
    original_shape = tuple(int(value) for value in field.shape)
    n_selected = int(torch.count_nonzero(selected).item())
    output = np.empty((n_selected, len(COMPONENTS)), dtype=np.float32)
    padding = int(padding_voxels)
    work = F.pad(field[None, None], (padding,) * 6)[0, 0] if padding else field
    work_shape = tuple(int(value) for value in work.shape)
    dk = torch.fft.rfftn(work)
    kx = torch.fft.fftfreq(work_shape[0], d=cell_mpc, device=field.device) * (2.0 * math.pi)
    ky = torch.fft.fftfreq(work_shape[1], d=cell_mpc, device=field.device) * (2.0 * math.pi)
    kz = torch.fft.rfftfreq(work_shape[2], d=cell_mpc, device=field.device) * (2.0 * math.pi)
    k2 = kx[:, None, None].square() + ky[None, :, None].square() + kz[None, None, :].square()
    inverse_k2 = 1.0 / torch.where(k2 > 0, k2, torch.ones_like(k2))
    inverse_k2[0, 0, 0] = 0.0
    axes = (kx, ky, kz)
    peak = int(torch.cuda.max_memory_allocated()) if field.device.type == "cuda" else 0
    for column, (left_axis, right_axis) in enumerate(COMPONENTS):
        left_shape = (-1, 1, 1) if left_axis == 0 else ((1, -1, 1) if left_axis == 1 else (1, 1, -1))
        right_shape = (-1, 1, 1) if right_axis == 0 else ((1, -1, 1) if right_axis == 1 else (1, 1, -1))
        component = torch.fft.irfftn(
            dk
            * inverse_k2
            * axes[left_axis].reshape(left_shape)
            * axes[right_axis].reshape(right_shape),
            s=work_shape,
        )
        if padding:
            component = component[
                padding:padding + original_shape[0],
                padding:padding + original_shape[1],
                padding:padding + original_shape[2],
            ]
        output[:, column] = component[selected].cpu().numpy()
        peak = max(
            peak,
            int(torch.cuda.max_memory_allocated()) if field.device.type == "cuda" else 0,
        )
        del component
    del work, dk, k2, inverse_k2
    if field.device.type == "cuda":
        torch.cuda.empty_cache()
    return output, {
        "padding_voxels": padding,
        "work_shape": list(work_shape),
        "selected_cells": n_selected,
        "maximum_cuda_memory_bytes": peak,
    }


def six_to_tensor(components: torch.Tensor) -> torch.Tensor:
    tensor = torch.empty(
        (len(components), 3, 3), dtype=components.dtype, device=components.device
    )
    for column, (left, right) in enumerate(COMPONENTS):
        tensor[:, left, right] = components[:, column]
        tensor[:, right, left] = components[:, column]
    return tensor


def class_recall_from_components(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float,
    chunk: int,
    device: str,
) -> dict:
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError("prediction and truth components must both have shape (N,6)")
    confusion = np.zeros((4, 4), dtype=np.int64)
    for left in range(0, len(truth), int(chunk)):
        right = min(left + int(chunk), len(truth))
        p = torch.from_numpy(prediction[left:right]).to(device)
        t = torch.from_numpy(truth[left:right]).to(device)
        p_class = torch.sum(torch.linalg.eigvalsh(six_to_tensor(p)) > threshold, dim=1)
        t_class = torch.sum(torch.linalg.eigvalsh(six_to_tensor(t)) > threshold, dim=1)
        encoded = (4 * t_class + p_class).long()
        confusion += torch.bincount(encoded, minlength=16).reshape(4, 4).cpu().numpy()
        del p, t, p_class, t_class, encoded
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion), support,
        out=np.full(4, np.nan, dtype=np.float64), where=support > 0,
    )
    predicted_count = confusion.sum(axis=0)
    return {
        "threshold": float(threshold),
        "class_order": list(CLASS_NAMES),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "true_cell_count": dict(zip(CLASS_NAMES, support.tolist())),
        "predicted_cell_count": dict(zip(CLASS_NAMES, predicted_count.tolist())),
        "recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "balanced_accuracy": float(np.nanmean(recall)),
        "exact_cell_accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
    }


def spectral_table(report: dict) -> list[dict]:
    rows = []
    for index, centre in enumerate(report["k_centres_h_mpc"]):
        rows.append({
            "k_low_h_mpc": float(report["k_edges_h_mpc"][index]),
            "k_high_h_mpc": float(report["k_edges_h_mpc"][index + 1]),
            "k_centre_h_mpc": float(centre),
            "mode_count": float(report["mode_count"][index]),
            "p_cross_over_p_true": float(report["cross_transfer"][index]),
            "r_k": float(report["cross_correlation_r"][index]),
        })
    return rows


def weighted_spectral_summary(rows: list[dict], low: float, high: float) -> dict:
    selected = [
        row for row in rows
        if low <= row["k_centre_h_mpc"] < high
        and row["mode_count"] > 0
        and np.isfinite(row["p_cross_over_p_true"])
        and np.isfinite(row["r_k"])
    ]
    if not selected:
        return {"bins": 0, "modes": 0, "p_cross_over_p_true": None, "r_k": None}
    weight = np.asarray([row["mode_count"] for row in selected], dtype=np.float64)
    return {
        "bins": len(selected),
        "modes": int(weight.sum()),
        "p_cross_over_p_true": float(np.average(
            [row["p_cross_over_p_true"] for row in selected], weights=weight
        )),
        "r_k": float(np.average([row["r_k"] for row in selected], weights=weight)),
    }


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DarkAI-like full-grid rescore requires an interactive GPU")
    if not 0.0 <= args.z_min < args.z_max:
        raise ValueError("require 0 <= z_min < z_max")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(
        args.output / ".darkai_like_rescore.lock",
        purpose="P8.9 frozen D0 DarkAI-like volume-grid rescore",
    )
    target_manifest = json.loads(args.target_manifest.read_text())
    stitched_manifest_path = args.stitched / "stitched_field_manifest.json"
    stitched_manifest = json.loads(stitched_manifest_path.read_text())
    if stitched_manifest.get("double_smoothing_applied") is not False:
        raise RuntimeError("stitched field does not certify no double smoothing")
    if stitched_manifest.get("status") != "PASS":
        raise RuntimeError("stitched field is not complete")
    component = target_manifest["components"]["NGC"]
    grid = component["grid"]
    shape = tuple(int(value) for value in grid["shape"])
    origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
    cell = float(grid["cell_mpc"])
    with h5py.File(component["file"], "r") as target:
        truth = torch.from_numpy(np.asarray(target["delta_r7"], dtype=np.float32)).to(args.device)
        support = torch.from_numpy(np.asarray(target["science_support"], dtype=np.uint8)).to(args.device)
    prediction_file = predicted_path(args.stitched)
    with h5py.File(prediction_file, "r") as predicted:
        prediction = torch.from_numpy(
            np.nan_to_num(np.asarray(predicted["predicted_delta_r7"], dtype=np.float32))
        ).to(args.device)
    if tuple(truth.shape) != shape or tuple(prediction.shape) != shape:
        raise RuntimeError("NGC prediction/target shape does not match target manifest")
    selected = selection_mask(
        support, shape=shape, origin_mpc=origin, cell_mpc=cell,
        z_min=args.z_min, z_max=args.z_max,
    )
    if torch.any(~torch.isfinite(prediction[selected])) or torch.any(~torch.isfinite(truth[selected])):
        raise RuntimeError("non-finite selected D0 or truth cells")
    selected_cells = int(torch.count_nonzero(selected).item())
    predicted_masked, predicted_mean = masked_mean_subtracted(prediction, selected)
    truth_masked, truth_mean = masked_mean_subtracted(truth, selected)
    edges = np.geomspace(args.k_min_h_mpc, args.k_max_h_mpc, args.spectral_bins + 1)
    sums = spectral_sums(
        predicted_masked, truth_masked, cell_mpc=cell, edges_h_mpc=edges,
    )
    spectra = spectra_report(sums, edges)
    rows = spectral_table(spectra)
    predicted_components, pred_fft = selected_tidal_components(
        predicted_masked, selected, cell_mpc=cell, padding_voxels=args.padding_voxels,
    )
    truth_components, truth_fft = selected_tidal_components(
        truth_masked, selected, cell_mpc=cell, padding_voxels=args.padding_voxels,
    )
    classes = {
        "darkai_sign_threshold": class_recall_from_components(
            predicted_components, truth_components, threshold=0.0,
            chunk=args.eig_chunk, device=args.device,
        ),
        "graphweb_threshold_0p2_secondary": class_recall_from_components(
            predicted_components, truth_components, threshold=0.2,
            chunk=args.eig_chunk, device=args.device,
        ),
    }
    del predicted_components, truth_components
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    cell_volume_mpc3 = cell ** 3
    report = {
        "schema_version": "p8-density-darkai-like-rescore-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_sha(),
        "status": "PASS",
        "model": "U-DENSITY-PHYS-v1",
        "rotation": 0,
        "seed": 42,
        "no_retraining": True,
        "subset": {
            "cap": "NGC",
            "cell_centre_redshift_open_interval": [float(args.z_min), float(args.z_max)],
            "science_support_required": True,
            "weighting": "one equal-volume vote per selected grid cell",
            "selected_cells": selected_cells,
            "cell_size_mpc": cell,
            "cell_volume_mpc3": cell_volume_mpc3,
            "selected_volume_mpc3": float(selected_cells * cell_volume_mpc3),
            "selected_volume_mpc_over_h_cubed": float(
                selected_cells * cell_volume_mpc3 * float(Planck18.h) ** 3
            ),
        },
        "spectra": {
            "definition": {
                "p_cross_over_p_true": "Re(F_pred F_true*) / |F_true|^2 per k shell",
                "r_k": "P_cross / sqrt(P_pred P_true)",
                "window": "binary NGC science-support cells with 0.15<z_cell<0.4",
                "mean_subtraction": "separate equal-volume means inside selected cells",
                "random_window_deconvolution": False,
            },
            "selected_means_before_subtraction": {
                "prediction": predicted_mean,
                "truth": truth_mean,
            },
            "table": rows,
            "mode_weighted_summary": {
                "k_0p002_to_0p1": weighted_spectral_summary(rows, 0.002, 0.1),
                "k_0p02_to_0p08": weighted_spectral_summary(rows, 0.02, 0.08),
                "k_0p08_to_0p2": weighted_spectral_summary(rows, 0.08, 0.2),
                "k_0p2_to_0p4": weighted_spectral_summary(rows, 0.2, 0.4),
            },
        },
        "grid_cell_classes": {
            "tensor_operator": "one fixed unsmoothed k_i*k_j/k^2 solve of each masked R=7 field",
            "double_smoothing_applied": False,
            "predicted_fft": pred_fft,
            "truth_fft": truth_fft,
            **classes,
        },
        "inputs": {
            "stitched_manifest": str(stitched_manifest_path),
            "stitched_manifest_sha256": sha256(stitched_manifest_path),
            "prediction_file": str(prediction_file),
            "prediction_sha256_from_stitched_manifest": stitched_manifest["field_files"]["NGC"]["sha256"],
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
            "truth_file": component["file"],
            "truth_sha256_from_target_manifest": component["file_sha256"],
        },
        "elapsed_seconds": float(time.time() - started),
    }
    csv_path = args.output / "spectra_pcross_over_ptrue_rk.csv"
    atomic_csv(csv_path, rows)
    report["spectra"]["csv"] = str(csv_path)
    atomic_json(args.output / "darkai_like_rescore.json", report)
    atomic_json(args.tracked_output, report)
    (args.output / "DARKAI_LIKE_RESCORE_COMPLETE").write_text(
        "frozen D0 NGC 0.15<z<0.4 equal-volume grid diagnostic PASS\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    run_lock.close()


if __name__ == "__main__":
    main()
