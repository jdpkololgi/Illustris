#!/usr/bin/env python3
"""Infer and exactly stitch the frozen P8.9 U-DENSITY-PHYS-v1 field."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_density_training_utils import (
    DensityUnitAdapter,
    extract_core_prediction,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_patch_recovery import torch_load
from workflows.abacus_tweb.p8_train_unet_patch import UNet3D


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
RUN = ROOT / "p8_density_phys_v1/d0_runs/rotation_0/seed_42/scientific_v1"
OUTPUT = ROOT / "p8_density_phys_v1/d0_stitched/rotation_0/seed_42"
CAP_NAME = {0: "SGC", 1: "NGC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "best_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--parity-cores", type=int, default=8)
    parser.add_argument("--expanded-halo-voxels", type=int, default=32)
    parser.add_argument("--parity-nrmse", type=float, default=0.02)
    parser.add_argument("--parity-p95-over-std", type=float, default=0.08)
    parser.add_argument("--parity-worst-core-nrmse", type=float, default=0.04)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def field_path(output: Path, cap: int) -> Path:
    return output / f"{CAP_NAME[int(cap)].lower()}_predicted_delta_r7.h5"


def initialize_fields(output: Path, adapter: DensityUnitAdapter, *, resume: bool) -> None:
    for cap, cap_name in CAP_NAME.items():
        path = field_path(output, cap)
        grid = adapter.target_manifest["components"][cap_name]["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        if path.exists():
            if not resume:
                raise RuntimeError(f"existing stitched field requires --resume: {path}")
            with h5py.File(path, "r") as handle:
                if tuple(handle["predicted_delta_r7"].shape) != shape:
                    raise RuntimeError(f"stitched field shape mismatch: {path}")
            continue
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset(
                "predicted_delta_r7",
                shape=shape,
                dtype="f4",
                chunks=(min(8, shape[0]), min(64, shape[1]), min(64, shape[2])),
                fillvalue=np.nan,
            )
            dataset.attrs["target"] = "R=7 Mpc/h smoothed matter contrast"
            dataset.attrs["double_smoothing_applied"] = False
            dataset.attrs["cap"] = cap_name
            handle.attrs["grid"] = json.dumps(grid, sort_keys=True)
            handle.flush()


def owner_partition_report(cores) -> dict:
    reports = {}
    for cap, cap_name in CAP_NAME.items():
        rows = np.flatnonzero(np.asarray(cores["cap"]) == cap)
        keys = np.asarray(cores["core_index"][rows], dtype=np.int64)
        starts = np.asarray(cores["voxel_start"][rows], dtype=np.int64)
        stops = np.asarray(cores["voxel_stop"][rows], dtype=np.int64)
        if len(np.unique(keys, axis=0)) != len(rows):
            raise RuntimeError(f"duplicate output owner core in {cap_name}")
        axis_reports = []
        for axis in range(3):
            ranges = {}
            for key, start, stop in zip(keys[:, axis], starts[:, axis], stops[:, axis], strict=True):
                pair = (int(start), int(stop))
                if int(key) in ranges and ranges[int(key)] != pair:
                    raise RuntimeError(f"inconsistent owner range in {cap_name} axis {axis}")
                ranges[int(key)] = pair
            ordered = [ranges[key] for key in sorted(ranges)]
            nonoverlap = all(left[1] <= right[0] for left, right in zip(ordered[:-1], ordered[1:]))
            if not nonoverlap:
                raise RuntimeError(f"overlapping owner ranges in {cap_name} axis {axis}")
            axis_reports.append({"ranges": len(ordered), "nonoverlap": True})
        reports[cap_name] = {
            "owners": int(len(rows)),
            "unique_core_indices": True,
            "axis_owner_ranges": axis_reports,
            "write_order_invariant_by_disjoint_half_open_ownership": True,
        }
    return reports


def infer_bounds(
    model: torch.nn.Module,
    adapter: DensityUnitAdapter,
    row: int,
    device: str,
    halo: int,
) -> tuple[np.ndarray, object]:
    patch, values, _ = adapter.extract_output_core(
        row, device, context_halo_voxels=halo
    )
    with torch.no_grad():
        scaled = extract_core_prediction(model(values), patch.core_slice)
    prediction = (
        scaled.detach().cpu().numpy() * np.float32(adapter.scaler["std"])
        + np.float32(adapter.scaler["mean"])
    )
    return prediction.astype(np.float32, copy=False), patch


def parity_rows(adapter: DensityUnitAdapter, count: int) -> np.ndarray:
    units = np.asarray(adapter.units)
    validation_fold = int(adapter.config["roles"]["validation_fold"])
    candidates = units[units["fold"] == validation_fold]
    chosen = []
    for cap in (0, 1):
        for shell in range(4):
            local = candidates[(candidates["cap"] == cap) & (candidates["shell"] == shell)]
            if len(local):
                chosen.append(int(local[len(local) // 2]["output_core_id"]))
    return np.asarray(chosen[:count], dtype=np.int64)


def convergence_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """P6-compatible, scale-normalized patch-convergence diagnostics."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = candidate - reference
    scale = max(float(np.std(reference)), 1e-6)
    absolute = np.abs(difference).ravel()
    return {
        "n": int(difference.size),
        "reference_std": scale,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "nrmse": float(np.sqrt(np.mean(np.square(difference))) / scale),
        "p95_abs_over_std": float(np.quantile(absolute, 0.95) / scale),
        "max_abs": float(np.max(absolute)),
        "max_abs_over_std": float(np.max(absolute) / scale),
    }


def trained_patch_parity(
    model: torch.nn.Module,
    adapter: DensityUnitAdapter,
    rows: np.ndarray,
    device: str,
    expanded_halo: int,
    nrmse_tolerance: float,
    p95_tolerance: float,
    worst_core_nrmse_tolerance: float,
) -> dict:
    context_residuals = []
    subdivision_residuals = []
    context_references = []
    subdivision_references = []
    context_core_nrmse = []
    subdivision_core_nrmse = []
    details = []
    for row in rows:
        base, _ = infer_bounds(model, adapter, int(row), device, 24)
        expanded, _ = infer_bounds(model, adapter, int(row), device, expanded_halo)
        context = expanded.astype(np.float64) - base.astype(np.float64)
        context_residuals.append(context.ravel())
        context_references.append(base.astype(np.float64).ravel())
        context_core = convergence_metrics(expanded, base)
        context_core_nrmse.append(context_core["nrmse"])

        start = np.asarray(adapter.cores["voxel_start"][row], dtype=np.int64)
        stop = np.asarray(adapter.cores["voxel_stop"][row], dtype=np.int64)
        axis = int(np.argmax(stop - start))
        split = int((start[axis] + stop[axis]) // 2)
        pieces = []
        for low, high in ((int(start[axis]), split), (split, int(stop[axis]))):
            sub_start = start.copy()
            sub_stop = stop.copy()
            sub_start[axis] = low
            sub_stop[axis] = high
            cap = int(adapter.cores["cap"][row])
            patch = adapter.field.extract_bounds(
                cap=cap,
                core_start=sub_start,
                core_stop=sub_stop,
                context_halo_voxels=24,
                channel_names=("counts", "exposure_apodized", "log_count_ratio"),
                alignment_voxels=8,
                core_id=int(adapter.cores["nominal_core_id"][row]),
                fold=int(adapter.cores["fold"][row]),
                authoritative_parent_id=np.empty(0, dtype=np.int64),
                authoritative_frac_index_global=np.empty((0, 3), dtype=np.float64),
            )
            from workflows.abacus_tweb.p8_train_unet_patch import model_inputs
            values, _ = model_inputs(patch, adapter.normalization, device)
            with torch.no_grad():
                scaled = extract_core_prediction(model(values), patch.core_slice)
            pieces.append(
                scaled.cpu().numpy() * np.float32(adapter.scaler["std"])
                + np.float32(adapter.scaler["mean"])
            )
        stitched = np.concatenate(pieces, axis=axis)
        subdivision = stitched.astype(np.float64) - base.astype(np.float64)
        subdivision_residuals.append(subdivision.ravel())
        subdivision_references.append(base.astype(np.float64).ravel())
        subdivision_core = convergence_metrics(stitched, base)
        subdivision_core_nrmse.append(subdivision_core["nrmse"])
        details.append({
            "output_core_id": int(row),
            "cap": CAP_NAME[int(adapter.cores["cap"][row])],
            "core_shape": list(base.shape),
            "subdivision_axis": axis,
            "subdivision_split_global_voxel": split,
            "expanded_context_max_abs": float(np.max(np.abs(context))),
            "expanded_context_nrmse": context_core["nrmse"],
            "subdivision_max_abs": float(np.max(np.abs(subdivision))),
            "subdivision_nrmse": subdivision_core["nrmse"],
        })
    context = np.concatenate(context_residuals) if context_residuals else np.empty(0)
    subdivision = np.concatenate(subdivision_residuals) if subdivision_residuals else np.empty(0)
    context_reference = (
        np.concatenate(context_references) if context_references else np.empty(0)
    )
    subdivision_reference = (
        np.concatenate(subdivision_references) if subdivision_references else np.empty(0)
    )
    context_metrics = (
        convergence_metrics(context + context_reference, context_reference)
        if len(context) else None
    )
    subdivision_metrics = (
        convergence_metrics(subdivision + subdivision_reference, subdivision_reference)
        if len(subdivision) else None
    )
    gates = {
        "nrmse": float(nrmse_tolerance),
        "p95_abs_over_std": float(p95_tolerance),
        "worst_core_nrmse": float(worst_core_nrmse_tolerance),
        "source": "P6 trained U-Net convergence contract",
    }
    report = {
        "cores": int(len(rows)),
        "output_core_ids": rows.tolist(),
        "base_halo_voxels": 24,
        "expanded_halo_voxels": int(expanded_halo),
        "gates": gates,
        "expanded_context": {
            **(context_metrics or {}),
            "worst_core_nrmse": (
                float(max(context_core_nrmse)) if context_core_nrmse else None
            ),
        },
        "subdivision": {
            **(subdivision_metrics or {}),
            "worst_core_nrmse": (
                float(max(subdivision_core_nrmse)) if subdivision_core_nrmse else None
            ),
        },
        "details": details,
    }
    def passes(metrics: dict | None) -> bool:
        return bool(
            metrics
            and metrics["nrmse"] <= gates["nrmse"]
            and metrics["p95_abs_over_std"] <= gates["p95_abs_over_std"]
            and metrics["worst_core_nrmse"] <= gates["worst_core_nrmse"]
        )

    report["expanded_context"]["pass"] = passes(report["expanded_context"])
    report["subdivision"]["pass"] = passes(report["subdivision"])
    report["pass"] = bool(
        len(context)
        and len(subdivision)
        and report["expanded_context"]["pass"]
        and report["subdivision"]["pass"]
    )
    return report


def support_coverage(adapter: DensityUnitAdapter, output: Path) -> dict:
    result = {}
    for cap, cap_name in CAP_NAME.items():
        target_path = Path(adapter.target_manifest["components"][cap_name]["file"])
        finite = supported = 0
        with h5py.File(target_path, "r") as truth, h5py.File(field_path(output, cap), "r") as pred:
            support = truth["science_support"]
            values = pred["predicted_delta_r7"]
            step = int(values.chunks[0])
            for left in range(0, values.shape[0], step):
                right = min(left + step, values.shape[0])
                local_support = np.asarray(support[left:right], dtype=bool)
                supported += int(np.count_nonzero(local_support))
                finite += int(np.count_nonzero(np.isfinite(np.asarray(values[left:right])[local_support])))
        result[cap_name] = {
            "supported_voxels": supported,
            "finite_predicted_supported_voxels": finite,
            "coverage_fraction": float(finite / max(supported, 1)),
        }
        if finite != supported:
            raise RuntimeError(f"incomplete stitched support for {cap_name}: {finite}/{supported}")
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("stitched density inference requires an interactive CUDA allocation")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch_load(args.checkpoint, args.device)
    for field, expected in (("model", "U-DENSITY-PHYS-v1"), ("rotation", 0), ("seed", 42)):
        if checkpoint.get(field) != expected:
            raise RuntimeError(f"best checkpoint {field} mismatch")
    model = UNet3D(in_channels=3, latent_channels=1, base=24).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with DensityUnitAdapter(rotation=0) as adapter:
        if checkpoint["target_scaler"] != adapter.scaler:
            raise RuntimeError("best checkpoint target scaler differs from frozen adapter")
        initialize_fields(args.output, adapter, resume=args.resume)
        partition = owner_partition_report(adapter.cores)
        parity = trained_patch_parity(
            model, adapter, parity_rows(adapter, args.parity_cores), args.device,
            args.expanded_halo_voxels, args.parity_nrmse,
            args.parity_p95_over_std, args.parity_worst_core_nrmse,
        )
        atomic_json(args.output / "trained_patch_parity.json", parity)
        if not parity["pass"]:
            raise RuntimeError("trained density patch context/subdivision parity failed")

        progress_path = args.output / "stitch_progress.json"
        start_row = 0
        if args.resume and progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["checkpoint_sha256"] != checkpoint_sha:
                raise RuntimeError("stitch resume checkpoint hash changed")
            start_row = int(progress["next_output_core_id"])
        handles = {
            cap: h5py.File(field_path(args.output, cap), "r+") for cap in CAP_NAME
        }
        try:
            with torch.no_grad():
                for row in range(start_row, len(adapter.cores["output_core_id"])):
                    prediction, _ = infer_bounds(model, adapter, row, args.device, 24)
                    cap = int(adapter.cores["cap"][row])
                    start = np.asarray(adapter.cores["voxel_start"][row], dtype=np.int64)
                    stop = np.asarray(adapter.cores["voxel_stop"][row], dtype=np.int64)
                    selection = tuple(slice(int(start[a]), int(stop[a])) for a in range(3))
                    handles[cap]["predicted_delta_r7"][selection] = prediction
                    if (row + 1) % args.progress_every == 0 or row + 1 == len(adapter.cores["cap"]):
                        for handle in handles.values():
                            handle.flush()
                        atomic_json(progress_path, {
                            "schema_version": "p8-density-stitch-progress-v1",
                            "checkpoint_sha256": checkpoint_sha,
                            "next_output_core_id": row + 1,
                            "output_cores": int(len(adapter.cores["cap"])),
                        })
        finally:
            for handle in handles.values():
                handle.close()
        coverage = support_coverage(adapter, args.output)

    files = {
        CAP_NAME[cap]: {
            "path": str(field_path(args.output, cap)),
            "sha256": sha256(field_path(args.output, cap)),
        }
        for cap in CAP_NAME
    }
    manifest = {
        "schema_version": "p8-density-stitched-field-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_sha(),
        "status": "PASS",
        "model": "U-DENSITY-PHYS-v1",
        "rotation": 0,
        "seed": 42,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_score_macro_shell_r2_delta_r7": float(checkpoint["score"]),
        "field_files": files,
        "owner_partition": partition,
        "trained_patch_parity": parity,
        "support_coverage": coverage,
        "unsupported_voxel_policy": "NaN in artifact; replaced by explicit science window zero only at FFT evaluation",
        "double_smoothing_applied": False,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(args.output / "stitched_field_manifest.json", manifest)
    (args.output / "D0_STITCHED_FIELD_READY").write_text(
        f"checkpoint={checkpoint_sha} epoch={checkpoint['epoch']}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
