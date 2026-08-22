#!/usr/bin/env python3
"""Freeze the capacity-matched R1 field-loader and normalization contract."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import (
    TRAINING_PHASES,
    VALIDATION_PHASE,
    atomic_json,
    sha256,
)
from workflows.abacus_tweb.p6_field_patch_utils import channel_transform


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
VISIBLE_PHASES = TRAINING_PHASES + (VALIDATION_PHASE,)
CHANNELS = ("counts", "exposure_apodized", "log_count_ratio")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"wrong existing symlink: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing contract path: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def build_phase_links(base: Path, output: Path) -> None:
    for phase in VISIBLE_PHASES:
        relative_symlink(base / "phases" / phase, output / "phases" / phase)


def build_field_adapter(base: Path, output: Path, root: Path, phase: str) -> dict:
    source = base / "adapters" / phase / "field"
    destination = output / "adapters" / phase / "field"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.npy"):
        relative_symlink(path, destination / path.name)
    overlay_manifest_path = root / phase / "p3b_random_response_v1/manifest.json"
    overlay = load(overlay_manifest_path)
    if not overlay.get("pass") or overlay.get("ph001_opened"):
        raise RuntimeError(f"{phase} P3b-R overlay is absent, failing, or blind-contaminated")
    old = load(source / "adapter_manifest.json")
    caps = {}
    for cap_name, record in old["caps"].items():
        component = overlay["components"][cap_name]
        caps[cap_name] = {
            **record,
            "field_path": component["file"],
            "field_sha256": component["file_sha256"],
            "selection_expected_to_input_by_shell": {
                shell: values["expected_count_sum"] / max(values["input_galaxies"], 1.0)
                for shell, values in component["support_atlas"].items()
            },
        }
    manifest = {
        **old,
        "schema_version": "p3br-r1-field-patch-adapter-v1",
        "stage": "P3b-R capacity-matched R1 field-patch view",
        "p3_manifest": str(overlay_manifest_path),
        "p3_manifest_sha256": sha256(overlay_manifest_path),
        "channel_order": list(CHANNELS),
        "caps": caps,
        "selection_channel_status": {
            "ready_for_u_patch_training": True,
            "reason": "stored P3b-R expected/log-ratio channels include frozen radial and random angular response",
            "failures_over_5pct": [],
        },
        "response_overlay_only": True,
        "ph001_opened": False,
        "pass": bool(old["pass"] and overlay["pass"]),
    }
    path = destination / "adapter_manifest.json"
    atomic_json(path, manifest)
    return {"path": str(path), "sha256": sha256(path), "pass": manifest["pass"]}


def moments(values: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(values, dtype=np.float64)
    return int(values.size), float(values.sum()), float(np.square(values).sum())


def fit_field_normalization(output: Path, adapters: dict, base: Path) -> dict:
    per_phase = {}
    for phase in TRAINING_PHASES:
        adapter = load(Path(adapters[phase]["path"]))
        accum = defaultdict(lambda: [0, 0.0, 0.0])
        for cap in ("NGC", "SGC"):
            with h5py.File(adapter["caps"][cap]["field_path"], "r") as handle:
                exposure_ds = handle["exposure_apodized"]
                for slices in exposure_ds.iter_chunks():
                    exposure = np.asarray(exposure_ds[slices], dtype=np.float32)
                    supported = exposure > np.float32(1.0e-4)
                    if not supported.any():
                        continue
                    for channel in ("counts", "log_count_ratio"):
                        values = channel_transform(
                            channel, np.asarray(handle[channel][slices], dtype=np.float32)[supported]
                        )
                        n, total, total2 = moments(values)
                        accum[channel][0] += n
                        accum[channel][1] += total
                        accum[channel][2] += total2
        per_phase[phase] = {}
        for channel, (n, total, total2) in accum.items():
            mean = total / n
            per_phase[phase][channel] = {
                "count": n,
                "mean": mean,
                "second_moment": total2 / n,
                "std": max(total2 / n - mean * mean, 0.0) ** 0.5,
            }
    base_transform = load(base / "transforms/field/field_transform.json")
    normalization = {"channels": {
        "counts": base_transform["normalization"]["channels"]["counts"],
        "exposure_apodized": {"policy": "identity"},
    }}
    for channel in ("log_count_ratio",):
        mean = float(np.mean([per_phase[phase][channel]["mean"] for phase in TRAINING_PHASES]))
        second = float(np.mean([
            per_phase[phase][channel]["second_moment"] for phase in TRAINING_PHASES
        ]))
        std = max(second - mean * mean, 0.0) ** 0.5
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise RuntimeError(f"invalid R1 normalization for {channel}")
        normalization["channels"][channel] = {"policy": "zscore", "mean": mean, "std": std}
    manifest = {
        "schema_version": "p3br-r1-field-transform-v1",
        "fit_phases": list(TRAINING_PHASES),
        "fit_policy": "equal phase mixture of exact random-supported voxel moments",
        "counts_normalization_policy": (
            "copied exactly from frozen R0 because the BRIGHT count field is unchanged; "
            "only the response-derived log-ratio scaler is refit on training phases"
        ),
        "channels": list(CHANNELS),
        "normalization": normalization,
        "per_phase_diagnostics": per_phase,
        "no_patch_local_statistics": True,
        "ph001_opened": False,
        "pass": True,
    }
    path = output / "transforms/field/field_transform.json"
    atomic_json(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-contract", type=Path, default=ROOT / "training_contract")
    parser.add_argument("--output", type=Path, default=ROOT / "training_contract_r1_random")
    args = parser.parse_args()
    marker = load(args.base_contract / "TRAINING_LOADER_READY.json")
    if not marker.get("pass"):
        raise RuntimeError("base P10 training loader is not ready")
    args.output.mkdir(parents=True, exist_ok=True)
    build_phase_links(args.base_contract, args.output)
    adapters = {
        phase: build_field_adapter(args.base_contract, args.output, args.root, phase)
        for phase in VISIBLE_PHASES
    }
    atomic_json(args.output / "adapter_inventory.json", {
        "schema_version": "p3br-r1-adapter-inventory-v1",
        "phases": {
            phase: {
                "field_manifest": row["path"],
                "field_manifest_sha256": row["sha256"],
            }
            for phase, row in adapters.items()
        },
        "ph001_product_built": False,
        "pass": all(row["pass"] for row in adapters.values()),
    })
    transforms = args.output / "transforms"
    transforms.mkdir(parents=True, exist_ok=True)
    relative_symlink(
        args.base_contract / "transforms/target_scaler.json",
        transforms / "target_scaler.json",
    )
    field = fit_field_normalization(args.output, adapters, args.base_contract)
    ready = {
        **marker,
        "schema_version": "p3br-r1-training-loader-ready-v1",
        "view": "R1 BRIGHT counts plus random-derived response, capacity matched to R0",
        "field_channels": list(CHANNELS),
        "adapters": adapters,
        "field_transform": str(args.output / "transforms/field/field_transform.json"),
        "field_transform_sha256": sha256(args.output / "transforms/field/field_transform.json"),
        "base_contract": str(args.base_contract),
        "base_contract_marker_sha256": sha256(args.base_contract / "TRAINING_LOADER_READY.json"),
        "ph001_product_built": False,
        "ph001_opened": False,
        "pass": bool(field["pass"] and all(row["pass"] for row in adapters.values())),
    }
    atomic_json(args.output / "TRAINING_LOADER_READY.json", ready)
    if not ready["pass"]:
        raise RuntimeError("P3b-R R1 loader freeze failed")
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
