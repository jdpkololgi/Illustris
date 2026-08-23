#!/usr/bin/env python3
"""Freeze the high-S/N P10 R3-RF empirical random-field products and loader."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_r3_random_field_contract import R3_RF_MODEL_CHANNELS
from workflows.abacus_tweb.p10_training_contract import (
    TRAINING_PHASES,
    VALIDATION_PHASE,
    atomic_json,
    sha256,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
VISIBLE_PHASES = TRAINING_PHASES + (VALIDATION_PHASE,)
CAPS = ("NGC", "SGC")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"wrong existing symlink: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def validate_component(path: Path) -> dict:
    checks = {
        "finite": True,
        "nonnegative_intensity": True,
        "binary_support": True,
        "zero_outside_support": True,
    }
    supported = 0
    intensity_sum = 0.0
    response_sum = 0.0
    with h5py.File(path, "r") as handle:
        required = set(R3_RF_MODEL_CHANNELS)
        if not required.issubset(handle):
            raise RuntimeError(f"{path}: missing R3-RF datasets {sorted(required - set(handle))}")
        if int(handle.attrs.get("random_realisation_count", -1)) != 18:
            raise RuntimeError(f"{path}: R3-RF requires the frozen all-18 random aggregate")
        support_ds = handle["support_random"]
        for slices in support_ds.iter_chunks():
            support = np.asarray(support_ds[slices], dtype=np.uint8)
            intensity = np.asarray(handle["expected_counts_random"][slices], dtype=np.float32)
            response = np.asarray(handle["angular_response"][slices], dtype=np.float32)
            checks["finite"] &= bool(np.isfinite(intensity).all() and np.isfinite(response).all())
            checks["nonnegative_intensity"] &= bool(np.all(intensity >= 0))
            checks["binary_support"] &= bool(np.all((support == 0) | (support == 1)))
            outside = support == 0
            checks["zero_outside_support"] &= bool(
                np.all(intensity[outside] == 0) and np.all(response[outside] == 0)
            )
            inside = support == 1
            supported += int(np.count_nonzero(inside))
            intensity_sum += float(np.sum(intensity[inside], dtype=np.float64))
            response_sum += float(np.sum(response[inside], dtype=np.float64))
    record = {
        "file": str(path),
        "file_sha256": sha256(path),
        "random_realisation_count": 18,
        "supported_voxels": supported,
        "expected_intensity_sum": intensity_sum,
        "mean_angular_response_on_support": response_sum / max(supported, 1),
        "checks": checks,
        "pass": bool(supported > 0 and all(checks.values())),
    }
    if not record["pass"]:
        raise RuntimeError(f"{path}: R3-RF component validation failed")
    return record


def freeze_products(root: Path, output: Path) -> dict:
    records = {}
    for phase in VISIBLE_PHASES:
        manifest_path = root / phase / "p3b_random_response_v1/manifest.json"
        manifest = load(manifest_path)
        if not manifest.get("pass") or manifest.get("ph001_opened"):
            raise RuntimeError(f"{phase}: P3b-R source is not freezeable")
        if int(manifest.get("random_realisation_count", -1)) != 18:
            raise RuntimeError(f"{phase}: P3b-R source is not the all-18 aggregate")
        records[phase] = {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path),
            "components": {
                cap: validate_component(Path(manifest["components"][cap]["file"]))
                for cap in CAPS
            },
            "pass": True,
        }
    marker = {
        "schema_version": "p10-r3-rf-products-ready-v1",
        "view": (
            "high-S/N empirical BRIGHT random field: all-18 expected intensity, "
            "voxel angular response and binary support"
        ),
        "radial_contract": "frozen BRIGHT ntilde(z) already embedded in expected_counts_random",
        "spatial_contract": "native P3 Cartesian voxels; no FAINT data and no clustering-random Z",
        "random_realisation_count": 18,
        "model_channels": list(R3_RF_MODEL_CHANNELS),
        "phases": records,
        "ph001_product_built": False,
        "ph001_opened": False,
        "pass": all(row["pass"] for row in records.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "R3_RANDOM_FIELD_PRODUCTS_READY.json", marker)
    return marker


def build_phase_links(base: Path, output: Path) -> None:
    for phase in VISIBLE_PHASES:
        relative_symlink(base / "phases" / phase, output / "phases" / phase)


def build_adapter(base: Path, output: Path, root: Path, phase: str) -> dict:
    source = base / "adapters" / phase / "field"
    destination = output / "adapters" / phase / "field"
    destination.mkdir(parents=True, exist_ok=True)
    geometry = {}
    for path in source.glob("*.npy"):
        relative_symlink(path, destination / path.name)
        linked = destination / path.name
        geometry[path.name] = {
            "source": str(path),
            "source_sha256": sha256(path),
            "linked": str(linked),
            "linked_sha256": sha256(linked),
            "exact_identity": linked.resolve() == path.resolve() and sha256(linked) == sha256(path),
        }
    source_manifest = root / phase / "p3b_random_response_v1/manifest.json"
    random_view = load(source_manifest)
    old = load(source / "adapter_manifest.json")
    caps = {}
    for cap, record in old["caps"].items():
        component = random_view["components"][cap]
        caps[cap] = {
            **record,
            "field_path": component["file"],
            "field_sha256": component["file_sha256"],
        }
    manifest = {
        **old,
        "schema_version": "p10-r3-rf-field-patch-adapter-v1",
        "stage": "P10 R3-RF high-S/N empirical random-field view",
        "p3_manifest": str(source_manifest),
        "p3_manifest_sha256": sha256(source_manifest),
        "channel_order": list(R3_RF_MODEL_CHANNELS),
        "caps": caps,
        "response_contract": {
            "bright_triplet": ["counts", "log_count_ratio", "exposure_apodized"],
            "random_triplet": [
                "expected_counts_random",
                "angular_response",
                "support_random",
            ],
            "interpretation": (
                "voxel-resolved all-18 random response; no FAINT context and no "
                "clustering-random redshift"
            ),
        },
        "geometry_arrays": geometry,
        "p3a_parent_core_index_parity": all(row["exact_identity"] for row in geometry.values()),
        "ph001_opened": False,
        "pass": bool(old["pass"] and all(row["exact_identity"] for row in geometry.values())),
    }
    path = destination / "adapter_manifest.json"
    atomic_json(path, manifest)
    return {"path": str(path), "sha256": sha256(path), "pass": manifest["pass"]}


def fit_random_intensity(adapters: dict) -> tuple[dict, dict]:
    per_phase = {}
    for phase in TRAINING_PHASES:
        adapter = load(Path(adapters[phase]["path"]))
        n = 0
        total = 0.0
        total2 = 0.0
        for cap in CAPS:
            with h5py.File(adapter["caps"][cap]["field_path"], "r") as handle:
                support_ds = handle["support_random"]
                for slices in support_ds.iter_chunks():
                    support = np.asarray(support_ds[slices], dtype=bool)
                    if not support.any():
                        continue
                    values = np.log1p(
                        np.asarray(handle["expected_counts_random"][slices], dtype=np.float32)[support]
                    ).astype(np.float64)
                    n += int(values.size)
                    total += float(values.sum())
                    total2 += float(np.square(values).sum())
        mean = total / n
        second = total2 / n
        per_phase[phase] = {
            "count": n,
            "mean": mean,
            "second_moment": second,
            "std": max(second - mean * mean, 0.0) ** 0.5,
        }
    mean = float(np.mean([per_phase[p]["mean"] for p in TRAINING_PHASES]))
    second = float(np.mean([per_phase[p]["second_moment"] for p in TRAINING_PHASES]))
    std = max(second - mean * mean, 0.0) ** 0.5
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError("invalid R3-RF random-intensity scaler")
    return {"policy": "zscore", "pretransform": "log1p", "mean": mean, "std": std}, per_phase


def freeze_loader(root: Path, products_root: Path, base: Path, output: Path) -> dict:
    base_marker = load(base / "TRAINING_LOADER_READY.json")
    products_marker = load(products_root / "R3_RANDOM_FIELD_PRODUCTS_READY.json")
    if not base_marker.get("pass") or not products_marker.get("pass"):
        raise RuntimeError("R1 base or R3-RF product marker does not pass")
    output.mkdir(parents=True, exist_ok=True)
    build_phase_links(base, output)
    adapters = {
        phase: build_adapter(base, output, root, phase)
        for phase in VISIBLE_PHASES
    }
    transforms = output / "transforms"
    transforms.mkdir(parents=True, exist_ok=True)
    relative_symlink(base / "transforms/target_scaler.json", transforms / "target_scaler.json")
    old = load(base / "transforms/field/field_transform.json")
    intensity, per_phase = fit_random_intensity(adapters)
    normalization = json.loads(json.dumps(old["normalization"]))
    normalization["channels"]["expected_counts_random"] = intensity
    normalization["channels"]["angular_response"] = {"policy": "identity"}
    normalization["channels"]["support_random"] = {"policy": "identity"}
    transform = {
        "schema_version": "p10-r3-rf-field-transform-v1",
        "fit_phases": list(TRAINING_PHASES),
        "channels": list(R3_RF_MODEL_CHANNELS),
        "normalization": normalization,
        "random_intensity_per_phase": per_phase,
        "random_triplet_model_mapping": {
            "expected_counts_random": "log1p then frozen equal-phase zscore",
            "angular_response": "clip(response - 1, -1, 20)",
            "support_random": "binary identity",
        },
        "no_patch_local_statistics": True,
        "ph001_opened": False,
        "pass": True,
    }
    transform_path = transforms / "field/field_transform.json"
    atomic_json(transform_path, transform)
    ready = {
        **base_marker,
        "schema_version": "p10-r3-rf-training-loader-ready-v1",
        "view": "R3-RF BRIGHT triplet plus high-S/N all-18 empirical random-field triplet",
        "field_channels": list(R3_RF_MODEL_CHANNELS),
        "adapters": adapters,
        "field_transform": str(transform_path),
        "field_transform_sha256": sha256(transform_path),
        "products_marker": str(products_root / "R3_RANDOM_FIELD_PRODUCTS_READY.json"),
        "products_marker_sha256": sha256(
            products_root / "R3_RANDOM_FIELD_PRODUCTS_READY.json"
        ),
        "base_contract": str(base),
        "base_contract_marker_sha256": sha256(base / "TRAINING_LOADER_READY.json"),
        "ph001_product_built": False,
        "ph001_opened": False,
        "throughput_canary_pass": False,
        "pass": bool(transform["pass"] and all(row["pass"] for row in adapters.values())),
    }
    atomic_json(output / "adapter_inventory.json", {
        "schema_version": "p10-r3-rf-adapter-inventory-v1",
        "phases": adapters,
        "ph001_product_built": False,
        "pass": all(row["pass"] for row in adapters.values()),
    })
    atomic_json(output / "TRAINING_LOADER_READY.json", ready)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--products-root", type=Path, default=ROOT / "r3_random_field_v1"
    )
    parser.add_argument(
        "--base-contract", type=Path, default=ROOT / "training_contract_r1_random"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "training_contract_r3_random_field"
    )
    args = parser.parse_args()
    freeze_products(args.root, args.products_root)
    ready = freeze_loader(
        args.root, args.products_root, args.base_contract, args.output
    )
    if not ready["pass"]:
        raise RuntimeError("R3-RF loader freeze failed")
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

