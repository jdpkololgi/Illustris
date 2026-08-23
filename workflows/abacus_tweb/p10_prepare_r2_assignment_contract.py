#!/usr/bin/env python3
"""Freeze the P10 R2 field adapter and normalization contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_build_r2_assignment_overlays import R2_MODEL_CHANNELS
from workflows.abacus_tweb.p10_training_contract import (
    TRAINING_PHASES,
    VALIDATION_PHASE,
    atomic_json,
    sha256,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
VISIBLE_PHASES = TRAINING_PHASES + (VALIDATION_PHASE,)


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

    overlay_path = root / phase / "p3c_assignment_response_v1/manifest.json"
    overlay = load(overlay_path)
    if not overlay.get("pass") or overlay.get("blind_phase_opened"):
        raise RuntimeError(f"{phase}: R2 overlay is absent, failing, or blind-contaminated")
    old = load(source / "adapter_manifest.json")
    caps = {}
    for cap, record in old["caps"].items():
        component = overlay["components"][cap]
        caps[cap] = {
            **record,
            "field_path": component["file"],
            "field_sha256": component["file_sha256"],
        }
    manifest = {
        **old,
        "schema_version": "p10-r2-field-patch-adapter-v1",
        "stage": "P10 R2 assignment-response field-patch view",
        "p3_manifest": str(overlay_path),
        "p3_manifest_sha256": sha256(overlay_path),
        "channel_order": list(R2_MODEL_CHANNELS),
        "caps": caps,
        "response_contract": {
            "M_mu": "inherited exactly from frozen R1 random-response overlay",
            "C_fibre": ["c_fibre_tileloc", "c_fibre_tiles"],
            "C_fibre_defined": "explicit binary flag; neutral response where zero",
            "C_z": "stored in overlay but omitted from model input because constant and non-informative",
        },
        "response_overlay_only": True,
        "geometry_arrays": geometry,
        "p3a_parent_core_index_parity": all(row["exact_identity"] for row in geometry.values()),
        "ph001_opened": False,
        "pass": bool(
            old["pass"]
            and overlay["pass"]
            and all(row["exact_identity"] for row in geometry.values())
        ),
    }
    path = destination / "adapter_manifest.json"
    atomic_json(path, manifest)
    return {"path": str(path), "sha256": sha256(path), "pass": manifest["pass"]}


def build_transform(base: Path, output: Path) -> dict:
    old = load(base / "transforms/field/field_transform.json")
    normalization = json.loads(json.dumps(old["normalization"]))
    for channel in ("c_fibre_tileloc", "c_fibre_tiles", "c_fibre_defined"):
        normalization["channels"][channel] = {"policy": "identity"}
    transform = {
        "schema_version": "p10-r2-field-transform-v1",
        "fit_phases": list(TRAINING_PHASES),
        "channels": list(R2_MODEL_CHANNELS),
        "normalization": normalization,
        "base_r1_transform": str(base / "transforms/field/field_transform.json"),
        "base_r1_transform_sha256": sha256(base / "transforms/field/field_transform.json"),
        "assignment_response_policy": (
            "bounded physical probabilities and binary definition flag use identity transforms; "
            "no patch-local or validation-phase statistics"
        ),
        "no_patch_local_statistics": True,
        "ph001_opened": False,
        "pass": True,
    }
    path = output / "transforms/field/field_transform.json"
    atomic_json(path, transform)
    return transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--base-contract", type=Path, default=ROOT / "training_contract_r1_random"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "training_contract_r2_assignment"
    )
    args = parser.parse_args()

    base_marker = load(args.base_contract / "TRAINING_LOADER_READY.json")
    if not base_marker.get("pass") or base_marker.get("ph001_opened"):
        raise RuntimeError("base R1 loader is not freezeable")
    overlay_marker = load(
        args.root / "r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json"
    )
    if not overlay_marker.get("pass") or overlay_marker.get("blind_phase_opened"):
        raise RuntimeError("R2 overlays are not freezeable")

    args.output.mkdir(parents=True, exist_ok=True)
    build_phase_links(args.base_contract, args.output)
    adapters = {
        phase: build_adapter(args.base_contract, args.output, args.root, phase)
        for phase in VISIBLE_PHASES
    }
    atomic_json(
        args.output / "adapter_inventory.json",
        {
            "schema_version": "p10-r2-adapter-inventory-v1",
            "phases": adapters,
            "ph001_product_built": False,
            "pass": all(record["pass"] for record in adapters.values()),
        },
    )
    transforms = args.output / "transforms"
    transforms.mkdir(parents=True, exist_ok=True)
    relative_symlink(
        args.base_contract / "transforms/target_scaler.json",
        transforms / "target_scaler.json",
    )
    field = build_transform(args.base_contract, args.output)
    ready = {
        **base_marker,
        "schema_version": "p10-r2-training-loader-ready-v1",
        "view": "R2 BRIGHT plus random M/mu and audited assignment response",
        "field_channels": list(R2_MODEL_CHANNELS),
        "adapters": adapters,
        "field_transform": str(args.output / "transforms/field/field_transform.json"),
        "field_transform_sha256": sha256(args.output / "transforms/field/field_transform.json"),
        "base_contract": str(args.base_contract),
        "base_contract_marker_sha256": sha256(
            args.base_contract / "TRAINING_LOADER_READY.json"
        ),
        "assignment_overlays_marker": str(
            args.root / "r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json"
        ),
        "assignment_overlays_marker_sha256": sha256(
            args.root / "r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json"
        ),
        "ph001_product_built": False,
        "ph001_opened": False,
        "throughput_canary_pass": False,
        "pass": bool(field["pass"] and all(record["pass"] for record in adapters.values())),
    }
    atomic_json(args.output / "TRAINING_LOADER_READY.json", ready)
    if not ready["pass"]:
        raise RuntimeError("R2 loader freeze failed")
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
