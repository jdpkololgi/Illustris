#!/usr/bin/env python3
"""Validate strict P10 sparse-random and cross-phase FAINT control roots.

The validation is intentionally input-only: ph001 and tidal targets are never
opened.  It verifies schema/provenance, HDF5 links and a representative frozen
six-channel U-PATCH extraction for every visible phase.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_build_strict_multitracer_controls import VISIBLE
from workflows.abacus_tweb.p10_multitracer_training import (
    P10MultitracerFieldAdapter,
    model_inputs,
)
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


BASE = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/v1")
CONTRACT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def representative_core(loader: P10PhaseBalancedLoader, phase: str) -> int:
    filename = "validation_core_id.npy" if phase == loader.validation_phase else "training_core_id.npy"
    core_ids = np.load(loader.root / "phases" / phase / filename, mmap_mode="r")
    if not len(core_ids):
        raise RuntimeError(f"{phase}: no eligible core for loader smoke test")
    return int(core_ids[0])


def validate_component(component: dict) -> dict:
    path = Path(component["file"])
    audit = component["strict_control_audit"]
    if not path.is_file() or not audit.get("pass"):
        raise RuntimeError(f"component does not pass: {path}")
    with h5py.File(path, "r") as handle:
        required = ("counts", "exposure_apodized", "exposure_binary", "response_angular")
        if any(name not in handle for name in required):
            raise RuntimeError(f"{path}: missing a required dataset/link")
        shapes = {name: tuple(handle[name].shape) for name in required}
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f"{path}: linked response arrays have different shapes")
        count_sum = float(handle["counts"].attrs.get("count_sum", handle.attrs["count_sum"]))
        if not np.isfinite(count_sum) or abs(count_sum - float(audit["count_sum"])) > 1.0e-3:
            raise RuntimeError(f"{path}: count-sum metadata mismatch")
        sample = np.asarray(handle["counts"][::max(shapes["counts"][0] // 7, 1), ::max(shapes["counts"][1] // 7, 1), ::max(shapes["counts"][2] // 7, 1)])
        if not np.all(np.isfinite(sample)) or np.any(sample < 0):
            raise RuntimeError(f"{path}: invalid sampled counts")
    return {
        "file": str(path),
        "file_sha256_registered": component["file_sha256"],
        "shape": list(shapes["counts"]),
        "count_sum": count_sum,
        "control": component["control"],
        "donor_phase": component.get("donor_phase"),
        "pass": True,
    }


def validate(root: Path) -> dict:
    ready_path = root / "P10_MULTITRACER_VIEWS_READY.json"
    ready = load(ready_path)
    if not ready.get("pass") or ready.get("sealed_phase_opened"):
        raise RuntimeError("strict-control readiness marker does not pass")
    if ready.get("sealed_phase") != "ph001" or (root / "phases/ph001").exists():
        raise RuntimeError("strict-control root violates the ph001 seal")
    base_selection = BASE / "selection_manifest.json"
    if sha256(root / "selection_manifest.json") != sha256(base_selection):
        raise RuntimeError("strict-control selection contract changed")

    loader = P10PhaseBalancedLoader(CONTRACT)
    phase_rows = {}
    loader_smoke = {}
    for phase in VISIBLE:
        phase_path = root / "phases" / phase / "PHASE_MULTITRACER_VIEWS_READY.json"
        row = load(phase_path)
        if not row.get("pass") or row.get("ph001_opened") or row.get("labels_read_by_builder"):
            raise RuntimeError(f"{phase}: phase contract does not pass input-only gates")
        phase_rows[phase] = {
            cap: validate_component(row["proxy"]["components"][cap])
            for cap in ("NGC", "SGC")
        }

        adapter = P10MultitracerFieldAdapter(
            loader=loader, phase=phase, root=root, view="proxy"
        )
        try:
            core_id = representative_core(loader, phase)
            bright, values, points = model_inputs(
                adapter, adapter.extract(core_id), "cpu"
            )
            if tuple(values.shape[:2]) != (1, 6):
                raise RuntimeError(f"{phase}: U-PATCH input is not six-channel")
            if points.shape[-2] != len(bright.authoritative_parent_id):
                raise RuntimeError(f"{phase}: authoritative point/parent mismatch")
            if not bool(values.isfinite().all()) or not bool(points.isfinite().all()):
                raise RuntimeError(f"{phase}: non-finite loader smoke tensor")
            loader_smoke[phase] = {
                "core_id": core_id,
                "cap": int(bright.cap),
                "input_shape": list(values.shape),
                "authoritative_points": int(points.shape[-2]),
                "pass": True,
            }
        finally:
            adapter.close()

    payload = {
        "schema_version": "p10-strict-multitracer-validation-v1",
        "created_utc": utc_now(),
        "root": str(root),
        "control": ready["control"],
        "control_name": ready["control_name"],
        "visible_phases": list(VISIBLE),
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "targets_opened_by_validator": False,
        "selection_manifest_sha256": sha256(root / "selection_manifest.json"),
        "phase_components": phase_rows,
        "loader_smoke": loader_smoke,
        "pass": True,
    }
    atomic_json(root / "STRICT_CONTROL_LOADER_SMOKE.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
