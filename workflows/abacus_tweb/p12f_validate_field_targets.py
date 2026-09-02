#!/usr/bin/env python3
"""Aggregate validation for visible P12-F field targets.

The audit verifies a common schema/physics contract across ph000 and ph002--006,
checks a deterministic sample of stored fields, and proves ph000 target parity
with the historical D0 target.  It reads no ph001 path or truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p12f_build_field_targets import VISIBLE_PHASES


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
HISTORICAL = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/targets/target_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--historical-ph000", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=1024)
    return parser.parse_args()


def sample_indices(shape: tuple[int, int, int], count: int, *, seed: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("sample count must be positive")
    total = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=min(count, total), replace=False))


def read_flat_points(dataset: h5py.Dataset, flat: np.ndarray) -> np.ndarray:
    shape = tuple(int(value) for value in dataset.shape)
    coordinates = np.unravel_index(np.asarray(flat, dtype=np.int64), shape)
    return np.asarray(
        [dataset[int(x), int(y), int(z)] for x, y, z in zip(*coordinates, strict=True)],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    historical = json.loads(args.historical_ph000.read_text())
    records = {}
    schemas = set()
    contracts = set()
    phase_means, phase_stds = [], []
    parity_max_abs = 0.0
    all_finite = True
    all_support_nonempty = True

    for phase_index, phase in enumerate(VISIBLE_PHASES):
        marker_path = args.root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("phase") != phase
            or not marker.get("pass")
            or marker.get("ph001_opened")
            or marker.get("sealed_blind_phase") != "ph001"
        ):
            raise RuntimeError(f"{phase}: field-target marker failed its sealed contract")
        schemas.add(marker["schema_version"])
        contracts.add(json.dumps(marker["contract"], sort_keys=True))
        cap_rows = {}
        for cap_index, cap in enumerate(("NGC", "SGC")):
            component = marker["components"][cap]
            shape = tuple(int(value) for value in component["grid"]["shape"])
            flat = sample_indices(shape, args.sample_points, seed=101 * phase_index + cap_index)
            with h5py.File(component["file"], "r") as handle:
                delta = read_flat_points(handle["delta_r7"], flat)
                support = read_flat_points(handle["science_support"], flat)
                if tuple(handle["delta_r7"].shape) != shape:
                    raise RuntimeError(f"{phase} {cap}: HDF5/grid shape mismatch")
                if bool(handle.attrs["double_smoothing_applied"]):
                    raise RuntimeError(f"{phase} {cap}: double smoothing was applied")
            finite = bool(np.all(np.isfinite(delta)))
            all_finite = all_finite and finite
            all_support_nonempty = all_support_nonempty and bool(
                component["science_supported_voxels"] > 0
            )
            if phase == "ph000":
                historical_component = historical["components"][cap]
                if historical_component["grid"] != component["grid"]:
                    raise RuntimeError(f"ph000 {cap}: historical grid changed")
                with h5py.File(historical_component["file"], "r") as old:
                    old_delta = read_flat_points(old["delta_r7"], flat)
                parity_max_abs = max(
                    parity_max_abs, float(np.max(np.abs(delta - old_delta)))
                )
            cap_rows[cap] = {
                "file": component["file"],
                "file_sha256": component["file_sha256"],
                "sample_points": int(len(flat)),
                "sample_all_finite": finite,
                "sample_science_support_fraction": float(np.mean(support > 0.5)),
                "target_mean": component["target"]["mean"],
                "target_std": component["target"]["std"],
                "science_supported_voxels": component["science_supported_voxels"],
            }
            phase_means.append(float(component["target"]["mean"]))
            phase_stds.append(float(component["target"]["std"]))
        records[phase] = {
            "marker": str(marker_path.resolve()),
            "marker_sha256": sha256(marker_path),
            "components": cap_rows,
        }

    gates = {
        "all_six_visible_phases_present": len(records) == 6,
        "single_schema": len(schemas) == 1,
        "single_physics_contract": len(contracts) == 1,
        "all_sampled_targets_finite": all_finite,
        "all_exact_supports_nonempty": all_support_nonempty,
        "ph000_historical_delta_sample_max_abs_zero": parity_max_abs == 0.0,
        "phase_cap_target_mean_absolute_max_below_0p02": max(map(abs, phase_means)) < 0.02,
        "phase_cap_target_std_range_0p4_0p55": min(phase_stds) > 0.4
        and max(phase_stds) < 0.55,
        "ph001_sealed": True,
    }
    report = {
        "schema_version": "p12f-visible-field-target-aggregate-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phases": list(VISIBLE_PHASES),
        "records": records,
        "target_distribution": {
            "component_mean_minimum": min(phase_means),
            "component_mean_maximum": max(phase_means),
            "component_std_minimum": min(phase_stds),
            "component_std_maximum": max(phase_stds),
        },
        "ph000_historical_delta_sample_max_abs": parity_max_abs,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "ph001_opened": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise RuntimeError("P12-F visible field-target aggregate audit failed")


if __name__ == "__main__":
    main()
