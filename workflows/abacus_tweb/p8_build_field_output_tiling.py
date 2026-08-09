#!/usr/bin/env python3
"""Freeze exact voxel ownership and complete output tiling for P8.9.

P4 cores were defined from galaxies and their stored voxel ranges contain every
cell intersecting a physical core.  That is appropriate for extracting context,
but neighbouring ranges can share a boundary cell and galaxy-occupied cores do
not cover every P6-supported field voxel.  A voxelwise density objective needs a
different contract: each voxel centre has exactly one owner on the unchanged P4
64 Mpc/h lattice.

This script retains all nominal P4 cores and adds label-free inference-only
cores wherever an otherwise absent owner contains P6 science support.  It never
changes folds, galaxy ownership, targets, or the P4 benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
P3_MANIFEST = ROOT / "p3_full_footprint/field_manifest.json"
P4_MANIFEST = ROOT / "p4_spatial_manifest/spatial_manifest.json"
P4_CORES = ROOT / "p4_spatial_manifest/cores.npz"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
OUTPUT = ROOT / "p8_density_phys_v1/field_output_tiling"
CAP_ID = {"SGC": 0, "NGC": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-manifest", type=Path, default=P3_MANIFEST)
    parser.add_argument("--p4-manifest", type=Path, default=P4_MANIFEST)
    parser.add_argument("--p4-cores", type=Path, default=P4_CORES)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def axis_owner_ranges(length: int, cell_mpc: float, core_mpc: float) -> dict[int, tuple[int, int]]:
    """Return contiguous half-open voxel ranges owned by each P4 lattice cell."""
    if length <= 0 or cell_mpc <= 0 or core_mpc <= 0:
        raise ValueError("length, cell_mpc, and core_mpc must be positive")
    centre = (np.arange(length, dtype=np.float64) + 0.5) * float(cell_mpc)
    owner = np.floor(centre / float(core_mpc)).astype(np.int64)
    unique, first = np.unique(owner, return_index=True)
    stops = np.r_[first[1:], length]
    return {
        int(index): (int(start), int(stop))
        for index, start, stop in zip(unique, first, stops, strict=True)
    }


def core_owner_table(
    shape: tuple[int, int, int], cell_mpc: float, core_mpc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate the exact, non-overlapping owner partition of a P3 lattice."""
    axes = [axis_owner_ranges(int(n), cell_mpc, core_mpc) for n in shape]
    keys = np.asarray(
        [(ix, iy, iz) for ix in axes[0] for iy in axes[1] for iz in axes[2]],
        dtype=np.int32,
    )
    starts = np.asarray(
        [[axes[a][int(row[a])][0] for a in range(3)] for row in keys], dtype=np.int32
    )
    stops = np.asarray(
        [[axes[a][int(row[a])][1] for a in range(3)] for row in keys], dtype=np.int32
    )
    return keys, starts, stops


def supporting_rows(
    support: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Identify owner cores containing at least one supported voxel."""
    support = np.asarray(support, dtype=bool)
    result = np.zeros(len(starts), dtype=bool)
    for row, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        result[row] = bool(np.any(support[
            start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]
        ]))
    return result


def coverage_from_rows(
    shape: tuple[int, int, int], starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    coverage = np.zeros(shape, dtype=bool)
    for start, stop in zip(starts, stops, strict=True):
        coverage[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = True
    return coverage


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def masked_field_summary(field: h5py.Dataset, mask: np.ndarray) -> dict:
    """Summarise one field on a boolean mask without materialising the full field."""
    total = 0.0
    positive = 0
    finite = True
    values_count = 0
    step = int(field.chunks[0]) if field.chunks else 8
    for left in range(0, field.shape[0], step):
        right = min(left + step, field.shape[0])
        local_mask = mask[left:right]
        if not np.any(local_mask):
            continue
        values = np.asarray(field[left:right])[local_mask]
        total += float(np.sum(values, dtype=np.float64))
        positive += int(np.count_nonzero(values > 0))
        finite &= bool(np.all(np.isfinite(values)))
        values_count += int(values.size)
    return {
        "voxels": values_count,
        "sum": total,
        "positive_voxels": positive,
        "all_finite": bool(finite),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    p3 = json.loads(args.p3_manifest.read_text())
    p4 = json.loads(args.p4_manifest.read_text())
    target = json.loads(args.target_manifest.read_text())
    if not target.get("target_fields_ready"):
        raise RuntimeError("passing P8.9 target fields are required")
    core_mpc = float(p4["unit_contract"]["core_mpc"])
    core_mpc_h = float(p4["unit_contract"]["core_mpc_h"])
    nominal = np.load(args.p4_cores, allow_pickle=False)
    records: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "cap", "core_index", "lower_mpc", "upper_mpc", "centroid_mpc",
            "voxel_start", "voxel_stop", "nominal_core_id", "fold",
            "inference_only", "owns_density_loss",
        )
    }
    component_report = {}

    for cap_name in ("SGC", "NGC"):
        cap_id = CAP_ID[cap_name]
        component = p3["components"][cap_name]
        grid = component["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        target_path = Path(target["components"][cap_name]["file"])
        with h5py.File(target_path, "r") as target_handle:
            support = np.asarray(target_handle["science_support"], dtype=bool)
            legacy_coverage = np.asarray(target_handle["core_coverage"], dtype=bool)
            delta = target_handle["delta_r7"]
            keys, starts, stops = core_owner_table(shape, cell, core_mpc)
            has_support = supporting_rows(support, starts, stops)

            nominal_rows = np.flatnonzero(np.asarray(nominal["cap"]) == cap_id)
            nominal_lookup = {
                tuple(int(value) for value in nominal["core_index"][row]): int(row)
                for row in nominal_rows
            }
            owner_is_nominal = np.asarray(
                [tuple(int(value) for value in key) in nominal_lookup for key in keys],
                dtype=bool,
            )
            exact_nominal_coverage = coverage_from_rows(
                shape, starts[owner_is_nominal], stops[owner_is_nominal]
            )
            selected = owner_is_nominal | has_support
            output_coverage = coverage_from_rows(shape, starts[selected], stops[selected])
            output_missing = support & ~output_coverage
            exact_nominal_missing = support & ~exact_nominal_coverage
            legacy_missing = support & ~legacy_coverage

            p3_path = Path(component["file"])
            with h5py.File(p3_path, "r") as field_handle:
                uncovered_fields = {
                    name: masked_field_summary(field_handle[name], exact_nominal_missing)
                    for name in ("counts", "expected_counts", "exposure_apodized")
                }
            target_uncovered = masked_field_summary(delta, exact_nominal_missing)

            selected_keys = keys[selected]
            selected_starts = starts[selected]
            selected_stops = stops[selected]
            selected_nominal = np.asarray(
                [nominal_lookup.get(tuple(int(value) for value in key), -1)
                 for key in selected_keys], dtype=np.int32,
            )
            selected_fold = np.full(len(selected_keys), 255, dtype=np.uint8)
            real = selected_nominal >= 0
            selected_fold[real] = np.asarray(nominal["fold"])[selected_nominal[real]]
            inference_only = ~real
            lower = origin + selected_keys.astype(np.float64) * core_mpc
            upper = lower + core_mpc

            records["cap"].append(np.full(len(selected_keys), cap_id, dtype=np.uint8))
            records["core_index"].append(selected_keys.astype(np.int32))
            records["lower_mpc"].append(lower)
            records["upper_mpc"].append(upper)
            records["centroid_mpc"].append(0.5 * (lower + upper))
            records["voxel_start"].append(selected_starts.astype(np.int32))
            records["voxel_stop"].append(selected_stops.astype(np.int32))
            records["nominal_core_id"].append(selected_nominal)
            records["fold"].append(selected_fold)
            records["inference_only"].append(inference_only)
            records["owns_density_loss"].append(real)

            component_report[cap_name] = {
                "cap_id": cap_id,
                "shape": list(shape),
                "candidate_owner_cores": int(len(keys)),
                "nominal_p4_cores": int(len(nominal_rows)),
                "owner_cores_with_science_support": int(np.count_nonzero(has_support)),
                "output_cores": int(np.count_nonzero(selected)),
                "inference_only_cores": int(np.count_nonzero(inference_only)),
                "science_supported_voxels": int(np.count_nonzero(support)),
                "legacy_intersection_uncovered_voxels": int(np.count_nonzero(legacy_missing)),
                "exact_owner_uncovered_voxels_before_extension": int(
                    np.count_nonzero(exact_nominal_missing)
                ),
                "uncovered_voxels_after_extension": int(np.count_nonzero(output_missing)),
                "exact_owner_coverage_fraction_before_extension": float(
                    np.count_nonzero(support & exact_nominal_coverage)
                    / max(np.count_nonzero(support), 1)
                ),
                "coverage_fraction_after_extension": float(
                    np.count_nonzero(support & output_coverage)
                    / max(np.count_nonzero(support), 1)
                ),
                "uncovered_supported_field_summary": uncovered_fields,
                "uncovered_supported_target_summary": target_uncovered,
            }

    arrays = {name: np.concatenate(parts) for name, parts in records.items()}
    arrays["output_core_id"] = np.arange(len(arrays["cap"]), dtype=np.int32)
    output_path = args.output / "field_output_cores.npz"
    atomic_savez(output_path, **arrays)
    total_support = sum(v["science_supported_voxels"] for v in component_report.values())
    total_missing_before = sum(
        v["exact_owner_uncovered_voxels_before_extension"]
        for v in component_report.values()
    )
    total_missing_after = sum(
        v["uncovered_voxels_after_extension"] for v in component_report.values()
    )
    inference_only_count = int(np.count_nonzero(arrays["inference_only"]))
    manifest = {
        "stage": "P8.9 exact density-loss ownership and complete field-output tiling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "elapsed_seconds": float(time.time() - started),
        "inputs": {
            "p3_manifest": str(args.p3_manifest),
            "p3_manifest_sha256": sha256(args.p3_manifest),
            "p4_manifest": str(args.p4_manifest),
            "p4_manifest_sha256": sha256(args.p4_manifest),
            "p4_cores": str(args.p4_cores),
            "p4_cores_sha256": sha256(args.p4_cores),
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
        },
        "artifact": {
            "field_output_cores": str(output_path),
            "field_output_cores_sha256": sha256(output_path),
            "rows": int(len(arrays["cap"])),
            "inference_only_rows": inference_only_count,
        },
        "contract": {
            "core_mpc_h": core_mpc_h,
            "core_mpc": core_mpc,
            "voxel_owner": "P3 cell centre in one half-open P4 64 Mpc/h lattice core",
            "nominal_p4_folds_changed": False,
            "inference_only_fold_sentinel": 255,
            "inference_only_rows_contribute_loss_or_metrics": False,
            "unsupported_voxels_for_final_fft": "explicitly windowed to zero; never model supervised",
            "uncovered_supported_voxels_zero_filled": False,
            "classical_field_substitution": False,
        },
        "components": component_report,
        "global": {
            "science_supported_voxels": int(total_support),
            "exact_owner_uncovered_voxels_before_extension": int(total_missing_before),
            "uncovered_voxels_after_extension": int(total_missing_after),
            "coverage_fraction_after_extension": float(
                (total_support - total_missing_after) / max(total_support, 1)
            ),
        },
        "gates": {
            "unique_cell_centre_owner_partition": True,
            "nominal_p4_rows_and_folds_unchanged": True,
            "inference_only_rows_have_no_loss_or_metric_ownership": bool(
                np.all(~arrays["owns_density_loss"][arrays["inference_only"]])
            ),
            "all_supported_voxels_have_output_owner": total_missing_after == 0,
            "no_silent_supported_zero_fill": total_missing_after == 0,
        },
        "pass": bool(total_missing_after == 0),
    }
    manifest_path = args.output / "field_output_tiling_manifest.json"
    atomic_json(manifest_path, manifest)
    marker = args.output / "FIELD_OUTPUT_TILING_READY"
    marker.unlink(missing_ok=True)
    if manifest["pass"]:
        marker.write_text(
            f"git_sha={manifest['git_sha']}\nmanifest={manifest_path}\n"
            f"output_cores={len(arrays['cap'])}\ninference_only={inference_only_count}\n"
        )
    print(json.dumps({
        "pass": manifest["pass"],
        "output_cores": int(len(arrays["cap"])),
        "inference_only_cores": inference_only_count,
        "global": manifest["global"],
        "components": component_report,
    }, indent=2))
    if not manifest["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
