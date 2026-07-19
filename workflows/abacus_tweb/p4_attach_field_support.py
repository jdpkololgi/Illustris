#!/usr/bin/env python3
"""Attach P3 exposure/convolutional support to the immutable P4 geometry.

Distances are Euclidean distances, in observer-frame comoving Mpc, from each P1
active galaxy's canonical P3 voxel to the nearest unsupported exposure voxel.
They are quality/support metadata only and never redefine core or fold ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt


CAPS = ((0, "SGC"), (1, "NGC"))


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def grouped_quantiles(group: np.ndarray, values: np.ndarray, n_group: int) -> dict[str, np.ndarray]:
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    sorted_values = values[order]
    starts = np.searchsorted(sorted_group, np.arange(n_group), side="left")
    stops = np.searchsorted(sorted_group, np.arange(n_group), side="right")
    result = {
        "min": np.full(n_group, np.nan, dtype=np.float32),
        "p10": np.full(n_group, np.nan, dtype=np.float32),
        "median": np.full(n_group, np.nan, dtype=np.float32),
    }
    for gid, (start, stop) in enumerate(zip(starts, stops)):
        if stop <= start:
            continue
        local = sorted_values[start:stop]
        result["min"][gid] = np.min(local)
        result["p10"][gid] = np.quantile(local, 0.10)
        result["median"][gid] = np.median(local)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"))
    ap.add_argument("--p3-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"))
    ap.add_argument("--p4-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/spatial_manifest.json"))
    ap.add_argument("--active-assignment", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz"))
    ap.add_argument("--cores", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/cores.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest"))
    args = ap.parse_args()
    started = time.time()
    p3 = json.loads(args.p3_manifest.read_text())
    p4 = json.loads(args.p4_manifest.read_text())
    if not p4.get("pass", False):
        raise RuntimeError("passing P4 geometry manifest required")
    if p4["inputs"]["p3_manifest_sha256"] != sha256(args.p3_manifest):
        raise RuntimeError("P4 geometry and P3 field identity differ")

    points = np.load(args.points, mmap_mode="r")
    assignment = np.load(args.active_assignment)
    cores = np.load(args.cores)
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    cap = np.asarray(assignment["cap"], dtype=np.uint8)
    core_id = np.asarray(assignment["core_id"], dtype=np.int32)
    eligible = np.asarray(assignment["supervised_eligible"], dtype=bool)
    distance = np.full(len(parent), np.nan, dtype=np.float32)
    supported = np.zeros(len(parent), dtype=bool)
    in_grid = np.zeros(len(parent), dtype=bool)
    core_exposure_fraction = np.full(len(cores["core_id"]), np.nan, dtype=np.float32)
    cap_reports = {}

    for cap_id, cap_name in CAPS:
        component = p3["components"][cap_name]
        origin = np.asarray(component["grid"]["origin_mpc"], dtype=np.float64)
        cell = float(component["grid"]["cell_mpc"])
        shape = np.asarray(component["grid"]["shape"], dtype=np.int64)
        selected = np.flatnonzero(cap == cap_id)
        xyz = np.asarray(points[parent[selected], :3], dtype=np.float64)
        voxel = np.floor((xyz - origin) / cell).astype(np.int64)
        valid = np.all((voxel >= 0) & (voxel < shape), axis=1)
        in_grid[selected] = valid
        with h5py.File(component["file"], "r") as handle:
            exposure = np.asarray(handle["exposure_binary"], dtype=bool)
        edt = distance_transform_edt(exposure, sampling=cell).astype(np.float32)
        valid_selected = selected[valid]
        valid_voxel = voxel[valid]
        supported[valid_selected] = exposure[
            valid_voxel[:, 0], valid_voxel[:, 1], valid_voxel[:, 2]]
        distance[valid_selected] = edt[
            valid_voxel[:, 0], valid_voxel[:, 1], valid_voxel[:, 2]]

        cap_cores = np.flatnonzero(np.asarray(cores["cap"], dtype=np.uint8) == cap_id)
        starts = np.asarray(cores["voxel_start"][cap_cores], dtype=np.int32)
        stops = np.asarray(cores["voxel_stop"][cap_cores], dtype=np.int32)
        for cid, lo, hi in zip(cap_cores, starts, stops):
            block = exposure[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            core_exposure_fraction[cid] = float(np.mean(block)) if block.size else np.nan
        del edt, exposure
        cap_reports[cap_name] = {
            "active_rows": int(len(selected)),
            "coordinates_in_grid": int(valid.sum()),
            "supported_active_rows": int(supported[selected].sum()),
            "supported_active_fraction": float(np.mean(supported[selected])),
            "eligible_supported_fraction": float(np.mean(supported[selected][eligible[selected]])),
            "distance_mpc_supported_quantiles": [float(v) for v in np.quantile(
                distance[selected][supported[selected]], [0.0, 0.1, 0.5, 0.9, 1.0])],
        }

    core_distance = grouped_quantiles(core_id[eligible], distance[eligible], len(cores["core_id"]))
    core_eligible_count = np.bincount(core_id[eligible], minlength=len(cores["core_id"]))
    core_supported_count = np.bincount(
        core_id[eligible], weights=supported[eligible].astype(np.int64),
        minlength=len(cores["core_id"]))
    core_supported_fraction = np.divide(
        core_supported_count, core_eligible_count,
        out=np.full(len(core_eligible_count), np.nan, dtype=np.float64),
        where=core_eligible_count > 0).astype(np.float32)

    support_path = args.out_dir / "field_support.npz"
    core_path = args.out_dir / "core_field_support.npz"
    atomic_savez(
        support_path, parent_node_id=parent, field_voxel_in_grid=in_grid,
        exposure_supported=supported, field_support_distance_mpc=distance,
        convolution_20mpc_safe=supported & (distance >= 20.0),
        convolution_40mpc_safe=supported & (distance >= 40.0),
        supervised_eligible=eligible,
    )
    atomic_savez(
        core_path, core_id=np.asarray(cores["core_id"], dtype=np.int32),
        core_exposure_voxel_fraction=core_exposure_fraction,
        eligible_supported_fraction=core_supported_fraction,
        eligible_distance_min_mpc=core_distance["min"],
        eligible_distance_p10_mpc=core_distance["p10"],
        eligible_distance_median_mpc=core_distance["median"],
    )
    gates = {
        "p3_identity_matches_geometry": p4["inputs"]["p3_manifest_sha256"] == sha256(args.p3_manifest),
        "all_active_coordinates_in_grid": bool(np.all(in_grid)),
        "both_caps_above_98pct_supported": all(
            v["eligible_supported_fraction"] >= 0.98 for v in cap_reports.values()),
        "support_distances_finite_for_supported": bool(np.isfinite(distance[supported]).all()),
        "unsupported_rows_have_zero_or_nan_distance": bool(
            np.all(np.isnan(distance[~in_grid]) | (distance[~in_grid] == 0))),
        "core_ids_match": np.array_equal(cores["core_id"], np.arange(len(cores["core_id"]))),
    }
    manifest = {
        "schema_version": 1, "stage": "P4 P3 field/exposure support attachment",
        "p4_geometry_manifest": str(args.p4_manifest),
        "p4_geometry_manifest_sha256": sha256(args.p4_manifest),
        "p3_manifest": str(args.p3_manifest), "p3_manifest_sha256": sha256(args.p3_manifest),
        "artifacts": {"galaxy_field_support": str(support_path),
                      "core_field_support": str(core_path)},
        "artifact_sha256": {"galaxy_field_support": sha256(support_path),
                            "core_field_support": sha256(core_path)},
        "caps": cap_reports,
        "global": {
            "p1_active_rows": int(len(parent)),
            "eligible_rows": int(eligible.sum()),
            "eligible_supported_fraction": float(np.mean(supported[eligible])),
            "eligible_convolution_20mpc_safe_fraction": float(
                np.mean((supported & (distance >= 20.0))[eligible])),
            "eligible_convolution_40mpc_safe_fraction": float(
                np.mean((supported & (distance >= 40.0))[eligible])),
        },
        "interpretation": (
            "Distance is to the nearest unsupported voxel in the target-free P3a occupancy "
            "exposure. It is a support/quality flag, not a learned input normalization and not "
            "a substitute for P6 context-growth parity."
        ),
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_dir / "field_support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=bool))
    if not manifest["pass"]:
        raise RuntimeError(f"P4 field support gates failed: {gates}")
    (args.out_dir / "P4_FIELD_SUPPORT_COMPLETE").write_text(
        f"manifest_sha256={sha256(manifest_path)}\n"
        f"eligible_rows={int(eligible.sum())}\n")


if __name__ == "__main__":
    main()
