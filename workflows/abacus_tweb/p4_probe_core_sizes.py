#!/usr/bin/env python3
"""P4 candidate core-size occupancy and conservative resource probe.

The scientific candidates are specified in Mpc/h.  Canonical coordinates are in
observer-frame comoving Mpc, so this script performs one explicit Planck18-h
conversion and records both values.  It does not round core sizes to the 5-Mpc P3
voxel lattice.

Graph context figures are conservative whole-neighbour-core upper estimates, not
GPU-memory guarantees.  P5 must replace them with exact K-hop adapter canaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


CAPS = ((0, "SGC"), (1, "NGC"))


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def core_indices(xyz: np.ndarray, origin: np.ndarray, size_mpc: float) -> np.ndarray:
    return np.floor((np.asarray(xyz, dtype=np.float64) - origin) / size_mpc).astype(np.int32)


def rows_to_counts(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse.astype(np.int64), counts.astype(np.int64)


def count_lookup(indices: np.ndarray, counts: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {tuple(int(v) for v in row): int(count) for row, count in zip(indices, counts)}


def conservative_context_counts(core_index: np.ndarray, context_lookup: dict,
                                size_mpc: float, halo_mpc: float) -> np.ndarray:
    reach = int(math.ceil(halo_mpc / size_mpc))
    offsets = np.asarray(
        [(i, j, k) for i in range(-reach, reach + 1)
         for j in range(-reach, reach + 1)
         for k in range(-reach, reach + 1)], dtype=np.int32)
    result = np.zeros(len(core_index), dtype=np.int64)
    for n, core in enumerate(core_index):
        result[n] = sum(context_lookup.get(tuple(int(v) for v in core + delta), 0)
                        for delta in offsets)
    return result


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if len(values) == 0:
        return {"n": 0, "min": None, "p05": None, "median": None,
                "p95": None, "p99": None, "max": None, "mean": None}
    q = np.quantile(values, [0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(len(values)), "min": int(np.min(values)),
        "p05": float(q[0]), "median": float(q[1]), "p95": float(q[2]),
        "p99": float(q[3]), "max": int(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path,
                    default=repo / "docs/evidence/p4/p4_spatial_schema_v1.json")
    ap.add_argument("--points", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"))
    ap.add_argument("--index", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"))
    ap.add_argument("--p3-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"))
    ap.add_argument("--p3-unit-audit", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json"))
    ap.add_argument("--p2-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/p2b_union_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/core_size_probe.json"))
    args = ap.parse_args()
    started = time.time()

    schema = json.loads(args.schema.read_text())
    p3 = json.loads(args.p3_manifest.read_text())
    unit = json.loads(args.p3_unit_audit.read_text())
    p2 = json.loads(args.p2_manifest.read_text())
    h = float(schema["coordinate_frame"]["h"])
    if not unit.get("pass", False):
        raise RuntimeError("P3 unit audit must pass before P4")
    if not np.isclose(h, float(unit["physical_scales"]["planck18_h"])):
        raise RuntimeError("P4 schema h differs from passing P3 unit audit")
    if schema["coordinate_frame"]["indexing_units"] != "Mpc":
        raise RuntimeError("P4 indexing coordinates must remain Mpc")
    if not np.isclose(float(schema["core_size_probe"]["field_cell_mpc"]),
                      float(json.loads(Path(p3["frozen_schema"]).read_text())["grid"]["cell_mpc"])):
        raise RuntimeError("P4/P3 field cell mismatch")

    points = np.load(args.points, mmap_mode="r")
    index = np.load(args.index)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    shell = np.asarray(index["shell"], dtype=np.int8)
    active = np.asarray(index["active"], dtype=bool) & np.asarray(index["valid_target"], dtype=bool)
    context = np.asarray(index["context"], dtype=bool)
    if points.shape != (len(cap), 4):
        raise RuntimeError("canonical points/index mismatch")

    mean_union_degree = 2.0 * float(p2["counts"]["union_pairs_context"]) / float(context.sum())
    candidates = {}
    minimum_high_z = int(schema["core_size_probe"]["minimum_high_z_active_per_occupied_core"])
    for target_mpch in schema["core_size_probe"]["candidate_mpc_h"]:
        size_mpc = float(target_mpch) / h
        report = {
            "scientific_core_mpc_h": float(target_mpch),
            "indexing_core_mpc": size_mpc,
            "conversion_identity_abs": abs(size_mpc * h - float(target_mpch)),
            "field_core_voxels_per_axis_noninteger": size_mpc / float(
                schema["core_size_probe"]["field_cell_mpc"]),
            "caps": {},
        }
        all_high_z = []
        all_core_context = []
        all_context2 = []
        all_context4 = []
        for cap_id, cap_name in CAPS:
            origin = np.asarray(p3["components"][cap_name]["grid"]["origin_mpc"], dtype=np.float64)
            context_ids = np.flatnonzero(context & (cap == cap_id))
            active_ids = np.flatnonzero(active & (cap == cap_id))
            ci_context = core_indices(points[context_ids, :3], origin, size_mpc)
            context_cores, _, context_counts = rows_to_counts(ci_context)
            lookup = count_lookup(context_cores, context_counts)
            ci_active = core_indices(points[active_ids, :3], origin, size_mpc)
            active_cores, inverse, active_counts = rows_to_counts(ci_active)
            high_z = np.bincount(
                inverse, weights=(shell[active_ids] == 3).astype(np.int64),
                minlength=len(active_cores)).astype(np.int64)
            context_in_core = np.asarray(
                [lookup.get(tuple(int(v) for v in row), 0) for row in active_cores],
                dtype=np.int64)
            context2 = conservative_context_counts(
                active_cores, lookup, size_mpc,
                2 * float(schema["core_size_probe"]["graph_radius_mpc"]))
            context4 = conservative_context_counts(
                active_cores, lookup, size_mpc,
                4 * float(schema["core_size_probe"]["graph_radius_mpc"]))
            high_z_occupied = high_z[high_z > 0]
            report["caps"][cap_name] = {
                "occupied_active_cores": int(len(active_cores)),
                "active_rows": int(len(active_ids)),
                "active_per_core": stats(active_counts),
                "context_in_core": stats(context_in_core),
                "high_z_occupied_cores": int(len(high_z_occupied)),
                "high_z_active_per_occupied_core": stats(high_z_occupied),
                "high_z_cores_meeting_minimum": int(np.sum(high_z_occupied >= minimum_high_z)),
                "high_z_fraction_meeting_minimum": float(
                    np.mean(high_z_occupied >= minimum_high_z)) if len(high_z_occupied) else 0.0,
                "conservative_context_nodes_2pass": stats(context2),
                "conservative_context_nodes_4pass": stats(context4),
            }
            all_high_z.append(high_z_occupied)
            all_core_context.append(context_in_core)
            all_context2.append(context2)
            all_context4.append(context4)
        high_z_all = np.concatenate(all_high_z)
        context_all = np.concatenate(all_core_context)
        context2_all = np.concatenate(all_context2)
        context4_all = np.concatenate(all_context4)
        core_cells = int(math.ceil(size_mpc / float(schema["core_size_probe"]["field_cell_mpc"])))
        field_context = float(schema["core_size_probe"]["field_context_mpc"])
        patch_cells = int(math.ceil((size_mpc + 2 * field_context)
                                    / float(schema["core_size_probe"]["field_cell_mpc"])))
        report["combined"] = {
            "high_z_active_per_occupied_core": stats(high_z_all),
            "high_z_fraction_meeting_minimum": float(np.mean(high_z_all >= minimum_high_z)),
            "context_in_core": stats(context_all),
            "conservative_context_nodes_2pass": stats(context2_all),
            "conservative_context_nodes_4pass": stats(context4_all),
            "mean_union_degree_global": mean_union_degree,
            "estimated_union_pairs_2pass_p95": float(
                np.quantile(context2_all, 0.95) * mean_union_degree / 2.0),
            "estimated_union_pairs_4pass_p95": float(
                np.quantile(context4_all, 0.95) * mean_union_degree / 2.0),
            "field_core_cells_per_axis_covering_bounds": core_cells,
            "field_patch_cells_per_axis_with_40mpc_context": patch_cells,
            "field_patch_voxels_with_40mpc_context": patch_cells ** 3,
        }
        candidates[str(int(target_mpch))] = report

    default_key = str(int(schema["core_size_probe"]["default_mpc_h"]))
    default = candidates[default_key]
    gates = {
        "p3_unit_audit_pass": bool(unit["pass"]),
        "candidate_conversions_exact": all(
            v["conversion_identity_abs"] < 1.0e-10 for v in candidates.values()),
        "both_caps_present_for_every_candidate": all(
            set(v["caps"]) == {"NGC", "SGC"} for v in candidates.values()),
        "all_active_rows_represented_each_candidate": all(
            sum(c["active_rows"] for c in v["caps"].values()) == int(active.sum())
            for v in candidates.values()),
        "default_high_z_median_meets_minimum": (
            default["combined"]["high_z_active_per_occupied_core"]["median"] >= minimum_high_z),
        "default_field_patch_is_small": (
            default["combined"]["field_patch_voxels_with_40mpc_context"] < 10_000_000),
    }
    payload = {
        "schema_version": 1,
        "stage": "P4 core-size resource probe",
        "schema": str(args.schema), "schema_sha256": sha256(args.schema),
        "inputs": {
            "points": str(args.points), "points_sha256": sha256(args.points),
            "canonical_index": str(args.index), "canonical_index_sha256": sha256(args.index),
            "p2_manifest": str(args.p2_manifest), "p2_manifest_sha256": sha256(args.p2_manifest),
            "p3_manifest": str(args.p3_manifest), "p3_manifest_sha256": sha256(args.p3_manifest),
            "p3_unit_audit": str(args.p3_unit_audit),
        },
        "unit_contract": {
            "h": h, "scientific_candidates_mpc_h": schema["core_size_probe"]["candidate_mpc_h"],
            "indexing_candidates_mpc": [float(v) / h for v in schema["core_size_probe"]["candidate_mpc_h"]],
            "field_cell_mpc": schema["core_size_probe"]["field_cell_mpc"],
            "warning": "core boundaries are exact in Mpc/h and need not align to 5-Mpc voxel edges",
        },
        "candidates": candidates,
        "provisional_selection": {
            "core_mpc_h": float(schema["core_size_probe"]["default_mpc_h"]),
            "core_mpc": float(schema["core_size_probe"]["default_mpc_h"]) / h,
            "reason": (
                "registered plan default; median high-z occupancy exceeds the frozen minimum "
                "whereas 32 Mpc/h fragments the labels. High-z cores will be batched or gradient-"
                "accumulated rather than changing their physical size. The 96 Mpc/h candidate "
                "raises conservative graph context substantially. "
                "P5 exact K-hop GPU canary may require lossless subdivision but must not change "
                "the authoritative scientific core ownership."
            ),
        },
        "limitations": [
            "whole-neighbour-core graph context counts are conservative upper estimates",
            "union-pair estimates use the global mean degree and are not memory allocations",
            "P5/P6 adapters remain responsible for exact context and GPU memory canaries",
        ],
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=bool))
    if not payload["pass"]:
        raise RuntimeError(f"P4 core-size probe failed: {gates}")


if __name__ == "__main__":
    main()
