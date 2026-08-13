#!/usr/bin/env python3
"""Build the deterministic P4 core/super-block/fold geometry manifest.

The graph and fields remain global canonical products.  This script creates only
ownership metadata: exact 64-Mpc/h half-open cores, four-core super-blocks, five
blocked folds, and per-galaxy authoritative core assignments.  Periodically repeated
images of the same underlying halo retain one deterministic supervised occurrence;
the other images remain available as context but cannot leak identical targets.

Architecture-specific graph/convolution/FFT context is attached by later P4/P5/P6
steps; it is never used to redefine the scientific core or fold ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import fitsio
import numpy as np


CAPS = ((0, "SGC"), (1, "NGC"))
FOLD_COUNT = 5


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


def core_indices(xyz: np.ndarray, origin: np.ndarray, size_mpc: float) -> np.ndarray:
    return np.floor((np.asarray(xyz, dtype=np.float64) - origin) / size_mpc).astype(np.int32)


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Stable unsigned rank used only to choose one periodic image per host."""
    with np.errstate(over="ignore"):
        x = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


class DisjointSet:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            nxt = int(self.parent[value])
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def roots(self) -> np.ndarray:
        return np.asarray([self.find(i) for i in range(len(self.parent))], dtype=np.int32)


def greedy_balanced_folds(group_counts: np.ndarray, group_context: np.ndarray,
                          group_cores: np.ndarray, *, context_weight: float = 0.05,
                          core_weight: float = 0.05) -> np.ndarray:
    """Deterministic multi-objective LPT assignment of spatial groups to folds."""
    n = len(group_counts)
    total = group_counts.sum(axis=0).astype(np.float64)
    target = np.maximum(total / FOLD_COUNT, 1.0)
    target_context = max(float(group_context.sum()) / FOLD_COUNT, 1.0)
    target_cores = max(float(group_cores.sum()) / FOLD_COUNT, 1.0)
    priority = group_counts.sum(axis=(1, 2)) + 1.0e-3 * group_context
    order = np.lexsort((np.arange(n), -priority))
    fold_counts = np.zeros((FOLD_COUNT, 2, 4), dtype=np.float64)
    fold_context = np.zeros(FOLD_COUNT, dtype=np.float64)
    fold_cores = np.zeros(FOLD_COUNT, dtype=np.float64)
    result = np.full(n, 255, dtype=np.uint8)
    for group in order:
        scores = []
        for fold in range(FOLD_COUNT):
            counts = fold_counts.copy()
            context = fold_context.copy()
            cores = fold_cores.copy()
            counts[fold] += group_counts[group]
            context[fold] += group_context[group]
            cores[fold] += group_cores[group]
            score = float(np.sum(((counts - target) / target) ** 2))
            score += context_weight * float(np.sum(((context - target_context) / target_context) ** 2))
            score += core_weight * float(np.sum(((cores - target_cores) / target_cores) ** 2))
            scores.append((score, float(fold_context[fold]), fold))
        chosen = min(scores)[2]
        result[group] = chosen
        fold_counts[chosen] += group_counts[group]
        fold_context[chosen] += group_context[group]
        fold_cores[chosen] += group_cores[group]
    if np.any(result == 255):
        raise RuntimeError("unassigned fold group")
    return result


def same_fold_run_bounds(super_index: np.ndarray, super_cap: np.ndarray,
                         super_fold: np.ndarray, cap_origins: dict[int, np.ndarray],
                         super_size_mpc: float) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(c), int(row[0]), int(row[1]), int(row[2])): i
        for i, (c, row) in enumerate(zip(super_cap, super_index))
    }
    lower = np.empty((len(super_index), 3), dtype=np.float64)
    upper = np.empty((len(super_index), 3), dtype=np.float64)
    for sid, (cap_id, row, fold) in enumerate(zip(super_cap, super_index, super_fold)):
        for axis in range(3):
            lo = int(row[axis])
            hi = int(row[axis])
            while True:
                neighbour = row.copy()
                neighbour[axis] = lo - 1
                other = lookup.get((int(cap_id), int(neighbour[0]), int(neighbour[1]),
                                    int(neighbour[2])))
                if other is None or super_fold[other] != fold:
                    break
                lo -= 1
            while True:
                neighbour = row.copy()
                neighbour[axis] = hi + 1
                other = lookup.get((int(cap_id), int(neighbour[0]), int(neighbour[1]),
                                    int(neighbour[2])))
                if other is None or super_fold[other] != fold:
                    break
                hi += 1
            lower[sid, axis] = cap_origins[int(cap_id)][axis] + lo * super_size_mpc
            upper[sid, axis] = cap_origins[int(cap_id)][axis] + (hi + 1) * super_size_mpc
    return lower, upper


def fold_summary(active_fold: np.ndarray, cap: np.ndarray, shell: np.ndarray,
                 active_ids: np.ndarray, super_fold: np.ndarray) -> dict:
    result = {}
    for fold in range(FOLD_COUNT):
        selected = active_fold == fold
        by_cap_shell = {
            cap_name: [int(np.sum(selected & (cap[active_ids] == cap_id)
                                        & (shell[active_ids] == shell_id)))
                       for shell_id in range(4)]
            for cap_id, cap_name in CAPS
        }
        result[str(fold)] = {
            "active_rows": int(selected.sum()),
            "super_blocks": int(np.sum(super_fold == fold)),
            "by_cap_shell": by_cap_shell,
        }
    return result


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path,
                    default=repo / "docs/evidence/p4/p4_spatial_schema_v1.json")
    ap.add_argument("--probe", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/core_size_probe.json"))
    ap.add_argument("--points", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"))
    ap.add_argument("--index", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"))
    ap.add_argument("--catalogue", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
        "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
        "ngrid2048_thr0p2_halo_xcom.fits"))
    ap.add_argument("--p1-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json"))
    ap.add_argument("--p2-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/p2b_union_manifest.json"))
    ap.add_argument("--p3-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest"))
    args = ap.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(args.schema.read_text())
    probe = json.loads(args.probe.read_text())
    p1 = json.loads(args.p1_manifest.read_text())
    p3 = json.loads(args.p3_manifest.read_text())
    if not probe.get("pass", False):
        raise RuntimeError("passing core-size probe required")
    if probe["schema_sha256"] != sha256(args.schema):
        raise RuntimeError("core-size probe was not produced from the frozen P4 schema")
    if p1["catalogue_id"] != schema["catalogue_id"]:
        raise RuntimeError("P1/P4 catalogue identity mismatch")
    h = float(schema["coordinate_frame"]["h"])
    core_mpc_h = float(probe["provisional_selection"]["core_mpc_h"])
    core_mpc = float(probe["provisional_selection"]["core_mpc"])
    if not np.isclose(core_mpc * h, core_mpc_h):
        raise RuntimeError("P4 core unit conversion failed")
    super_factor = int(schema["super_blocks"]["cores_per_axis"])
    super_mpc = super_factor * core_mpc

    points = np.load(args.points, mmap_mode="r")
    index = np.load(args.index)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    shell = np.asarray(index["shell"], dtype=np.int8)
    context = np.asarray(index["context"], dtype=bool)
    p1_active = (np.asarray(index["active"], dtype=bool)
                 & np.asarray(index["valid_target"], dtype=bool))
    targetid = np.asarray(index["targetid"], dtype=np.int64)
    if points.shape != (len(cap), 4):
        raise RuntimeError("points/index mismatch")
    if np.any(p1_active & ~context):
        raise RuntimeError("active galaxies must be a subset of context")
    cap_origins = {
        cap_id: np.asarray(p3["components"][cap_name]["grid"]["origin_mpc"], dtype=np.float64)
        for cap_id, cap_name in CAPS
    }

    # Full-sky construction periodically repeats a small fraction of underlying
    # box halos at ~one box length. Retain exactly one supervised image per key.
    host = fitsio.read(str(args.catalogue), columns=["FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
    if len(host) != len(cap):
        raise RuntimeError("host table/index mismatch")
    p1_active_ids = np.flatnonzero(p1_active)
    file_all = np.asarray(host["FILE_NUM"][p1_active_ids], dtype=np.int64)
    box_all = np.asarray(host["BOX_INDEX"][p1_active_ids], dtype=np.int64)
    halo_all = np.asarray(host["HALO_INDEX"][p1_active_ids], dtype=np.int64)
    host_order = np.lexsort((halo_all, box_all, file_all))
    host_same = (
        (file_all[host_order][1:] == file_all[host_order][:-1])
        & (box_all[host_order][1:] == box_all[host_order][:-1])
        & (halo_all[host_order][1:] == halo_all[host_order][:-1])
    )
    repeated_left = p1_active_ids[host_order[:-1][host_same]]
    repeated_right = p1_active_ids[host_order[1:][host_same]]
    repeated_distances = np.linalg.norm(
        np.asarray(points[repeated_left, :3], dtype=np.float64)
        - np.asarray(points[repeated_right, :3], dtype=np.float64), axis=1)
    starts = np.r_[0, np.flatnonzero(~host_same) + 1]
    stops = np.r_[starts[1:], len(host_order)]
    eligible = p1_active.copy()
    repeated_groups = 0
    for start, stop in zip(starts, stops):
        if stop - start <= 1:
            continue
        repeated_groups += 1
        members = p1_active_ids[host_order[start:stop]]
        ranks = splitmix64(targetid[members])
        keeper = int(members[np.argmin(ranks)])
        eligible[members] = False
        eligible[keeper] = True
    periodic_duplicate = p1_active & ~eligible
    minimum_periodic_separation = float(
        schema["supervised_eligibility"]["minimum_periodic_image_separation_mpc"])

    parent_core = np.full(len(cap), -1, dtype=np.int32)
    core_cap_parts, core_index_parts, core_context_parts = [], [], []
    core_active_parts, core_p1_active_parts, core_shell_parts = [], [], []
    core_offset = 0
    for cap_id, _ in CAPS:
        context_ids = np.flatnonzero(context & (cap == cap_id))
        active_ids_cap = np.flatnonzero(eligible & (cap == cap_id))
        p1_active_ids_cap = np.flatnonzero(p1_active & (cap == cap_id))
        indices = core_indices(points[context_ids, :3], cap_origins[cap_id], core_mpc)
        unique, inverse, counts = np.unique(
            indices, axis=0, return_inverse=True, return_counts=True)
        parent_core[context_ids] = core_offset + inverse.astype(np.int32)
        active_local = inverse[np.searchsorted(context_ids, active_ids_cap)]
        active_counts = np.bincount(active_local, minlength=len(unique)).astype(np.int64)
        p1_active_local = inverse[np.searchsorted(context_ids, p1_active_ids_cap)]
        p1_active_counts = np.bincount(
            p1_active_local, minlength=len(unique)).astype(np.int64)
        shell_counts = np.stack([
            np.bincount(active_local, weights=(shell[active_ids_cap] == sid).astype(np.int64),
                        minlength=len(unique)).astype(np.int64)
            for sid in range(4)], axis=1)
        core_cap_parts.append(np.full(len(unique), cap_id, dtype=np.uint8))
        core_index_parts.append(unique.astype(np.int32))
        core_context_parts.append(counts.astype(np.int64))
        core_active_parts.append(active_counts)
        core_p1_active_parts.append(p1_active_counts)
        core_shell_parts.append(shell_counts)
        core_offset += len(unique)
    if np.any(parent_core[context] < 0):
        raise RuntimeError("unassigned context core")

    core_cap = np.concatenate(core_cap_parts)
    core_index = np.concatenate(core_index_parts)
    core_context = np.concatenate(core_context_parts)
    core_active = np.concatenate(core_active_parts)
    core_p1_active = np.concatenate(core_p1_active_parts)
    core_shell = np.concatenate(core_shell_parts)
    core_id = np.arange(len(core_cap), dtype=np.int32)
    core_lower = np.stack([
        cap_origins[int(c)] + row * core_mpc for c, row in zip(core_cap, core_index)
    ]).astype(np.float64)
    core_upper = core_lower + core_mpc
    core_centroid = 0.5 * (core_lower + core_upper)

    super_index_per_core = np.floor_divide(core_index, super_factor).astype(np.int32)
    super_key = np.column_stack([core_cap.astype(np.int32), super_index_per_core])
    super_unique, core_super = np.unique(super_key, axis=0, return_inverse=True)
    core_super = core_super.astype(np.int32)
    super_cap = super_unique[:, 0].astype(np.uint8)
    super_index = super_unique[:, 1:].astype(np.int32)
    super_id = np.arange(len(super_unique), dtype=np.int32)
    super_core_count = np.bincount(core_super, minlength=len(super_id)).astype(np.int64)
    super_context = np.bincount(
        core_super, weights=core_context, minlength=len(super_id)).astype(np.int64)
    super_active = np.bincount(
        core_super, weights=core_active, minlength=len(super_id)).astype(np.int64)
    super_shell = np.stack([
        np.bincount(core_super, weights=core_shell[:, sid], minlength=len(super_id)).astype(np.int64)
        for sid in range(4)], axis=1)
    parent_super = np.full(len(cap), -1, dtype=np.int32)
    parent_super[context] = core_super[parent_core[context]]

    active_ids = np.flatnonzero(eligible)
    active_super = parent_super[active_ids]
    active_core = parent_core[active_ids]
    file_num = np.asarray(host["FILE_NUM"][active_ids], dtype=np.int64)
    box = np.asarray(host["BOX_INDEX"][active_ids], dtype=np.int64)
    halo = np.asarray(host["HALO_INDEX"][active_ids], dtype=np.int64)
    order = np.lexsort((halo, box, file_num))
    same = (
        (file_num[order][1:] == file_num[order][:-1])
        & (box[order][1:] == box[order][:-1])
        & (halo[order][1:] == halo[order][:-1])
    )
    dsu = DisjointSet(len(super_id))
    crossing_pairs = 0
    for left, right in zip(order[:-1][same], order[1:][same]):
        a, b = int(active_super[left]), int(active_super[right])
        if a != b:
            crossing_pairs += 1
            dsu.union(a, b)
    roots = dsu.roots()
    component_root, super_component = np.unique(roots, return_inverse=True)
    super_component = super_component.astype(np.int32)
    n_component = len(component_root)

    component_counts = np.zeros((n_component, 2, 4), dtype=np.int64)
    for sid in range(len(super_id)):
        component_counts[super_component[sid], int(super_cap[sid])] += super_shell[sid]
    component_context = np.bincount(
        super_component, weights=super_context, minlength=n_component).astype(np.int64)
    component_cores = np.bincount(
        super_component, weights=super_core_count, minlength=n_component).astype(np.int64)
    active_xyz = np.asarray(points[active_ids, :3], dtype=np.float64)
    fold_candidates = (
        ("frozen_lpt_v1", 0.05, 0.05),
        # Geometry-only fallback activated only if the frozen LPT result fails a
        # registered fold balance gate. It removes a redundant context-count
        # penalty while retaining cap/shell and core-count balancing.
        ("boundary_balance_fallback_v1", 0.0, 0.05),
    )
    fold_candidate_audit = []
    selected = None
    for candidate_name, context_weight, core_weight in fold_candidates:
        candidate_component_fold = greedy_balanced_folds(
            component_counts, component_context, component_cores,
            context_weight=context_weight, core_weight=core_weight,
        )
        candidate_super_fold = candidate_component_fold[super_component]
        candidate_core_fold = candidate_super_fold[core_super]
        candidate_active_fold = candidate_core_fold[active_core]
        candidate_lower, candidate_upper = same_fold_run_bounds(
            super_index, super_cap, candidate_super_fold, cap_origins, super_mpc)
        candidate_distance = np.min(
            np.minimum(active_xyz - candidate_lower[active_super],
                       candidate_upper[active_super] - active_xyz), axis=1)
        if np.any(candidate_distance < -1.0e-5):
            raise RuntimeError("active point lies outside assigned same-fold run")
        candidate_distance = np.maximum(candidate_distance, 0.0).astype(np.float32)
        candidate_info = fold_summary(
            candidate_active_fold, cap, shell, active_ids, candidate_super_fold)
        candidate_counts = np.asarray(
            [candidate_info[str(i)]["active_rows"] for i in range(FOLD_COUNT)])
        candidate_dimensions = np.asarray([
            [[candidate_info[str(f)]["by_cap_shell"][name][s] for s in range(4)]
              for _, name in CAPS] for f in range(FOLD_COUNT)
        ])
        dimension_mean = candidate_dimensions.mean(axis=0)
        candidate_dimension_deviation = float(np.max(
            np.abs(candidate_dimensions - dimension_mean) / np.maximum(dimension_mean, 1.0)))
        candidate_super_counts = np.asarray(
            [np.sum(candidate_super_fold == fold) for fold in range(FOLD_COUNT)])
        candidate_medians = np.asarray([
            np.median(candidate_distance[candidate_active_fold == fold])
            for fold in range(FOLD_COUNT)
        ])
        candidate_gates = {
            "five_nonempty_folds": set(np.unique(candidate_active_fold).tolist()) == set(range(FOLD_COUNT)),
            "fold_active_count_max_min_below_1p05": (
                float(candidate_counts.max() / candidate_counts.min()) < 1.05),
            "cap_shell_relative_deviation_below_10pct": candidate_dimension_deviation < 0.10,
            "fold_occupied_superblock_ratio_below_1p25": (
                float(candidate_super_counts.max() / candidate_super_counts.min()) < 1.25),
            "fold_distance_medians_matched_below_25pct": (
                float(candidate_medians.max() / candidate_medians.min()) < 1.25),
        }
        fold_candidate_audit.append({
            "name": candidate_name,
            "context_weight": context_weight,
            "core_weight": core_weight,
            "distance_mpc_medians": [float(value) for value in candidate_medians],
            "distance_median_ratio": float(candidate_medians.max() / candidate_medians.min()),
            "active_count_ratio": float(candidate_counts.max() / candidate_counts.min()),
            "occupied_superblock_ratio": float(
                candidate_super_counts.max() / candidate_super_counts.min()),
            "cap_shell_max_relative_deviation": candidate_dimension_deviation,
            "gates": candidate_gates,
            "pass": all(candidate_gates.values()),
        })
        if all(candidate_gates.values()):
            selected = (
                candidate_name, candidate_component_fold, candidate_super_fold,
                candidate_core_fold, candidate_active_fold, candidate_lower,
                candidate_upper, candidate_distance, candidate_info,
                candidate_counts, candidate_dimensions, candidate_dimension_deviation,
            )
            break
    if selected is None:
        raise RuntimeError(f"no deterministic fold candidate passed: {fold_candidate_audit}")
    (fold_candidate_name, component_fold, super_fold, core_fold, active_fold,
     run_lower, run_upper, active_fold_distance, fold_info, fold_counts,
     dimension_counts, max_dimension_relative_deviation) = selected

    # Exact no-host-crossing readback after host-linked super-block assignment.
    host_fold_mismatch = int(np.sum(active_fold[order][1:][same] != active_fold[order][:-1][same]))
    if len(np.unique(targetid)) != len(targetid):
        raise RuntimeError("TARGETID uniqueness failed before P4")

    radius = float(schema["core_size_probe"]["graph_radius_mpc"])

    # P3 voxel ranges intersecting each exact core. Bounds are not silently snapped.
    voxel_start = np.empty_like(core_index)
    voxel_stop = np.empty_like(core_index)
    for cap_id, cap_name in CAPS:
        selected = core_cap == cap_id
        origin = cap_origins[cap_id]
        shape = np.asarray(p3["components"][cap_name]["grid"]["shape"], dtype=np.int32)
        cell = float(p3["components"][cap_name]["grid"]["cell_mpc"])
        voxel_start[selected] = np.maximum(
            0, np.floor((core_lower[selected] - origin) / cell).astype(np.int32))
        voxel_stop[selected] = np.minimum(
            shape, np.ceil((core_upper[selected] - origin) / cell).astype(np.int32))

    context_ids = np.flatnonzero(context)
    context_core = parent_core[context_ids]
    context_super = core_super[context_core]
    context_fold = core_fold[context_core]
    super_lower = np.stack([
        cap_origins[int(c)] + row * super_mpc for c, row in zip(super_cap, super_index)
    ]).astype(np.float64)
    super_upper = super_lower + super_mpc
    core_path = args.out_dir / "cores.npz"
    super_path = args.out_dir / "super_blocks.npz"
    context_path = args.out_dir / "context_assignment.npz"
    active_path = args.out_dir / "active_assignment.npz"
    atomic_savez(
        core_path, core_id=core_id, cap=core_cap, core_index=core_index,
        lower_mpc=core_lower, upper_mpc=core_upper, centroid_mpc=core_centroid,
        superblock_id=core_super, fold=core_fold, context_count=core_context,
        p1_active_count=core_p1_active, active_count=core_active,
        active_count_by_shell=core_shell,
        voxel_start=voxel_start, voxel_stop=voxel_stop,
    )
    atomic_savez(
        super_path, superblock_id=super_id, cap=super_cap, super_index=super_index,
        lower_mpc=super_lower, upper_mpc=super_upper, fold=super_fold,
        host_component=super_component, core_count=super_core_count,
        context_count=super_context, active_count=super_active,
        active_count_by_shell=super_shell, same_fold_run_lower_mpc=run_lower,
        same_fold_run_upper_mpc=run_upper,
    )
    atomic_savez(
        context_path, parent_node_id=context_ids.astype(np.int64),
        core_id=context_core, superblock_id=context_super, fold=context_fold,
        cap=cap[context_ids], shell=shell[context_ids], p1_active=p1_active[context_ids],
        supervised_eligible=eligible[context_ids],
    )
    all_active_ids = np.flatnonzero(p1_active)
    all_active_core = parent_core[all_active_ids]
    all_active_super = core_super[all_active_core]
    all_active_fold = core_fold[all_active_core]
    all_active_xyz = np.asarray(points[all_active_ids, :3], dtype=np.float64)
    all_active_fold_distance = np.min(
        np.minimum(all_active_xyz - run_lower[all_active_super],
                   run_upper[all_active_super] - all_active_xyz), axis=1).astype(np.float32)
    all_active_fold_distance = np.maximum(all_active_fold_distance, 0.0)
    atomic_savez(
        active_path, parent_node_id=all_active_ids.astype(np.int64),
        targetid=targetid[all_active_ids], core_id=all_active_core,
        superblock_id=all_active_super, fold=all_active_fold, cap=cap[all_active_ids],
        shell=shell[all_active_ids], supervised_eligible=eligible[all_active_ids],
        periodic_duplicate_image=periodic_duplicate[all_active_ids],
        distance_to_conservative_fold_boundary_mpc=all_active_fold_distance,
        radius_2pass_split_safe=all_active_fold_distance >= 2 * radius,
        radius_4pass_split_safe=all_active_fold_distance >= 4 * radius,
        exact_union_khop_support=np.full(len(all_active_ids), -1, dtype=np.int8),
        field_support_distance_mpc=np.full(len(all_active_ids), np.nan, dtype=np.float32),
        fft_support_status=np.zeros(len(all_active_ids), dtype=np.uint8),
    )

    rotations = {
        str(rotation): {
            "train_folds": [f for f in range(FOLD_COUNT)
                            if f not in {rotation, (rotation + 1) % FOLD_COUNT}],
            "validation_fold": (rotation + 1) % FOLD_COUNT,
            "development_test_fold": rotation,
        } for rotation in range(FOLD_COUNT)
    }
    rotations_path = args.out_dir / "rotations.json"
    rotations_path.write_text(json.dumps(rotations, indent=2, sort_keys=True) + "\n")
    gates = {
        "core_units_exact": np.isclose(core_mpc * h, core_mpc_h),
        "every_context_row_has_one_core": int(np.sum(parent_core[context] >= 0)) == int(context.sum()),
        "every_p1_active_row_has_one_core": len(all_active_ids) == int(p1_active.sum()),
        "every_eligible_active_row_has_one_core": len(active_ids) == int(eligible.sum()),
        "core_context_counts_close": int(core_context.sum()) == int(context.sum()),
        "core_p1_active_counts_close": int(core_p1_active.sum()) == int(p1_active.sum()),
        "core_active_counts_close": int(core_active.sum()) == int(eligible.sum()),
        "five_nonempty_folds": set(np.unique(active_fold).tolist()) == set(range(FOLD_COUNT)),
        "every_fold_has_both_caps_all_shells": bool(np.all(dimension_counts > 0)),
        "fold_active_count_max_min_below_1p05": float(fold_counts.max() / fold_counts.min()) < 1.05,
        "cap_shell_relative_deviation_below_10pct": max_dimension_relative_deviation < 0.10,
        "targetids_unique": len(np.unique(targetid)) == len(targetid),
        "repeated_hosts_do_not_cross_folds": host_fold_mismatch == 0,
        "periodic_duplicates_are_box_length_separated": (
            len(repeated_distances) == 0
            or float(np.min(repeated_distances)) >= minimum_periodic_separation),
        "one_supervised_occurrence_per_host": int(eligible.sum()) == len(host_order) - int(host_same.sum()),
        "voxel_ranges_positive": bool(np.all(voxel_stop > voxel_start)),
        "rotations_are_3_1_1": all(len(v["train_folds"]) == 3 for v in rotations.values()),
    }
    manifest_path = args.out_dir / "spatial_manifest.json"
    payload = {
        "schema_version": 1, "stage": "P4 shared fixed-comoving spatial manifest geometry",
        "catalogue_id": schema["catalogue_id"],
        "unit_contract": {"h": h, "core_mpc_h": core_mpc_h, "core_mpc": core_mpc,
                          "superblock_mpc_h": super_factor * core_mpc_h,
                          "superblock_mpc": super_mpc,
                          "indexing_coordinates": "observer-frame comoving Mpc"},
        "counts": {"context_rows": int(context.sum()),
                   "p1_active_rows": int(p1_active.sum()),
                   "active_rows": int(eligible.sum()),
                   "periodic_duplicate_images_context_only": int(periodic_duplicate.sum()),
                   "repeated_host_groups": int(repeated_groups),
                   "context_occupied_cores": int(len(core_id)),
                   "active_occupied_cores": int(np.sum(core_active > 0)),
                   "super_blocks": int(len(super_id)), "host_components": int(n_component),
                   "eligible_repeated_host_pairs_linking_superblocks": int(crossing_pairs)},
        "periodic_image_audit": {
            "adjacent_repeated_host_pairs": int(host_same.sum()),
            "distance_mpc_min": float(np.min(repeated_distances)) if len(repeated_distances) else None,
            "distance_mpc_median": float(np.median(repeated_distances)) if len(repeated_distances) else None,
            "distance_mpc_max": float(np.max(repeated_distances)) if len(repeated_distances) else None,
            "policy": schema["supervised_eligibility"]["periodic_image_policy"]},
        "folds": fold_info,
        "fold_assignment": {
            "selected": fold_candidate_name,
            "selection_policy": (
                "use frozen LPT when all registered balance gates pass; otherwise use the "
                "first deterministic geometry-only fallback that passes every same gate"
            ),
            "candidates": fold_candidate_audit,
        },
        "fold_balance": {"active_max_min_ratio": float(fold_counts.max() / fold_counts.min()),
                         "max_cap_shell_relative_deviation": max_dimension_relative_deviation},
        "rotations": str(rotations_path),
        "artifacts": {
            "cores": str(core_path), "super_blocks": str(super_path),
            "context_assignment": str(context_path), "active_assignment": str(active_path)},
        "inputs": {
            "schema": str(args.schema), "schema_sha256": sha256(args.schema),
            "core_size_probe": str(args.probe), "core_size_probe_sha256": sha256(args.probe),
            "p1_manifest": str(args.p1_manifest), "p1_manifest_sha256": sha256(args.p1_manifest),
            "p2_manifest": str(args.p2_manifest), "p2_manifest_sha256": sha256(args.p2_manifest),
            "p3_manifest": str(args.p3_manifest), "p3_manifest_sha256": sha256(args.p3_manifest),
            "points": str(args.points), "canonical_index": str(args.index)},
        "support_status": {
            "radius_physical_split_flags": "attached for 2 and 4 passes",
            "exact_p2b_union_khop": "reserved=-1; P5 reverse dependency traversal pending",
            "p3_field_distance": "reserved=NaN; P4 support attachment pending",
            "fft": "reserved=0; P7 convergence pending and non-blocking for GraphNet/U-Net"},
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    payload["artifact_sha256"] = {key: sha256(Path(path))
                                  for key, path in payload["artifacts"].items()}
    payload["rotations_sha256"] = sha256(rotations_path)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=bool))
    if not payload["pass"]:
        raise RuntimeError(f"P4 geometry gates failed: {gates}")
    marker = args.out_dir / "P4_GEOMETRY_COMPLETE"
    marker.write_text(
        f"stage=P4_GEOMETRY_COMPLETE\nmanifest_sha256={sha256(manifest_path)}\n"
        f"core_mpc_h={core_mpc_h}\nactive_rows={int(eligible.sum())}\n")


if __name__ == "__main__":
    main()
