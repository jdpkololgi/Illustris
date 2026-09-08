#!/usr/bin/env python3
"""Prepare E2E source/geometry products using bounded JSON metadata reads only.

No source array, FITS table, target value, HDF5 payload or sealed-phase path is
opened. Geometry proposals are not support-selected or assigned to data splits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess

REPO = Path(__file__).resolve().parents[2]
VISIBLE = {"ph000", "ph002", "ph003", "ph004", "ph005", "ph006"}
MAX_JSON_BYTES = 2 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(path: Path) -> Path:
    """Reject the reserved phase even through a symlink, before opening it."""
    path = Path(path)
    if "ph001" in str(path) or "ph001" in str(path.resolve()):
        raise PermissionError("E2E preparation forbids ph001")
    return path


def read_json(path: Path) -> tuple[dict, dict]:
    path = safe_path(path)
    if path.suffix != ".json" or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"not a bounded JSON metadata source: {path}")
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()), "bytes": len(raw), "sha256": digest(raw)
    }


def stat_payload(path: str, recorded_sha256: str | None = None) -> dict:
    path = safe_path(Path(path))
    exists = path.is_file()
    return {
        "path": str(path.resolve()), "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "recorded_sha256": recorded_sha256,
        "payload_read": False, "payload_hash_verified": False,
    }


def validate_config(config: dict) -> None:
    if config.get("schema_version") != "e2e-field-data-prep-v1":
        raise ValueError("unexpected preparation schema")
    phases = config["development_pool"] + config["legacy_benchmarks"]
    if len(phases) != len(set(phases)) or not set(phases) <= VISIBLE:
        raise PermissionError("duplicate, reserved or unregistered phase")
    if "ph006" in config["development_pool"]:
        raise ValueError("ph006 must remain a legacy benchmark")
    if config["external_confirmation"] is not None or config["future_blind"] is not None:
        raise ValueError("this preparer cannot assign fresh evidence")
    c = config["coordinate_contract"]
    if c["cell_mpc"] != 5.0 or c["observer_h"] != 0.6766:
        raise ValueError("inherited observer-grid units changed")
    if not math.isclose(c["cell_mpc"] * c["observer_h"], c["cell_mpc_h"], abs_tol=1e-12):
        raise ValueError("Mpc and Mpc/h are inconsistent")
    if c["target_epoch"] != 0.2 or c["smoothing_mpc_h"] != 7.0:
        raise ValueError("target epoch/smoothing changed")
    g = config["geometry_proposals"]
    if len(g["parent_side_voxels"]) > 2 or not 1 <= g["anchors_per_cap"] <= 8:
        raise ValueError("preparation geometry budget exceeded")
    for n in g["parent_side_voxels"]:
        if not isinstance(n, int) or not 32 <= n <= 128 or n % g["child_core_side_voxels"]:
            raise ValueError("parent must tile into integral child cores")
    if g["child_core_side_voxels"] <= 0 or g["alignment_voxels"] <= 0:
        raise ValueError("invalid core/alignment")
    if g["child_halo_voxels"] < 0 or g["observation_halo_voxels"] < 0:
        raise ValueError("negative halo")
    if any(config["bounded_io"][k] for k in (
        "read_hdf5_payload", "read_npz_payload", "hash_large_source_payloads"
    )):
        raise ValueError("this entrypoint only supports metadata reads")


def grids_equal(a: dict, b: dict) -> bool:
    return (a["shape"] == b["shape"]
            and math.isclose(a["cell_mpc"], b["cell_mpc"], abs_tol=1e-12)
            and all(abs(x - y) <= 1e-9 for x, y in zip(a["origin_mpc"], b["origin_mpc"])))


def periodic_intervals(lo: float, hi: float, box: float) -> list[list[float]]:
    """Half-open source-box footprint, including crossings of the box edge."""
    if hi <= lo:
        raise ValueError("empty footprint")
    if hi - lo >= box:
        return [[0.0, box]]
    start, width = lo % box, hi - lo
    if start + width <= box:
        return [[start, start + width]]
    return [[start, box], [0.0, start + width - box]]


def footprints_overlap(a: list, b: list) -> bool:
    return all(any(max(x[0], y[0]) < min(x[1], y[1]) for x in aa for y in bb)
               for aa, bb in zip(a, b))


def propose_parents(phase: str, cap: str, grid: dict, config: dict) -> list[dict]:
    g, c = config["geometry_proposals"], config["coordinate_contract"]
    largest = max(g["parent_side_voxels"])
    halo, alignment = g["observation_halo_voxels"], g["alignment_voxels"]
    seed = int(digest(f"{g['seed']}:{phase}:{cap}".encode())[:16], 16)
    rng = random.Random(seed)
    center_axes = [list(range(largest // 2 + halo,
                            n - largest // 2 - halo + 1, alignment)) for n in grid["shape"]]
    if not all(center_axes):
        raise ValueError("parent/context geometry exceeds source lattice")
    centers = set()
    attempts = 0
    while len(centers) < g["anchors_per_cap"] and attempts < 100:
        centers.add(tuple(rng.choice(axis) for axis in center_axes))
        attempts += 1
    rows = []
    for ordinal, center in enumerate(sorted(centers)):
        anchor = f"{phase}_{cap}_anchor{ordinal:02d}"
        for side in g["parent_side_voxels"]:
            start = [v - side // 2 for v in center]
            stop = [v + side // 2 for v in center]
            obs_start = [v - halo for v in start]
            obs_stop = [v + halo for v in stop]
            origin, cell, h = grid["origin_mpc"], grid["cell_mpc"], c["observer_h"]
            lo = [o + v * cell for o, v in zip(origin, obs_start)]
            hi = [o + v * cell for o, v in zip(origin, obs_stop)]
            footprint = [periodic_intervals(x * h + c["source_box_offset_mpc_h"],
                                           y * h + c["source_box_offset_mpc_h"],
                                           c["source_box_size_mpc_h"]) for x, y in zip(lo, hi)]
            parent_id = f"{anchor}_n{side:03d}"
            children = []
            core, chalo = g["child_core_side_voxels"], g["child_halo_voxels"]
            for ijk in itertools.product(range(side // core), repeat=3):
                begin = [s + i * core for s, i in zip(start, ijk)]
                end = [b + core for b in begin]
                children.append({
                    "child_id": parent_id + "_" + "_".join(map(str, ijk)),
                    "core_start": begin, "core_stop": end,
                    "evaluation_start": [max(p, b - chalo) for p, b in zip(start, begin)],
                    "evaluation_stop": [min(p, e + chalo) for p, e in zip(stop, end)],
                })
            rows.append({
                "parent_id": parent_id, "anchor_id": anchor, "phase": phase, "cap": cap,
                "role": "legacy_benchmark" if phase == "ph006" else "development_pool_unassigned",
                "start": start, "stop": stop, "context_start": obs_start, "context_stop": obs_stop,
                "side_mpc": side * cell, "side_mpc_h": side * cell * h,
                "source_box_context_footprint_mpc_h": footprint,
                "source_box_mapping_status": "inherited_mapping_requires_independent_audit",
                "support_screened": False, "split_assigned": False,
                "tidal_boundary_closure_passed": False,
                "children": children,
            })
    return rows


def build_products(config: dict) -> dict[str, dict]:
    validate_config(config)
    registry_path = REPO / config["phase_registry"]
    registry, reg_record = read_json(registry_path)
    root, adapters = Path(config["phase_root"]), Path(config["response_contract_root"])
    sources, parents = [], []
    for phase in config["development_pool"] + config["legacy_benchmarks"]:
        # Phase allowlisting precedes any phase-specific path resolution or read.
        if phase not in VISIBLE or phase not in registry["phases"]:
            raise PermissionError(f"invalid phase: {phase}")
        marker, marker_record = read_json(root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json")
        adapter, adapter_record = read_json(adapters / "adapters" / phase / "field/adapter_manifest.json")
        response, response_record = read_json(Path(adapter["p3_manifest"]))
        if marker.get("phase") != phase or not marker.get("pass") or not adapter.get("pass"):
            raise ValueError(f"{phase}: incomplete source metadata")
        if response.get("phase") != phase or not response.get("pass"):
            raise ValueError(f"{phase}: response metadata mismatch")
        if response_record["sha256"] != adapter["p3_manifest_sha256"]:
            raise ValueError(f"{phase}: response manifest hash differs from adapter")
        contract = marker["contract"]
        if contract.get("double_smoothing_applied") or contract["target_epoch"] != 0.2:
            raise ValueError("invalid target convention")
        for cap in ("NGC", "SGC"):
            t, a = marker["components"][cap], adapter["caps"][cap]
            grid = t["grid"]
            if not grids_equal(grid, a) or grid["cell_mpc"] != config["coordinate_contract"]["cell_mpc"]:
                raise ValueError(f"{phase}/{cap}: target and response grids differ")
            target_file = stat_payload(t["file"], t.get("file_sha256"))
            response_file = stat_payload(a["field_path"], a.get("field_sha256"))
            sources.append({
                "phase": phase, "cap": cap, "grid": grid,
                "target_metadata": marker_record, "adapter_metadata": adapter_record,
                "response_metadata": response_record, "target_file": target_file,
                "response_file": response_file,
                "target_dataset": "delta_r7", "response_channels": adapter["channel_order"],
                "source_payloads_present": target_file["exists"] and response_file["exists"],
                "target_values_read": False, "normalizers_inherited": False,
                "source_epoch": 0.2, "source_sampling": "nearest_native_grid_cell_trace",
                "native_tensor_reference": "not_materialized",
            })
            parents.extend(propose_parents(phase, cap, grid, config))
    conflicts = []
    for i, a in enumerate(parents):
        for b in parents[i + 1:]:
            if a["phase"] == b["phase"] and footprints_overlap(
                a["source_box_context_footprint_mpc_h"], b["source_box_context_footprint_mpc_h"]
            ):
                conflicts.append([a["parent_id"], b["parent_id"]])
    common = {"schema_version": "e2e-field-preparation-products-v1", "training_ready": False}
    return {
        "source_inventory.json": {**common, "phase_registry": reg_record,
                                  "coordinate_contract": config["coordinate_contract"], "sources": sources},
        "parent_geometry_proposals.json": {**common, "selection": "lattice_only_not_science_sample",
                                            "parents": parents},
        "source_box_overlap_proposals.json": {
            **common, "status": "provisional_inherited_coordinate_mapping",
            "rule": "conflicting parents and all descendants must not cross data roles",
            "conflicting_parent_pairs": conflicts,
            "limitations": ["does not validate coordinate mapping", "not a complete phase leakage audit",
                            "no galaxy, halo or native source identity payload read"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/e2e_field_data_prep_v1.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config, config_record = read_json(args.config)
    products = build_products(config)
    output = safe_path(args.output).resolve()
    # Publishing only below this programme's repository evidence namespace.
    allowed = (REPO / "docs/evidence/e2e_field_v1").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("output must be a new child of docs/evidence/e2e_field_v1")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, payload in products.items():
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with (output / name).open("xb") as stream:
            stream.write(raw)
        hashes[name] = {"sha256": digest(raw), "bytes": len(raw)}
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config_record,
        "source_sha256": digest(Path(__file__).read_bytes()),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "products": hashes, "preparation_complete": True, "training_ready": False,
        "source_array_payloads_read": 0, "ph001_accessed": False,
        "parent_count": len(products["parent_geometry_proposals.json"]["parents"]),
        "deferred_compute": config["deferred_compute"],
    }
    with (output / "PREPARATION_COMPLETE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
