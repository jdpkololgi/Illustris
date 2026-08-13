#!/usr/bin/env python3
"""Build a phase P2 Delaunay graph with independent cap checkpoints.

The canonical P1 graph is global *within each catalogue* but NGC and SGC are
disconnected physical components.  Each cap is therefore an atomic, resumable
work unit.  ``merge`` concatenates verified cap products while retaining the
exact P1 row/node IDs; it never reconstructs a patch graph or graph metrics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_abacus_graph import _build_graph_artifacts  # noqa: E402
from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file  # noqa: E402
from shared.resource_requirements import require_cpu_mpi_slurm  # noqa: E402

CAPS = {"NGC": 1, "SGC": 0}


class PhaseGraphError(RuntimeError):
    """The phase graph violates the canonical P1 or disconnected-cap contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def defaults(registry: dict[str, Any], phase: str) -> tuple[Path, Path, Path]:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "p1_canonical/points.npy", root / "p1_canonical/canonical_index.npz", root / "p2_graph"


def validate_cap_artifacts(marker: Path) -> dict[str, Any]:
    payload = json.loads(marker.read_text())
    for record in payload["artifacts"].values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise PhaseGraphError(f"cap artifact missing/hash mismatch: {path}")
    if not payload.get("pass", False):
        raise PhaseGraphError(f"cap marker is not passing: {marker}")
    return payload


def build_cap(phase: str, cap_name: str, points_path: Path, index_path: Path, out_dir: Path) -> None:
    require_cpu_mpi_slurm("p10_build_phase_graph.py", min_tasks=1)
    cap_id = CAPS[cap_name]
    cap_dir = out_dir / "caps" / cap_name.lower()
    marker = cap_dir / "CAP_GRAPH_COMPLETE.json"
    paths = {
        "edges": cap_dir / "edges_global_idx.npy",
        "tetrahedra": cap_dir / "tetrahedra_global_idx.npy",
        "volumes": cap_dir / "tetrahedra_volumes.npy",
    }
    if marker.is_file():
        print(json.dumps(validate_cap_artifacts(marker), indent=2, sort_keys=True))
        return
    complete_arrays = all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    if cap_dir.exists() and any(cap_dir.iterdir()) and not complete_arrays:
        raise PhaseGraphError(f"ambiguous partial cap directory: {cap_dir}")
    cap_dir.mkdir(parents=True, exist_ok=True)
    points = np.load(points_path, mmap_mode="r")
    index = np.load(index_path)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    parent = np.flatnonzero(cap == cap_id).astype(np.int64)
    if points.shape != (len(cap), 4) or not len(parent):
        raise PhaseGraphError("P1 points/index/cap mismatch")
    if complete_arrays:
        print(f"[{phase}/{cap_name}] recovering completed arrays without marker", flush=True)
        global_edges = np.load(paths["edges"], mmap_mode="r")
        global_tetra = np.load(paths["tetrahedra"], mmap_mode="r")
        volumes = np.load(paths["volumes"], mmap_mode="r")
    else:
        local_points = np.asarray(points[parent], dtype=np.float64)
        print(f"[{phase}/{cap_name}] Delaunay rows={len(parent):,}", flush=True)
        edges, tetrahedra, volumes = _build_graph_artifacts(
            local_points, alpha_sq=math.inf, split_hemispheres=False,
        )
        global_edges = parent[edges].astype(np.int32, copy=False)
        global_tetra = parent[tetrahedra].astype(np.int32, copy=False)
    if len(global_edges) and np.any(cap[global_edges] != cap_id):
        raise PhaseGraphError("cap graph produced cross-cap edge")
    if not complete_arrays:
        atomic_npy(paths["edges"], global_edges)
        atomic_npy(paths["tetrahedra"], global_tetra)
        atomic_npy(paths["volumes"], volumes)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "schema_version": "p10-p2-cap-graph-v1", "created_utc": utc_now(),
        "phase": phase, "cap": cap_name, "cap_id": cap_id, "git_sha": git_sha,
        "p1": {"points": str(points_path.resolve()), "points_sha256": sha256_file(points_path),
               "index": str(index_path.resolve()), "index_sha256": sha256_file(index_path)},
        "counts": {"parent_rows": len(cap), "cap_rows": len(parent),
                   "edges": len(global_edges), "tetrahedra": len(global_tetra)},
        "index_contract": "local Delaunay indices mapped back to immutable P1 parent rows",
        "artifacts": {
            name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                   "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "gates": {"nonempty": len(global_edges) > 0 and len(global_tetra) > 0,
                  "volume_alignment": len(volumes) == len(global_tetra),
                  "finite_positive_volumes": bool(np.isfinite(volumes).all() and np.all(volumes > 0)),
                  "endpoint_bounds": bool(len(global_edges) == 0 or global_edges.max() < len(cap)),
                  "no_cross_cap_edges": bool(
                      len(global_edges) == 0 or np.all(cap[global_edges] == cap_id))},
    }
    payload["pass"] = all(payload["gates"].values())
    if not payload["pass"]:
        raise PhaseGraphError(f"cap graph gates failed: {payload['gates']}")
    atomic_json(marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def concatenate_npy(paths: list[Path], output: Path, trailing_shape: tuple[int, ...], dtype: np.dtype) -> None:
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    total = sum(len(array) for array in arrays)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npy")
    merged = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=dtype, shape=(total, *trailing_shape),
    )
    start = 0
    for array in arrays:
        stop = start + len(array)
        merged[start:stop] = array
        start = stop
    merged.flush()
    del merged, arrays
    os.replace(temporary, output)


def merge_caps(phase: str, points_path: Path, index_path: Path, out_dir: Path) -> None:
    marker = out_dir / "GRAPH_COMPLETE.json"
    prefix = f"{phase}_bgs_bright_full_delaunay"
    metadata_path = out_dir / f"{prefix}_metadata.json"
    if marker.is_file() and metadata_path.is_file():
        payload = json.loads(marker.read_text())
        for path in (metadata_path, out_dir / f"{prefix}_edges_combined_idx.npy",
                     out_dir / f"{prefix}_tetrahedra_idx.npy",
                     out_dir / f"{prefix}_tetrahedra_volumes.npy"):
            if not path.is_file():
                raise PhaseGraphError(f"complete graph is missing {path}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    cap_payloads = {
        name: validate_cap_artifacts(out_dir / "caps" / name.lower() / "CAP_GRAPH_COMPLETE.json")
        for name in CAPS
    }
    if any(payload["phase"] != phase for payload in cap_payloads.values()):
        raise PhaseGraphError("cap phase mismatch")
    points = np.load(points_path, mmap_mode="r")
    index = np.load(index_path)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    edge_path = out_dir / f"{prefix}_edges_combined_idx.npy"
    tetra_path = out_dir / f"{prefix}_tetrahedra_idx.npy"
    volume_path = out_dir / f"{prefix}_tetrahedra_volumes.npy"
    if any(path.exists() for path in (edge_path, tetra_path, volume_path, metadata_path)):
        raise PhaseGraphError("ambiguous partial merged graph; cap checkpoints remain reusable")
    concatenate_npy(
        [Path(cap_payloads[name]["artifacts"]["edges"]["path"]) for name in CAPS],
        edge_path, (2,), np.dtype("int32"),
    )
    concatenate_npy(
        [Path(cap_payloads[name]["artifacts"]["tetrahedra"]["path"]) for name in CAPS],
        tetra_path, (4,), np.dtype("int32"),
    )
    concatenate_npy(
        [Path(cap_payloads[name]["artifacts"]["volumes"]["path"]) for name in CAPS],
        volume_path, (), np.dtype("float64"),
    )
    edges = np.load(edge_path, mmap_mode="r")
    tetra = np.load(tetra_path, mmap_mode="r")
    volumes = np.load(volume_path, mmap_mode="r")
    cross = int(np.count_nonzero(cap[edges[:, 0]] != cap[edges[:, 1]]))
    if cross or len(volumes) != len(tetra) or points.shape != (len(cap), 4):
        raise PhaseGraphError("merged graph violates cap/volume/P1 identity")
    metadata = {
        "prefix": prefix, "mode": "delaunay", "alpha_sq": None,
        "split_hemispheres": True, "source": "P10/P1 points",
        "source_path": str(points_path.resolve()), "n_points": len(points),
        "n_point_columns": 4, "n_edges": len(edges), "n_tetrahedra": len(tetra),
        "catalog_filters": {"apply_y1y5_filter": False,
                            "exclude_invalid_box_index": False, "r_mag_app_lt": None},
        "files": {"points": str(points_path.resolve()),
                  "points_xyz": str(points_path.resolve()),
                  "edges": edge_path.name, "tetrahedra_idx": tetra_path.name,
                  "tetrahedra_volumes": volume_path.name},
        "phase": phase, "canonical_index": str(index_path.resolve()),
        "cap_checkpoints": {name: str((out_dir / "caps" / name.lower() / "CAP_GRAPH_COMPLETE.json").resolve())
                            for name in CAPS},
    }
    atomic_json(metadata_path, metadata)
    payload = {
        "schema_version": "p10-p2-graph-v1", "created_utc": utc_now(),
        "phase": phase, "prefix": prefix, "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "p1_points_sha256": sha256_file(points_path), "p1_index_sha256": sha256_file(index_path),
        "counts": {"nodes": len(points), "edges": len(edges), "tetrahedra": len(tetra),
                   "NGC_nodes": int(np.sum(cap == 1)), "SGC_nodes": int(np.sum(cap == 0)),
                   "cross_cap_edges": cross},
        "artifacts": {
            "edges": {"path": str(edge_path.resolve()), "sha256": sha256_file(edge_path)},
            "tetrahedra": {"path": str(tetra_path.resolve()), "sha256": sha256_file(tetra_path)},
            "volumes": {"path": str(volume_path.resolve()), "sha256": sha256_file(volume_path)},
        },
        "pass": True,
    }
    atomic_json(marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stage", choices=("cap", "merge"), required=True)
    parser.add_argument("--cap", choices=tuple(CAPS))
    parser.add_argument("--points", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise PhaseGraphError(f"unregistered phase: {args.phase}")
    points, index, out_dir = defaults(registry, args.phase)
    points, index, out_dir = args.points or points, args.index or index, args.out_dir or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "cap":
        if not args.cap:
            raise PhaseGraphError("--stage cap requires --cap")
        build_cap(args.phase, args.cap, points, index, out_dir)
    else:
        if args.cap:
            raise PhaseGraphError("--cap is invalid for --stage merge")
        merge_caps(args.phase, points, index, out_dir)


if __name__ == "__main__":
    try:
        main()
    except (PhaseGraphError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
