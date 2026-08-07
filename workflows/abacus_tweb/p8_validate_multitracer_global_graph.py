#!/usr/bin/env python3
"""Validate a globally indexed, hemisphere-disconnected multitracer graph."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
PRODUCT = "bf_proxy_response_v1"
GRAPH_PRODUCT = f"{PRODUCT}_photsys_marginal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--product", default=PRODUCT)
    parser.add_argument("--graph-product", default=GRAPH_PRODUCT)
    parser.add_argument("--prefix", default="bf_proxy_delaunay")
    parser.add_argument("--chunk-edges", type=int, default=5_000_000)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate(args: argparse.Namespace) -> dict:
    graph_dir = args.root / "graph" / args.graph_product / "global"
    metadata_path = graph_dir / f"{args.prefix}_metadata.json"
    catalogue_manifest_path = args.root / "catalogues" / args.product / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    catalogue = json.loads(catalogue_manifest_path.read_text())
    files = {name: graph_dir / path for name, path in metadata["files"].items()}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing graph artifacts: {missing}")

    points = np.load(files["points"], mmap_mode="r")
    source_points = np.load(catalogue["points"], mmap_mode="r")
    edges = np.load(files["edges"], mmap_mode="r")
    tetrahedra = np.load(files["tetrahedra_idx"], mmap_mode="r")
    volumes = np.load(files["tetrahedra_volumes"], mmap_mode="r")
    n_points = int(metadata["n_points"])
    n_edges = int(metadata["n_edges"])
    n_tetrahedra = int(metadata["n_tetrahedra"])
    gates = {
        "catalogue_passed": bool(catalogue["pass"]),
        "split_hemispheres": bool(metadata["split_hemispheres"]),
        "point_count_matches_catalogue": n_points == int(catalogue["total_rows"]),
        "point_shape": tuple(points.shape) == (n_points, 4),
        "source_point_shape": tuple(source_points.shape) == tuple(points.shape),
        "edge_shape": tuple(edges.shape) == (n_edges, 2),
        "tetrahedron_shape": tuple(tetrahedra.shape) == (n_tetrahedra, 4),
        "volume_shape": tuple(volumes.shape) == (n_tetrahedra,),
        "nonempty_edges": n_edges > 0,
        "nonempty_tetrahedra": n_tetrahedra > 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"graph shape gates failed: {gates}")

    point_identity = True
    for start in range(0, n_points, args.chunk_edges):
        stop = min(start + args.chunk_edges, n_points)
        if not np.array_equal(points[start:stop], source_points[start:stop]):
            point_identity = False
            break
    cross_cap_edges = 0
    edge_min, edge_max = n_points, -1
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    for start in range(0, n_edges, args.chunk_edges):
        block = np.asarray(edges[start : start + args.chunk_edges], dtype=np.int64)
        edge_min = min(edge_min, int(block.min(initial=n_points)))
        edge_max = max(edge_max, int(block.max(initial=-1)))
        cross_cap_edges += int(np.count_nonzero(cap[block[:, 0]] != cap[block[:, 1]]))
    finite_positive_volumes = True
    for start in range(0, n_tetrahedra, args.chunk_edges):
        block = np.asarray(volumes[start : start + args.chunk_edges], dtype=np.float64)
        if not np.all(np.isfinite(block) & (block > 0)):
            finite_positive_volumes = False
            break
    gates.update(
        {
            "points_exactly_match_catalogue": point_identity,
            "edge_indices_in_range": edge_min >= 0 and edge_max < n_points,
            "zero_cross_cap_edges": cross_cap_edges == 0,
            "tetrahedron_volumes_finite_positive": finite_positive_volumes,
        }
    )
    report = {
        "schema_version": "p8-multitracer-global-graph-validation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "product": args.product,
        "graph_product": args.graph_product,
        "metadata": str(metadata_path),
        "catalogue_manifest": str(catalogue_manifest_path),
        "n_points": n_points,
        "n_edges": n_edges,
        "n_tetrahedra": n_tetrahedra,
        "edge_index_min": edge_min,
        "edge_index_max": edge_max,
        "cross_cap_edges": cross_cap_edges,
        "gates": gates,
        "pass": all(gates.values()),
    }
    output = graph_dir / "global_graph_validation.json"
    atomic_json(output, report)
    if not report["pass"]:
        raise RuntimeError(f"global graph validation failed: {gates}")
    return report


def main() -> None:
    report = validate(parse_args())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
