#!/usr/bin/env python3
"""Audit/promote the existing full-footprint ph000 path1 graph for P1b/P2b."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.units as u
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z-core", nargs=2, type=float, default=(0.15, 0.55))
    ap.add_argument("--z-buffer", nargs=2, type=float, default=(0.10, 0.60))
    ap.add_argument("--sample-size", type=int, default=4096)
    args = ap.parse_args()

    meta_path = args.graph_dir / f"{args.prefix}_metadata.json"
    gnn_meta_path = args.graph_dir / f"{args.prefix}_cugraph_gnn_metadata.json"
    graph_meta = json.loads(meta_path.read_text())
    gnn_meta = json.loads(gnn_meta_path.read_text())

    columns = [
        "TARGETID", "RA", "DEC", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX",
        "LAMBDA1", "LAMBDA2", "LAMBDA3",
    ]
    table = fitsio.read(str(args.parent), columns=columns)
    n = len(table)
    points = np.load(args.graph_dir / graph_meta["files"]["points"], mmap_mode="r")
    xyz = np.load(args.graph_dir / graph_meta["files"]["points_xyz"], mmap_mode="r")
    edges = np.load(args.graph_dir / graph_meta["files"]["edges"], mmap_mode="r")
    cap = np.asarray(points[:, 3], dtype=np.int8)

    if points.shape != (n, 4) or xyz.shape != (n, 3):
        raise RuntimeError(f"row mismatch: parent={n}, points={points.shape}, xyz={xyz.shape}")
    if graph_meta["source_path"] != str(args.parent):
        raise RuntimeError("graph metadata source does not equal requested parent")

    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(n, size=min(args.sample_size, n), replace=False))
    sky = SkyCoord(
        ra=np.asarray(table["RA"][sample], dtype=np.float64) * u.deg,
        dec=np.asarray(table["DEC"][sample], dtype=np.float64) * u.deg,
        distance=Planck18.comoving_distance(np.asarray(table["Z"][sample], dtype=np.float64)),
        frame="icrs",
    )
    expected_xyz = np.column_stack([
        sky.cartesian.x.to_value(u.Mpc),
        sky.cartesian.y.to_value(u.Mpc),
        sky.cartesian.z.to_value(u.Mpc),
    ])
    expected_cap = (sky.galactic.b.to_value(u.deg) > 0).astype(np.int8)
    max_xyz_delta = float(np.max(np.abs(np.asarray(xyz[sample]) - expected_xyz)))
    cap_sample_match = bool(np.array_equal(cap[sample], expected_cap))

    z = np.asarray(table["Z"], dtype=np.float64)
    lambdas = np.column_stack([table["LAMBDA1"], table["LAMBDA2"], table["LAMBDA3"]])
    valid = (np.asarray(table["BOX_INDEX"]) >= 0) & np.isfinite(lambdas).all(axis=1)
    sentinel = (z >= 0.585) & (z < 0.595)
    in_buffer = (z >= args.z_buffer[0]) & (z < args.z_buffer[1]) & ~sentinel
    active = valid & (z >= args.z_core[0]) & (z < args.z_core[1])

    shell_counts = {}
    for lo, hi in ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55)):
        mask = valid & (z >= lo) & (z < hi)
        shell_counts[f"{lo:.2f}_{hi:.2f}"] = {
            "all": int(mask.sum()),
            "NGC": int((mask & (cap == 1)).sum()),
            "SGC": int((mask & (cap == 0)).sum()),
        }

    cross_cap_edges = ngc_edges = sgc_edges = 0
    chunk = 5_000_000
    for start in range(0, len(edges), chunk):
        edge_chunk = np.asarray(edges[start:start + chunk], dtype=np.int64)
        c0 = cap[edge_chunk[:, 0]]
        c1 = cap[edge_chunk[:, 1]]
        cross_cap_edges += int((c0 != c1).sum())
        ngc_edges += int(((c0 == 1) & (c1 == 1)).sum())
        sgc_edges += int(((c0 == 0) & (c1 == 0)).sum())

    with np.load(gnn_meta["outputs"]["gnn_arrays_npz"]) as gnn:
        x = gnn["x"]
        edge_index = gnn["edge_index"]
        edge_attr = gnn["edge_attr"]
        gnn_shapes = {
            "x": list(x.shape),
            "edge_index": list(edge_index.shape),
            "edge_attr": list(edge_attr.shape),
        }
        node_features_finite = bool(np.isfinite(x).all())
        edge_features_finite = bool(np.isfinite(edge_attr).all())
        edge_index_exact = bool(
            edge_index.shape == (2, len(edges))
            and np.array_equal(edge_index[0], edges[:, 0])
            and np.array_equal(edge_index[1], edges[:, 1])
        )

    counts = {
        "parent_rows": n,
        "NGC_rows": int((cap == 1).sum()),
        "SGC_rows": int((cap == 0).sum()),
        "buffer_rows": int(in_buffer.sum()),
        "active_rows": int(active.sum()),
        "active_NGC": int((active & (cap == 1)).sum()),
        "active_SGC": int((active & (cap == 0)).sum()),
        "invalid_target_rows_in_core": int(
            (((z >= args.z_core[0]) & (z < args.z_core[1])) & ~valid).sum()
        ),
        "sentinel_rows": int(sentinel.sum()),
        "delaunay_pairs": int(len(edges)),
        "NGC_delaunay_pairs": ngc_edges,
        "SGC_delaunay_pairs": sgc_edges,
        "cross_cap_pairs": cross_cap_edges,
    }
    alignment = {
        "metadata_source_exact": True,
        "sample_size": int(len(sample)),
        "max_abs_xyz_mpc": max_xyz_delta,
        "cap_sample_exact": cap_sample_match,
        "targetid_unique": bool(len(np.unique(table["TARGETID"])) == n),
        "gnn_edge_index_exact": edge_index_exact,
    }
    gates = {
        "row_alignment": points.shape == (n, 4) and xyz.shape == (n, 3),
        "coordinate_alignment": max_xyz_delta < 1e-5,
        "cap_alignment": cap_sample_match,
        "targetid_unique": alignment["targetid_unique"],
        "zero_cross_cap_edges": cross_cap_edges == 0,
        "gnn_edge_index_exact": edge_index_exact,
        "features_finite": node_features_finite and edge_features_finite,
        "both_caps_have_active_rows": counts["active_NGC"] > 0 and counts["active_SGC"] > 0,
    }
    payload = {
        "schema_version": 1,
        "scope": "P1b/P2b full ph000 path1 NGC+SGC audit",
        "parent": str(args.parent),
        "graph_metadata": str(meta_path),
        "gnn_metadata": str(gnn_meta_path),
        "counts": counts,
        "shell_counts": shell_counts,
        "alignment": alignment,
        "features": {
            "node_columns": gnn_meta["node_feature_columns"],
            "edge_columns": gnn_meta["edge_feature_columns"],
            "shapes": gnn_shapes,
            "node_features_finite": node_features_finite,
            "edge_features_finite": edge_features_finite,
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise RuntimeError("full-footprint audit failed one or more gates")


if __name__ == "__main__":
    main()
