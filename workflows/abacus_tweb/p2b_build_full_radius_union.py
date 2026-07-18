#!/usr/bin/env python3
"""Build the P2b fixed-radius augmentation for the full NGC+SGC canonical graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DENSITY_COL = 2


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--canonical-index", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--radius-mpc", type=float, default=14.78)
    args = ap.parse_args()
    started = time.time()

    meta_path = args.graph_dir / f"{args.prefix}_metadata.json"
    gnn_meta_path = args.graph_dir / f"{args.prefix}_cugraph_gnn_metadata.json"
    graph_meta = json.loads(meta_path.read_text())
    gnn_meta = json.loads(gnn_meta_path.read_text())
    points = np.load(args.graph_dir / graph_meta["files"]["points"], mmap_mode="r")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    index = np.load(args.canonical_index)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    if len(points) != len(cap):
        raise RuntimeError("canonical index does not align to graph rows")

    with np.load(gnn_meta["outputs"]["gnn_arrays_npz"]) as gnn:
        x = np.asarray(gnn["x"], dtype=np.float32)
        delaunay = np.asarray(gnn["edge_index"], dtype=np.int64)
    n = len(points)
    if delaunay.shape[0] != 2 or int(delaunay.max()) >= n:
        raise RuntimeError("invalid Delaunay edge index")
    du = np.minimum(delaunay[0], delaunay[1])
    dv = np.maximum(delaunay[0], delaunay[1])
    dkeys = np.sort(du * np.int64(n) + dv)
    dcontext = context[du] & context[dv]
    n_delaunay_context = int(dcontext.sum())
    n_delaunay_context_ngc = int((dcontext & (cap[du] == 1)).sum())
    n_delaunay_context_sgc = int((dcontext & (cap[du] == 0)).sum())
    if len(np.unique(dkeys)) != len(dkeys):
        raise RuntimeError("Delaunay graph contains duplicate undirected pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    component_results = {}
    total_radius = total_overlap = total_new = 0
    for cap_id, name in ((1, "NGC"), (0, "SGC")):
        ids = np.flatnonzero(context & (cap == cap_id)).astype(np.int64)
        print(f"[{name}] building cKDTree for {len(ids):,} context nodes", flush=True)
        tree = cKDTree(xyz[ids])
        local_pairs = tree.query_pairs(args.radius_mpc, output_type="ndarray")
        pairs = ids[local_pairs].astype(np.int32)
        del local_pairs, tree
        if len(pairs):
            pairs.sort(axis=1)
        radius_all = int(len(pairs))
        keys = (
            pairs[:, 0].astype(np.int64) * np.int64(n)
            + pairs[:, 1].astype(np.int64)
        )
        pos = np.searchsorted(dkeys, keys)
        overlap = (pos < len(dkeys)) & (dkeys[np.minimum(pos, len(dkeys) - 1)] == keys)
        new = pairs[~overlap]
        del pairs, keys, pos, overlap

        vec = xyz[new[:, 1]] - xyz[new[:, 0]]
        length = np.linalg.norm(vec, axis=1)
        unit = vec / np.maximum(length, 1e-12)[:, None]
        density = x[:, DENSITY_COL].astype(np.float64)
        contrast = np.ones(len(new), dtype=np.float64)
        valid_density = density[new[:, 0]] > 0
        contrast[valid_density] = (
            density[new[:, 1]][valid_density] / density[new[:, 0]][valid_density]
        )
        attr = np.column_stack([length, unit, contrast]).astype(np.float32)
        if not np.isfinite(attr).all():
            raise RuntimeError(f"{name} radius edge attributes are non-finite")
        if (cap[new[:, 0]] != cap_id).any() or (cap[new[:, 1]] != cap_id).any():
            raise RuntimeError(f"{name} output contains a cross-cap edge")
        if (~context[new[:, 0]]).any() or (~context[new[:, 1]]).any():
            raise RuntimeError(f"{name} output contains an endpoint outside context")

        pair_path = args.out_dir / f"{name.lower()}_radius_only_pairs.npy"
        attr_path = args.out_dir / f"{name.lower()}_radius_only_edge_attr.npy"
        np.save(pair_path, new)
        np.save(attr_path, attr)
        n_overlap = radius_all - len(new)
        component_results[name] = {
            "context_nodes": int(len(ids)),
            "radius_pairs_all": radius_all,
            "overlap_with_delaunay": int(n_overlap),
            "radius_only_pairs": int(len(new)),
            "pairs": str(pair_path),
            "pairs_sha256": sha256(pair_path),
            "edge_attr": str(attr_path),
            "edge_attr_sha256": sha256(attr_path),
        }
        total_radius += radius_all
        total_overlap += n_overlap
        total_new += len(new)
        print(
            f"[{name}] radius={radius_all:,} overlap={n_overlap:,} "
            f"new={len(new):,} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        del new, attr, vec, length, unit, contrast, ids

    payload = {
        "schema_version": 1,
        "stage": "P2b radius augmentation",
        "representation": (
            "existing full-footprint Delaunay graph/attributes plus per-cap radius-only "
            "pairs/attributes, all indexed by parent graph node ID"
        ),
        "parent_graph_metadata": str(meta_path),
        "parent_gnn_metadata": str(gnn_meta_path),
        "canonical_index": str(args.canonical_index),
        "radius_mpc": args.radius_mpc,
        "components": component_results,
        "counts": {
            "parent_nodes": n,
            "delaunay_pairs_all_parent_rows": int(delaunay.shape[1]),
            "delaunay_pairs_context": n_delaunay_context,
            "NGC_delaunay_pairs_context": n_delaunay_context_ngc,
            "SGC_delaunay_pairs_context": n_delaunay_context_sgc,
            "context_nodes": int(context.sum()),
            "radius_pairs_all_context": int(total_radius),
            "radius_delaunay_overlap": int(total_overlap),
            "radius_only_pairs": int(total_new),
            "union_pairs_context": int(n_delaunay_context + total_new),
        },
        "assembly_contract": {
            "patch_edges": (
                "fancy-index parent Delaunay edges and attributes, then append matching "
                "per-cap radius-only pairs and attributes; never join NGC to SGC"
            ),
            "edge_attr_columns": [
                "edge_length", "x_dir", "y_dir", "z_dir", "density_contrast"
            ],
            "cross_cap_pairs": 0,
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_dir / "p2b_union_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    complete_path = args.out_dir / "UNION_COMPLETE"
    complete_path.write_text(
        f"P2b radius={args.radius_mpc} context={int(context.sum())} "
        f"radius_only={total_new} cross_cap=0\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
