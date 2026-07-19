#!/usr/bin/env python3
"""Build the immutable P5 union-edge table and incident-edge CSR index.

This is a promotion/subsetting operation over P1b/P2b/P4 artifacts.  It does
not construct a new graph and does not recompute a graph metric.  The output
is optimized for lazy exact patch extraction by parent graph node ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from numba import njit


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@njit(cache=True)
def add_degrees(pairs: np.ndarray, degree: np.ndarray) -> None:
    for row in range(len(pairs)):
        degree[pairs[row, 0]] += 1
        degree[pairs[row, 1]] += 1


@njit(cache=True)
def fill_incident(
    pairs: np.ndarray, first_edge_id: int, cursor: np.ndarray, incident: np.ndarray
) -> None:
    for row in range(len(pairs)):
        edge_id = first_edge_id + row
        left = pairs[row, 0]
        right = pairs[row, 1]
        incident[cursor[left]] = edge_id
        cursor[left] += 1
        incident[cursor[right]] = edge_id
        cursor[right] += 1


def open_npy(path: Path, dtype, shape):
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    return partial, np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def finish_npy(partial: Path, array: np.memmap, final: Path) -> None:
    array.flush()
    del array
    os.replace(partial, final)


def chunks(length: int, rows: int):
    for start in range(0, length, rows):
        yield start, min(start + rows, length)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-arrays", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_cugraph_gnn_arrays.npz"))
    ap.add_argument("--delaunay-pairs", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_edges_combined_idx.npy"))
    ap.add_argument("--canonical-index", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"))
    ap.add_argument("--p2-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/p2b_union_manifest.json"))
    ap.add_argument("--radius-ngc", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/ngc_radius_only_pairs.npy"))
    ap.add_argument("--radius-ngc-attr", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/ngc_radius_only_edge_attr.npy"))
    ap.add_argument("--radius-sgc", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/sgc_radius_only_pairs.npy"))
    ap.add_argument("--radius-sgc-attr", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/sgc_radius_only_edge_attr.npy"))
    ap.add_argument("--p4-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/spatial_manifest.json"))
    ap.add_argument("--active-assignment", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz"))
    ap.add_argument("--graph-support", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/graph_support_active.npz"))
    ap.add_argument("--cores", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/cores.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter"))
    ap.add_argument("--chunk-rows", type=int, default=5_000_000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("GRAPH_PATCH_READY", "parity_report.json"):
        stale_path = args.out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    p2 = json.loads(args.p2_manifest.read_text())
    p4 = json.loads(args.p4_manifest.read_text())
    if not p4.get("pass"):
        raise RuntimeError("P4 must pass before building P5")
    if p4["inputs"]["p2_manifest_sha256"] != sha256(args.p2_manifest):
        raise RuntimeError("P4/P2 manifest identity mismatch")

    index = np.load(args.canonical_index)
    context = np.asarray(index["context"], dtype=bool)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    n_parent = len(context)
    gnn = np.load(args.gnn_arrays)
    x = np.asarray(gnn["x"], dtype=np.float32)
    edge_index = np.asarray(gnn["edge_index"], dtype=np.int64)
    delaunay_attr = np.asarray(gnn["edge_attr"], dtype=np.float32)
    source_delaunay = np.load(args.delaunay_pairs, mmap_mode="r")
    if x.shape != (n_parent, 7) or edge_index.shape[1] != len(source_delaunay):
        raise RuntimeError("P1/P2 parent node or edge count mismatch")

    # Validate the independent pair file against the GNN source in chunks and
    # count the context-only Delaunay rows without a full-size temporary mask.
    delaunay_context_count = 0
    for start, stop in chunks(len(source_delaunay), args.chunk_rows):
        pairs = np.asarray(source_delaunay[start:stop], dtype=np.int64)
        if not np.array_equal(pairs, edge_index[:, start:stop].T):
            raise RuntimeError(f"Delaunay pair/GNN edge order mismatch at {start}")
        delaunay_context_count += int(np.sum(context[pairs[:, 0]] & context[pairs[:, 1]]))

    radius_sources = [
        ("NGC", np.load(args.radius_ngc, mmap_mode="r"),
         np.load(args.radius_ngc_attr, mmap_mode="r"), 1),
        ("SGC", np.load(args.radius_sgc, mmap_mode="r"),
         np.load(args.radius_sgc_attr, mmap_mode="r"), 0),
    ]
    total_edges = delaunay_context_count + sum(len(pairs) for _, pairs, _, _ in radius_sources)
    if total_edges != int(p2["counts"]["union_pairs_context"]):
        raise RuntimeError("P2 union count mismatch")

    node_path = args.out_dir / "node_features.npy"
    np.save(node_path, x, allow_pickle=False)
    del x
    pair_path = args.out_dir / "union_pairs.npy"
    attr_path = args.out_dir / "union_edge_features.npy"
    pair_partial, pair_out = open_npy(pair_path, np.int32, (total_edges, 2))
    attr_partial, attr_out = open_npy(attr_path, np.float32, (total_edges, 5))
    write = 0
    for start, stop in chunks(len(source_delaunay), args.chunk_rows):
        pairs = np.asarray(source_delaunay[start:stop], dtype=np.int64)
        valid = context[pairs[:, 0]] & context[pairs[:, 1]]
        count = int(valid.sum())
        pair_out[write:write + count] = pairs[valid].astype(np.int32)
        attr_out[write:write + count] = delaunay_attr[start:stop][valid]
        write += count
    source_offsets = {"delaunay_context": [0, write]}
    for name, pairs, attrs, cap_id in radius_sources:
        start_write = write
        if len(pairs) != len(attrs):
            raise RuntimeError(f"{name} radius pair/attribute count mismatch")
        for start, stop in chunks(len(pairs), args.chunk_rows):
            block = np.asarray(pairs[start:stop], dtype=np.int64)
            if (
                np.any(~context[block[:, 0]]) or np.any(~context[block[:, 1]])
                or np.any(cap[block[:, 0]] != cap_id) or np.any(cap[block[:, 1]] != cap_id)
            ):
                raise RuntimeError(f"{name} radius block violates P2 context/cap contract")
            n = len(block)
            pair_out[write:write + n] = block.astype(np.int32)
            attr_out[write:write + n] = np.asarray(attrs[start:stop], dtype=np.float32)
            write += n
        source_offsets[f"radius_{name.lower()}"] = [start_write, write]
    if write != total_edges:
        raise RuntimeError("union edge write count mismatch")
    finish_npy(pair_partial, pair_out, pair_path)
    finish_npy(attr_partial, attr_out, attr_path)
    del delaunay_attr, edge_index, gnn

    # Persistent incident-edge CSR.  Each canonical undirected edge occurs
    # exactly twice, once in each endpoint's incident list.
    union_pairs = np.load(pair_path, mmap_mode="r")
    degree = np.zeros(n_parent, dtype=np.int64)
    for start, stop in chunks(total_edges, args.chunk_rows):
        add_degrees(np.asarray(union_pairs[start:stop], dtype=np.int32), degree)
    offsets = np.empty(n_parent + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    if int(offsets[-1]) != 2 * total_edges:
        raise RuntimeError("CSR degree sum mismatch")
    offsets_path = args.out_dir / "incident_offsets.npy"
    np.save(offsets_path, offsets, allow_pickle=False)
    incident_path = args.out_dir / "incident_edge_id.npy"
    incident_partial, incident = open_npy(
        incident_path, np.int32, (2 * total_edges,)
    )
    cursor = offsets[:-1].copy()
    for start, stop in chunks(total_edges, args.chunk_rows):
        fill_incident(
            np.asarray(union_pairs[start:stop], dtype=np.int32), start, cursor, incident
        )
    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("CSR fill cursor mismatch")
    finish_npy(incident_partial, incident, incident_path)
    del cursor, degree

    # Compact P4 core lookup, retaining all P1-active rows but exposing
    # supervised eligibility and strict K-support separately.
    active = np.load(args.active_assignment)
    support = np.load(args.graph_support)
    cores = np.load(args.cores)
    active_parent = np.asarray(active["parent_node_id"], dtype=np.int64)
    if not np.array_equal(active_parent, np.asarray(support["parent_node_id"], dtype=np.int64)):
        raise RuntimeError("P4 active/graph support row order mismatch")
    core_id = np.asarray(active["core_id"], dtype=np.int32)
    order = np.argsort(core_id, kind="stable")
    sorted_core = core_id[order]
    n_cores = len(cores["core_id"])
    if not np.array_equal(np.asarray(cores["core_id"]), np.arange(n_cores)):
        raise RuntimeError("P4 core IDs must be dense for the adapter lookup")
    core_counts = np.bincount(sorted_core, minlength=n_cores)
    core_offsets = np.empty(n_cores + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(core_counts, out=core_offsets[1:])
    compact = {
        "core_active_offsets.npy": core_offsets,
        "core_active_parent.npy": active_parent[order].astype(np.int32),
        "core_active_eligible.npy": np.asarray(active["supervised_eligible"], dtype=bool)[order],
        "core_active_safe2hop.npy": np.asarray(support["safe_2pass"], dtype=bool)[order],
        "core_active_safe4hop.npy": np.asarray(support["safe_4pass"], dtype=bool)[order],
        "core_fold.npy": np.asarray(cores["fold"], dtype=np.uint8),
        "core_cap.npy": np.asarray(cores["cap"], dtype=np.uint8),
    }
    for name, array in compact.items():
        np.save(args.out_dir / name, array, allow_pickle=False)

    artifacts = {
        name: {"path": str(args.out_dir / name), "sha256": sha256(args.out_dir / name)}
        for name in [
            "node_features.npy", "union_pairs.npy", "union_edge_features.npy",
            "incident_offsets.npy", "incident_edge_id.npy", *compact.keys()
        ]
    }
    gates = {
        "p4_passes": bool(p4.get("pass")),
        "p2_identity_matches_p4": p4["inputs"]["p2_manifest_sha256"] == sha256(args.p2_manifest),
        "parent_node_count_matches": len(np.load(node_path, mmap_mode="r")) == n_parent,
        "union_pair_count_matches": len(np.load(pair_path, mmap_mode="r")) == total_edges,
        "union_attr_count_matches": len(np.load(attr_path, mmap_mode="r")) == total_edges,
        "incident_count_is_twice_edges": len(np.load(incident_path, mmap_mode="r")) == 2 * total_edges,
        "csr_offsets_complete": int(np.load(offsets_path, mmap_mode="r")[-1]) == 2 * total_edges,
        "active_lookup_complete": int(core_offsets[-1]) == len(active_parent),
        "no_graph_metrics_recomputed": True,
    }
    manifest = {
        "schema_version": 1,
        "stage": "P5 canonical GraphNet patch adapter index",
        "representation": (
            "immutable P1b node features plus P2b parent-Delaunay/context and radius-only "
            "pairs copied into one canonical table; incident CSR enables exact lazy K-hop views"
        ),
        "inputs": {
            "gnn_arrays": str(args.gnn_arrays), "gnn_arrays_sha256": sha256(args.gnn_arrays),
            "canonical_index": str(args.canonical_index),
            "canonical_index_sha256": sha256(args.canonical_index),
            "p2_manifest": str(args.p2_manifest), "p2_manifest_sha256": sha256(args.p2_manifest),
            "p4_manifest": str(args.p4_manifest), "p4_manifest_sha256": sha256(args.p4_manifest),
            "active_assignment": str(args.active_assignment),
            "graph_support": str(args.graph_support),
        },
        "counts": {
            "parent_nodes": n_parent, "context_nodes": int(context.sum()),
            "delaunay_context_pairs": delaunay_context_count,
            "union_pairs": total_edges, "directed_messages_per_full_graph": 2 * total_edges,
            "p4_active_rows": len(active_parent), "p4_cores": n_cores,
        },
        "source_edge_offsets_half_open": source_offsets,
        "node_feature_columns": [
            "Degree", "Clustering", "Density", "Neigh Density",
            "I_eig1", "I_eig2", "I_eig3"
        ],
        "edge_feature_columns": [
            "edge_length", "x_dir", "y_dir", "z_dir", "density_contrast"
        ],
        "p4_support_semantics": "source keys safe_2pass and safe_4pass are literal 2-hop and 4-hop masks; model-pass mapping is architecture-specific",
        "edge_direction_contract": (
            "canonical u->v attributes followed by v->u with direction negated and "
            "density contrast inverted, identical to build_abacus_sbi_cache.py"
        ),
        "artifacts": artifacts,
        "gates": gates,
        "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if not manifest["pass"]:
        raise RuntimeError(f"P5 adapter build failed gates: {gates}")


if __name__ == "__main__":
    main()
