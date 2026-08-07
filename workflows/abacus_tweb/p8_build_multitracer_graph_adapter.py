#!/usr/bin/env python3
"""Build a P5-compatible patch adapter for a Bright+Faint union graph.

The graph and its metrics are global within the response-aware multitracer
catalogue.  P4 core ownership is not rebuilt: the frozen Bright prefix retains
its exact parent IDs, while appended Faint rows are context-only.  Patch views
therefore gain Faint message-passing context without changing the supervised or
evaluated population.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from workflows.abacus_tweb.p5_build_graph_patch_adapter import (
    add_degrees,
    chunks,
    fill_incident,
    finish_npy,
    open_npy,
    sha256,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--catalogue-index", type=Path,
        default=ROOT / "catalogues/bf_proxy_response_v1/catalogue_index.npz",
    )
    parser.add_argument("--p2-manifest", type=Path, required=True)
    parser.add_argument("--active-assignment", type=Path, default=P4 / "active_assignment.npz")
    parser.add_argument("--cores", type=Path, default=P4 / "cores.npz")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "graph/bf_proxy_response_v1/adapter"
    )
    parser.add_argument("--chunk-rows", type=int, default=5_000_000)
    return parser.parse_args()


def bright_prefix_rows(index: np.lib.npyio.NpzFile) -> int:
    tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
    bright_parent = np.asarray(index["bright_parent_id"], dtype=np.int64)
    faint_at = np.flatnonzero(tracer == 1)
    n_bright = int(faint_at[0]) if len(faint_at) else len(tracer)
    if np.any(tracer[:n_bright] != 0) or np.any(tracer[n_bright:] != 1):
        raise RuntimeError("multitracer catalogue is not Bright-prefix/Faint-suffix")
    if not np.array_equal(bright_parent[:n_bright], np.arange(n_bright)):
        raise RuntimeError("Bright parent identity prefix changed")
    if np.any(bright_parent[n_bright:] != -1):
        raise RuntimeError("Faint context rows carry Bright parent IDs")
    return n_bright


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.graph_dir / f"{args.prefix}_metadata.json"
    gnn_metadata_path = args.graph_dir / f"{args.prefix}_cugraph_gnn_metadata.json"
    graph_metadata = json.loads(metadata_path.read_text())
    gnn_metadata = json.loads(gnn_metadata_path.read_text())
    p2 = json.loads(args.p2_manifest.read_text())
    index = np.load(args.catalogue_index)
    context = np.asarray(index["context"], dtype=bool)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
    n_parent = len(context)
    n_bright = bright_prefix_rows(index)

    gnn_path = Path(gnn_metadata["outputs"]["gnn_arrays_npz"])
    with np.load(gnn_path) as gnn:
        node = np.asarray(gnn["x"], dtype=np.float32)
        edge_index = np.asarray(gnn["edge_index"], dtype=np.int64)
        delaunay_attr = np.asarray(gnn["edge_attr"], dtype=np.float32)
    source_delaunay = np.load(
        args.graph_dir / graph_metadata["files"]["edges"], mmap_mode="r"
    )
    if node.shape != (n_parent, 7):
        raise RuntimeError(f"expected {(n_parent, 7)} node metrics, found {node.shape}")
    if edge_index.shape != (2, len(source_delaunay)):
        raise RuntimeError("Delaunay GNN array shape mismatch")

    radius_sources = []
    for cap_name, cap_id in (("NGC", 1), ("SGC", 0)):
        component = p2["components"][cap_name]
        radius_sources.append(
            (
                cap_name,
                np.load(component["pairs"], mmap_mode="r"),
                np.load(component["edge_attr"], mmap_mode="r"),
                cap_id,
            )
        )

    delaunay_context_count = 0
    for start, stop in chunks(len(source_delaunay), args.chunk_rows):
        pair = np.asarray(source_delaunay[start:stop], dtype=np.int64)
        if not np.array_equal(pair, edge_index[:, start:stop].T):
            raise RuntimeError(f"Delaunay pair/GNN order mismatch at {start}")
        delaunay_context_count += int(np.sum(context[pair[:, 0]] & context[pair[:, 1]]))
    total_edges = delaunay_context_count + sum(len(pair) for _, pair, _, _ in radius_sources)
    if total_edges != int(p2["counts"]["union_pairs_context"]):
        raise RuntimeError("P2 multitracer union edge count mismatch")

    node_path = args.out_dir / "node_features.npy"
    np.save(node_path, node, allow_pickle=False)
    pair_path = args.out_dir / "union_pairs.npy"
    attr_path = args.out_dir / "union_edge_features.npy"
    pair_partial, pair_out = open_npy(pair_path, np.int32, (total_edges, 2))
    attr_partial, attr_out = open_npy(attr_path, np.float32, (total_edges, 5))
    write = 0
    for start, stop in chunks(len(source_delaunay), args.chunk_rows):
        pair = np.asarray(source_delaunay[start:stop], dtype=np.int64)
        valid = context[pair[:, 0]] & context[pair[:, 1]]
        count = int(valid.sum())
        pair_out[write:write + count] = pair[valid].astype(np.int32)
        attr_out[write:write + count] = delaunay_attr[start:stop][valid]
        write += count
    source_offsets = {"delaunay_context": [0, write]}
    for name, pair, attr, cap_id in radius_sources:
        first = write
        if len(pair) != len(attr):
            raise RuntimeError(f"{name} radius pair/attribute mismatch")
        for start, stop in chunks(len(pair), args.chunk_rows):
            block = np.asarray(pair[start:stop], dtype=np.int64)
            if (
                np.any(~context[block[:, 0]])
                or np.any(~context[block[:, 1]])
                or np.any(cap[block[:, 0]] != cap_id)
                or np.any(cap[block[:, 1]] != cap_id)
            ):
                raise RuntimeError(f"{name} radius block violates context/cap contract")
            rows = len(block)
            pair_out[write:write + rows] = block.astype(np.int32)
            attr_out[write:write + rows] = np.asarray(attr[start:stop], dtype=np.float32)
            write += rows
        source_offsets[f"radius_{name.lower()}"] = [first, write]
    if write != total_edges:
        raise RuntimeError("union edge write count mismatch")
    finish_npy(pair_partial, pair_out, pair_path)
    finish_npy(attr_partial, attr_out, attr_path)
    del node, edge_index, delaunay_attr

    union_pairs = np.load(pair_path, mmap_mode="r")
    degree = np.zeros(n_parent, dtype=np.int64)
    for start, stop in chunks(total_edges, args.chunk_rows):
        add_degrees(np.asarray(union_pairs[start:stop], dtype=np.int32), degree)
    offsets = np.empty(n_parent + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    if int(offsets[-1]) != 2 * total_edges:
        raise RuntimeError("incident degree sum mismatch")
    offsets_path = args.out_dir / "incident_offsets.npy"
    np.save(offsets_path, offsets, allow_pickle=False)
    incident_path = args.out_dir / "incident_edge_id.npy"
    incident_partial, incident = open_npy(incident_path, np.int32, (2 * total_edges,))
    cursor = offsets[:-1].copy()
    for start, stop in chunks(total_edges, args.chunk_rows):
        fill_incident(
            np.asarray(union_pairs[start:stop], dtype=np.int32), start, cursor, incident
        )
    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("incident CSR cursor mismatch")
    finish_npy(incident_partial, incident, incident_path)

    active = np.load(args.active_assignment)
    cores = np.load(args.cores)
    active_parent = np.asarray(active["parent_node_id"], dtype=np.int64)
    if np.any(active_parent >= n_bright):
        raise RuntimeError("P4 authoritative IDs leave the frozen Bright prefix")
    core_id = np.asarray(active["core_id"], dtype=np.int32)
    order = np.argsort(core_id, kind="stable")
    n_cores = len(cores["core_id"])
    core_counts = np.bincount(core_id[order], minlength=n_cores)
    core_offsets = np.empty(n_cores + 1, dtype=np.int64)
    core_offsets[0] = 0
    np.cumsum(core_counts, out=core_offsets[1:])
    compact = {
        "core_active_offsets.npy": core_offsets,
        "core_active_parent.npy": active_parent[order].astype(np.int32),
        "core_active_eligible.npy": np.asarray(active["supervised_eligible"], dtype=bool)[order],
        # Bright-only strict-hop masks do not apply after adding Faint edges.
        "core_active_safe2hop.npy": np.zeros(len(active_parent), dtype=bool),
        "core_active_safe4hop.npy": np.zeros(len(active_parent), dtype=bool),
        "core_fold.npy": np.asarray(cores["fold"], dtype=np.uint8),
        "core_cap.npy": np.asarray(cores["cap"], dtype=np.uint8),
        "tracer_type.npy": tracer,
        "node_cap.npy": cap,
        "node_context.npy": context,
    }
    for name, values in compact.items():
        np.save(args.out_dir / name, values, allow_pickle=False)

    artifact_names = [
        "node_features.npy", "union_pairs.npy", "union_edge_features.npy",
        "incident_offsets.npy", "incident_edge_id.npy", *compact,
    ]
    artifacts = {
        name: {"path": str(args.out_dir / name), "sha256": sha256(args.out_dir / name)}
        for name in artifact_names
    }
    gates = {
        "bright_prefix_identity": True,
        "faint_context_never_supervised": bool(np.all(active_parent < n_bright)),
        "parent_node_count_matches": len(np.load(node_path, mmap_mode="r")) == n_parent,
        "union_pair_count_matches": len(np.load(pair_path, mmap_mode="r")) == total_edges,
        "incident_count_twice_edges": len(np.load(incident_path, mmap_mode="r")) == 2 * total_edges,
        "both_tracers_present": set(np.unique(tracer)) == {0, 1},
        "cross_cap_edges_absent": bool(
            all(
                np.all(cap[np.asarray(union_pairs[start:stop])[:, 0]]
                       == cap[np.asarray(union_pairs[start:stop])[:, 1]])
                for start, stop in chunks(total_edges, args.chunk_rows)
            )
        ),
        "graph_metrics_global_not_patchwise": True,
    }
    manifest = {
        "schema_version": "p8-multitracer-graph-adapter-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "global Bright+Faint graph metrics and exact masked patch views",
        "counts": {
            "parent_nodes": n_parent,
            "bright_nodes": n_bright,
            "faint_nodes": int(n_parent - n_bright),
            "context_nodes": int(context.sum()),
            "delaunay_context_pairs": delaunay_context_count,
            "union_pairs": total_edges,
            "p4_authoritative_bright_rows": int(len(active_parent)),
            "p4_cores": n_cores,
        },
        "supervision_contract": {
            "loss_and_evaluation": "frozen authoritative BGS_BRIGHT core rows only",
            "BGS_FAINT": "message-passing context only",
            "faint_predictions_released": False,
            "strict_hop_masks": "disabled; old Bright-only graph masks are not reused",
        },
        "source_edge_offsets_half_open": source_offsets,
        "node_feature_columns": [
            "Degree", "Clustering", "Density", "Neigh Density",
            "I_eig1", "I_eig2", "I_eig3",
        ],
        "edge_feature_columns": [
            "edge_length", "x_dir", "y_dir", "z_dir", "density_contrast",
        ],
        "inputs": {
            "graph_metadata": str(metadata_path),
            "graph_metadata_sha256": sha256(metadata_path),
            "gnn_metadata": str(gnn_metadata_path),
            "gnn_metadata_sha256": sha256(gnn_metadata_path),
            "catalogue_index": str(args.catalogue_index),
            "catalogue_index_sha256": sha256(args.catalogue_index),
            "p2_manifest": str(args.p2_manifest),
            "p2_manifest_sha256": sha256(args.p2_manifest),
            "active_assignment": str(args.active_assignment),
            "active_assignment_sha256": sha256(args.active_assignment),
        },
        "artifacts": artifacts,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not manifest["pass"]:
        raise RuntimeError(f"multitracer graph adapter gates failed: {gates}")
    (args.out_dir / "MULTITRACER_GRAPH_PATCH_READY").write_text(
        f"manifest_sha256={sha256(manifest_path)}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
