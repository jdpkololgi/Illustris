#!/usr/bin/env python3
"""Compute exact P2b union-graph hop distance to another P4 fold.

For every canonical context node, ``min_hops_to_other_fold`` is the minimum
number of union-graph message-passing steps needed to reach a context node owned
by a different fold.  A value of 255 means no crossing was reached within the
registered maximum K.  This is computed by streaming the global parent Delaunay
and P2b radius-only arrays; no patch graph is reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


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


def edge_chunks(path: Path, rows: int):
    edges = np.load(path, mmap_mode="r")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise RuntimeError(f"invalid edge array {path}: {edges.shape}")
    for start in range(0, len(edges), rows):
        yield np.asarray(edges[start:start + rows], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-index", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"))
    ap.add_argument("--context-assignment", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/context_assignment.npz"))
    ap.add_argument("--active-assignment", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz"))
    ap.add_argument("--p4-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/spatial_manifest.json"))
    ap.add_argument("--p2-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/p2b_union_manifest.json"))
    ap.add_argument("--delaunay", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_edges_combined_idx.npy"))
    ap.add_argument("--radius-ngc", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/ngc_radius_only_pairs.npy"))
    ap.add_argument("--radius-sgc", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/sgc_radius_only_pairs.npy"))
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--edge-chunk", type=int, default=5_000_000)
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest"))
    args = ap.parse_args()
    started = time.time()
    p4 = json.loads(args.p4_manifest.read_text())
    p2 = json.loads(args.p2_manifest.read_text())
    if not p4.get("pass", False):
        raise RuntimeError("passing P4 geometry required")
    if p4["inputs"]["p2_manifest_sha256"] != sha256(args.p2_manifest):
        raise RuntimeError("P4/P2 identity mismatch")
    if args.max_k < 1 or args.max_k > 16:
        raise ValueError("max-k must be between 1 and 16")

    index = np.load(args.canonical_index)
    n_parent = len(index["cap"])
    context_mask = np.asarray(index["context"], dtype=bool)
    context = np.load(args.context_assignment)
    context_parent = np.asarray(context["parent_node_id"], dtype=np.int64)
    parent_fold = np.full(n_parent, -1, dtype=np.int8)
    parent_fold[context_parent] = np.asarray(context["fold"], dtype=np.int8)
    if int(np.sum(parent_fold >= 0)) != int(context_mask.sum()):
        raise RuntimeError("context assignment does not cover canonical context")

    edge_paths = (args.delaunay, args.radius_ngc, args.radius_sgc)
    expected_pairs = int(p2["counts"]["union_pairs_context"])
    hop = np.full(n_parent, 255, dtype=np.uint8)
    context_pairs = 0
    cross_fold_pairs = 0
    cross_cap_pairs = 0
    cap = np.asarray(index["cap"], dtype=np.uint8)

    # One-hop boundary seeds.
    for path in edge_paths:
        for edge in edge_chunks(path, args.edge_chunk):
            u, v = edge[:, 0], edge[:, 1]
            valid = (parent_fold[u] >= 0) & (parent_fold[v] >= 0)
            context_pairs += int(valid.sum())
            cross_cap_pairs += int(np.sum(valid & (cap[u] != cap[v])))
            cross = valid & (parent_fold[u] != parent_fold[v])
            cross_fold_pairs += int(cross.sum())
            hop[u[cross]] = 1
            hop[v[cross]] = 1

    reached_by_k = {"1": int(np.sum((hop == 1) & context_mask))}
    for step in range(2, args.max_k + 1):
        frontier = hop == step - 1
        new = np.zeros(n_parent, dtype=bool)
        for path in edge_paths:
            for edge in edge_chunks(path, args.edge_chunk):
                u, v = edge[:, 0], edge[:, 1]
                same_fold = (parent_fold[u] >= 0) & (parent_fold[u] == parent_fold[v])
                from_u = same_fold & frontier[u] & (hop[v] == 255)
                from_v = same_fold & frontier[v] & (hop[u] == 255)
                new[v[from_u]] = True
                new[u[from_v]] = True
        new &= context_mask & (hop == 255)
        hop[new] = step
        reached_by_k[str(step)] = int(new.sum())

    active = np.load(args.active_assignment)
    active_parent = np.asarray(active["parent_node_id"], dtype=np.int64)
    eligible = np.asarray(active["supervised_eligible"], dtype=bool)
    active_fold = np.asarray(active["fold"], dtype=np.uint8)
    active_cap = np.asarray(active["cap"], dtype=np.uint8)
    active_shell = np.asarray(active["shell"], dtype=np.int8)
    active_hop = hop[active_parent]
    report = {}
    for k in (2, 4):
        if k > args.max_k:
            continue
        safe = active_hop > k
        by_fold = [float(np.mean(safe[eligible & (active_fold == fold)]))
                   for fold in range(5)]
        by_cap_shell = {
            f"cap{cap_id}_shell{shell_id}": float(np.mean(
                safe[eligible & (active_cap == cap_id) & (active_shell == shell_id)]))
            for cap_id in (0, 1) for shell_id in range(4)
        }
        report[str(k)] = {
            "eligible_safe_rows": int(np.sum(safe & eligible)),
            "eligible_safe_fraction": float(np.mean(safe[eligible])),
            "eligible_safe_fraction_by_fold": by_fold,
            "eligible_safe_fraction_by_cap_shell": by_cap_shell,
        }

    context_path = args.out_dir / "graph_support_context.npz"
    active_path = args.out_dir / "graph_support_active.npz"
    atomic_savez(
        context_path, parent_node_id=context_parent,
        min_hops_to_other_fold=hop[context_parent],
        safe_2pass=hop[context_parent] > 2,
        safe_4pass=hop[context_parent] > 4,
    )
    atomic_savez(
        active_path, parent_node_id=active_parent,
        min_hops_to_other_fold=active_hop,
        safe_2pass=active_hop > 2, safe_4pass=active_hop > 4,
        supervised_eligible=eligible,
    )
    gates = {
        "p2_identity_matches_geometry": p4["inputs"]["p2_manifest_sha256"] == sha256(args.p2_manifest),
        "union_context_pair_count_matches": context_pairs == expected_pairs,
        "no_cross_cap_union_edges": cross_cap_pairs == 0,
        "cross_fold_boundary_exists": cross_fold_pairs > 0,
        "hop_values_bounded": bool(np.all((hop[context_mask] <= args.max_k)
                                           | (hop[context_mask] == 255))),
        "all_folds_have_safe_2pass_rows": all(v > 0 for v in report["2"]["eligible_safe_fraction_by_fold"]),
        "all_cap_shell_strata_have_safe_2pass_rows": all(
            v > 0 for v in report["2"]["eligible_safe_fraction_by_cap_shell"].values()),
    }
    manifest = {
        "schema_version": 1, "stage": "P4 exact P2b union-graph split support",
        "p4_geometry_manifest": str(args.p4_manifest),
        "p4_geometry_manifest_sha256": sha256(args.p4_manifest),
        "p2_manifest": str(args.p2_manifest), "p2_manifest_sha256": sha256(args.p2_manifest),
        "edge_inputs": {str(path): {"sha256": sha256(path),
                                    "pairs": int(len(np.load(path, mmap_mode="r")))}
                        for path in edge_paths},
        "counts": {"parent_nodes": n_parent, "context_nodes": int(context_mask.sum()),
                   "union_context_pairs": context_pairs,
                   "cross_fold_pairs": cross_fold_pairs,
                   "cross_cap_pairs": cross_cap_pairs,
                   "new_context_nodes_by_hop": reached_by_k},
        "support": report,
        "artifacts": {"context": str(context_path), "active": str(active_path)},
        "artifact_sha256": {"context": sha256(context_path), "active": sha256(active_path)},
        "interpretation": (
            "safe_Kpass means a K-step union-graph message-passing path cannot reach a node "
            "owned by another fold. Global graph-metric construction remains a separately "
            "declared, label-free representation-level transductive choice."
        ),
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_dir / "graph_support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=bool))
    if not manifest["pass"]:
        raise RuntimeError(f"P4 graph support gates failed: {gates}")
    (args.out_dir / "P4_GRAPH_SUPPORT_COMPLETE").write_text(
        f"manifest_sha256={sha256(manifest_path)}\nmax_k={args.max_k}\n")


if __name__ == "__main__":
    main()
