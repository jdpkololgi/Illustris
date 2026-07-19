#!/usr/bin/env python3
"""Validate P5 exact-context, padding, subdivision, and P1a model parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from p5_build_graph_patch_adapter import add_degrees, fill_incident
from p5_graph_patch_utils import (
    CanonicalGraphPatchAdapter,
    assemble_patch,
    core_prediction_map,
    make_bidirectional,
    pad_patch,
)


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def incident_csr(pairs: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    degree = np.zeros(n_nodes, dtype=np.int64)
    add_degrees(np.asarray(pairs, dtype=np.int32), degree)
    offsets = np.empty(n_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    incident = np.empty(2 * len(pairs), dtype=np.int32)
    cursor = offsets[:-1].copy()
    fill_incident(np.asarray(pairs, dtype=np.int32), 0, cursor, incident)
    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("temporary CSR fill failed")
    return offsets, incident


def standardize_for_parity(x: np.ndarray, edge_attr: np.ndarray):
    x = np.asarray(x, dtype=np.float32).copy()
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    x = ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    return x, np.asarray(edge_attr, dtype=np.float32)


def jraph_embeddings(
    x: np.ndarray,
    pairs: np.ndarray,
    edge_attr: np.ndarray,
    num_passes: int,
    params=None,
    edge_stats=None,
):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import jraph

    from shared.graph_net_models import make_gnn_encoder

    senders, receivers, attrs = make_bidirectional(pairs, edge_attr)
    attrs = attrs.copy()
    logged = {col: np.log(np.maximum(attrs[:, col], 1e-6)) for col in (0, 4)}
    if edge_stats is None:
        edge_stats = {
            str(col): [float(logged[col].mean(dtype=np.float64)),
                       max(float(logged[col].std(dtype=np.float64)), 1e-6)]
            for col in (0, 4)
        }
    for col in (0, 4):
        mean, std = edge_stats[str(col)]
        attrs[:, col] = ((logged[col] - mean) / std).astype(np.float32)
    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(x), edges=jnp.asarray(attrs),
        senders=jnp.asarray(senders), receivers=jnp.asarray(receivers),
        n_node=jnp.asarray([len(x)], dtype=jnp.int32),
        n_edge=jnp.asarray([len(senders)], dtype=jnp.int32), globals=None,
    )
    transformed = hk.transform(
        lambda value: make_gnn_encoder(
            num_passes=num_passes, latent_size=8, num_heads=2, dropout_rate=0.0
        )(value, is_training=False)
    )
    if params is None:
        params = transformed.init(jax.random.PRNGKey(271828), graph)
    apply = jax.jit(lambda value: transformed.apply(params, None, value))
    embedding = np.asarray(apply(graph))
    # Ensure asynchronous device failures surface before writing a PASS report.
    jax.block_until_ready(embedding)
    return embedding, params, str(jax.default_backend()), str(jax.devices()[0]), edge_stats


def p1a_parity(args: argparse.Namespace) -> dict:
    with np.load(args.p1a_arrays) as data:
        raw_x = np.asarray(data["x"], dtype=np.float32)
        pairs = np.asarray(data["edge_index"].T, dtype=np.int32)
        raw_attr = np.asarray(data["edge_attr"], dtype=np.float32)
    x, edge_attr = standardize_for_parity(raw_x, raw_attr)
    offsets, incident = incident_csr(pairs, len(x))
    points = np.load(args.p1a_points, mmap_mode="r")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    if len(xyz) != len(x):
        raise RuntimeError("P1a point/GNN rows do not align")
    centre = np.median(xyz, axis=0)
    core_half_width_mpc = 32.0
    offset = np.abs(xyz - centre)
    candidate = np.flatnonzero(np.all(offset < core_half_width_mpc, axis=1))
    if len(candidate) < args.p1a_core_nodes:
        candidate = np.argsort(np.linalg.norm(xyz - centre, axis=1), kind="stable")
    core_boundary_all = core_half_width_mpc - np.max(offset, axis=1)
    ranked = candidate[np.argsort(core_boundary_all[candidate], kind="stable")]
    take = np.linspace(0, len(ranked) - 1, args.p1a_core_nodes, dtype=int)
    core = np.sort(ranked[take]).astype(np.int64)

    full_embedding, params, backend, device, edge_stats = jraph_embeddings(
        x, pairs, edge_attr, args.num_passes
    )
    full_prediction = full_embedding @ np.asarray(
        [[0.7, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.2, 0.2, 0.6],
         [0.3, -0.5, 0.2], [0.2, 0.1, -0.4], [-0.1, 0.3, 0.5],
         [0.4, 0.2, 0.1], [-0.3, 0.1, 0.2]], dtype=np.float32
    )
    patch = assemble_patch(
        core_id=-1, fold=-1, core_parent_ids=core, loss_parent_ids=core,
        num_passes=args.num_passes, dependency_hops=2 * args.num_passes,
        node_features=x, union_pairs=pairs,
        union_edge_features=edge_attr, offsets=offsets, incident_edge_id=incident,
    )
    patch_pairs = np.column_stack([
        patch.senders[:patch.n_edge // 2], patch.receivers[:patch.n_edge // 2]
    ])
    patch_attr = patch.edge_features[:patch.n_edge // 2]
    patch_embedding, _, _, _, _ = jraph_embeddings(
        patch.node_features, patch_pairs, patch_attr, args.num_passes,
        params=params, edge_stats=edge_stats
    )
    patch_prediction = patch_embedding @ np.asarray(
        [[0.7, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.2, 0.2, 0.6],
         [0.3, -0.5, 0.2], [0.2, 0.1, -0.4], [-0.1, 0.3, 0.5],
         [0.4, 0.2, 0.1], [-0.3, 0.1, 0.2]], dtype=np.float32
    )
    local = np.searchsorted(patch.parent_node_id, core)
    emb_abs = np.abs(full_embedding[core] - patch_embedding[local])
    pred_abs = np.abs(full_prediction[core] - patch_prediction[local])
    boundary_proxy = core_boundary_all[core]
    per_node = np.max(emb_abs, axis=1)
    slope = float(np.polyfit(boundary_proxy, per_node, 1)[0]) if len(core) > 2 else 0.0

    # Exact core subdivision.  Every half receives its own complete K-hop
    # context and shares the unchanged full-graph parameters.
    halves = [np.sort(core)[::2], np.sort(core)[1::2]]
    subdivision = []
    for half in halves:
        sub = assemble_patch(
            core_id=-1, fold=-1, core_parent_ids=half, loss_parent_ids=half,
            num_passes=args.num_passes, dependency_hops=2 * args.num_passes,
        node_features=x, union_pairs=pairs,
            union_edge_features=edge_attr, offsets=offsets, incident_edge_id=incident,
        )
        sub_pairs = np.column_stack([
            sub.senders[:sub.n_edge // 2], sub.receivers[:sub.n_edge // 2]
        ])
        sub_emb, _, _, _, _ = jraph_embeddings(
            sub.node_features, sub_pairs, sub.edge_features[:sub.n_edge // 2],
            args.num_passes, params=params, edge_stats=edge_stats
        )
        subdivision.append((sub, sub_emb))
    forward = core_prediction_map([v[0] for v in subdivision], [v[1] for v in subdivision])
    reverse = core_prediction_map(
        [v[0] for v in subdivision[::-1]], [v[1] for v in subdivision[::-1]]
    )
    subdivision_abs = []
    order_abs = []
    for parent in core:
        subdivision_abs.append(np.max(np.abs(forward[int(parent)] - full_embedding[parent])))
        order_abs.append(np.max(np.abs(forward[int(parent)] - reverse[int(parent)])))

    gates = {
        "embedding_max_abs_below_5e-5": float(emb_abs.max()) < 5e-5,
        "prediction_max_abs_below_5e-5": float(pred_abs.max()) < 5e-5,
        "subdivision_max_abs_below_5e-5": float(max(subdivision_abs)) < 5e-5,
        "patch_order_exact": float(max(order_abs)) == 0.0,
        "no_boundary_error_trend": abs(slope) < 1e-6,
    }
    return {
        "source": "P1a RA120-160 canonical union wedge",
        "backend": backend, "device": device,
        "num_passes": args.num_passes, "dependency_hops": 2 * args.num_passes,
        "dependency_reason": "Jraph GraphNetwork aggregates sent edges whose attention is receiver-normalized",
        "latent_size": 8,
        "full_nodes": len(x), "full_undirected_pairs": len(pairs),
        "core_nodes": len(core), "core_half_width_mpc": core_half_width_mpc,
        "core_boundary_distance_min_mpc": float(boundary_proxy.min()),
        "core_boundary_distance_max_mpc": float(boundary_proxy.max()),
        "patch_nodes": patch.n_node,
        "patch_directed_edges": patch.n_edge,
        "embedding_max_abs": float(emb_abs.max()),
        "embedding_mean_abs": float(emb_abs.mean()),
        "prediction_max_abs": float(pred_abs.max()),
        "prediction_mean_abs": float(pred_abs.mean()),
        "subdivision_max_abs": float(max(subdivision_abs)),
        "patch_order_max_abs": float(max(order_abs)),
        "boundary_proxy_error_slope": slope,
        "gates": gates, "pass": all(gates.values()),
    }


def full_adapter_smoke(args: argparse.Namespace) -> dict:
    adapter = CanonicalGraphPatchAdapter(args.adapter_root)
    manifest = json.loads((args.adapter_root / "adapter_manifest.json").read_text())
    if not manifest.get("pass"):
        raise RuntimeError("passing adapter build manifest required")
    safe_count = np.zeros(len(adapter.core_offsets) - 1, dtype=np.int64)
    for core_id in range(len(safe_count)):
        start = int(adapter.core_offsets[core_id])
        stop = int(adapter.core_offsets[core_id + 1])
        eligible = np.asarray(adapter.core_eligible[start:stop], dtype=bool)
        safe = np.asarray(adapter.core_safe4hop[start:stop], dtype=bool)
        safe_count[core_id] = int(np.sum(eligible & safe))
    safe_core = np.flatnonzero(safe_count > 0)
    chosen = []
    for cap in (0, 1):
        for fold in range(5):
            candidates = safe_core[
                (adapter.core_cap[safe_core] == cap)
                & (adapter.core_fold[safe_core] == fold)
            ]
            if not len(candidates):
                raise RuntimeError(f"no strict four-hop core for cap={cap} fold={fold}")
            chosen.append(int(candidates[np.argmax(safe_count[candidates])]))
    if args.smoke_cores > len(chosen):
        remaining = np.setdiff1d(safe_core, np.asarray(chosen, dtype=np.int64))
        positions = np.linspace(0, len(remaining) - 1, args.smoke_cores - len(chosen), dtype=int)
        chosen.extend(remaining[positions].tolist())
    sample = np.asarray(chosen, dtype=np.int64)
    records = []
    largest = None
    for core_id in sample:
        patch = adapter.extract(int(core_id), 2)
        padded = pad_patch(patch)
        original = patch.n_edge // 2
        edge_id = patch.union_edge_id[:original]
        feature_identity = bool(
            np.array_equal(
                patch.node_features,
                np.asarray(adapter.node_features[patch.parent_node_id], dtype=np.float32),
            )
            and np.array_equal(
                patch.edge_features[:original],
                np.asarray(adapter.union_edge_features[edge_id], dtype=np.float32),
            )
        )
        masks_valid = bool(
            int(padded["node_mask"].sum()) == patch.n_node
            and int(padded["edge_mask"].sum()) == patch.n_edge
            and int(padded["authoritative_core_mask"].sum())
            == int(patch.authoritative_core_mask.sum())
            and int(padded["loss_mask"].sum()) == int(patch.loss_mask.sum())
        )
        records.append({
            "core_id": int(core_id), "fold": patch.fold,
            "cap": int(adapter.core_cap[core_id]),
            "nodes": patch.n_node, "directed_edges": patch.n_edge,
            "model_passes": patch.num_passes, "dependency_hops": patch.dependency_hops,
            "authoritative_core_nodes": int(patch.authoritative_core_mask.sum()),
            "strict_loss_nodes": int(patch.loss_mask.sum()),
            "bucket_nodes": int(len(padded["nodes"])),
            "bucket_edges": int(len(padded["edges"])),
            "canonical_feature_identity": feature_identity,
            "padding_masks_valid": masks_valid,
        })
        if largest is None or patch.n_edge > largest.n_edge:
            largest = patch

    assert largest is not None
    split = adapter.subdivide_exact(
        largest.core_id, 2,
        max_nodes=max(1, largest.n_node // 2),
        max_edges=max(1, largest.n_edge // 2),
    )
    covered = np.concatenate([
        part.parent_node_id[part.authoritative_core_mask] for part in split
    ])
    original_core = largest.parent_node_id[largest.authoritative_core_mask]
    subdivision_unique = (
        len(covered) == len(np.unique(covered))
        and np.array_equal(np.sort(covered), np.sort(original_core))
    )
    gates = {
        "all_sample_features_are_canonical": all(
            row["canonical_feature_identity"] for row in records),
        "all_padding_masks_are_exact": all(row["padding_masks_valid"] for row in records),
        "every_sample_has_core_nodes": all(row["authoritative_core_nodes"] > 0 for row in records),
        "every_sample_has_strict_loss_nodes": all(row["strict_loss_nodes"] > 0 for row in records),
        "both_caps_all_folds_represented": {
            (row["cap"], row["fold"]) for row in records
        } == {(cap, fold) for cap in (0, 1) for fold in range(5)},
        "subdivision_preserves_unique_core_ownership": bool(subdivision_unique),
        "subdivision_never_truncates": all(part.n_node > 0 and part.n_edge > 0 for part in split),
    }
    return {
        "sample_policy": "maximum strict-four-hop core in every cap-fold stratum plus deterministic fill",
        "sample_count": len(records), "records": records,
        "largest_sample_core": largest.core_id,
        "largest_sample_nodes": largest.n_node,
        "largest_sample_directed_edges": largest.n_edge,
        "subpatches": len(split),
        "gates": gates, "pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter"))
    ap.add_argument("--p1a-arrays", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
        "path1_wedge_union_r10hmpc_gnn_arrays.npz"))
    ap.add_argument("--p1a-points", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
        "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points.npy"))
    ap.add_argument("--num-passes", type=int, default=2)
    ap.add_argument("--p1a-core-nodes", type=int, default=16)
    ap.add_argument("--smoke-cores", type=int, default=12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    adapter_manifest = args.adapter_root / "adapter_manifest.json"
    p1a = p1a_parity(args)
    smoke = full_adapter_smoke(args)
    gates = {
        "adapter_manifest_passes": bool(json.loads(adapter_manifest.read_text()).get("pass")),
        "p1a_full_graph_patch_parity": bool(p1a["pass"]),
        "full_adapter_smoke_passes": bool(smoke["pass"]),
    }
    report = {
        "schema_version": 1,
        "stage": "P5 GraphNet patch adapter parity",
        "adapter_manifest": str(adapter_manifest),
        "adapter_manifest_sha256": sha256(adapter_manifest),
        "p1a": p1a, "full_p2b_p4_smoke": smoke,
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.adapter_root / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass"]:
        raise RuntimeError(f"P5 parity failed: {gates}")
    marker = args.adapter_root / "GRAPH_PATCH_READY"
    marker.write_text(
        f"adapter_manifest_sha256={sha256(adapter_manifest)}\n"
        f"parity_report_sha256={sha256(report_path)}\n"
        f"model_passes={args.num_passes}\n"
        f"dependency_hops={2 * args.num_passes}\n"
    )


if __name__ == "__main__":
    main()
