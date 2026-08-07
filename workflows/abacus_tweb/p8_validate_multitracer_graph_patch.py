#!/usr/bin/env python3
"""Validate the multitracer graph adapter under the corrected P5 contract.

The generic P5 validator requires non-empty strict-hop masks in every cap/fold.
Those masks are diagnostic-only and intentionally disabled for the multitracer
adapter: using them for ownership would remove nearly the entire sparse shell.
This validator retains the parts that matter for deployment:

* production-shape full-graph/patch model parity on the canonical P1a graph;
* exact identity with globally computed multitracer node/edge products;
* authoritative loss ownership restricted to the frozen Bright prefix;
* Faint nodes present only as message-passing context; and
* exact, unique Bright-core ownership after recursive subdivision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p5_graph_patch_utils import (
    CanonicalGraphPatchAdapter,
    pad_patch,
)
from workflows.abacus_tweb.p5_validate_graph_patch_adapter import p1a_parity, sha256


DEFAULT_ADAPTER = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/graph/"
    "bf_proxy_response_v1_photsys_marginal/adapter"
)
DEFAULT_P1A_ARRAYS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
    "path1_wedge_union_r10hmpc_gnn_arrays.npz"
)
DEFAULT_P1A_POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
    "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points.npy"
)


def sample_cores(adapter: CanonicalGraphPatchAdapter) -> np.ndarray:
    """Select the largest non-empty core in every cap/fold stratum."""
    counts = np.diff(np.asarray(adapter.core_offsets, dtype=np.int64))
    chosen = []
    for cap in (0, 1):
        for fold in range(5):
            candidate = np.flatnonzero(
                (np.asarray(adapter.core_cap) == cap)
                & (np.asarray(adapter.core_fold) == fold)
                & (counts > 0)
            )
            if not len(candidate):
                raise RuntimeError(f"no non-empty authoritative core for cap={cap} fold={fold}")
            chosen.append(int(candidate[np.argmax(counts[candidate])]))
    return np.asarray(chosen, dtype=np.int64)


def adapter_smoke(root: Path, num_passes: int) -> dict:
    adapter = CanonicalGraphPatchAdapter(root)
    manifest = json.loads((root / "adapter_manifest.json").read_text())
    if not manifest.get("pass"):
        raise RuntimeError("passing multitracer adapter manifest required")
    tracer = np.load(root / "tracer_type.npy", mmap_mode="r")
    bright_rows = int(manifest["counts"]["bright_nodes"])
    records = []
    largest = None
    for core_id in sample_cores(adapter):
        patch = adapter.extract(
            int(core_id), num_passes, dependency_hops_per_pass=2,
            loss_policy="authoritative",
        )
        padded = pad_patch(patch)
        original_edges = patch.n_edge // 2
        edge_id = patch.union_edge_id[:original_edges]
        loss_parent = patch.parent_node_id[patch.loss_mask]
        patch_tracer = np.asarray(tracer[patch.parent_node_id], dtype=np.uint8)
        record = {
            "core_id": int(core_id), "cap": int(adapter.core_cap[core_id]),
            "fold": int(adapter.core_fold[core_id]), "nodes": patch.n_node,
            "directed_edges": patch.n_edge,
            "authoritative_bright_nodes": int(patch.loss_mask.sum()),
            "bright_context_nodes": int(np.count_nonzero(patch_tracer == 0)),
            "faint_context_nodes": int(np.count_nonzero(patch_tracer == 1)),
            "canonical_node_identity": bool(np.array_equal(
                patch.node_features,
                np.asarray(adapter.node_features[patch.parent_node_id], dtype=np.float32),
            )),
            "canonical_edge_identity": bool(np.array_equal(
                patch.edge_features[:original_edges],
                np.asarray(adapter.union_edge_features[edge_id], dtype=np.float32),
            )),
            "loss_is_bright_prefix": bool(np.all(loss_parent < bright_rows)),
            "loss_tracer_is_bright": bool(np.all(np.asarray(tracer[loss_parent]) == 0)),
            "padding_masks_exact": bool(
                int(padded["node_mask"].sum()) == patch.n_node
                and int(padded["edge_mask"].sum()) == patch.n_edge
                and int(padded["loss_mask"].sum()) == int(patch.loss_mask.sum())
            ),
        }
        records.append(record)
        if largest is None or patch.n_edge > largest.n_edge:
            largest = patch
    assert largest is not None
    parts = adapter.subdivide_exact(
        largest.core_id, num_passes,
        max_nodes=max(1, largest.n_node // 2),
        max_edges=max(1, largest.n_edge // 2),
        dependency_hops_per_pass=2, loss_policy="authoritative",
    )
    covered = np.concatenate(
        [part.parent_node_id[part.authoritative_core_mask] for part in parts]
    )
    original = largest.parent_node_id[largest.authoritative_core_mask]
    subdivision_exact = bool(
        len(covered) == len(np.unique(covered))
        and np.array_equal(np.sort(covered), np.sort(original))
    )
    gates = {
        "all_node_features_are_global": all(row["canonical_node_identity"] for row in records),
        "all_edge_features_are_global": all(row["canonical_edge_identity"] for row in records),
        "all_loss_nodes_are_bright": all(
            row["loss_is_bright_prefix"] and row["loss_tracer_is_bright"] for row in records
        ),
        "all_samples_contain_faint_context": all(row["faint_context_nodes"] > 0 for row in records),
        "all_padding_masks_are_exact": all(row["padding_masks_exact"] for row in records),
        "both_caps_all_folds_represented": {
            (row["cap"], row["fold"]) for row in records
        } == {(cap, fold) for cap in (0, 1) for fold in range(5)},
        "subdivision_preserves_unique_bright_ownership": subdivision_exact,
        "strict_hop_not_used_for_primary_loss": manifest["supervision_contract"]["strict_hop_masks"].startswith("disabled"),
    }
    return {
        "sample_policy": "largest non-empty authoritative core in every cap-fold stratum",
        "records": records, "largest_sample_core": int(largest.core_id),
        "largest_sample_nodes": largest.n_node,
        "largest_sample_directed_edges": largest.n_edge,
        "subpatches": len(parts), "gates": gates, "pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--p1a-arrays", type=Path, default=DEFAULT_P1A_ARRAYS)
    parser.add_argument("--p1a-points", type=Path, default=DEFAULT_P1A_POINTS)
    parser.add_argument("--num-passes", type=int, default=2)
    parser.add_argument("--p1a-core-nodes", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--parity-atol", type=float, default=5.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    parity = p1a_parity(args)
    smoke = adapter_smoke(args.adapter_root, args.num_passes)
    adapter_manifest = args.adapter_root / "adapter_manifest.json"
    gates = {
        "adapter_manifest_passes": bool(json.loads(adapter_manifest.read_text()).get("pass")),
        "production_shape_full_graph_patch_parity": bool(parity["pass"]),
        "multitracer_adapter_smoke_passes": bool(smoke["pass"]),
    }
    report = {
        "schema_version": "p8-multitracer-graph-parity-v1",
        "stage": "P8 multitracer GraphNet patch parity",
        "adapter_manifest": str(adapter_manifest),
        "adapter_manifest_sha256": sha256(adapter_manifest),
        "p1a_production_shape_parity": parity,
        "multitracer_adapter": smoke,
        "mask_contract": {
            "primary_loss": "all authoritative Bright core nodes",
            "strict_hop": "disabled diagnostic; never a loss or primary-score gate",
        },
        "gates": gates, "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.adapter_root / "multitracer_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    marker = args.adapter_root / "MULTITRACER_GRAPH_PATCH_PARITY_READY"
    if not report["pass"]:
        if marker.exists():
            marker.unlink()
        raise RuntimeError(f"multitracer graph parity failed: {gates}")
    marker.write_text(
        f"adapter_manifest_sha256={sha256(adapter_manifest)}\n"
        f"parity_report_sha256={sha256(report_path)}\n"
        f"model_passes={args.num_passes}\n"
        f"dependency_hops={2 * args.num_passes}\n"
        f"latent_size={args.latent_size}\nnum_heads={args.num_heads}\n"
    )


if __name__ == "__main__":
    main()
