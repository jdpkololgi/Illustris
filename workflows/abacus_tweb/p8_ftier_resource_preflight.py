#!/usr/bin/env python3
"""Test whether frozen v2_A F-PATCH can enter the P8 full-cap screen.

The P7 FFT convergence gate requires a 72-voxel field halo.  A faithful F-tier
patch must graph-encode every galaxy scattered into that field context (plus its
encoder dependency context), not merely the authoritative output-core nodes.
This preflight counts the irreducible nodes/edges and records a conservative
forward-activation lower bound before spending a GPU allocation on an OOM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_field_patch_utils import CAP_NAME, CanonicalFieldPatchAdapter
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P5_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter")
P6_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
SELECTION = P6_ROOT / "fullcap_selection_v1/selection_manifest.json"
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)


def representative_cores(assignment, core_count: int) -> list[dict]:
    auth = np.asarray(assignment["supervised_eligible"], dtype=bool)
    result = []
    for cap_id in (0, 1):
        for shell_id in range(4):
            selected = auth & (assignment["cap"] == cap_id) & (assignment["shell"] == shell_id)
            core, count = np.unique(assignment["core_id"][selected], return_counts=True)
            if not len(core):
                raise RuntimeError(f"no representative core for cap={cap_id} shell={shell_id}")
            order = np.argsort(count, kind="stable")
            chosen = int(core[order[len(order) // 2]])
            result.append({
                "core_id": chosen,
                "cap": cap_id,
                "shell": shell_id,
                "authoritative_nodes": int(count[core == chosen][0]),
            })
    return result[:core_count] if core_count < len(result) else result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--halo-voxels", type=int, default=72)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--edge-features", type=int, default=5)
    parser.add_argument("--max-forward-activation-gib", type=float, default=60.0)
    parser.add_argument("--chunk", type=int, default=5_000_000)
    parser.add_argument("--representatives", type=int, default=8)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p5-root", type=Path, default=P5_ROOT)
    parser.add_argument("--p6-root", type=Path, default=P6_ROOT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--points", type=Path, default=POINTS)
    args = parser.parse_args()
    assignment = np.load(args.assignment, mmap_mode="r")
    points = np.load(args.points, mmap_mode="r")
    records = representative_cores(assignment, args.representatives)
    membership = np.zeros((len(records), len(points)), dtype=bool)
    with CanonicalFieldPatchAdapter(
        args.p6_root, selection_manifest=args.selection, rotation=args.rotation
    ) as field:
        for index, record in enumerate(records):
            patch = field.extract(
                record["core_id"],
                args.halo_voxels,
                ("counts", "exposure_apodized", "log_count_ratio"),
                alignment_voxels=8,
            )
            cap_spec = field.manifest["caps"][CAP_NAME[patch.cap]]
            origin = np.asarray(cap_spec["origin_mpc"], dtype=np.float64)
            cell = float(cap_spec["cell_mpc"])
            lower = origin + patch.context_start * cell
            upper = origin + patch.context_stop * cell
            membership[index] = (
                (np.asarray(points[:, 3], dtype=np.int8) == patch.cap)
                & np.all(np.asarray(points[:, :3]) >= lower, axis=1)
                & np.all(np.asarray(points[:, :3]) < upper, axis=1)
            )
            record.update({
                "field_shape": [int(v) for v in patch.values.shape[1:]],
                "field_voxels": int(np.prod(patch.values.shape[1:])),
                "galaxies_in_required_field_context": int(membership[index].sum()),
                "context_lower_mpc": lower.tolist(),
                "context_upper_mpc": upper.tolist(),
            })
    pairs = np.load(args.p5_root / "union_pairs.npy", mmap_mode="r")
    edge_count = np.zeros(len(records), dtype=np.int64)
    for start in range(0, len(pairs), args.chunk):
        stop = min(start + args.chunk, len(pairs))
        block = np.asarray(pairs[start:stop], dtype=np.int64)
        for index in range(len(records)):
            edge_count[index] += np.sum(
                membership[index, block[:, 0]] & membership[index, block[:, 1]]
            )
    for index, record in enumerate(records):
        directed = 2 * int(edge_count[index])
        # Minimal simultaneous forward tensors for one attention layer:
        # embedded edge W, pair concat (2W+F), message W, logits H, weighted W.
        features_per_directed_edge = (
            args.width + (2 * args.width + args.edge_features)
            + args.width + args.heads + args.width
        )
        edge_bytes = directed * features_per_directed_edge * 4
        node_bytes = record["galaxies_in_required_field_context"] * args.width * 4
        field_bytes = record["field_voxels"] * (args.width + 2) * 4
        lower_bound = edge_bytes + node_bytes + field_bytes
        record.update({
            "induced_undirected_edges_before_encoder_halo": int(edge_count[index]),
            "directed_edges_before_encoder_halo": directed,
            "minimum_forward_activation_bytes": int(lower_bound),
            "minimum_forward_activation_gib": float(lower_bound / 2**30),
            "lower_bound_omits": [
                "five-hop graph context beyond the field box",
                "autograd saved tensors and gradients",
                "optimizer state",
                "U-Net decoder activations",
                "FFT padding and tensor fields",
            ],
        })
    threshold = args.max_forward_activation_gib
    feasible = all(row["minimum_forward_activation_gib"] <= threshold for row in records)
    payload = {
        "schema_version": 1,
        "stage": "P8 F-PATCH v2_A full-cap resource preflight",
        "rotation": args.rotation,
        "frozen_architecture": {
            "encoder": "five-layer EGNNAttnEncoder",
            "width": args.width,
            "heads": args.heads,
            "field_halo_voxels": args.halo_voxels,
            "fft_padding_voxels": 20,
            "fft_apodization_voxels": 20,
        },
        "records": records,
        "registered_forward_activation_limit_gib": threshold,
        "screen_feasible_without_redesign": feasible,
        "decision": (
            "ENTER_ONE_SEED_SCREEN" if feasible else "NO_GO_FROZEN_V2_A_RESOURCE_INFEASIBLE"
        ),
        "scientific_note": (
            "This is a deployment-feasibility no-go, not evidence that the field-to-physics "
            "factorisation lacks predictive value. A redesigned U-Physics or multiresolution "
            "graph-field model is a separately named future candidate."
        ),
        "inputs": {
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "p5_adapter": str(args.p5_root / "adapter_manifest.json"),
            "p5_adapter_sha256": sha256(args.p5_root / "adapter_manifest.json"),
            "p6_adapter": str(args.p6_root / "adapter_manifest.json"),
            "p6_adapter_sha256": sha256(args.p6_root / "adapter_manifest.json"),
            "selection": str(args.selection),
            "selection_sha256": sha256(args.selection),
        },
    }
    output = args.p8_root / "f_patch" / "resource_preflight.json"
    atomic_json(output, payload)
    marker = output.parent / (
        "F_PATCH_PREFLIGHT_READY" if feasible else "F_PATCH_PREFLIGHT_NO_GO"
    )
    marker.write_text(f"decision={payload['decision']}\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
