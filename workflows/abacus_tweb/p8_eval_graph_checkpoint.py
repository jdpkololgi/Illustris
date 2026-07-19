#!/usr/bin/env python3
"""Resume exact complete-fold G-PATCH evaluation from a pre-evaluation checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p5_graph_patch_utils import CanonicalGraphPatchAdapter
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    unscale_increments,
)
from workflows.abacus_tweb.p8_train_graph_patch import GraphPatchNet, predict_fold


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P5_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-group-cores", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p5-root", type=Path, default=P5_ROOT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    args = parser.parse_args()
    output = args.p8_root / "g_patch" / f"rotation_{args.rotation}" / f"seed_{args.seed}"
    checkpoint_path = output / "pre_evaluation_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    feature_dir = args.p8_root / "g_patch_features" / f"rotation_{args.rotation}"
    feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    validation_core = np.load(rotation_dir / "validation_core_id.npy")
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")

    model = GraphPatchNet(
        latent_size=args.latent_size, heads=args.heads, dropout=args.dropout
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    adapter = CanonicalGraphPatchAdapter(args.p5_root)
    adapter.node_features = np.load(feature_dir / "node_features_8d.npy", mmap_mode="r")
    started = time.time()
    parent, scaled, failures, maximum_nodes, maximum_edges = predict_fold(
        model,
        adapter,
        validation_core,
        feature_manifest["edge"],
        args.device,
        args.validation_group_cores,
    )
    eigenvalues = increments_to_eigenvalues(
        unscale_increments(scaled, scaler)
    ).astype(np.float32)
    report = evaluate_complete_fold(
        parent_node_id=parent,
        predicted_eigenvalues=eigenvalues,
        truth_by_parent=truth,
        assignment=assignment,
        validation_fold=roles["validation_fold"],
        runtime={
            "training_step": int(checkpoint["step"]),
            "evaluation_elapsed_seconds": time.time() - started,
            "patch_failures": failures,
            "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "maximum_patch_nodes": maximum_nodes,
            "maximum_patch_directed_edges": maximum_edges,
            "validation_group_cores": args.validation_group_cores,
            "resumed_from_pre_evaluation_checkpoint": True,
        },
    )
    torch.save(
        {
            "state_dict": checkpoint["state_dict"],
            "rotation": args.rotation,
            "seed": args.seed,
            "step": int(checkpoint["step"]),
            "score": report["primary_macro_r2_lambda1"],
            "scaler": scaler,
            "feature_manifest": feature_manifest,
        },
        output / "best_checkpoint.pt",
    )
    np.save(output / "best_validation_parent_node_id.npy", parent)
    np.save(output / "best_validation_eigenvalues.npy", eigenvalues)
    atomic_json(output / "best_validation_report.json", report)
    summary = {
        "schema_version": 1,
        "model": "G-PATCH",
        "rotation": args.rotation,
        "seed": args.seed,
        "status": "screen_complete",
        "best_step": int(checkpoint["step"]),
        "best_primary_macro_r2_lambda1": report["primary_macro_r2_lambda1"],
        "history": [{
            "step": int(checkpoint["step"]),
            "training_loss": float(checkpoint["training_loss"]),
            "primary_macro_r2_lambda1": report["primary_macro_r2_lambda1"],
            "per_shell_lambda1_r2": {
                name: row["lambda1"]["r2"] for name, row in report["per_shell"].items()
            },
        }],
        "training": {
            "shell_row_exposure": np.asarray(checkpoint["shell_row_exposure"]).tolist(),
            "evaluation_elapsed_seconds": time.time() - started,
        },
        "architecture": {
            "lineage": "two-pass receiver-normalized attentional GraphNetwork",
            "num_passes": 2,
            "dependency_hops": 4,
            "latent_size": args.latent_size,
            "heads": args.heads,
            "dropout": args.dropout,
            "validation_group_cores": args.validation_group_cores,
        },
    }
    atomic_json(output / "screen_summary.json", summary)
    (output / "G_PATCH_SCREEN_COMPLETE").write_text(
        f"rotation={args.rotation} seed={args.seed} "
        f"score={report['primary_macro_r2_lambda1']:.8f}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
