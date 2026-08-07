#!/usr/bin/env python3
"""Train G-PATCH on Bright targets with appended Faint graph context."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from workflows.abacus_tweb.p5_graph_patch_utils import CanonicalGraphPatchAdapter
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    unscale_increments,
)
from workflows.abacus_tweb.p8_train_graph_patch import (
    DEPENDENCY_HOPS_PER_PASS,
    NUM_PASSES,
    GraphPatchNet,
    predict_fold,
    transformed_patch,
)


P8 = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
MT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument(
        "--graph-product",
        help="Graph artifact directory name; defaults to --product.",
    )
    parser.add_argument("--run-name", default="screen")
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--loss-log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-group-cores", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8)
    parser.add_argument("--root", type=Path, default=MT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("multitracer G-PATCH requires a GPU allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output = (
        args.root / "models/g_patch" / args.product
        / f"rotation_{args.rotation}" / f"seed_{args.seed}"
        / args.run_name
    )
    output.mkdir(parents=True, exist_ok=True)
    graph_product = args.graph_product or args.product
    adapter_root = args.root / "graph" / graph_product / "adapter"
    feature_dir = (
        args.root / "models/g_patch_features" / args.product
        / f"rotation_{args.rotation}"
    )
    feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    if not feature_manifest["pass"]:
        raise RuntimeError("multitracer graph feature transform did not pass")
    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    training_core = np.load(rotation_dir / "training_core_id.npy")
    core_weight = np.load(rotation_dir / "training_core_weight.npy").astype(np.float64)
    core_probability = core_weight / core_weight.sum()
    validation_core = np.load(rotation_dir / "validation_core_id.npy")
    row_weight = np.load(rotation_dir / "active_training_weight.npy", mmap_mode="r")
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    parent_weight[active_parent] = row_weight
    target_scaled = scale_increments(linear_increments(np.asarray(truth)), scaler)

    model = GraphPatchNet(
        node_features=10, edge_features=5, latent_size=args.latent_size,
        heads=args.heads, dropout=args.dropout, passes=NUM_PASSES,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    adapter = CanonicalGraphPatchAdapter(adapter_root)
    adapter.node_features = np.load(feature_dir / "node_features_10d.npy", mmap_mode="r")
    edge_spec = feature_manifest["edge"]
    history, loss_window = [], []
    best_score, best_step, best_state = -np.inf, -1, None
    maximum_nodes = maximum_edges = maximum_memory = 0
    started = time.time()

    for step in range(1, args.steps + 1):
        core_id = int(rng.choice(training_core, p=core_probability))
        patch = adapter.extract(
            core_id, NUM_PASSES,
            dependency_hops_per_pass=DEPENDENCY_HOPS_PER_PASS,
            loss_policy="authoritative",
        )
        maximum_nodes = max(maximum_nodes, patch.n_node)
        maximum_edges = max(maximum_edges, patch.n_edge)
        tensors = transformed_patch(
            patch, edge_spec, int(adapter.core_cap[core_id]), args.device
        )
        parent = patch.parent_node_id[patch.loss_mask]
        weight = torch.from_numpy(parent_weight[parent]).to(args.device)
        target = torch.from_numpy(target_scaled[parent]).to(args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*tensors)[patch.loss_mask]
        per_row = torch.mean((prediction - target) ** 2, dim=1)
        loss = torch.sum(weight * per_row) / torch.sum(weight).clamp_min(1.0e-12)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        maximum_memory = max(maximum_memory, int(torch.cuda.max_memory_allocated()))
        loss_window.append(float(loss.detach().cpu()))
        if step % args.loss_log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "training_loss_window_mean": float(np.mean(loss_window)),
                "training_loss_window_min": float(np.min(loss_window)),
                "training_loss_window_max": float(np.max(loss_window)),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            with (output / "loss_trace.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            loss_window.clear()
        if step % args.eval_every == 0 or step == args.steps:
            torch.save(
                {
                    "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step,
                    "product": args.product, "rotation": args.rotation, "seed": args.seed,
                },
                output / "latest_checkpoint.pt",
            )
            val_parent, val_scaled, failures, val_nodes, val_edges = predict_fold(
                model, adapter, validation_core, edge_spec, args.device,
                args.validation_group_cores,
            )
            maximum_nodes = max(maximum_nodes, val_nodes)
            maximum_edges = max(maximum_edges, val_edges)
            val_eigen = increments_to_eigenvalues(
                unscale_increments(val_scaled, scaler)
            ).astype(np.float32)
            report = evaluate_complete_fold(
                parent_node_id=val_parent,
                predicted_eigenvalues=val_eigen,
                truth_by_parent=truth,
                assignment=assignment,
                validation_fold=roles["validation_fold"],
                runtime={
                    "training_step": step, "elapsed_seconds": time.time() - started,
                    "patch_failures": failures, "maximum_patch_nodes": maximum_nodes,
                    "maximum_patch_directed_edges": maximum_edges,
                    "maximum_cuda_memory_bytes": maximum_memory,
                },
            )
            score = report["primary_macro_r2_lambda1"]
            history.append({"step": step, "macro_r2_lambda1": score})
            print(json.dumps(history[-1]), flush=True)
            if score > best_score:
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {"state_dict": best_state, "step": step, "score": score},
                    output / "best_checkpoint.pt",
                )
                np.save(output / "best_validation_parent_node_id.npy", val_parent)
                np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                atomic_json(output / "best_validation_report.json", report)
    if best_state is None:
        raise RuntimeError("multitracer G-PATCH produced no validation checkpoint")
    summary = {
        "schema_version": "p8-multitracer-g-patch-v1",
        "model": "G-PATCH-BRIGHT_TARGET-FAINT_CONTEXT",
        "product": args.product,
        "graph_product": graph_product,
        "rotation": args.rotation,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "best_primary_macro_r2_lambda1": best_score,
        "history": history,
        "node_features": 10,
        "supervision_contract": "BGS_BRIGHT only; Faint context is never scored",
        "maximum_patch_nodes": maximum_nodes,
        "maximum_patch_directed_edges": maximum_edges,
        "maximum_cuda_memory_bytes": maximum_memory,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "screen_summary.json", summary)
    (output / "MULTITRACER_G_PATCH_SCREEN_COMPLETE").write_text(
        f"product={args.product} score={best_score:.8f}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
