#!/usr/bin/env python3
"""Train and evaluate the deterministic two-pass G-PATCH P8 control.

The implementation is a PyTorch-faithful form of the established Jraph
GraphNetwork: updated edges receive receiver-normalized multi-head attention,
the weighted edges are summed at both senders and receivers, and node/edge
residuals are applied for two passes.  P5 proves that this topology needs four
exact graph dependency hops for patch/full-graph equivalence.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p5_graph_patch_utils import (
    CanonicalGraphPatchAdapter,
    assemble_patch,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    sha256,
    unscale_increments,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P5_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter")
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
NUM_PASSES = 2
DEPENDENCY_HOPS_PER_PASS = 2


class MLP(nn.Module):
    def __init__(self, input_size: int, latent_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.GELU(),
            nn.LayerNorm(latent_size),
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class AttentionPass(nn.Module):
    def __init__(self, latent_size: int, heads: int, dropout: float):
        super().__init__()
        if latent_size % heads:
            raise ValueError("latent size must be divisible by attention heads")
        self.latent_size = latent_size
        self.heads = heads
        self.head_size = latent_size // heads
        self.edge = MLP(3 * latent_size, latent_size, dropout)
        self.node = MLP(3 * latent_size, latent_size, dropout)
        self.attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * latent_size, self.head_size),
                    nn.GELU(),
                    nn.Linear(self.head_size, self.head_size),
                    nn.GELU(),
                    nn.Linear(self.head_size, 1),
                )
                for _ in range(heads)
            ]
        )

    @staticmethod
    def segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        result = values.new_zeros((size, values.shape[1]))
        result.index_add_(0, index, values)
        return result

    @staticmethod
    def receiver_softmax(
        logits: torch.Tensor, receivers: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        maximum = logits.new_full((n_nodes, logits.shape[1]), -torch.inf)
        expanded = receivers[:, None].expand(-1, logits.shape[1])
        maximum.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
        shifted = torch.exp(logits - maximum[receivers])
        denominator = logits.new_zeros((n_nodes, logits.shape[1]))
        denominator.index_add_(0, receivers, shifted)
        return shifted / denominator[receivers].clamp_min(1e-12)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender_nodes = nodes[senders]
        receiver_nodes = nodes[receivers]
        updated_edges = self.edge(torch.cat((edges, sender_nodes, receiver_nodes), dim=1))
        attention_input = torch.cat(
            (updated_edges, sender_nodes, receiver_nodes), dim=1
        )
        logits = torch.cat([head(attention_input) for head in self.attention], dim=1)
        weights = self.receiver_softmax(logits, receivers, len(nodes))
        weighted_edges = (
            updated_edges.view(-1, self.heads, self.head_size)
            * weights[:, :, None]
        ).reshape(-1, self.latent_size)
        sent = self.segment_sum(weighted_edges, senders, len(nodes))
        received = self.segment_sum(weighted_edges, receivers, len(nodes))
        updated_nodes = self.node(torch.cat((nodes, sent, received), dim=1))
        return nodes + updated_nodes, edges + weighted_edges


class GraphPatchNet(nn.Module):
    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 5,
        latent_size: int = 80,
        heads: int = 8,
        dropout: float = 0.1,
        passes: int = NUM_PASSES,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_features, latent_size)
        self.edge_embed = nn.Linear(edge_features, latent_size)
        self.passes = nn.ModuleList(
            [AttentionPass(latent_size, heads, dropout) for _ in range(passes)]
        )
        self.output = nn.Linear(latent_size, 3)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.node_embed(nodes)
        edges = self.edge_embed(edges)
        for layer in self.passes:
            nodes, edges = layer(nodes, edges, senders, receivers)
        return self.output(nodes)


def transformed_patch(patch, edge_spec: dict, cap_id: int, device: str):
    edge = np.asarray(patch.edge_features, dtype=np.float32).copy()
    spec = edge_spec[str(int(cap_id))]
    edge[:, 0] = (
        np.log(np.maximum(edge[:, 0] / spec["edge_length_si_median"], 1e-6))
        - spec["log_length_mean"]
    ) / spec["log_length_std"]
    edge[:, 4] = (
        np.log(np.maximum(edge[:, 4], 1e-6))
        - spec["log_density_contrast_mean"]
    ) / spec["log_density_contrast_std"]
    return (
        torch.from_numpy(np.asarray(patch.node_features)).to(device),
        torch.from_numpy(edge).to(device),
        torch.from_numpy(np.asarray(patch.senders, dtype=np.int64)).to(device),
        torch.from_numpy(np.asarray(patch.receivers, dtype=np.int64)).to(device),
    )


def grouped_patch(adapter, core_ids: np.ndarray):
    """Extract one exact graph view for several authoritative cores."""
    core_ids = np.asarray(core_ids, dtype=np.int64)
    caps = np.asarray(adapter.core_cap[core_ids], dtype=np.int8)
    if len(np.unique(caps)) != 1:
        raise ValueError("a grouped graph patch cannot cross Galactic caps")
    parents = np.unique(
        np.concatenate([adapter.core_nodes(int(core))[0] for core in core_ids])
    )
    return assemble_patch(
        core_id=-1,
        fold=int(adapter.core_fold[int(core_ids[0])]),
        core_parent_ids=parents,
        loss_parent_ids=parents,
        strict_parent_ids=np.empty(0, dtype=np.int64),
        loss_policy="authoritative_group",
        num_passes=NUM_PASSES,
        dependency_hops=NUM_PASSES * DEPENDENCY_HOPS_PER_PASS,
        node_features=adapter.node_features,
        union_pairs=adapter.union_pairs,
        union_edge_features=adapter.union_edge_features,
        offsets=adapter.offsets,
        incident_edge_id=adapter.incident_edge_id,
    )


def predict_fold(
    model,
    adapter,
    core_ids,
    edge_spec,
    device,
    group_cores,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    model.eval()
    parent_parts, prediction_parts = [], []
    failures = 0
    maximum_nodes = maximum_edges = 0
    groups = []
    core_ids = np.asarray(core_ids, dtype=np.int64)
    for cap_id in (0, 1):
        cap_core = np.sort(core_ids[np.asarray(adapter.core_cap[core_ids]) == cap_id])
        groups.extend(
            cap_core[start : start + group_cores]
            for start in range(0, len(cap_core), group_cores)
        )
    with torch.inference_mode():
        for group in groups:
            patch = grouped_patch(adapter, group)
            maximum_nodes = max(maximum_nodes, patch.n_node)
            maximum_edges = max(maximum_edges, patch.n_edge)
            tensors = transformed_patch(
                patch, edge_spec, int(adapter.core_cap[int(group[0])]), device
            )
            prediction = model(*tensors)[patch.authoritative_core_mask].cpu().numpy()
            parent_parts.append(patch.parent_node_id[patch.authoritative_core_mask])
            prediction_parts.append(prediction)
    return (
        np.concatenate(parent_parts),
        np.concatenate(prediction_parts),
        failures,
        maximum_nodes,
        maximum_edges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--loss-log-every", type=int, default=25,
                        help="steps between windowed training-loss records (no validation cost)")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-group-cores", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p5-root", type=Path, default=P5_ROOT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("G-PATCH requires a CUDA interactive allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output = args.p8_root / "g_patch" / f"rotation_{args.rotation}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    feature_dir = args.p8_root / "g_patch_features" / f"rotation_{args.rotation}"
    feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    if not feature_manifest["pass"]:
        raise RuntimeError("G-PATCH feature transform did not pass")
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    training_core = np.load(rotation_dir / "training_core_id.npy")
    training_core_weight = np.load(rotation_dir / "training_core_weight.npy").astype(np.float64)
    training_probability = training_core_weight / training_core_weight.sum()
    validation_core = np.load(rotation_dir / "validation_core_id.npy")
    row_weight = np.load(rotation_dir / "active_training_weight.npy", mmap_mode="r")
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    parent_weight[active_parent] = row_weight
    parent_shell = np.full(len(truth), -1, dtype=np.int8)
    parent_shell[active_parent] = np.asarray(assignment["shell"], dtype=np.int8)
    target_scaled = scale_increments(linear_increments(np.asarray(truth)), scaler)

    model = GraphPatchNet(
        latent_size=args.latent_size,
        heads=args.heads,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_score, best_step, stale = -np.inf, -1, 0
    best_state = None
    history = []
    loss_trace: list[dict] = []
    loss_window: list[float] = []
    shell_exposure = np.zeros(4, dtype=np.int64)
    maximum_nodes = maximum_edges = maximum_memory = 0
    started = time.time()

    adapter = CanonicalGraphPatchAdapter(args.p5_root)
    adapter.node_features = np.load(
        feature_dir / "node_features_8d.npy", mmap_mode="r"
    )
    edge_spec = feature_manifest["edge"]
    for step in range(1, args.steps + 1):
        core_id = int(rng.choice(training_core, p=training_probability))
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
        loss_per_row = torch.mean((prediction - target) ** 2, dim=1)
        loss = torch.sum(weight * loss_per_row) / torch.sum(weight).clamp_min(1e-12)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        shell_exposure += np.bincount(parent_shell[parent], minlength=4)[:4]
        maximum_memory = max(maximum_memory, int(torch.cuda.max_memory_allocated()))

        # Training-curve logging. Costs no validation work, so it is decoupled from
        # --eval-every: the short screens recorded a single instantaneous single-patch
        # loss and therefore had no learning curve at all. A windowed mean is logged so
        # the curve reflects optimization rather than which patch was drawn.
        loss_window.append(float(loss.detach().cpu()))
        if step % args.loss_log_every == 0 or step == args.steps:
            loss_trace.append({
                "step": step,
                "training_loss_window_mean": float(np.mean(loss_window)),
                "training_loss_window_min": float(np.min(loss_window)),
                "training_loss_window_max": float(np.max(loss_window)),
                "window": len(loss_window),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            })
            with open(output / "loss_trace.jsonl", "a") as handle:
                handle.write(json.dumps(loss_trace[-1]) + "\n")
            loss_window.clear()

        if step % args.eval_every == 0 or step == args.steps:
            # Persist before the expensive exact complete-fold graph assembly.
            # Evaluation can therefore be resumed if an interactive allocation ends.
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rotation": args.rotation,
                    "seed": args.seed,
                    "step": step,
                    "training_loss": float(loss.detach().cpu()),
                    "shell_row_exposure": shell_exposure,
                    "maximum_nodes": maximum_nodes,
                    "maximum_edges": maximum_edges,
                    "maximum_memory": maximum_memory,
                },
                output / "pre_evaluation_checkpoint.pt",
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
                    "training_step": step,
                    "elapsed_seconds": time.time() - started,
                    "patch_failures": failures,
                    "maximum_cuda_memory_bytes": maximum_memory,
                    "maximum_patch_nodes": maximum_nodes,
                    "maximum_patch_directed_edges": maximum_edges,
                },
            )
            score = report["primary_macro_r2_lambda1"]
            history.append({
                "step": step,
                "training_loss": float(loss.detach().cpu()),
                "primary_macro_r2_lambda1": score,
                "per_shell_lambda1_r2": {
                    name: row["lambda1"]["r2"]
                    for name, row in report["per_shell"].items()
                },
            })
            print(json.dumps(history[-1]), flush=True)
            if score > best_score:
                best_score, best_step, stale = score, step, 0
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "state_dict": best_state,
                        "rotation": args.rotation,
                        "seed": args.seed,
                        "step": step,
                        "score": score,
                        "scaler": scaler,
                        "feature_manifest": feature_manifest,
                    },
                    output / "best_checkpoint.pt",
                )
                np.save(output / "best_validation_parent_node_id.npy", val_parent)
                np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                atomic_json(output / "best_validation_report.json", report)
            else:
                stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("G-PATCH did not produce a complete validation checkpoint")
    final = {
        "schema_version": 1,
        "model": "G-PATCH",
        "rotation": args.rotation,
        "seed": args.seed,
        "status": "screen_complete",
        "best_step": best_step,
        "best_primary_macro_r2_lambda1": best_score,
        "steps_run": step,
        "history": history,
        "loss_trace": loss_trace,
        "architecture": {
            "lineage": "two-pass receiver-normalized attentional GraphNetwork",
            "implementation": "PyTorch faithful Jraph GraphNetwork form",
            "num_passes": NUM_PASSES,
            "dependency_hops_per_pass": DEPENDENCY_HOPS_PER_PASS,
            "dependency_hops": NUM_PASSES * DEPENDENCY_HOPS_PER_PASS,
            "latent_size": args.latent_size,
            "heads": args.heads,
            "dropout": args.dropout,
            "node_features": 8,
            "edge_features": 5,
            "validation_group_cores": args.validation_group_cores,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "training": {
            "objective": "sqrt-shell-weighted mean MSE over authoritative core rows",
            "core_sampling": "probability proportional to frozen core weight",
            "shell_row_exposure": shell_exposure.tolist(),
            "maximum_cuda_memory_bytes": maximum_memory,
            "maximum_patch_nodes": maximum_nodes,
            "maximum_patch_directed_edges": maximum_edges,
            "elapsed_seconds": time.time() - started,
        },
        "inputs": {
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "p5_adapter": str(args.p5_root / "adapter_manifest.json"),
            "p5_adapter_sha256": sha256(args.p5_root / "adapter_manifest.json"),
            "feature_manifest": str(feature_dir / "feature_manifest.json"),
            "feature_manifest_sha256": sha256(feature_dir / "feature_manifest.json"),
        },
    }
    atomic_json(output / "screen_summary.json", final)
    (output / "G_PATCH_SCREEN_COMPLETE").write_text(
        f"rotation={args.rotation} seed={args.seed} score={best_score:.8f}\n"
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
