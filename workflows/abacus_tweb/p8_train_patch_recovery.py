#!/usr/bin/env python3
"""Epoch-aware, resumable P8 recovery trainer for G-PATCH and U-PATCH.

This is deliberately separate from the frozen 2,000-step smoke trainers.  A
recovery epoch visits every eligible P4 training core exactly once.  Patch
losses retain the frozen square-root shell objective and are scaled so that
their arithmetic epoch mean is the globally row-weighted MSE.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p8_train_graph_patch as graph_impl
from workflows.abacus_tweb import p8_train_unet_cic_residual as residual_impl
from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    evaluate_complete_fold,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    append_jsonl,
    epoch_order,
    improved,
    patch_objective,
    reconcile_loss_trace,
    rewrite_jsonl,
    should_stop,
    validate_resume_order,
    validate_warm_start_contract,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
OUTPUT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1")


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=device)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def jsonable_arguments(arguments: dict) -> dict:
    """Convert checkpoint arguments into stable JSON provenance."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
    }


def validation_losses(
    predicted_scaled: np.ndarray,
    parent_node_id: np.ndarray,
    target_scaled: np.ndarray,
    parent_shell: np.ndarray,
) -> dict:
    row_loss = np.mean(
        (np.asarray(predicted_scaled, np.float64)
         - np.asarray(target_scaled[parent_node_id], np.float64)) ** 2,
        axis=1,
    )
    result = {"all_rows_scaled_mse": float(np.mean(row_loss))}
    result["per_shell_scaled_mse"] = {
        str(shell): float(np.mean(row_loss[parent_shell[parent_node_id] == shell]))
        for shell in range(4)
    }
    return result


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    cursor: int,
    order: np.ndarray,
    global_step: int,
    accumulator: EpochLossAccumulator,
    shell_numerator: np.ndarray,
    shell_denominator: np.ndarray,
    shell_rows: np.ndarray,
    history: list[dict],
    best_score: float,
    best_epoch: int,
    early_best_score: float,
    stale_epochs: int,
    maximum_memory: int,
    args,
) -> dict:
    payload = {
        "schema_version": 1,
        "git_revision": args.git_revision,
        "model": args.model,
        "rotation": args.rotation,
        "seed": args.seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "cursor": int(cursor),
        "epoch_order": np.asarray(order, dtype=np.int64),
        "global_step": int(global_step),
        "epoch_accumulator": accumulator.as_dict(),
        "shell_numerator": np.asarray(shell_numerator, dtype=np.float64),
        "shell_denominator": np.asarray(shell_denominator, dtype=np.float64),
        "shell_rows": np.asarray(shell_rows, dtype=np.int64),
        "history": history,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "early_best_score": float(early_best_score),
        "stale_epochs": int(stale_epochs),
        "maximum_memory": int(maximum_memory),
        "arguments": vars(args),
        # A resumed allocation must continue the same dropout sequence, not
        # merely recover the same weights and epoch cursor.
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("graph", "unet", "unet_cic_residual"), required=True
    )
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--loss-log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validation-group-cores", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unet-base", type=int, default=24)
    parser.add_argument("--unet-latent-channels", type=int, default=32)
    parser.add_argument(
        "--lambda1-max-sigma",
        type=float,
        default=1.0,
        help=(
            "maximum absolute lambda1 correction in training-sigma units; "
            "used only by unet_cic_residual"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="control_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument(
        "--backbone-checkpoint", type=Path,
        help="frozen U-PATCH best checkpoint used only to initialize U-CIC-RESID-v1",
    )
    parser.add_argument("--epoch-seed-offset", type=int, default=0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--assignment", type=Path, default=graph_impl.ASSIGNMENT)
    parser.add_argument("--p5-root", type=Path, default=graph_impl.P5_ROOT)
    parser.add_argument("--unet-adapter", type=Path, default=unet_impl.ADAPTER)
    parser.add_argument("--selection", type=Path, default=unet_impl.SELECTION)
    args = parser.parse_args()
    if args.epochs <= 0 or args.min_epochs <= 0:
        parser.error("epochs and min-epochs must be positive")
    if args.min_epochs > args.epochs:
        parser.error("min-epochs cannot exceed epochs")
    if args.epoch_seed_offset < 0:
        parser.error("epoch-seed-offset must be non-negative")
    if (
        args.warm_start_checkpoint is not None
        and not args.warm_start_checkpoint.is_file()
    ):
        parser.error(f"warm-start checkpoint not found: {args.warm_start_checkpoint}")
    if args.warm_start_checkpoint is None and args.epoch_seed_offset != 0:
        parser.error("epoch-seed-offset requires warm-start-checkpoint")
    if args.warm_start_checkpoint is not None:
        if args.run_name != "convergence_extension_v1":
            parser.error(
                "the pre-registered warm start requires "
                "--run-name convergence_extension_v1"
            )
        if args.epochs != 20 or not np.isclose(args.lr, 2e-4):
            parser.error(
                "the convergence extension requires --epochs 20 --lr 2e-4"
            )
        if not args.disable_early_stopping:
            parser.error(
                "the convergence extension requires --disable-early-stopping"
            )
    if args.model == "unet_cic_residual":
        if args.backbone_checkpoint is None or not args.backbone_checkpoint.is_file():
            parser.error("unet_cic_residual requires an existing --backbone-checkpoint")
        if args.warm_start_checkpoint is not None:
            parser.error("the first residual screen cannot use --warm-start-checkpoint")
        if not np.isfinite(args.lambda1_max_sigma) or args.lambda1_max_sigma <= 0:
            parser.error("unet_cic_residual requires --lambda1-max-sigma > 0")
    elif args.backbone_checkpoint is not None:
        parser.error("--backbone-checkpoint is exclusive to unet_cic_residual")
    elif not np.isclose(args.lambda1_max_sigma, 1.0):
        parser.error("--lambda1-max-sigma is exclusive to unet_cic_residual")
    if args.patience <= 0 or args.loss_log_every <= 0 or args.checkpoint_every <= 0:
        parser.error("patience and logging/checkpoint intervals must be positive")
    if not args.canary and args.min_epochs < 5:
        parser.error("scientific recovery runs require --min-epochs >= 5")
    if args.canary and args.epochs != 1:
        parser.error("a canary must use exactly one complete epoch")
    return args


def main() -> None:
    args = parse_args()
    args.git_revision = git_revision()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P8 recovery training requires a CUDA interactive allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = (
        args.output_root / args.run_name / args.model
        / f"rotation_{args.rotation}" / f"seed_{args.seed}"
    )
    checkpoint_path = output / "recovery_checkpoint.pt"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"non-empty recovery output requires --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rotation_dir = args.p8_root / f"rotation_{args.rotation}"
    roles = json.loads((rotation_dir / "roles.json").read_text())
    scaler = json.loads((rotation_dir / "target_scaler.json").read_text())
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    assignment = np.load(args.assignment, mmap_mode="r")
    training_core = np.load(rotation_dir / "training_core_id.npy").astype(np.int64)
    training_core_weight = np.load(
        rotation_dir / "training_core_weight.npy"
    ).astype(np.float64)
    validation_core = np.load(rotation_dir / "validation_core_id.npy").astype(np.int64)
    row_weight = np.load(rotation_dir / "active_training_weight.npy", mmap_mode="r")
    active_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    parent_weight[active_parent] = row_weight
    parent_shell = np.full(len(truth), -1, dtype=np.int8)
    parent_shell[active_parent] = np.asarray(assignment["shell"], dtype=np.int8)
    target_scaled = scale_increments(linear_increments(np.asarray(truth)), scaler)
    mean_core_weight = float(np.mean(training_core_weight))
    expected_core_weight = {
        int(core): float(weight)
        for core, weight in zip(training_core, training_core_weight)
    }

    feature_manifest = None
    normalization = None
    edge_spec = None
    adapter = None
    cic_by_parent = None
    cic_anchor = None
    backbone_start = None
    if args.model == "graph":
        feature_dir = args.p8_root / "g_patch_features" / f"rotation_{args.rotation}"
        feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
        if not feature_manifest["pass"]:
            raise RuntimeError("G-PATCH feature transform did not pass")
        model = graph_impl.GraphPatchNet(
            latent_size=args.latent_size,
            heads=args.heads,
            dropout=args.dropout,
        ).to(args.device)
        adapter = graph_impl.CanonicalGraphPatchAdapter(args.p5_root)
        adapter.node_features = np.load(
            feature_dir / "node_features_8d.npy", mmap_mode="r"
        )
        edge_spec = feature_manifest["edge"]
    elif args.model == "unet":
        selection_manifest = json.loads(args.selection.read_text())
        normalization = selection_manifest["rotations"][str(args.rotation)]["normalization"]
        model = unet_impl.UPatch(
            base=args.unet_base,
            latent_channels=args.unet_latent_channels,
        ).to(args.device)
        adapter = unet_impl.CanonicalFieldPatchAdapter(
            args.unet_adapter,
            selection_manifest=args.selection,
            rotation=args.rotation,
        )
    else:
        selection_manifest = json.loads(args.selection.read_text())
        normalization = selection_manifest["rotations"][str(args.rotation)]["normalization"]
        cic_by_parent, cic_anchor = residual_impl.load_cic_anchor(
            args.p8_root, args.rotation, len(truth)
        )
        model = residual_impl.UCICResidual(
            scaler,
            base=args.unet_base,
            latent_channels=args.unet_latent_channels,
            lambda1_max_sigma=args.lambda1_max_sigma,
        ).to(args.device)
        backbone_start = residual_impl.load_unet_backbone(
            model, args.backbone_checkpoint, args.device
        )
        if backbone_start["model"] != "unet":
            raise RuntimeError("residual backbone is not a U-PATCH checkpoint")
        if int(backbone_start["rotation"]) != args.rotation:
            raise RuntimeError("residual backbone rotation mismatch")
        if int(backbone_start["seed"]) != args.seed:
            raise RuntimeError("residual backbone seed mismatch")
        backbone_start["checkpoint_sha256"] = sha256(args.backbone_checkpoint)
        adapter = unet_impl.CanonicalFieldPatchAdapter(
            args.unet_adapter,
            selection_manifest=args.selection,
            rotation=args.rotation,
        )

    warm_start = None
    if args.warm_start_checkpoint is not None:
        parent_best = torch_load(args.warm_start_checkpoint, args.device)
        validate_warm_start_contract(
            parent_best,
            model=args.model,
            rotation=args.rotation,
            seed=args.seed,
        )
        if args.epoch_seed_offset != int(parent_best["epoch"]):
            raise RuntimeError(
                "epoch-seed-offset must equal the parent best epoch: "
                f"{args.epoch_seed_offset} != {parent_best['epoch']}"
            )
        model.load_state_dict(parent_best["state_dict"])
        parent_directory = args.warm_start_checkpoint.parent
        parent_manifest_path = parent_directory / "run_manifest.json"
        parent_recovery_path = parent_directory / "recovery_checkpoint.pt"
        if not parent_manifest_path.is_file() or not parent_recovery_path.is_file():
            raise RuntimeError(
                "warm-start provenance requires sibling run_manifest.json and "
                "recovery_checkpoint.pt"
            )
        parent_manifest = json.loads(parent_manifest_path.read_text())
        parent_recovery = torch_load(parent_recovery_path, "cpu")
        for field in ("model", "rotation", "seed"):
            if parent_recovery.get(field) != getattr(args, field):
                raise RuntimeError(
                    f"parent recovery {field} mismatch: "
                    f"{parent_recovery.get(field)} != {getattr(args, field)}"
                )
        warm_start = {
            "checkpoint": str(args.warm_start_checkpoint),
            "checkpoint_sha256": sha256(args.warm_start_checkpoint),
            "parent_recovery_checkpoint": str(parent_recovery_path),
            "parent_recovery_checkpoint_sha256": sha256(parent_recovery_path),
            "parent_run_manifest": str(parent_manifest_path),
            "parent_run_manifest_sha256": sha256(parent_manifest_path),
            "parent_git_revision": parent_manifest["git_revision"],
            "parent_epoch": int(parent_best["epoch"]),
            "parent_global_step": int(parent_best["global_step"]),
            "parent_score": float(parent_best["score"]),
            "parent_arguments": jsonable_arguments(parent_recovery["arguments"]),
            "optimizer_policy": "fresh AdamW and fresh cosine scheduler",
        }
        del parent_best, parent_recovery

    zero_residual_parity = None
    if args.model == "unet_cic_residual":
        parity_patch = adapter.extract(
            int(training_core[0]),
            unet_impl.HALO_VOXELS,
            unet_impl.CHANNELS,
            alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
        )
        parity_parent = parity_patch.authoritative_parent_id
        parity_cic = np.asarray(cic_by_parent[parity_parent], dtype=np.float32)
        if not np.all(np.isfinite(parity_cic)):
            raise RuntimeError("checkpoint-zero parity core has missing CIC anchors")
        parity_values, parity_points = unet_impl.model_inputs(
            parity_patch, normalization, args.device
        )
        zero_residual_parity = residual_impl.checkpoint_zero_parity(
            model,
            parity_values,
            parity_points,
            torch.from_numpy(parity_cic).to(args.device),
        )
        zero_residual_parity["core_id"] = int(training_core[0])
        zero_residual_parity["rows"] = int(len(parity_parent))
        atomic_json(output / "checkpoint_zero_parity.json", zero_residual_parity)
        if not zero_residual_parity["pass"]:
            raise RuntimeError(f"U-CIC residual null-model parity failed: {zero_residual_parity}")
        del parity_patch, parity_values, parity_points

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_updates = args.epochs * len(training_core)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates
    )
    history: list[dict] = []
    best_score = -np.inf
    best_epoch = -1
    early_best_score = -np.inf
    stale_epochs = 0
    start_epoch = 1
    start_cursor = 0
    global_step = 0
    resume_order = None
    resume_accumulator = EpochLossAccumulator()
    resume_shell_numerator = np.zeros(4, dtype=np.float64)
    resume_shell_denominator = np.zeros(4, dtype=np.float64)
    resume_shell_rows = np.zeros(4, dtype=np.int64)
    maximum_memory = 0

    if args.resume:
        if not checkpoint_path.exists():
            raise RuntimeError(f"resume checkpoint not found: {checkpoint_path}")
        state = torch_load(checkpoint_path, args.device)
        for field in ("model", "rotation", "seed"):
            expected = getattr(args, field)
            if state[field] != expected:
                raise RuntimeError(
                    f"resume {field} mismatch: {state[field]} != {expected}"
                )
        if state.get("git_revision") not in (None, args.git_revision):
            raise RuntimeError(
                "resume code revision mismatch: "
                f"{state['git_revision']} != {args.git_revision}"
            )
        frozen_resume_fields = (
            "epochs",
            "min_epochs",
            "patience",
            "min_delta",
            "lr",
            "latent_size",
            "heads",
            "dropout",
            "unet_base",
            "unet_latent_channels",
            "lambda1_max_sigma",
            "canary",
            "run_name",
            "warm_start_checkpoint",
            "backbone_checkpoint",
            "epoch_seed_offset",
            "disable_early_stopping",
            "p8_root",
            "assignment",
            "p5_root",
            "unet_adapter",
            "selection",
        )
        checkpoint_arguments = state["arguments"]
        for field in frozen_resume_fields:
            if checkpoint_arguments.get(field) != getattr(args, field):
                raise RuntimeError(
                    f"resume argument {field} mismatch: "
                    f"{checkpoint_arguments.get(field)} != {getattr(args, field)}"
                )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        start_cursor = int(state["cursor"])
        global_step = int(state["global_step"])
        resume_order = np.asarray(state["epoch_order"], dtype=np.int64)
        validate_resume_order(resume_order, training_core, start_cursor)
        resume_accumulator = EpochLossAccumulator.from_dict(state["epoch_accumulator"])
        resume_shell_numerator = np.asarray(state["shell_numerator"], dtype=np.float64)
        resume_shell_denominator = np.asarray(state["shell_denominator"], dtype=np.float64)
        resume_shell_rows = np.asarray(state["shell_rows"], dtype=np.int64)
        history = list(state["history"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        early_best_score = float(state["early_best_score"])
        stale_epochs = int(state["stale_epochs"])
        maximum_memory = int(state["maximum_memory"])
        reconcile_loss_trace(
            output / "loss_trace.jsonl", maximum_global_step=global_step
        )
        rewrite_jsonl(output / "epoch_history.jsonl", history)
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
    else:
        (output / "loss_trace.jsonl").write_text("")
        (output / "epoch_history.jsonl").write_text("")

    run_manifest = {
        "schema_version": 1,
        "stage": "P8 exposure-aware recovery",
        "git_revision": args.git_revision,
        "model": args.model,
        "rotation": args.rotation,
        "seed": args.seed,
        "roles": roles,
        "eligible_training_cores": int(len(training_core)),
        "validation_cores": int(len(validation_core)),
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "mean_core_weight": mean_core_weight,
        "sampler": (
            "deterministic probability-proportional-to-core-weight permutation; "
            "without replacement; 100% eligible cores per completed epoch"
        ),
        "objective": "complete-core epoch; sum(w_i mse_i)/mean(training core weight)",
        "canary": args.canary,
        "resume": args.resume,
        "warm_start": warm_start,
        "cic_anchor": cic_anchor,
        "backbone_start": backbone_start,
        "checkpoint_zero_parity": zero_residual_parity,
        "arguments": jsonable_arguments(vars(args)),
        "assignment": str(args.assignment),
        "assignment_sha256": sha256(args.assignment),
    }
    atomic_json(output / "run_manifest.json", run_manifest)

    loss_window_numerator = 0.0
    loss_window_denominator = 0.0
    loss_window_objective: list[float] = []
    started = time.time()
    stopped_early = False

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if epoch == start_epoch and resume_order is not None:
                order = resume_order
                cursor0 = start_cursor
                accumulator = resume_accumulator
                shell_numerator = resume_shell_numerator.copy()
                shell_denominator = resume_shell_denominator.copy()
                shell_rows = resume_shell_rows.copy()
            else:
                order = epoch_order(
                    training_core,
                    seed=args.seed,
                    epoch=args.epoch_seed_offset + epoch,
                    core_weight=training_core_weight,
                )
                cursor0 = 0
                accumulator = EpochLossAccumulator()
                shell_numerator = np.zeros(4, dtype=np.float64)
                shell_denominator = np.zeros(4, dtype=np.float64)
                shell_rows = np.zeros(4, dtype=np.int64)

            for cursor in range(cursor0, len(order)):
                core_id = int(order[cursor])
                if args.model == "graph":
                    patch = adapter.extract(
                        core_id,
                        graph_impl.NUM_PASSES,
                        dependency_hops_per_pass=graph_impl.DEPENDENCY_HOPS_PER_PASS,
                        loss_policy="authoritative",
                    )
                    tensors = graph_impl.transformed_patch(
                        patch,
                        edge_spec,
                        int(adapter.core_cap[core_id]),
                        args.device,
                    )
                    parent = patch.parent_node_id[patch.loss_mask]
                    prediction = model(*tensors)[patch.loss_mask]
                elif args.model == "unet":
                    patch = adapter.extract(
                        core_id,
                        unet_impl.HALO_VOXELS,
                        unet_impl.CHANNELS,
                        alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
                    )
                    parent = patch.authoritative_parent_id
                    values, points = unet_impl.model_inputs(
                        patch, normalization, args.device
                    )
                    prediction = model(values, points)
                else:
                    patch = adapter.extract(
                        core_id,
                        unet_impl.HALO_VOXELS,
                        unet_impl.CHANNELS,
                        alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
                    )
                    parent = patch.authoritative_parent_id
                    cic_np = np.asarray(cic_by_parent[parent], dtype=np.float32)
                    if not np.all(np.isfinite(cic_np)):
                        raise RuntimeError(f"core {core_id} has missing CIC anchor rows")
                    values, points = unet_impl.model_inputs(
                        patch, normalization, args.device
                    )
                    prediction, _, _ = model(
                        values, points, torch.from_numpy(cic_np).to(args.device)
                    )

                weight_np = np.asarray(parent_weight[parent], dtype=np.float32)
                actual_weight = float(np.sum(weight_np, dtype=np.float64))
                expected_weight = expected_core_weight[core_id]
                if not np.isclose(actual_weight, expected_weight, rtol=2e-6, atol=1e-7):
                    raise RuntimeError(
                        f"core {core_id} weight mismatch: {actual_weight} != {expected_weight}"
                    )
                weight = torch.from_numpy(weight_np).to(args.device)
                target = torch.from_numpy(target_scaled[parent]).to(args.device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_per_row = torch.mean((prediction - target) ** 2, dim=1)
                weighted_numerator = torch.sum(weight * loss_per_row)
                loss = patch_objective(
                    weighted_numerator, mean_core_weight=mean_core_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                loss_np = loss_per_row.detach().cpu().numpy()
                accumulator.add(loss_np, weight_np)
                loss_window_numerator += float(np.sum(weight_np * loss_np))
                loss_window_denominator += actual_weight
                loss_window_objective.append(float(loss.detach().cpu()))
                shells = parent_shell[parent]
                for shell in range(4):
                    select = shells == shell
                    if np.any(select):
                        shell_numerator[shell] += float(
                            np.sum(weight_np[select] * loss_np[select])
                        )
                        shell_denominator[shell] += float(np.sum(weight_np[select]))
                        shell_rows[shell] += int(np.sum(select))
                maximum_memory = max(
                    maximum_memory, int(torch.cuda.max_memory_allocated())
                )

                if (
                    global_step % args.loss_log_every == 0
                    or cursor + 1 == len(order)
                ):
                    row = {
                        "epoch": epoch,
                        "effective_epoch": args.epoch_seed_offset + epoch,
                        "cursor": cursor + 1,
                        "global_step": global_step,
                        "training_weighted_mse_window": (
                            loss_window_numerator / loss_window_denominator
                        ),
                        "optimizer_objective_window_mean": float(
                            np.mean(loss_window_objective)
                        ),
                        "window_patches": len(loss_window_objective),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    append_jsonl(output / "loss_trace.jsonl", row)
                    loss_window_numerator = 0.0
                    loss_window_denominator = 0.0
                    loss_window_objective.clear()

                if (
                    global_step % args.checkpoint_every == 0
                    or cursor + 1 == len(order)
                ):
                    atomic_torch_save(
                        checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            cursor=cursor + 1,
                            order=order,
                            global_step=global_step,
                            accumulator=accumulator,
                            shell_numerator=shell_numerator,
                            shell_denominator=shell_denominator,
                            shell_rows=shell_rows,
                            history=history,
                            best_score=best_score,
                            best_epoch=best_epoch,
                            early_best_score=early_best_score,
                            stale_epochs=stale_epochs,
                            maximum_memory=maximum_memory,
                            args=args,
                        ),
                        checkpoint_path,
                    )

            if accumulator.patches != len(training_core):
                raise RuntimeError(
                    f"epoch {epoch} incomplete: {accumulator.patches}/{len(training_core)} cores"
                )

            if args.model == "graph":
                val_parent, val_scaled, failures, val_nodes, val_edges = (
                    graph_impl.predict_fold(
                        model,
                        adapter,
                        validation_core,
                        edge_spec,
                        args.device,
                        args.validation_group_cores,
                    )
                )
                validation_runtime = {
                    "maximum_patch_nodes": int(val_nodes),
                    "maximum_patch_directed_edges": int(val_edges),
                }
            elif args.model == "unet":
                val_parent, val_scaled, failures = unet_impl.predict_fold(
                    model,
                    adapter,
                    validation_core,
                    normalization,
                    args.device,
                )
                validation_runtime = {}
            else:
                val_parent, val_scaled, failures = residual_impl.predict_fold(
                    model,
                    adapter,
                    validation_core,
                    normalization,
                    cic_by_parent,
                    args.device,
                )
                validation_runtime = {
                    "classical_anchor": "train-affine full-cap CIC",
                    "residual_parameterization": (
                        "additive lambda1 plus multiplicative positive eigengaps"
                    ),
                    "lambda1_max_sigma": float(args.lambda1_max_sigma),
                }
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
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.time() - started,
                    "patch_failures": failures,
                    "maximum_cuda_memory_bytes": maximum_memory,
                    **validation_runtime,
                },
            )
            score = float(report["primary_macro_r2_lambda1"])
            validation_loss = validation_losses(
                val_scaled, val_parent, target_scaled, parent_shell
            )
            shell_loss = np.divide(
                shell_numerator,
                shell_denominator,
                out=np.full(4, np.nan),
                where=shell_denominator > 0,
            )
            epoch_row = {
                "epoch": epoch,
                "effective_epoch": args.epoch_seed_offset + epoch,
                "global_step": global_step,
                "training_weighted_mse": accumulator.mean,
                "training_loss_numerator": accumulator.weighted_numerator,
                "training_weight_denominator": accumulator.weight_denominator,
                "eligible_cores": int(len(training_core)),
                "unique_cores_seen": int(accumulator.patches),
                "repeat_cores": 0,
                "training_weighted_mse_by_shell": shell_loss.tolist(),
                "training_loss_numerator_by_shell": shell_numerator.tolist(),
                "training_weight_denominator_by_shell": shell_denominator.tolist(),
                "training_rows_by_shell": shell_rows.tolist(),
                "unique_core_fraction": 1.0,
                "validation": validation_loss,
                "primary_macro_r2_lambda1": score,
                "per_shell_lambda1_r2": {
                    name: row["lambda1"]["r2"]
                    for name, row in report["per_shell"].items()
                },
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": time.time() - started,
            }
            history.append(epoch_row)
            append_jsonl(output / "epoch_history.jsonl", epoch_row)
            print(json.dumps(epoch_row), flush=True)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                atomic_torch_save(
                    {
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "model": args.model,
                        "rotation": args.rotation,
                        "seed": args.seed,
                        "epoch": epoch,
                        "effective_epoch": args.epoch_seed_offset + epoch,
                        "global_step": global_step,
                        "score": score,
                        "scaler": scaler,
                        "normalization": normalization,
                        "feature_manifest": feature_manifest,
                        "warm_start": warm_start,
                        "cic_anchor": cic_anchor,
                        "backbone_start": backbone_start,
                        "checkpoint_zero_parity": zero_residual_parity,
                    },
                    output / "best_checkpoint.pt",
                )
                np.save(output / "best_validation_parent_node_id.npy", val_parent)
                np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                atomic_json(output / "best_validation_report.json", report)

            if improved(score, early_best_score, args.min_delta):
                early_best_score = score
                stale_epochs = 0
            else:
                stale_epochs += 1

            next_epoch = epoch + 1
            next_order = (
                epoch_order(
                    training_core,
                    seed=args.seed,
                    epoch=args.epoch_seed_offset + next_epoch,
                    core_weight=training_core_weight,
                )
                if next_epoch <= args.epochs else order
            )
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # Storing epochs+1 after the final validation makes a
                    # resume idempotent even if the allocation disappears
                    # before the summary/marker is written.
                    epoch=next_epoch,
                    cursor=0,
                    order=next_order,
                    global_step=global_step,
                    accumulator=EpochLossAccumulator(),
                    shell_numerator=np.zeros(4),
                    shell_denominator=np.zeros(4),
                    shell_rows=np.zeros(4, dtype=np.int64),
                    history=history,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    early_best_score=early_best_score,
                    stale_epochs=stale_epochs,
                    maximum_memory=maximum_memory,
                    args=args,
                ),
                checkpoint_path,
            )
            if (
                not args.canary
                and not args.disable_early_stopping
                and should_stop(
                    epoch=epoch,
                    stale_epochs=stale_epochs,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                )
            ):
                stopped_early = True
                break
    finally:
        if args.model in ("unet", "unet_cic_residual") and adapter is not None:
            adapter.close()

    final_epoch = history[-1]["epoch"] if history else start_epoch - 1
    extension_delta = (
        float(best_score - warm_start["parent_score"])
        if warm_start is not None and np.isfinite(best_score)
        else None
    )
    if args.canary:
        status = "CANARY_COMPLETE"
    elif stopped_early:
        status = "CONVERGED_EARLY_STOP"
    elif warm_start is not None and args.disable_early_stopping:
        if best_epoch >= final_epoch - 2:
            status = "NOT_CONVERGED_EXTENSION_CAP"
        elif extension_delta is not None and extension_delta < args.min_delta:
            status = "EXTENSION_COMPLETE_NO_MATERIAL_GAIN"
        else:
            status = "EXTENSION_COMPLETE"
    else:
        status = "NOT_CONVERGED_MAX_EPOCHS"
    final = {
        **run_manifest,
        "status": status,
        "epochs_completed": int(final_epoch),
        "global_steps": int(global_step),
        "best_epoch": int(best_epoch),
        "best_primary_macro_r2_lambda1": float(best_score),
        "extension_delta_from_parent": extension_delta,
        "history": history,
        "maximum_cuda_memory_bytes": int(maximum_memory),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
    }
    atomic_json(output / "recovery_summary.json", final)
    marker = "CANARY_COMPLETE" if args.canary else status
    (output / marker).write_text(
        f"model={args.model} rotation={args.rotation} seed={args.seed} "
        f"epoch={final_epoch} best={best_score:.8f}\n"
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
