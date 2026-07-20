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
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p8_train_graph_patch as graph_impl
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
    parser.add_argument("--model", choices=("graph", "unet"), required=True)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="control_v1")
    parser.add_argument("--resume", action="store_true")
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
    if args.patience <= 0 or args.loss_log_every <= 0 or args.checkpoint_every <= 0:
        parser.error("patience and logging/checkpoint intervals must be positive")
    if not args.canary and args.min_epochs < 5:
        parser.error("scientific recovery runs require --min-epochs >= 5")
    if args.canary and args.epochs != 1:
        parser.error("a canary must use exactly one complete epoch")
    return args


def main() -> None:
    args = parse_args()
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
    else:
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
        "objective": "complete-core epoch; sum(w_i mse_i)/mean(training core weight)",
        "canary": args.canary,
        "resume": args.resume,
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
                    epoch=epoch,
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
                else:
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
            else:
                val_parent, val_scaled, failures = unet_impl.predict_fold(
                    model,
                    adapter,
                    validation_core,
                    normalization,
                    args.device,
                )
                validation_runtime = {}
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
                        "global_step": global_step,
                        "score": score,
                        "scaler": scaler,
                        "normalization": normalization,
                        "feature_manifest": feature_manifest,
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
                    epoch=next_epoch,
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
            if not args.canary and should_stop(
                epoch=epoch,
                stale_epochs=stale_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
            ):
                stopped_early = True
                break
    finally:
        if args.model == "unet" and adapter is not None:
            adapter.close()

    final_epoch = history[-1]["epoch"] if history else start_epoch - 1
    if args.canary:
        status = "CANARY_COMPLETE"
    elif stopped_early:
        status = "CONVERGED_EARLY_STOP"
    else:
        status = "NOT_CONVERGED_MAX_EPOCHS"
    final = {
        **run_manifest,
        "status": status,
        "epochs_completed": int(final_epoch),
        "global_steps": int(global_step),
        "best_epoch": int(best_epoch),
        "best_primary_macro_r2_lambda1": float(best_score),
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
