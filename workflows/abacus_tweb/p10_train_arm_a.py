#!/usr/bin/env python3
"""Train deterministic P10 Arm-A U-PATCH or G-PATCH across five phases.

The trainer consumes the frozen ``TRAINING_LOADER_READY`` contract.  Every
scientific epoch visits every eligible core in ph000/ph002--ph005 exactly once,
uses an equal-phase square-root-shell-weighted objective, and evaluates exactly
all authoritative ph006 rows.  ph001 is never opened.

Runs are initialized from scratch and are exactly resumable at a patch cursor.
``--max-runtime-seconds`` permits a sequence of bounded interactive allocations;
exit code 75 means that an atomic checkpoint was written and another allocation
may continue the same optimizer trajectory.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
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
from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    acquire_run_lock,
    atomic_json,
    evaluate_complete_phase,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    append_jsonl,
    improved,
    reconcile_loss_trace,
    rewrite_jsonl,
    should_stop,
)
from workflows.abacus_tweb.p8_train_patch_recovery import (
    atomic_torch_save,
    git_revision,
    torch_load,
)
from workflows.abacus_tweb.p10_training_contract import (
    P10PhaseBalancedLoader,
    PatchRef,
    epoch_hash,
    phase_equal_patch_objective,
    resume_state,
    validate_resume_state,
)


CONTRACT_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract"
)
OUTPUT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training")
CONTINUE_EXIT_CODE = 75
TARGET_CHUNK = 1_000_000


@dataclass
class PhaseRuntime:
    phase: str
    target_scaled: np.ndarray
    parent_weight: np.ndarray
    parent_shell: np.ndarray
    expected_core_weight: dict[int, float]
    weight_denominator: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable_arguments(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def source_contract() -> dict[str, str]:
    paths = (
        Path(__file__),
        REPO_ROOT / "workflows/abacus_tweb/p10_training_contract.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_train_graph_patch.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_train_unet_patch.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_deterministic_common.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_epoch_training.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def frozen_arguments(args: argparse.Namespace) -> dict:
    """Arguments that can change weights or scientific model selection."""
    ignored = {
        "auto_resume",
        "checkpoint_every",
        "device",
        "loss_log_every",
        "max_runtime_seconds",
        "validation_reserve_seconds",
    }
    return {
        key: value
        for key, value in jsonable_arguments(args).items()
        if key not in ignored
    }


def scaled_targets(eigenvalues: np.ndarray, scaler: dict) -> np.ndarray:
    result = np.empty((len(eigenvalues), 3), dtype=np.float32)
    for start in range(0, len(eigenvalues), TARGET_CHUNK):
        stop = min(start + TARGET_CHUNK, len(eigenvalues))
        result[start:stop] = scale_increments(
            linear_increments(np.asarray(eigenvalues[start:stop])), scaler
        )
    return result


def prepare_phase_runtime(
    loader: P10PhaseBalancedLoader,
    phase: str,
    scaler: dict,
    *,
    training: bool,
) -> PhaseRuntime:
    record = loader.phase_records[phase]
    assignment = np.load(Path(record["inputs"]["assignment"]), mmap_mode="r")
    truth = loader.targets_by_parent(phase)
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    row_weight = np.asarray(loader.row_weights(phase), dtype=np.float32)
    if len(row_weight) != len(parent):
        raise RuntimeError(f"{phase} row-weight/assignment length mismatch")
    parent_weight = np.zeros(len(truth), dtype=np.float32)
    parent_weight[parent] = row_weight
    parent_shell = np.full(len(truth), -1, dtype=np.int8)
    parent_shell[parent] = np.asarray(assignment["shell"], dtype=np.int8)
    assignment.close()

    expected: dict[int, float] = {}
    if training:
        phase_root = loader.root / "phases" / phase
        core_id = np.load(phase_root / "training_core_id.npy")
        core_weight = np.load(phase_root / "training_core_weight.npy")
        expected = {
            int(core): float(weight)
            for core, weight in zip(core_id, core_weight)
        }
    return PhaseRuntime(
        phase=phase,
        target_scaled=scaled_targets(truth, scaler),
        parent_weight=parent_weight,
        parent_shell=parent_shell,
        expected_core_weight=expected,
        weight_denominator=float(record["phase_weight_denominator"]),
    )


def fresh_accumulators(phases: tuple[str, ...]) -> dict[str, EpochLossAccumulator]:
    return {phase: EpochLossAccumulator() for phase in phases}


def accumulators_as_dict(
    accumulators: dict[str, EpochLossAccumulator],
) -> dict[str, dict]:
    return {phase: accumulator.as_dict() for phase, accumulator in accumulators.items()}


def accumulators_from_dict(
    phases: tuple[str, ...], rows: dict | None
) -> dict[str, EpochLossAccumulator]:
    rows = rows or {}
    return {
        phase: EpochLossAccumulator.from_dict(rows.get(phase))
        for phase in phases
    }


def should_pause(started: float, args: argparse.Namespace, *, reserve: float = 0.0) -> bool:
    if args.max_runtime_seconds is None:
        return False
    return time.monotonic() - started >= args.max_runtime_seconds - reserve


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    refs: tuple[PatchRef, ...],
    epoch: int,
    cursor: int,
    global_step: int,
    phase_accumulators: dict[str, EpochLossAccumulator],
    shell_numerator: np.ndarray,
    shell_denominator: np.ndarray,
    shell_rows: np.ndarray,
    objective_sum: float,
    objective_steps: int,
    history: list[dict],
    best_score: float,
    best_epoch: int,
    early_best_score: float,
    stale_epochs: int,
    maximum_memory: int,
    args: argparse.Namespace,
    sources: dict[str, str],
) -> dict:
    payload = {
        "schema_version": "p10-arm-a-checkpoint-v1",
        "created_utc": utc_now(),
        "git_revision_at_write": git_revision(),
        "source_contract": sources,
        "model": args.model,
        "seed": int(args.seed),
        "frozen_arguments": frozen_arguments(args),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "resume": resume_state(
            seed=args.seed,
            epoch=epoch,
            cursor=cursor,
            refs=refs,
            loss_accumulator=accumulators_as_dict(phase_accumulators),
        ),
        "global_step": int(global_step),
        "shell_numerator": np.asarray(shell_numerator, dtype=np.float64),
        "shell_denominator": np.asarray(shell_denominator, dtype=np.float64),
        "shell_rows": np.asarray(shell_rows, dtype=np.int64),
        "objective_sum": float(objective_sum),
        "objective_steps": int(objective_steps),
        "history": history,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "early_best_score": float(early_best_score),
        "stale_epochs": int(stale_epochs),
        "maximum_memory": int(maximum_memory),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "graph"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unet-base", type=int, default=24)
    parser.add_argument("--unet-latent-channels", type=int, default=32)
    parser.add_argument("--validation-group-cores", type=int, default=8)
    parser.add_argument("--loss-log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--validation-reserve-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="technical GPU canary only; do not use for a scientific run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="arm_a_r0_v1")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--contract-root", type=Path, default=CONTRACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.epochs <= 0 or args.min_epochs <= 0 or args.min_epochs > args.epochs:
        parser.error("require 0 < min-epochs <= epochs")
    if args.patience <= 0 or args.min_delta < 0:
        parser.error("patience must be positive and min-delta non-negative")
    if args.loss_log_every <= 0 or args.checkpoint_every <= 0:
        parser.error("logging and checkpoint intervals must be positive")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("max-runtime-seconds must be positive")
    if args.validation_reserve_seconds < 0:
        parser.error("validation-reserve-seconds must be non-negative")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        parser.error("stop-after-updates must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P10 Arm A requires a CUDA interactive allocation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = args.output_root / args.run_name / args.model / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_run_lock(output / ".run.lock", purpose="P10 Arm-A training")
    checkpoint_path = output / "arm_a_checkpoint.pt"
    complete_marker = output / "ARM_A_TRAINING_COMPLETE.json"
    if complete_marker.exists():
        print(complete_marker.read_text(), flush=True)
        return
    resume = bool(args.auto_resume and checkpoint_path.exists())
    if any(path.name != ".run.lock" for path in output.iterdir()) and not resume:
        raise RuntimeError(f"non-empty Arm-A output requires --auto-resume: {output}")

    loader = P10PhaseBalancedLoader(args.contract_root, include_blind=False)
    phases = tuple(loader.training_phases)
    validation_phase = loader.validation_phase
    if loader.blind_phase == validation_phase or loader.blind_phase in phases:
        raise RuntimeError("sealed blind phase entered a visible P10 role")
    scaler = loader.target_scaler
    sources = source_contract()
    graph_transform_path = args.contract_root / "transforms/graph/graph_transform.json"
    graph_transform = json.loads(graph_transform_path.read_text())
    if not graph_transform.get("pass"):
        raise RuntimeError("frozen graph transform does not pass")
    edge_spec = graph_transform["edge"]
    normalization = loader.field_normalization

    runtime = {
        phase: prepare_phase_runtime(loader, phase, scaler, training=True)
        for phase in phases
    }
    validation_runtime = prepare_phase_runtime(
        loader, validation_phase, scaler, training=False
    )
    validation_refs = loader.validation_refs()
    validation_core = np.asarray([ref.core_id for ref in validation_refs], dtype=np.int64)
    validation_record = loader.phase_records[validation_phase]
    validation_assignment_path = Path(validation_record["inputs"]["assignment"])

    if args.model == "unet":
        model = unet_impl.UPatch(
            base=args.unet_base,
            latent_channels=args.unet_latent_channels,
        ).to(args.device)
    else:
        model = graph_impl.GraphPatchNet(
            latent_size=args.latent_size,
            heads=args.heads,
            dropout=args.dropout,
        ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch_length = len(loader.training_epoch(seed=args.seed, epoch=1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * epoch_length
    )

    history: list[dict] = []
    best_score = -np.inf
    best_epoch = -1
    early_best_score = -np.inf
    stale_epochs = 0
    global_step = 0
    epoch = 1
    cursor = 0
    phase_accumulators = fresh_accumulators(phases)
    shell_numerator = np.zeros((len(phases), 4), dtype=np.float64)
    shell_denominator = np.zeros((len(phases), 4), dtype=np.float64)
    shell_rows = np.zeros((len(phases), 4), dtype=np.int64)
    objective_sum = 0.0
    objective_steps = 0
    maximum_memory = 0

    if resume:
        state = torch_load(checkpoint_path, args.device)
        if state.get("schema_version") != "p10-arm-a-checkpoint-v1":
            raise RuntimeError("unsupported Arm-A checkpoint schema")
        if state["source_contract"] != sources:
            raise RuntimeError("Arm-A source files changed since the checkpoint")
        if state["frozen_arguments"] != frozen_arguments(args):
            raise RuntimeError("Arm-A scientific arguments changed on resume")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        resume_row = state["resume"]
        epoch = int(resume_row["epoch"])
        cursor = int(resume_row["cursor"])
        refs = loader.training_epoch(seed=args.seed, epoch=epoch)
        validate_resume_state(resume_row, refs)
        phase_accumulators = accumulators_from_dict(
            phases, resume_row.get("loss_accumulator")
        )
        shell_numerator = np.asarray(state["shell_numerator"], dtype=np.float64)
        shell_denominator = np.asarray(state["shell_denominator"], dtype=np.float64)
        shell_rows = np.asarray(state["shell_rows"], dtype=np.int64)
        objective_sum = float(state["objective_sum"])
        objective_steps = int(state["objective_steps"])
        history = list(state["history"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        early_best_score = float(state["early_best_score"])
        stale_epochs = int(state["stale_epochs"])
        global_step = int(state["global_step"])
        maximum_memory = int(state["maximum_memory"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [row.cpu() for row in state["cuda_rng_state_all"]]
            )
        reconcile_loss_trace(output / "loss_trace.jsonl", maximum_global_step=global_step)
        rewrite_jsonl(output / "epoch_history.jsonl", history)
    else:
        (output / "loss_trace.jsonl").write_text("")
        (output / "epoch_history.jsonl").write_text("")

    run_manifest = {
        "schema_version": "p10-arm-a-run-v1",
        "created_utc": utc_now(),
        "stage": "P10 deterministic multi-phase Arm A final-view R0",
        "model": args.model,
        "seed": args.seed,
        "fresh_initialization": True,
        "warm_start": None,
        "training_phases": list(phases),
        "validation_and_selection_phase": validation_phase,
        "sealed_blind_phase": loader.blind_phase,
        "blind_truth_accessed": False,
        "phase_is_model_input": False,
        "training_cores_per_epoch": epoch_length,
        "validation_cores": int(len(validation_core)),
        "sampler": loader.manifest["epoch"],
        "objective": loader.manifest["objective"],
        "view": "V_final R0 frozen P3a/P8 input contract",
        "arguments": jsonable_arguments(args),
        "frozen_arguments": frozen_arguments(args),
        "git_revision_at_launch": git_revision(),
        "source_contract": sources,
        "training_ready_marker": str(args.contract_root / "TRAINING_LOADER_READY.json"),
        "training_ready_marker_sha256": sha256(
            args.contract_root / "TRAINING_LOADER_READY.json"
        ),
        "target_scaler": str(args.contract_root / "transforms/target_scaler.json"),
        "target_scaler_sha256": sha256(
            args.contract_root / "transforms/target_scaler.json"
        ),
        "graph_transform_sha256": sha256(graph_transform_path),
        "field_transform_sha256": sha256(
            args.contract_root / "transforms/field/field_transform.json"
        ),
    }
    atomic_json(output / "run_manifest.json", run_manifest)

    started = time.monotonic()
    loss_window_objective: list[float] = []
    loss_window_phase = {phase: EpochLossAccumulator() for phase in phases}

    def save_checkpoint(refs: tuple[PatchRef, ...], checkpoint_cursor: int) -> None:
        atomic_torch_save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                refs=refs,
                epoch=epoch,
                cursor=checkpoint_cursor,
                global_step=global_step,
                phase_accumulators=phase_accumulators,
                shell_numerator=shell_numerator,
                shell_denominator=shell_denominator,
                shell_rows=shell_rows,
                objective_sum=objective_sum,
                objective_steps=objective_steps,
                history=history,
                best_score=best_score,
                best_epoch=best_epoch,
                early_best_score=early_best_score,
                stale_epochs=stale_epochs,
                maximum_memory=maximum_memory,
                args=args,
                sources=sources,
            ),
            checkpoint_path,
        )

    def pause(refs: tuple[PatchRef, ...], checkpoint_cursor: int, reason: str) -> None:
        save_checkpoint(refs, checkpoint_cursor)
        atomic_json(
            output / "ALLOCATION_PAUSED.json",
            {
                "schema_version": "p10-arm-a-allocation-pause-v1",
                "created_utc": utc_now(),
                "reason": reason,
                "epoch": epoch,
                "cursor": checkpoint_cursor,
                "epoch_length": len(refs),
                "global_step": global_step,
                "resume_command_required": True,
            },
        )
        raise SystemExit(CONTINUE_EXIT_CODE)

    try:
        while epoch <= args.epochs:
            refs = loader.training_epoch(seed=args.seed, epoch=epoch)
            if cursor > len(refs):
                raise RuntimeError("checkpoint cursor exceeds reconstructed epoch")
            if cursor == 0 and sum(a.patches for a in phase_accumulators.values()) != 0:
                raise RuntimeError("new epoch has non-empty phase accumulators")

            for position in range(cursor, len(refs)):
                if should_pause(started, args):
                    pause(refs, position, "interactive allocation runtime budget")
                ref = refs[position]
                phase = ref.phase
                phase_state = runtime[phase]
                model.train()
                if args.model == "graph":
                    adapter = loader.graph_adapter(phase)
                    patch = adapter.extract(
                        ref.core_id,
                        graph_impl.NUM_PASSES,
                        dependency_hops_per_pass=graph_impl.DEPENDENCY_HOPS_PER_PASS,
                        loss_policy="authoritative",
                    )
                    tensors = graph_impl.transformed_patch(
                        patch,
                        edge_spec,
                        int(adapter.core_cap[ref.core_id]),
                        args.device,
                    )
                    parent = patch.parent_node_id[patch.loss_mask]
                    prediction = model(*tensors)[patch.loss_mask]
                else:
                    adapter = loader.field_adapter(phase)
                    patch = adapter.extract(
                        ref.core_id,
                        unet_impl.HALO_VOXELS,
                        unet_impl.CHANNELS,
                        alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
                    )
                    parent = patch.authoritative_parent_id
                    values, points = unet_impl.model_inputs(
                        patch, normalization, args.device
                    )
                    prediction = model(values, points)

                weight_np = np.asarray(
                    phase_state.parent_weight[parent], dtype=np.float32
                )
                actual_weight = float(np.sum(weight_np, dtype=np.float64))
                expected_weight = phase_state.expected_core_weight[ref.core_id]
                if not np.isclose(actual_weight, expected_weight, rtol=2e-6, atol=1e-7):
                    raise RuntimeError(
                        f"{phase} core {ref.core_id} weight mismatch: "
                        f"{actual_weight} != {expected_weight}"
                    )
                target_np = phase_state.target_scaled[parent]
                weight = torch.from_numpy(weight_np).to(args.device)
                target = torch.from_numpy(target_np).to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss_per_row = torch.mean((prediction - target) ** 2, dim=1)
                weighted_numerator = torch.sum(weight * loss_per_row)
                loss = phase_equal_patch_objective(
                    weighted_numerator,
                    phase_weight_denominator=phase_state.weight_denominator,
                    phase_objective_scale=ref.phase_objective_scale,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                loss_np = loss_per_row.detach().cpu().numpy()
                phase_accumulators[phase].add(loss_np, weight_np)
                loss_window_phase[phase].add(loss_np, weight_np)
                objective_value = float(loss.detach().cpu())
                objective_sum += objective_value
                objective_steps += 1
                loss_window_objective.append(objective_value)
                phase_index = phases.index(phase)
                shells = phase_state.parent_shell[parent]
                for shell in range(4):
                    selected = shells == shell
                    if np.any(selected):
                        shell_numerator[phase_index, shell] += float(
                            np.sum(weight_np[selected] * loss_np[selected])
                        )
                        shell_denominator[phase_index, shell] += float(
                            np.sum(weight_np[selected])
                        )
                        shell_rows[phase_index, shell] += int(np.sum(selected))
                maximum_memory = max(
                    maximum_memory, int(torch.cuda.max_memory_allocated())
                )

                if global_step % args.loss_log_every == 0 or position + 1 == len(refs):
                    row = {
                        "epoch": epoch,
                        "cursor": position + 1,
                        "epoch_length": len(refs),
                        "global_step": global_step,
                        "optimizer_objective_window_mean": float(
                            np.mean(loss_window_objective)
                        ),
                        "phase_weighted_mse_window": {
                            phase_name: accumulator.mean
                            for phase_name, accumulator in loss_window_phase.items()
                            if accumulator.patches
                        },
                        "phase_patches_window": {
                            phase_name: accumulator.patches
                            for phase_name, accumulator in loss_window_phase.items()
                        },
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    append_jsonl(output / "loss_trace.jsonl", row)
                    loss_window_objective.clear()
                    loss_window_phase = {
                        phase_name: EpochLossAccumulator() for phase_name in phases
                    }

                if global_step % args.checkpoint_every == 0 or position + 1 == len(refs):
                    save_checkpoint(refs, position + 1)

                if (
                    args.stop_after_updates is not None
                    and global_step >= args.stop_after_updates
                ):
                    save_checkpoint(refs, position + 1)
                    marker = {
                        "schema_version": "p10-arm-a-technical-canary-v1",
                        "created_utc": utc_now(),
                        "model": args.model,
                        "global_step": global_step,
                        "epoch": epoch,
                        "cursor": position + 1,
                        "pass": True,
                    }
                    atomic_json(output / "TECHNICAL_CANARY_COMPLETE.json", marker)
                    print(json.dumps(marker, indent=2), flush=True)
                    return

            cursor = len(refs)
            if any(accumulator.patches == 0 for accumulator in phase_accumulators.values()):
                raise RuntimeError("complete epoch omitted a training phase")
            if sum(accumulator.patches for accumulator in phase_accumulators.values()) != len(refs):
                raise RuntimeError("complete epoch patch accounting mismatch")
            if should_pause(
                started, args, reserve=args.validation_reserve_seconds
            ):
                pause(refs, cursor, "validation deferred to next interactive allocation")

            if args.model == "graph":
                val_parent, val_scaled, failures, val_nodes, val_edges = (
                    graph_impl.predict_fold(
                        model,
                        loader.graph_adapter(validation_phase),
                        validation_core,
                        edge_spec,
                        args.device,
                        args.validation_group_cores,
                    )
                )
                validation_details = {
                    "maximum_patch_nodes": int(val_nodes),
                    "maximum_patch_directed_edges": int(val_edges),
                }
            else:
                val_parent, val_scaled, failures = unet_impl.predict_fold(
                    model,
                    loader.field_adapter(validation_phase),
                    validation_core,
                    normalization,
                    args.device,
                )
                validation_details = {}
            val_eigen = increments_to_eigenvalues(
                unscale_increments(val_scaled, scaler)
            ).astype(np.float32)
            validation_assignment = np.load(validation_assignment_path, mmap_mode="r")
            validation_truth = loader.targets_by_parent(validation_phase)
            report = evaluate_complete_phase(
                parent_node_id=val_parent,
                predicted_eigenvalues=val_eigen,
                truth_by_parent=validation_truth,
                assignment=validation_assignment,
                phase=validation_phase,
                runtime={
                    "epoch": epoch,
                    "global_step": global_step,
                    "patch_failures": failures,
                    "maximum_cuda_memory_bytes": maximum_memory,
                    **validation_details,
                },
            )
            validation_assignment.close()
            validation_row_loss = np.mean(
                (
                    np.asarray(val_scaled, dtype=np.float64)
                    - np.asarray(
                        validation_runtime.target_scaled[val_parent], dtype=np.float64
                    )
                ) ** 2,
                axis=1,
            )
            validation_loss = {
                "all_rows_scaled_mse": float(np.mean(validation_row_loss)),
                "per_shell_scaled_mse": {
                    str(shell): float(
                        np.mean(
                            validation_row_loss[
                                validation_runtime.parent_shell[val_parent] == shell
                            ]
                        )
                    )
                    for shell in range(4)
                },
            }
            phase_shell_mse = np.divide(
                shell_numerator,
                shell_denominator,
                out=np.full_like(shell_numerator, np.nan),
                where=shell_denominator > 0,
            )
            score = float(report["primary_macro_r2_lambda1"])
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_sha256": epoch_hash(refs),
                "complete_epoch_coverage": True,
                "unique_cores_seen": len(refs),
                "repeat_cores": 0,
                "phase_patches": {
                    phase_name: phase_accumulators[phase_name].patches
                    for phase_name in phases
                },
                "phase_weighted_mse": {
                    phase_name: phase_accumulators[phase_name].mean
                    for phase_name in phases
                },
                "phase_shell_weighted_mse": {
                    phase_name: phase_shell_mse[index].tolist()
                    for index, phase_name in enumerate(phases)
                },
                "phase_shell_rows": {
                    phase_name: shell_rows[index].tolist()
                    for index, phase_name in enumerate(phases)
                },
                "equal_phase_optimizer_objective_mean": (
                    objective_sum / objective_steps
                ),
                "validation": validation_loss,
                "primary_macro_r2_lambda1": score,
                "diagnostic_first_three_shell_macro_r2_lambda1": report[
                    "diagnostic_first_three_shell_macro_r2_lambda1"
                ],
                "worst_shell_r2_lambda1": report["worst_shell_r2_lambda1"],
                "per_shell_lambda1_r2": {
                    name: report["per_shell"][name]["lambda1"]["r2"]
                    for name in SHELL_NAMES
                },
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "invocation_elapsed_seconds": time.monotonic() - started,
            }
            history.append(epoch_row)
            append_jsonl(output / "epoch_history.jsonl", epoch_row)
            print(json.dumps(epoch_row), flush=True)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                atomic_torch_save(
                    {
                        "schema_version": "p10-arm-a-best-v1",
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "model": args.model,
                        "seed": args.seed,
                        "epoch": epoch,
                        "global_step": global_step,
                        "score": score,
                        "scaler": scaler,
                        "normalization": normalization if args.model == "unet" else None,
                        "graph_transform": graph_transform if args.model == "graph" else None,
                        "source_contract": sources,
                        "training_phases": list(phases),
                        "validation_phase": validation_phase,
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
            stopped_early = (
                not args.disable_early_stopping
                and should_stop(
                    epoch=epoch,
                    stale_epochs=stale_epochs,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                )
            )
            completed_epoch = epoch
            epoch += 1
            cursor = 0
            phase_accumulators = fresh_accumulators(phases)
            shell_numerator = np.zeros((len(phases), 4), dtype=np.float64)
            shell_denominator = np.zeros((len(phases), 4), dtype=np.float64)
            shell_rows = np.zeros((len(phases), 4), dtype=np.int64)
            objective_sum = 0.0
            objective_steps = 0
            if epoch <= args.epochs:
                next_refs = loader.training_epoch(seed=args.seed, epoch=epoch)
                save_checkpoint(next_refs, 0)
            if stopped_early:
                break

        final_epoch = history[-1]["epoch"] if history else 0
        status = (
            "CONVERGED_EARLY_STOP"
            if final_epoch < args.epochs
            else (
                "NOT_CONVERGED_MAX_EPOCHS"
                if best_epoch >= max(1, final_epoch - 2)
                else "TRAINING_COMPLETE"
            )
        )
        final = {
            **run_manifest,
            "completed_utc": utc_now(),
            "status": status,
            "epochs_completed": int(final_epoch),
            "global_steps": int(global_step),
            "best_epoch": int(best_epoch),
            "best_primary_macro_r2_lambda1": float(best_score),
            "history": history,
            "maximum_cuda_memory_bytes": int(maximum_memory),
            "checkpoint": str(checkpoint_path),
        }
        atomic_json(output / "arm_a_summary.json", final)
        atomic_json(complete_marker, final)
        paused = output / "ALLOCATION_PAUSED.json"
        if paused.exists():
            paused.unlink()
        print(json.dumps(final, indent=2), flush=True)
    finally:
        for adapter in loader._field.values():
            adapter.close()
        lock_handle.close()


if __name__ == "__main__":
    main()
