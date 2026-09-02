#!/usr/bin/env python3
"""Checkpoint-resumable Gaussian or v-diffusion training on frozen P12-F1b rows."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f_gaussian_controls import (
    ConditionalGaussianUNet,
    gaussian_nll,
)
from workflows.sbi.p12f_score_diffusion import (
    ConditionalVDiffusionUNet,
    diffusion_training_pair,
)
from workflows.sbi.p12f_train_conditional_field_flow import (
    FieldTargetStore,
    fit_target_scaler,
    selected_core_contract,
    target_scaler_core_contract,
    target_tensor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
DEFAULT_CONTRACT = Path(
    "/global/homes/d/dkololgi/p11_contracts/"
    "training_contract_r1_random_repair_v2_20260901"
)
DEFAULT_PHASE_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
DEFAULT_OUTPUT = DEFAULT_PHASE_ROOT / "p12f_matched_challengers_v1"
METHODS = ("gaussian", "diffusion")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def matched_flow_config(config: dict) -> dict:
    parent_path = REPO_ROOT / config["parent_flow_config"]
    parent = json.loads(parent_path.read_text())
    if parent.get("schema_version") != "p12f-conditional-field-flow-canary-v1":
        raise RuntimeError("P12-F1b parent config schema changed")
    matched = config["matched_contract"]
    output = deepcopy(parent)
    output["canary"].update(
        {
            "training_cores_per_phase": int(matched["training_cores_per_phase"]),
            "nested_reference_training_cores_per_phase": 64,
            "target_scaler_cores_per_phase": int(
                matched["target_scaler_cores_per_phase"]
            ),
            "validation_cores": int(matched["technical_panel_cores"]),
            "updates": int(matched["updates"]),
            "posterior_draws": int(matched["posterior_draws"]),
        }
    )
    output["roles"] = config["roles"]
    return output


def build_model(method: str, *, base: int, device: str) -> torch.nn.Module:
    if method == "gaussian":
        return ConditionalGaussianUNet(condition_channels=3, base=base).to(device)
    if method == "diffusion":
        return ConditionalVDiffusionUNet(condition_channels=3, base=base).to(device)
    raise ValueError(f"unsupported challenger method {method}")


def challenger_loss(
    method: str,
    model: torch.nn.Module,
    condition: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    core_slice: tuple[slice, slice, slice],
) -> tuple[torch.Tensor, dict]:
    core = (slice(None), slice(None)) + core_slice
    if support.shape != target.shape or support.dtype != torch.bool:
        raise ValueError("exact support must be a boolean tensor aligned to target")
    science = torch.zeros_like(support)
    science[core] = support[core]
    if not torch.any(science):
        raise RuntimeError("P12-F training core has no M=1 science voxels")
    if method == "gaussian":
        mean, log_std = model(condition)
        loss = gaussian_nll(mean, log_std, target, science)
        return loss, {
            "mean_abs_log_std": float(torch.mean(torch.abs(log_std[core])).detach().cpu())
        }
    if method == "diffusion":
        # Use the global Torch RNG: it is part of the exact checkpoint contract.
        noisy, time_value, target_v, _ = diffusion_training_pair(target)
        predicted_v = model(noisy, time_value, condition)
        loss = torch.mean(torch.square(predicted_v[science] - target_v[science]))
        return loss, {"mean_time": float(time_value.mean().detach().cpu())}
    raise ValueError(f"unsupported challenger method {method}")


def epoch_refs(selected: dict[str, np.ndarray], phases: tuple[str, ...], *, seed: int, epoch: int):
    refs = [
        (phase, int(core_id))
        for phase in phases
        for core_id in np.asarray(selected[phase], dtype=np.int64)
    ]
    np.random.default_rng(seed + 1009 * epoch).shuffle(refs)
    return refs


def checkpoint_payload(
    *,
    method: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    frozen_digest: str,
    scaler: dict,
    loss_sum: float,
    loss_count: int,
) -> dict:
    return {
        "schema_version": "p12f-matched-challenger-checkpoint-v1",
        "method": method,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "frozen_digest": frozen_digest,
        "target_scaler": scaler,
        "loss_sum": float(loss_sum),
        "loss_count": int(loss_count),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "ph001_opened": False,
    }


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("matched P12-F challenger requires a CUDA allocation")
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f-matched-challengers-v1":
        raise RuntimeError("unsupported matched-challenger config")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F blind phase contract changed")
    phases = tuple(config["roles"]["training"])
    validation = config["roles"]["validation_and_selection"]
    if "ph001" in phases + (validation,):
        raise PermissionError("ph001 entered matched challenger roles")
    flow_config = matched_flow_config(config)
    output = args.output_root / args.run_name / args.method
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    terminal = output / "P12F_TRAINED.json"
    if terminal.exists():
        existing = json.loads(terminal.read_text())
        if existing.get("pass") and not existing.get("ph001_opened"):
            print(json.dumps(existing, indent=2), flush=True)
            return
    if any(output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty challenger output requires --resume or a new run name")

    torch.manual_seed(int(flow_config["canary"]["seed"]))
    np.random.seed(int(flow_config["canary"]["seed"]))
    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    if tuple(loader.training_phases) != phases or loader.validation_phase != validation:
        raise RuntimeError("matched challenger phase roles disagree with loader")
    selected = selected_core_contract(loader, flow_config)
    scaler_selected = target_scaler_core_contract(loader, selected, flow_config)
    store = FieldTargetStore(args.phase_root, phases + (validation,))
    scaler = fit_target_scaler(loader, store, scaler_selected, flow_config)
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/sbi/p12f_gaussian_controls.py",
        REPO_ROOT / "workflows/sbi/p12f_score_diffusion.py",
        REPO_ROOT / "workflows/sbi/p12f_train_conditional_field_flow.py",
        args.config.resolve(),
        (REPO_ROOT / config["parent_flow_config"]).resolve(),
    )
    source_hashes = {str(path): sha256(path) for path in source_paths}
    target_markers = {
        phase: sha256(
            args.phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
        )
        for phase in phases + (validation,)
    }
    frozen = {
        "config_sha256": sha256(args.config),
        "parent_config_sha256": sha256(REPO_ROOT / config["parent_flow_config"]),
        "training_ready_sha256": sha256(
            args.contract_root / "TRAINING_LOADER_READY.json"
        ),
        "source_hashes": source_hashes,
        "target_markers": target_markers,
        "selected_core_ids": {
            phase: [int(value) for value in selected[phase]]
            for phase in phases + (validation,)
        },
        "scaler_core_ids": {
            phase: [int(value) for value in scaler_selected[phase]] for phase in phases
        },
        "target_scaler": scaler,
        "method": args.method,
    }
    frozen_digest = sha256_bytes(json.dumps(frozen, sort_keys=True).encode())
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("frozen_digest") != frozen_digest:
            raise RuntimeError("matched challenger resume contract changed")
    else:
        atomic_json(
            manifest_path,
            {
                "schema_version": "p12f-matched-challenger-run-v1",
                "created_utc": utc_now(),
                "git_revision_at_launch": git_revision(),
                "frozen_digest": frozen_digest,
                "frozen": frozen,
                "ph001_opened": False,
            },
        )
    model = build_model(
        args.method,
        base=int(config["matched_contract"]["unet_base"]),
        device=args.device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(flow_config["canary"]["learning_rate"]),
        weight_decay=float(flow_config["canary"]["weight_decay"]),
    )
    update = 0
    loss_sum = 0.0
    loss_count = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("resume requested without checkpoint")
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if state.get("frozen_digest") != frozen_digest or state.get("method") != args.method:
            raise RuntimeError("challenger checkpoint contract mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        update = int(state["update"])
        loss_sum = float(state["loss_sum"])
        loss_count = int(state["loss_count"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    total_updates = int(config["matched_contract"]["updates"])
    refs_per_epoch = sum(len(selected[phase]) for phase in phases)
    halo = int(flow_config["patch"]["context_halo_voxels"])
    alignment = int(flow_config["patch"]["alignment_voxels"])
    normalization = loader.field_normalization
    trace_path = output / "loss_trace.jsonl"
    started = time.monotonic()
    last_gradient = None
    while update < total_updates:
        epoch = update // refs_per_epoch
        refs = epoch_refs(
            selected, phases, seed=int(flow_config["canary"]["seed"]) + 440, epoch=epoch
        )
        phase, core_id = refs[update % refs_per_epoch]
        adapter = loader.field_adapter(phase)
        patch = adapter.extract(core_id, halo, CHANNELS, alignment_voxels=alignment)
        condition, _ = model_inputs(patch, normalization, args.device)
        target_patch = store.extract(phase, patch)
        target = target_tensor(target_patch["delta"], scaler, args.device)
        support = torch.as_tensor(
            target_patch["support"][None, None],
            device=args.device,
            dtype=torch.bool,
        )
        loss, auxiliary = challenger_loss(
            args.method,
            model,
            condition,
            target,
            support,
            patch.core_slice,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(flow_config["canary"]["gradient_clip"])
            ).detach().cpu()
        )
        optimizer.step()
        update += 1
        loss_value = float(loss.detach().cpu())
        loss_sum += loss_value
        loss_count += 1
        finite = bool(
            np.isfinite(loss_value)
            and np.isfinite(last_gradient)
            and all(torch.isfinite(parameter).all() for parameter in model.parameters())
        )
        if not finite:
            raise RuntimeError(f"non-finite {args.method} state at update {update}")
        if update == 1 or update % 25 == 0 or update == total_updates:
            row = {
                "update": update,
                "method": args.method,
                "phase": phase,
                "core_id": core_id,
                "loss": loss_value,
                "preclip_gradient_norm": last_gradient,
                "elapsed_seconds": time.monotonic() - started,
                **auxiliary,
            }
            with trace_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        should_checkpoint = (
            update % args.checkpoint_every == 0
            or update == total_updates
            or time.monotonic() - started >= args.max_wall_seconds
        )
        if should_checkpoint:
            atomic_checkpoint(
                checkpoint_path,
                checkpoint_payload(
                    method=args.method,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    frozen_digest=frozen_digest,
                    scaler=scaler,
                    loss_sum=loss_sum,
                    loss_count=loss_count,
                ),
            )
        if time.monotonic() - started >= args.max_wall_seconds and update < total_updates:
            atomic_json(
                output / "P12F_PAUSED.json",
                {
                    "schema_version": "p12f-matched-challenger-paused-v1",
                    "method": args.method,
                    "update": update,
                    "frozen_digest": frozen_digest,
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "ph001_opened": False,
                    "pass": True,
                },
            )
            store.close()
            loader.close()
            raise SystemExit(75)

    reloaded = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    reload_model = build_model(
        args.method,
        base=int(config["matched_contract"]["unet_base"]),
        device=args.device,
    )
    reload_model.load_state_dict(reloaded["model"], strict=True)
    parity = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), reload_model.state_dict().values())
    )
    if not parity:
        raise RuntimeError("matched challenger checkpoint reload parity failed")
    marker = {
        "schema_version": "p12f-matched-challenger-trained-v1",
        "created_utc": utc_now(),
        "method": args.method,
        "updates": total_updates,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "frozen_digest": frozen_digest,
        "mean_training_loss": loss_sum / max(loss_count, 1),
        "checkpoint_reload_parity": parity,
        "last_preclip_gradient_norm": last_gradient,
        "ready_for_common_evaluator": True,
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(terminal, marker)
    store.close()
    loader.close()
    print(json.dumps(marker, indent=2), flush=True)


def sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
