#!/usr/bin/env python3
"""Train and freeze the P12-F3 low-mode conditional Gaussian controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_conditional_models import (
    ALL_PATCH_CHANNELS,
    ARMS,
    ConditionalLowModeGaussianUNet,
    conditional_gaussian_nll,
    low_mode_target,
    proxy_condition,
    science_mask,
)
from workflows.sbi.p12f3_fourier_modes import build_fourier_layout
from workflows.sbi.p12f3_train_fourier_lowmode_flow import load_config as load_parent_config
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import epoch_references, load_g1_model, target_tensor
from workflows.sbi.p12f_gaussian_controls import (
    finalize_residual_filter,
    residual_filter_accumulator,
    update_residual_filter_accumulator,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_conditional_calibration_v1.json"
DEFAULT_OUTPUT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_conditional_calibration_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("train", "fit-covariance"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-name", default="seed42_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_config(path: Path) -> tuple[dict, dict, Path]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "p12f3-conditional-calibration-v1":
        raise RuntimeError("unsupported conditional Gaussian contract")
    if (
        config["roles"]["validation"] != "ph006"
        or config["roles"]["sealed_blind_test"] != "ph001"
        or "ph001" in config["roles"]["training"]
        or config["scope"].get("ph001_opened")
        or "ph001" in json.dumps(config["sources"]).lower()
    ):
        raise PermissionError("conditional Gaussian phase boundary changed")
    parent_path = REPO_ROOT / config["sources"]["parent_config"]
    parent = load_parent_config(parent_path)
    return config, parent, parent_path


def split_selected(selected: dict[str, list[int]], phases: tuple[str, ...], fraction: float, seed: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    training: dict[str, list[int]] = {}
    validation: dict[str, list[int]] = {}
    for phase_index, phase in enumerate(phases):
        values = np.asarray(selected[phase], dtype=np.int64)
        order = np.random.default_rng(seed + 7919 * phase_index).permutation(len(values))
        count = max(1, int(round(len(values) * fraction)))
        held = set(map(int, values[order[:count]]))
        validation[phase] = [int(value) for value in values if int(value) in held]
        training[phase] = [int(value) for value in values if int(value) not in held]
    return training, validation


def shuffle_seed(base: int, phase: str, core_id: int) -> int:
    return int(base + 100_003 * int(phase[2:]) + core_id)


def build_example(*, loader, store, g1_model, scaler: dict, phase: str, core_id: int, config: dict, parent: dict, arm: str, device: str):
    patch = loader.field_adapter(phase).extract(
        core_id,
        int(parent["patch"]["conditioning_halo_voxels"]),
        ALL_PATCH_CHANNELS,
        alignment_voxels=int(parent["patch"]["alignment_voxels"]),
    )
    condition, g1_mean, _ = proxy_condition(
        patch, loader.field_normalization, g1_model, device=device, arm=arm,
        shuffle_seed=shuffle_seed(int(config["training"]["seed"]), phase, core_id) if arm == "proxy7_shuffled" else None,
    )
    target_data = store.extract(phase, patch)
    target = target_tensor(target_data["delta"], scaler, device)
    residual = target - g1_mean
    layout = build_fourier_layout(
        tuple(residual.shape[-3:]),
        voxel_mpc_h=float(config["target"]["voxel_mpc_h"]),
        band_edges_h_mpc=tuple(float(value) for value in config["target"]["band_edges_h_mpc"]),
    )
    target_low = low_mode_target(residual, layout)
    mask = science_mask(target_data["support"], patch.core_slice, device)
    return condition, target_low, mask, layout, patch, target_data


def source_contract(config: dict, config_path: Path, parent_path: Path, selected: dict, phases: tuple[str, ...], arm: str) -> dict:
    sources = {name: Path(value) for name, value in config["sources"].items() if name not in ("parent_config",)}
    return {
        "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "parent_config": str(parent_path.resolve()), "parent_config_sha256": sha256(parent_path),
        "source_hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(REPO_ROOT / "workflows/sbi/p12f3_conditional_models.py"): sha256(REPO_ROOT / "workflows/sbi/p12f3_conditional_models.py"),
        },
        "conditioning_ready_sha256": sha256(sources["conditioning_contract"] / "TRAINING_LOADER_READY.json"),
        "g1_checkpoint_sha256": sha256(sources["g1_checkpoint"]),
        "target_markers": {
            phase: sha256(sources["phase_root"] / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json")
            for phase in phases + (config["roles"]["validation"],)
        },
        "selected_core_ids": selected,
        "arm": arm,
        "ph001_opened": False,
    }


def checkpoint_payload(model, optimizer, update: int, frozen_digest: str, loss_sum: float, loss_count: int) -> dict:
    return {
        "schema_version": "p12f3-conditional-gaussian-checkpoint-v1",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "update": int(update), "frozen_digest": frozen_digest,
        "loss_sum": float(loss_sum), "loss_count": int(loss_count),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "ph001_opened": False,
    }


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


@torch.inference_mode()
def validation_nll(model, refs: dict[str, list[int]], *, loader, store, g1_model, scaler, config, parent, arm, device) -> dict:
    model.eval()
    total = 0.0; count = 0; by_phase: dict[str, float] = {}
    for phase, core_ids in refs.items():
        subtotal = 0.0
        for core_id in core_ids:
            condition, target, mask, _, _, _ = build_example(
                loader=loader, store=store, g1_model=g1_model, scaler=scaler,
                phase=phase, core_id=core_id, config=config, parent=parent, arm=arm, device=device,
            )
            location, log_scale = model(condition)
            value = float(conditional_gaussian_nll(location, log_scale, target, mask).cpu())
            subtotal += value; total += value; count += 1
        by_phase[phase] = subtotal / max(len(core_ids), 1)
    return {"mean_nll": total / max(count, 1), "cores": count, "by_phase": by_phase}


def train(args: argparse.Namespace, config: dict, parent: dict, parent_path: Path) -> None:
    _, _, phases, _, _, loader, store, selected = _open_common(parent)
    training, internal = split_selected(
        selected, phases, float(config["training"]["internal_validation_fraction_per_phase"]), int(config["training"]["seed"]),
    )
    output = args.output_root / "gaussian" / args.arm / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    terminal_path = output / "P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json"
    canary_path = output / "TECHNICAL_CANARY_COMPLETE.json"
    if terminal_path.exists():
        print(terminal_path.read_text(), flush=True); store.close(); loader.close(); return
    frozen = source_contract(config, args.config, parent_path, selected, phases, args.arm)
    frozen["training_core_ids"] = training; frozen["internal_validation_core_ids"] = internal
    frozen_digest = digest(frozen)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("frozen_digest") != frozen_digest:
            raise RuntimeError("conditional Gaussian resume contract changed")
    elif any(output.iterdir()):
        raise RuntimeError("non-empty conditional Gaussian output has no valid manifest")
    else:
        atomic_json(manifest_path, {
            "schema_version": "p12f3-conditional-gaussian-run-v1", "created_utc": utc_now(),
            "git_revision_at_launch": git_revision(), "frozen_digest": frozen_digest,
            "frozen": frozen, "truth_files_read": [f"{phase} training delta_R7" for phase in phases],
            "ph001_opened": False,
        })
    torch.manual_seed(int(config["training"]["seed"])); np.random.seed(int(config["training"]["seed"]))
    g1_model, scaler = load_g1_model(parent, args.device)
    model = ConditionalLowModeGaussianUNet(base=int(config["gaussian_control"]["unet_base"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    update = 0; loss_sum = 0.0; loss_count = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if state.get("frozen_digest") != frozen_digest or state.get("ph001_opened"):
            raise RuntimeError("unsafe conditional Gaussian checkpoint")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        update = int(state["update"]); loss_sum = float(state["loss_sum"]); loss_count = int(state["loss_count"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        atomic_checkpoint(checkpoint_path, checkpoint_payload(model, optimizer, 0, frozen_digest, 0.0, 0))
    total = int(config["training"]["gaussian_selection_updates"])
    stop = total if args.stop_after_updates is None else int(args.stop_after_updates)
    if stop <= update or stop > total:
        raise ValueError("invalid conditional Gaussian stop update")
    refs_per_epoch = sum(len(training[phase]) for phase in phases)
    trace_path = output / "loss_trace.jsonl"
    started = time.monotonic(); last_gradient = float("nan")
    try:
        while update < stop:
            epoch = update // refs_per_epoch; ordinal = update % refs_per_epoch
            phase, core_id = epoch_references(training, phases, seed=int(config["training"]["seed"]), epoch=epoch)[ordinal]
            condition, target, mask, _, _, _ = build_example(
                loader=loader, store=store, g1_model=g1_model, scaler=scaler,
                phase=phase, core_id=core_id, config=config, parent=parent, arm=args.arm, device=args.device,
            )
            model.train(); location, log_scale = model(condition)
            loss = conditional_gaussian_nll(location, log_scale, target, mask)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            last_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])).detach().cpu())
            if not torch.isfinite(loss) or not np.isfinite(last_gradient):
                raise FloatingPointError("conditional Gaussian loss/gradient is not finite")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("conditional Gaussian parameter is not finite")
            update += 1; value = float(loss.detach().cpu()); loss_sum += value; loss_count += 1
            if update % int(config["training"]["loss_log_every_updates"]) == 0 or update == stop:
                with trace_path.open("a") as stream:
                    stream.write(json.dumps({
                        "update": update, "epoch_equivalent": update / refs_per_epoch,
                        "loss": value, "mean_loss": loss_sum / loss_count,
                        "preclip_gradient_norm": last_gradient, "phase": phase,
                        "core_id": core_id, "elapsed_seconds": time.monotonic() - started,
                    }, sort_keys=True) + "\n")
            if update % int(config["training"]["checkpoint_every_updates"]) == 0 or update == stop:
                atomic_checkpoint(checkpoint_path, checkpoint_payload(model, optimizer, update, frozen_digest, loss_sum, loss_count))
            if time.monotonic() - started >= args.max_wall_seconds and update < stop:
                atomic_json(output / "PAUSED.json", {"schema_version":"p12f3-conditional-gaussian-pause-v1","update":update,"frozen_digest":frozen_digest,"ph001_opened":False})
                raise SystemExit(75)
        if stop == int(config["training"]["technical_canary_updates"]):
            marker = {
                "schema_version":"p12f3-conditional-gaussian-canary-v1", "pass":True,
                "arm":args.arm,"update":update,"finite":True,"frozen_digest":frozen_digest,
                "checkpoint":str(checkpoint_path.resolve()),"checkpoint_sha256":sha256(checkpoint_path),"ph001_opened":False,
            }
            atomic_json(canary_path, marker); print(json.dumps(marker, indent=2), flush=True); return
        validation = validation_nll(
            model, internal, loader=loader, store=store, g1_model=g1_model, scaler=scaler,
            config=config, parent=parent, arm=args.arm, device=args.device,
        )
        marker = {
            "schema_version":"p12f3-conditional-gaussian-trained-v1", "created_utc":utc_now(),
            "pass":True,"arm":args.arm,"updates":update,"mean_training_nll":loss_sum/max(loss_count,1),
            "internal_validation":validation,"last_preclip_gradient_norm":last_gradient,
            "checkpoint":str(checkpoint_path.resolve()),"checkpoint_sha256":sha256(checkpoint_path),
            "frozen_digest":frozen_digest,"target_scaler":scaler,"ph006_used_for_fit":False,"ph001_opened":False,
        }
        atomic_json(terminal_path, marker); print(json.dumps(marker, indent=2), flush=True)
    finally:
        store.close(); loader.close()


def fit_covariance(args: argparse.Namespace, config: dict, parent: dict, parent_path: Path) -> None:
    _, _, phases, _, _, loader, store, selected = _open_common(parent)
    training, internal = split_selected(
        selected,
        phases,
        float(config["training"]["internal_validation_fraction_per_phase"]),
        int(config["training"]["seed"]),
    )
    output = args.output_root / "gaussian" / args.arm / args.run_name
    filter_path = output / "conditional_residual_filter.json"
    if filter_path.exists():
        print(filter_path.read_text(), flush=True); store.close(); loader.close(); return
    marker_path = output / "P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json"
    marker = json.loads(marker_path.read_text())
    checkpoint_path = Path(marker["checkpoint"])
    if not marker.get("pass") or marker.get("ph001_opened") or marker.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise RuntimeError("conditional covariance received unsafe location/scale model")
    g1_model, scaler = load_g1_model(parent, args.device)
    model = ConditionalLowModeGaussianUNet(base=int(config["gaussian_control"]["unet_base"])).to(args.device)
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"]); model.eval().requires_grad_(False)
    accumulator = residual_filter_accumulator(32)
    refs = [(phase, int(core)) for phase in phases for core in training[phase]]
    supported = 0
    with torch.inference_mode():
        for ordinal, (phase, core_id) in enumerate(refs):
            condition, target, _, _, _, target_data = build_example(
                loader=loader, store=store, g1_model=g1_model, scaler=scaler,
                phase=phase, core_id=core_id, config=config, parent=parent, arm=args.arm, device=args.device,
            )
            location, log_scale = model(condition)
            standardized = ((target - location) * torch.exp(-log_scale))[0, 0].cpu().numpy()
            support = np.asarray(target_data["support"], dtype=bool)
            supported += int(support.sum())
            update_residual_filter_accumulator(accumulator, np.where(support, standardized, 0.0))
            if ordinal == 0 or (ordinal + 1) % 100 == 0 or ordinal + 1 == len(refs):
                print(json.dumps({"stage":"fit-covariance","arm":args.arm,"core":ordinal+1,"total":len(refs)}), flush=True)
    contract = finalize_residual_filter(accumulator)
    contract.update({
        "schema_version":"p12f3-conditional-standardized-residual-filter-v1",
        "created_utc":utc_now(),"pass":True,"arm":args.arm,
        "location_scale_checkpoint":str(checkpoint_path.resolve()),"location_scale_checkpoint_sha256":sha256(checkpoint_path),
        "config":str(args.config.resolve()),"config_sha256":sha256(args.config),
        "training_phases":list(phases),"training_core_ids":training,
        "internal_validation_core_ids_excluded":internal,"supported_voxels":supported,
        "validation_phase_used_for_fit":False,"truth_files_read":[f"{phase} training delta_R7" for phase in phases],
        "ph001_opened":False,
    })
    atomic_json(filter_path, contract); print(json.dumps(contract, indent=2), flush=True)
    store.close(); loader.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("conditional Gaussian work requires a CUDA allocation")
    config, parent, parent_path = load_config(args.config)
    if args.stage == "train": train(args, config, parent, parent_path)
    else: fit_covariance(args, config, parent, parent_path)


if __name__ == "__main__":
    main()
