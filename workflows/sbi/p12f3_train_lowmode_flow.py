#!/usr/bin/env python3
"""Train a matched local/wide P12-F3 conditional low-mode residual flow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import h5py
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f3_hierarchical_lowmode import (
    build_low_mode_model,
    crop_tensor_to_patch,
    prepare_low_mode_example,
    rectified_flow_training_pair,
    sample_heun,
    spectral_split,
)
from workflows.sbi.p12f_gaussian_controls import ConditionalGaussianUNet


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_hierarchical_lowmode_v1.json"
DEFAULT_OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
    "p12f3_hierarchical_lowmode_v1"
)


class FieldTargetStore:
    """Minimal visible-phase target reader without importing the evaluation stack."""

    def __init__(self, phase_root: Path, phases: tuple[str, ...]):
        self.phase_root = Path(phase_root)
        self.manifests: dict[str, dict] = {}
        self.targets: dict[tuple[str, int], h5py.File] = {}
        self.responses: dict[tuple[str, int], h5py.File] = {}
        for phase in phases:
            if phase == "ph001":
                raise PermissionError("ph001 is sealed for P12-F3")
            marker = self.phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
            payload = json.loads(marker.read_text())
            if (
                not payload.get("pass")
                or payload.get("phase") != phase
                or payload.get("ph001_opened")
            ):
                raise RuntimeError(f"{phase}: P12-F field target marker does not pass")
            self.manifests[phase] = payload

    def _component(self, phase: str, cap: int) -> dict:
        return self.manifests[phase]["components"]["NGC" if int(cap) == 1 else "SGC"]

    def _target(self, phase: str, cap: int) -> h5py.File:
        key = (phase, int(cap))
        if key not in self.targets:
            self.targets[key] = h5py.File(self._component(phase, cap)["file"], "r")
        return self.targets[key]

    def _response(self, phase: str, cap: int) -> h5py.File:
        key = (phase, int(cap))
        if key not in self.responses:
            self.responses[key] = h5py.File(
                self._component(phase, cap)["support_random_source"], "r"
            )
        return self.responses[key]

    def extract(self, phase: str, patch) -> dict[str, np.ndarray]:
        selection = tuple(
            slice(int(left), int(right))
            for left, right in zip(patch.context_start, patch.context_stop)
        )
        output = {
            "delta": np.asarray(
                self._target(phase, patch.cap)["delta_r7"][selection], dtype=np.float32
            ),
            "support": np.asarray(
                self._response(phase, patch.cap)["support_random"][selection], dtype=bool
            ),
        }
        if output["delta"].shape != patch.values.shape[1:]:
            raise RuntimeError("P12-F3 target/conditioning geometry mismatch")
        return output

    def close(self) -> None:
        for handle in (*self.targets.values(), *self.responses.values()):
            handle.close()
        self.targets.clear()
        self.responses.clear()


def target_tensor(values: np.ndarray, scaler: dict, device: str) -> torch.Tensor:
    scaled = (
        np.asarray(values, dtype=np.float32) - np.float32(scaler["mean"])
    ) / np.float32(scaler["std"])
    return torch.from_numpy(np.ascontiguousarray(scaled[None, None])).to(device)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=("local_h8", "wide_h24"), required=True)
    parser.add_argument("--run-name", default="seed42_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=13_200.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "p12f3-hierarchical-lowmode-v1":
        raise RuntimeError("unsupported P12-F3 low-mode contract")
    roles = config.get("roles", {})
    scope = config.get("scope", {})
    if (
        roles.get("validation") != "ph006"
        or roles.get("sealed_blind_test") != "ph001"
        or "ph001" in roles.get("training", [])
        or scope.get("ph001_opened")
        or not scope.get("does_not_reopen_p12f_v2_selection")
    ):
        raise PermissionError("P12-F3 phase or production boundary changed")
    if "ph001" in json.dumps(config.get("sources", {})).lower():
        raise PermissionError("ph001 entered P12-F3 source paths")
    return config


def validate_g1(config: dict) -> tuple[dict, dict, dict]:
    source = {name: Path(value) for name, value in config["sources"].items()}
    trained = json.loads(source["g1_trained_marker"].read_text())
    manifest = json.loads(source["g1_run_manifest"].read_text())
    filter_contract = json.loads(source["g1_filter"].read_text())
    if (
        trained.get("schema_version") != "p12f-matched-challenger-trained-v1"
        or trained.get("method") != "gaussian"
        or not trained.get("pass")
        or trained.get("ph001_opened")
        or int(trained.get("updates", -1)) != 10_000
        or trained.get("frozen_digest") != manifest.get("frozen_digest")
        or trained.get("checkpoint_sha256") != sha256(source["g1_checkpoint"])
    ):
        raise RuntimeError("frozen G1 checkpoint contract does not pass")
    if filter_contract.get("schema_version") != "p12f-g1-radial-residual-filter-v2":
        raise RuntimeError("unsupported frozen G1 residual filter")
    return trained, manifest, filter_contract


def load_g1_model(config: dict, device: str) -> tuple[ConditionalGaussianUNet, dict]:
    checkpoint_path = Path(config["sources"]["g1_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint.get("schema_version") != "p12f-matched-challenger-checkpoint-v1"
        or checkpoint.get("method") != "gaussian"
        or checkpoint.get("ph001_opened")
        or int(checkpoint.get("update", -1)) != 10_000
    ):
        raise RuntimeError("frozen G1 checkpoint payload is invalid")
    model = ConditionalGaussianUNet(condition_channels=3, base=8).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    scaler = checkpoint.get("target_scaler")
    if not isinstance(scaler, dict) or float(scaler.get("std", 0.0)) <= 0:
        raise RuntimeError("frozen G1 target scaler is missing")
    return model, scaler


def epoch_references(
    selected: dict[str, list[int]], phases: tuple[str, ...], *, seed: int, epoch: int
) -> list[tuple[str, int]]:
    refs = [(phase, int(core)) for phase in phases for core in selected[phase]]
    np.random.default_rng(seed + 1009 * epoch).shuffle(refs)
    return refs


def _paired_patch_guard(wide_patch, arm_patch) -> None:
    if (
        wide_patch.core_id != arm_patch.core_id
        or wide_patch.cap != arm_patch.cap
        or not np.array_equal(wide_patch.core_start, arm_patch.core_start)
        or not np.array_equal(wide_patch.core_stop, arm_patch.core_stop)
        or not np.array_equal(
            wide_patch.authoritative_parent_id, arm_patch.authoritative_parent_id
        )
    ):
        raise RuntimeError("local and wide patches do not share an authoritative core")


def build_example(
    *,
    loader: P10RandomResponseLoader,
    store: FieldTargetStore,
    g1_model: ConditionalGaussianUNet,
    scaler: dict,
    phase: str,
    core_id: int,
    arm_halo: int,
    common_halo: int,
    alignment: int,
    coarse_factor: int,
    voxel_mpc_h: float,
    maximum_k_h_mpc: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    adapter = loader.field_adapter(phase)
    wide_patch = adapter.extract(
        core_id, common_halo, CHANNELS, alignment_voxels=alignment
    )
    arm_patch = adapter.extract(
        core_id, arm_halo, CHANNELS, alignment_voxels=alignment
    )
    _paired_patch_guard(wide_patch, arm_patch)
    wide_condition, _ = model_inputs(wide_patch, loader.field_normalization, device)
    arm_condition, _ = model_inputs(arm_patch, loader.field_normalization, device)
    target_data = store.extract(phase, wide_patch)
    target = target_tensor(target_data["delta"], scaler, device)
    with torch.inference_mode():
        g1_mean, _ = g1_model(wide_condition)
        low_wide, _ = spectral_split(
            target - g1_mean,
            voxel_mpc_h=voxel_mpc_h,
            maximum_k_h_mpc=maximum_k_h_mpc,
        )
    low_arm = crop_tensor_to_patch(
        low_wide,
        source_start=wide_patch.context_start,
        target_start=arm_patch.context_start,
        target_stop=arm_patch.context_stop,
    )
    condition, low_target, science = prepare_low_mode_example(
        condition=arm_condition,
        low_residual=low_arm,
        core_slice=arm_patch.core_slice,
        coarse_factor=coarse_factor,
    )
    metadata = {
        "phase": phase,
        "core_id": int(core_id),
        "wide_shape": list(map(int, wide_patch.values.shape[1:])),
        "arm_shape": list(map(int, arm_patch.values.shape[1:])),
        "coarse_shape": list(map(int, low_target.shape[-3:])),
        "coarse_core_cells": int(science.sum().item()),
    }
    return condition, low_target, science, metadata


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    frozen_digest: str,
    loss_sum: float,
    loss_count: int,
) -> dict:
    return {
        "schema_version": "p12f3-lowmode-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "frozen_digest": frozen_digest,
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


def restore_rng_states(checkpoint: dict) -> None:
    """Restore CPU/CUDA RNG state without inheriting checkpoint map location.

    ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all`` require CPU
    byte tensors.  Coercing explicitly keeps resume safe even if a caller has
    loaded the checkpoint onto a CUDA device.
    """
    cpu_state = torch.as_tensor(checkpoint["torch_rng"], dtype=torch.uint8).cpu()
    torch.set_rng_state(cpu_state)
    cuda_states = checkpoint.get("cuda_rng") or []
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state, dtype=torch.uint8).cpu() for state in cuda_states]
        )


def state_reload_exact(model: torch.nn.Module, *, condition_channels: int, base: int) -> bool:
    clone = build_low_mode_model(condition_channels=condition_channels, base=base).cpu()
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    clone.load_state_dict(cpu_state)
    return all(torch.equal(clone.state_dict()[name], value) for name, value in cpu_state.items())


def technical_probe(
    *,
    model: torch.nn.Module,
    example: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
    draws: int,
    steps: int,
    seed: int,
    output: Path,
) -> dict:
    condition, target, science, metadata = example
    generator = torch.Generator(device=condition.device)
    generator.manual_seed(seed)
    model.eval()
    samples = sample_heun(
        model, condition, draws=draws, steps=steps, generator=generator
    )
    selected = science[0, 0]
    draw_std = float(samples[:, selected].std(dim=0, unbiased=True).mean().cpu())
    arrays = {
        "low_mode_target": target[0, 0].detach().cpu().numpy().astype(np.float32),
        "low_mode_draws": samples.detach().cpu().numpy().astype(np.float32),
        "coarse_core_mask": selected.detach().cpu().numpy().astype(np.uint8),
    }
    temporary = output / "technical_probe.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output / "technical_probe.npz")
    return {
        "metadata": metadata,
        "draws": int(draws),
        "all_finite": bool(torch.isfinite(samples).all()),
        "mean_core_draw_std": draw_std,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3 low-mode flow requires a CUDA allocation")
    config = load_config(args.config)
    trained, g1_manifest, _ = validate_g1(config)
    phases = tuple(config["roles"]["training"])
    validation = config["roles"]["validation"]
    sources = {name: Path(value) for name, value in config["sources"].items()}
    loader = P10RandomResponseLoader(sources["conditioning_contract"], include_blind=False)
    if tuple(loader.training_phases) != phases or loader.validation_phase != validation:
        raise RuntimeError("P12-F3 conditioning roles disagree with the frozen loader")
    selected = g1_manifest["frozen"]["selected_core_ids"]
    if set(selected) != set(phases + (validation,)):
        raise RuntimeError("P12-F3 did not inherit the exact G1 core contract")
    expected_train = int(config["training"]["training_cores_per_phase"])
    if any(len(selected[phase]) != expected_train for phase in phases):
        raise RuntimeError("P12-F3/G1 training core count mismatch")

    total_updates = int(config["training"]["science_updates"])
    stop_after = total_updates if args.stop_after_updates is None else int(args.stop_after_updates)
    if stop_after <= 0 or stop_after > total_updates:
        raise ValueError("stop-after-updates lies outside the frozen training budget")
    arm = config["arms"][args.arm]
    arm_halo = int(arm["conditioning_halo_voxels"])
    common_halo = int(config["patch"]["common_target_halo_voxels"])
    if arm_halo > common_halo:
        raise ValueError("conditioning halo exceeds common target context")

    output = args.output_root / args.run_name / args.arm
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    pause_path = output / "PAUSED.json"
    trace_path = output / "loss_trace.jsonl"
    manifest_path = output / "run_manifest.json"
    terminal_path = output / "P12F3_LOWMODE_TRAINED.json"
    canary_path = output / "TECHNICAL_CANARY_COMPLETE.json"
    if terminal_path.exists():
        print(terminal_path.read_text(), flush=True)
        return
    if any(output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty P12-F3 output requires --resume or a new run name")

    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/sbi/p12f3_hierarchical_lowmode.py",
        REPO_ROOT / "workflows/sbi/p12f_gaussian_controls.py",
        args.config.resolve(),
    ]
    target_markers = {
        phase: sha256(sources["phase_root"] / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json")
        for phase in phases + (validation,)
    }
    frozen = {
        "arm": args.arm,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "source_hashes": {str(path): sha256(path) for path in source_paths},
        "training_ready_sha256": sha256(sources["conditioning_contract"] / "TRAINING_LOADER_READY.json"),
        "g1_checkpoint_sha256": trained["checkpoint_sha256"],
        "g1_frozen_digest": trained["frozen_digest"],
        "g1_run_manifest_sha256": sha256(sources["g1_run_manifest"]),
        "g1_filter_sha256": sha256(sources["g1_filter"]),
        "target_markers": target_markers,
        "selected_core_ids": selected,
        "ph001_opened": False,
    }
    frozen_digest = canonical_digest(frozen)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("frozen_digest") != frozen_digest:
            raise RuntimeError("P12-F3 resume contract changed")
    else:
        atomic_json(
            manifest_path,
            {
                "schema_version": "p12f3-lowmode-run-v1",
                "created_utc": utc_now(),
                "git_revision_at_launch": git_revision(),
                "frozen_digest": frozen_digest,
                "frozen": frozen,
                "truth_files_read": ["visible-phase P12-F delta_R7 targets"],
                "ph001_opened": False,
            },
        )
        trace_path.write_text("")

    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    g1_model, scaler = load_g1_model(config, args.device)
    if scaler != g1_manifest["frozen"]["target_scaler"]:
        raise RuntimeError("G1 checkpoint and manifest target scalers differ")
    model = build_low_mode_model(
        condition_channels=int(config["model"]["condition_channels"]),
        base=int(config["model"]["unet_base"]),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    update = 0
    loss_sum = 0.0
    loss_count = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema_version") != "p12f3-lowmode-checkpoint-v1"
            or checkpoint.get("frozen_digest") != frozen_digest
            or checkpoint.get("ph001_opened")
        ):
            raise RuntimeError("P12-F3 checkpoint is stale or unsafe")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        update = int(checkpoint["update"])
        loss_sum = float(checkpoint["loss_sum"])
        loss_count = int(checkpoint["loss_count"])
        restore_rng_states(checkpoint)
    else:
        atomic_checkpoint(
            checkpoint_path,
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                update=0,
                frozen_digest=frozen_digest,
                loss_sum=0.0,
                loss_count=0,
            ),
        )

    store = FieldTargetStore(sources["phase_root"], phases + (validation,))
    started = time.monotonic()
    log_every = int(config["training"]["loss_log_every_updates"])
    checkpoint_every = int(config["training"]["checkpoint_every_updates"])
    alignment = int(config["patch"]["alignment_voxels"])
    coarse_factor = int(config["target"]["coarse_factor"])
    voxel_mpc_h = float(config["target"]["voxel_mpc_h"])
    maximum_k = float(config["target"]["maximum_k_h_mpc_inclusive"])
    refs_per_epoch = sum(len(selected[phase]) for phase in phases)
    last_gradient = None
    finite = True
    metadata = None

    try:
        while update < stop_after:
            epoch = update // refs_per_epoch
            ordinal = update % refs_per_epoch
            refs = epoch_references(selected, phases, seed=seed, epoch=epoch)
            phase, core_id = refs[ordinal]
            condition, target, science, metadata = build_example(
                loader=loader,
                store=store,
                g1_model=g1_model,
                scaler=scaler,
                phase=phase,
                core_id=core_id,
                arm_halo=arm_halo,
                common_halo=common_halo,
                alignment=alignment,
                coarse_factor=coarse_factor,
                voxel_mpc_h=voxel_mpc_h,
                maximum_k_h_mpc=maximum_k,
                device=args.device,
            )
            model.train()
            state, time_value, velocity, _ = rectified_flow_training_pair(target)
            predicted = model(state, time_value, condition)
            loss = torch.mean(torch.square(predicted[science] - velocity[science]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip"])
                ).detach().cpu()
            )
            finite_before_step = bool(
                torch.isfinite(loss)
                and np.isfinite(last_gradient)
                and all(torch.isfinite(parameter).all() for parameter in model.parameters())
            )
            if not finite_before_step:
                raise FloatingPointError("non-finite P12-F3 loss, gradient or parameter")
            optimizer.step()
            finite = bool(
                all(torch.isfinite(parameter).all() for parameter in model.parameters())
            )
            if not finite:
                raise FloatingPointError("P12-F3 optimizer produced a non-finite parameter")
            update += 1
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value
            loss_count += 1
            if update % log_every == 0 or update == stop_after:
                with trace_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "update": update,
                                "epoch_equivalent": update / refs_per_epoch,
                                "loss": loss_value,
                                "mean_loss": loss_sum / loss_count,
                                "preclip_gradient_norm": last_gradient,
                                "elapsed_seconds": time.monotonic() - started,
                                **metadata,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if update % checkpoint_every == 0 or update == stop_after:
                atomic_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        update=update,
                        frozen_digest=frozen_digest,
                        loss_sum=loss_sum,
                        loss_count=loss_count,
                    ),
                )
            if time.monotonic() - started >= args.max_wall_seconds and update < stop_after:
                atomic_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        update=update,
                        frozen_digest=frozen_digest,
                        loss_sum=loss_sum,
                        loss_count=loss_count,
                    ),
                )
                atomic_json(
                    pause_path,
                    {
                        "schema_version": "p12f3-lowmode-pause-v1",
                        "created_utc": utc_now(),
                        "arm": args.arm,
                        "update": update,
                        "stop_after_updates": stop_after,
                        "frozen_digest": frozen_digest,
                        "ph001_opened": False,
                    },
                )
                raise SystemExit(75)

        probe_core = int(selected[validation][0])
        example = build_example(
            loader=loader,
            store=store,
            g1_model=g1_model,
            scaler=scaler,
            phase=validation,
            core_id=probe_core,
            arm_halo=arm_halo,
            common_halo=common_halo,
            alignment=alignment,
            coarse_factor=coarse_factor,
            voxel_mpc_h=voxel_mpc_h,
            maximum_k_h_mpc=maximum_k,
            device=args.device,
        )
        probe = technical_probe(
            model=model,
            example=example,
            draws=int(config["training"]["technical_probe_draws"]),
            steps=int(config["model"]["ode_steps"]),
            seed=seed + 73001,
            output=output,
        )
        reload_ok = state_reload_exact(
            model,
            condition_channels=int(config["model"]["condition_channels"]),
            base=int(config["model"]["unet_base"]),
        )
        minimum_std = float(config["technical_gate"]["minimum_probe_draw_std"])
        passed = bool(
            finite
            and reload_ok
            and probe["all_finite"]
            and probe["mean_core_draw_std"] >= minimum_std
        )
        technical_updates = int(config["training"]["technical_canary_updates"])
        marker = {
            "schema_version": (
                "p12f3-lowmode-technical-canary-v1"
                if update == technical_updates
                else "p12f3-lowmode-bounded-probe-v1"
            ),
            "created_utc": utc_now(),
            "arm": args.arm,
            "update": update,
            "technical_canary_updates": technical_updates,
            "science_updates": total_updates,
            "mean_training_loss": loss_sum / max(loss_count, 1),
            "last_preclip_gradient_norm": last_gradient,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_reload_exact": reload_ok,
            "probe": probe,
            "frozen_digest": frozen_digest,
            "ph001_opened": False,
            "pass": passed,
        }
        if update == technical_updates:
            atomic_json(canary_path, marker)
        if update == total_updates:
            marker["schema_version"] = "p12f3-lowmode-trained-v1"
            atomic_json(terminal_path, marker)
        if not passed:
            raise RuntimeError("P12-F3 technical gate failed")
        if pause_path.exists():
            pause_path.unlink()
        print(json.dumps(marker, indent=2), flush=True)
    finally:
        store.close()
        loader.close()


if __name__ == "__main__":
    main()
