#!/usr/bin/env python3
"""Fit whitening and train the P12-F3-L2 direct Fourier low-mode flow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f3_fourier_modes import (
    ConditionalFourierVelocityUNet,
    build_fourier_layout,
    empty_whitening_accumulator,
    equal_band_flow_loss,
    finalize_whitening,
    hermitian_max_error,
    lowpass_exact,
    pack_fourier_components,
    rectified_flow_pair,
    sample_fourier_heun,
    spectral_lowpass_reference,
    update_whitening_accumulator,
    whiten_components,
)
from workflows.sbi.p12f3_train_lowmode_flow import (
    FieldTargetStore,
    epoch_references,
    load_g1_model,
    restore_rng_states,
    target_tensor,
    validate_g1,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_fourier_lowmode_v1.json"
DEFAULT_OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_fourier_lowmode_v1"
)


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
    parser.add_argument("--stage", choices=("fit-whitening", "train"), required=True)
    parser.add_argument("--run-name", default="seed42_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "p12f3-fourier-lowmode-v1":
        raise RuntimeError("unsupported P12-F3-L2 contract")
    roles = config["roles"]
    if (
        roles["validation"] != "ph006"
        or roles["sealed_blind_test"] != "ph001"
        or "ph001" in roles["training"]
        or config["scope"].get("ph001_opened")
        or "ph001" in json.dumps(config["sources"]).lower()
    ):
        raise PermissionError("P12-F3-L2 phase boundary changed")
    edges = config["target"]["band_edges_h_mpc"]
    if len(edges) != 3 or edges[0] != 0.0:
        raise RuntimeError("P12-F3-L2 requires exactly two registered non-DC bands")
    if config["target"].get("pooling") or config["target"].get("interpolation"):
        raise RuntimeError("P12-F3-L2 forbids pooled/interpolated stochastic coordinates")
    return config


def _source_contract(
    config: dict, config_path: Path, selected: dict[str, list[int]], phases: tuple[str, ...]
) -> dict:
    sources = {key: Path(value) for key, value in config["sources"].items()}
    visible = phases + (config["roles"]["validation"],)
    return {
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "source_hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(REPO_ROOT / "workflows/sbi/p12f3_fourier_modes.py"): sha256(
                REPO_ROOT / "workflows/sbi/p12f3_fourier_modes.py"
            ),
        },
        "training_ready_sha256": sha256(
            sources["conditioning_contract"] / "TRAINING_LOADER_READY.json"
        ),
        "g1_checkpoint_sha256": sha256(sources["g1_checkpoint"]),
        "g1_run_manifest_sha256": sha256(sources["g1_run_manifest"]),
        "target_markers": {
            phase: sha256(
                sources["phase_root"]
                / phase
                / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
            )
            for phase in visible
        },
        "selected_core_ids": selected,
        "ph001_opened": False,
    }


def build_example(
    *,
    loader: P10RandomResponseLoader,
    store: FieldTargetStore,
    g1_model,
    scaler: dict,
    phase: str,
    core_id: int,
    halo: int,
    alignment: int,
    voxel_mpc_h: float,
    band_edges: tuple[float, ...],
    device: str,
    whitening: dict | None,
):
    patch = loader.field_adapter(phase).extract(
        core_id, halo, CHANNELS, alignment_voxels=alignment
    )
    condition, _ = model_inputs(patch, loader.field_normalization, device)
    target = target_tensor(store.extract(phase, patch)["delta"], scaler, device)
    with torch.inference_mode():
        mean, _ = g1_model(condition)
        residual = target - mean
    layout = build_fourier_layout(
        residual.shape[-3:], voxel_mpc_h=voxel_mpc_h, band_edges_h_mpc=band_edges
    )
    raw = pack_fourier_components(residual, layout)
    vector = raw if whitening is None else whiten_components(raw, whitening, layout)
    reconstructed = lowpass_exact(residual, layout)
    reference = spectral_lowpass_reference(residual, layout)
    roundtrip = float(torch.max(torch.abs(reconstructed - reference)).cpu())
    metadata = {
        "phase": phase,
        "core_id": int(core_id),
        "shape": list(layout.shape),
        "modes": layout.modes,
        "components": layout.components,
        "band_mode_counts": np.bincount(layout.mode_band, minlength=2).astype(int).tolist(),
        "roundtrip_abs_error": roundtrip,
        "hermitian_abs_error": hermitian_max_error(raw, layout),
    }
    return condition, vector, layout, metadata


def _open_common(config: dict):
    trained, g1_manifest, _ = validate_g1(config)
    phases = tuple(config["roles"]["training"])
    validation = config["roles"]["validation"]
    sources = {key: Path(value) for key, value in config["sources"].items()}
    loader = P10RandomResponseLoader(sources["conditioning_contract"], include_blind=False)
    if tuple(loader.training_phases) != phases or loader.validation_phase != validation:
        raise RuntimeError("P12-F3-L2 loader roles differ from the frozen contract")
    selected = g1_manifest["frozen"]["selected_core_ids"]
    expected = int(config["training"]["training_cores_per_phase"])
    if set(selected) != set(phases + (validation,)) or any(
        len(selected[phase]) != expected for phase in phases
    ):
        raise RuntimeError("P12-F3-L2 did not inherit the exact G1 core identities")
    store = FieldTargetStore(sources["phase_root"], phases + (validation,))
    return trained, g1_manifest, phases, validation, sources, loader, store, selected


def fit_whitening(config: dict, args: argparse.Namespace) -> None:
    (
        _, g1_manifest, phases, _, sources, loader, store, selected,
    ) = _open_common(config)
    output = sources["whitening"].parent
    output.mkdir(parents=True, exist_ok=True)
    marker_path = sources["whitening"]
    if marker_path.exists():
        print(marker_path.read_text(), flush=True)
        store.close(); loader.close(); return
    if any(output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty whitening output requires --resume or cleanup")
    progress_path = output / "WHITENING_PROGRESS.json"
    source_contract = _source_contract(config, args.config, selected, phases)
    frozen_digest = canonical_digest(source_contract)
    accumulator = empty_whitening_accumulator(4)
    completed = 0
    maximum_roundtrip = 0.0
    maximum_hermitian = 0.0
    shape_counts: dict[str, int] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("frozen_digest") != frozen_digest:
            raise RuntimeError("Fourier whitening resume contract changed")
        completed = int(progress["completed"])
        for key in accumulator:
            accumulator[key][...] = np.asarray(progress["accumulator"][key], dtype=accumulator[key].dtype)
        maximum_roundtrip = float(progress["maximum_roundtrip_abs_error"])
        maximum_hermitian = float(progress["maximum_hermitian_abs_error"])
        shape_counts = {str(key): int(value) for key, value in progress["shape_counts"].items()}
    refs = [(phase, int(core)) for phase in phases for core in selected[phase]]
    g1_model, scaler = load_g1_model(config, args.device)
    started = time.monotonic()
    halo = int(config["patch"]["conditioning_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    voxel = float(config["target"]["voxel_mpc_h"])
    edges = tuple(float(value) for value in config["target"]["band_edges_h_mpc"])
    try:
        for ordinal in range(completed, len(refs)):
            phase, core_id = refs[ordinal]
            _, raw, layout, metadata = build_example(
                loader=loader, store=store, g1_model=g1_model, scaler=scaler,
                phase=phase, core_id=core_id, halo=halo, alignment=alignment,
                voxel_mpc_h=voxel, band_edges=edges, device=args.device, whitening=None,
            )
            update_whitening_accumulator(accumulator, raw, layout)
            maximum_roundtrip = max(maximum_roundtrip, metadata["roundtrip_abs_error"])
            maximum_hermitian = max(maximum_hermitian, metadata["hermitian_abs_error"])
            shape_key = "x".join(map(str, layout.shape))
            shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
            completed = ordinal + 1
            if completed % 25 == 0 or completed == len(refs):
                atomic_json(progress_path, {
                    "schema_version": "p12f3l2-fourier-whitening-progress-v1",
                    "frozen_digest": frozen_digest,
                    "completed": completed,
                    "total": len(refs),
                    "accumulator": {key: value.tolist() for key, value in accumulator.items()},
                    "maximum_roundtrip_abs_error": maximum_roundtrip,
                    "maximum_hermitian_abs_error": maximum_hermitian,
                    "shape_counts": shape_counts,
                    "ph001_opened": False,
                })
                print(json.dumps({"stage": "whitening", "completed": completed, "total": len(refs), "elapsed_seconds": time.monotonic()-started}), flush=True)
            if time.monotonic() - started >= args.max_wall_seconds and completed < len(refs):
                raise SystemExit(75)
        whitening = finalize_whitening(accumulator)
        gate = config["representation_gate"]
        passed = bool(
            maximum_roundtrip <= float(gate["maximum_roundtrip_abs_error"])
            and maximum_hermitian <= float(gate["maximum_hermitian_abs_error"])
            and min(whitening["count"]) >= int(gate["minimum_group_count"])
            and np.all(np.isfinite(whitening["mean"] + whitening["std"]))
            and min(whitening["std"]) > 0
        )
        marker = {
            "schema_version": "p12f3l2-fourier-whitening-v1",
            "created_utc": utc_now(),
            "git_revision": git_revision(),
            "pass": passed,
            "fit_phases": list(phases),
            "validation_phase_used_for_fit": False,
            "training_cores": completed,
            "whitening": whitening,
            "maximum_roundtrip_abs_error": maximum_roundtrip,
            "maximum_hermitian_abs_error": maximum_hermitian,
            "shape_counts": shape_counts,
            "frozen_digest": frozen_digest,
            "frozen": source_contract,
            "g1_target_scaler": g1_manifest["frozen"]["target_scaler"],
            "truth_files_read": [f"{phase} delta_R7 training targets" for phase in phases],
            "ph001_opened": False,
        }
        atomic_json(marker_path, marker)
        if not passed:
            raise RuntimeError("P12-F3-L2 representation/whitening gate failed")
        print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    finally:
        store.close(); loader.close()


def _checkpoint_payload(model, optimizer, update, frozen_digest, loss_sum, loss_count):
    return {
        "schema_version": "p12f3l2-fourier-flow-checkpoint-v1",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "update": int(update), "frozen_digest": frozen_digest,
        "loss_sum": float(loss_sum), "loss_count": int(loss_count),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "ph001_opened": False,
    }


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def train(config: dict, args: argparse.Namespace) -> None:
    trained, g1_manifest, phases, validation, sources, loader, store, selected = _open_common(config)
    whitening_marker = json.loads(sources["whitening"].read_text())
    if (
        whitening_marker.get("schema_version") != "p12f3l2-fourier-whitening-v1"
        or not whitening_marker.get("pass")
        or whitening_marker.get("validation_phase_used_for_fit")
        or whitening_marker.get("ph001_opened")
        or whitening_marker.get("fit_phases") != list(phases)
    ):
        raise RuntimeError("P12-F3-L2 whitening contract is incomplete or unsafe")
    whitening = whitening_marker["whitening"]
    total = int(config["training"]["science_updates"])
    stop_after = total if args.stop_after_updates is None else int(args.stop_after_updates)
    if stop_after <= 0 or stop_after > total:
        raise ValueError("invalid P12-F3-L2 stop-after update")
    output = args.output_root / args.run_name / "wide_h24"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    trace_path = output / "loss_trace.jsonl"
    pause_path = output / "PAUSED.json"
    canary_path = output / "TECHNICAL_CANARY_COMPLETE.json"
    terminal_path = output / "P12F3L2_FOURIER_FLOW_TRAINED.json"
    if terminal_path.exists():
        print(terminal_path.read_text(), flush=True); store.close(); loader.close(); return
    if any(output.iterdir()) and not args.resume:
        raise RuntimeError("non-empty Fourier-flow output requires --resume or new run name")
    frozen = _source_contract(config, args.config, selected, phases)
    frozen.update({
        "whitening": str(sources["whitening"].resolve()),
        "whitening_sha256": sha256(sources["whitening"]),
        "g1_frozen_digest": trained["frozen_digest"],
    })
    frozen_digest = canonical_digest(frozen)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("frozen_digest") != frozen_digest:
            raise RuntimeError("P12-F3-L2 resume contract changed")
    else:
        atomic_json(manifest_path, {
            "schema_version": "p12f3l2-fourier-flow-run-v1",
            "created_utc": utc_now(), "git_revision_at_launch": git_revision(),
            "frozen_digest": frozen_digest, "frozen": frozen,
            "truth_files_read": ["visible-phase P12-F delta_R7 targets"],
            "ph001_opened": False,
        })
        trace_path.write_text("")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    g1_model, scaler = load_g1_model(config, args.device)
    if scaler != g1_manifest["frozen"]["target_scaler"]:
        raise RuntimeError("G1 target scaler changed")
    model = ConditionalFourierVelocityUNet(
        condition_channels=int(config["model"]["condition_channels"]),
        base=int(config["model"]["unet_base"]),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    update = 0; loss_sum = 0.0; loss_count = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "p12f3l2-fourier-flow-checkpoint-v1" or checkpoint.get("frozen_digest") != frozen_digest or checkpoint.get("ph001_opened"):
            raise RuntimeError("unsafe P12-F3-L2 checkpoint")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        update = int(checkpoint["update"]); loss_sum = float(checkpoint["loss_sum"]); loss_count = int(checkpoint["loss_count"])
        restore_rng_states(checkpoint)
    else:
        _atomic_checkpoint(checkpoint_path, _checkpoint_payload(model, optimizer, 0, frozen_digest, 0.0, 0))

    refs_per_epoch = sum(len(selected[phase]) for phase in phases)
    halo = int(config["patch"]["conditioning_halo_voxels"])
    alignment = int(config["patch"]["alignment_voxels"])
    voxel = float(config["target"]["voxel_mpc_h"])
    edges = tuple(float(value) for value in config["target"]["band_edges_h_mpc"])
    started = time.monotonic(); last_gradient = None; metadata = None
    try:
        while update < stop_after:
            epoch = update // refs_per_epoch; ordinal = update % refs_per_epoch
            phase, core_id = epoch_references(selected, phases, seed=seed, epoch=epoch)[ordinal]
            condition, target, layout, metadata = build_example(
                loader=loader, store=store, g1_model=g1_model, scaler=scaler,
                phase=phase, core_id=core_id, halo=halo, alignment=alignment,
                voxel_mpc_h=voxel, band_edges=edges, device=args.device, whitening=whitening,
            )
            state, time_value, velocity = rectified_flow_pair(target)
            model.train(); predicted = model(state, time_value, condition, layout=layout, whitening=whitening)
            loss = equal_band_flow_loss(predicted, velocity, layout, 2)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            last_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])).detach().cpu())
            if not torch.isfinite(loss) or not np.isfinite(last_gradient) or not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite P12-F3-L2 loss/gradient/parameter")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("optimizer produced non-finite P12-F3-L2 parameter")
            update += 1; value = float(loss.detach().cpu()); loss_sum += value; loss_count += 1
            if update % int(config["training"]["loss_log_every_updates"]) == 0 or update == stop_after:
                with trace_path.open("a") as handle:
                    handle.write(json.dumps({
                        "update": update, "epoch_equivalent": update/refs_per_epoch,
                        "loss": value, "mean_loss": loss_sum/loss_count,
                        "preclip_gradient_norm": last_gradient,
                        "elapsed_seconds": time.monotonic()-started, **metadata,
                    }, sort_keys=True)+"\n")
            if update % int(config["training"]["checkpoint_every_updates"]) == 0 or update == stop_after:
                _atomic_checkpoint(checkpoint_path, _checkpoint_payload(model, optimizer, update, frozen_digest, loss_sum, loss_count))
            if time.monotonic()-started >= args.max_wall_seconds and update < stop_after:
                _atomic_checkpoint(checkpoint_path, _checkpoint_payload(model, optimizer, update, frozen_digest, loss_sum, loss_count))
                atomic_json(pause_path, {"schema_version":"p12f3l2-fourier-flow-pause-v1","update":update,"frozen_digest":frozen_digest,"ph001_opened":False})
                raise SystemExit(75)

        probe_core = int(selected[validation][0])
        condition, target, layout, metadata = build_example(
            loader=loader, store=store, g1_model=g1_model, scaler=scaler,
            phase=validation, core_id=probe_core, halo=halo, alignment=alignment,
            voxel_mpc_h=voxel, band_edges=edges, device=args.device, whitening=whitening,
        )
        generator = torch.Generator(device=condition.device).manual_seed(seed+73001)
        model.eval(); draws = sample_fourier_heun(
            model, condition, layout=layout, whitening=whitening,
            draws=int(config["training"]["technical_probe_draws"]),
            steps=int(config["model"]["ode_steps"]), generator=generator,
        )
        draw_std = float(draws.std(dim=0, unbiased=True).mean().cpu())
        clone = ConditionalFourierVelocityUNet(
            condition_channels=int(config["model"]["condition_channels"]),
            base=int(config["model"]["unet_base"]),
        )
        cpu_state = {name:value.detach().cpu() for name,value in model.state_dict().items()}
        clone.load_state_dict(cpu_state)
        reload_ok = all(torch.equal(clone.state_dict()[name], value) for name,value in cpu_state.items())
        passed = bool(torch.isfinite(draws).all() and draw_std >= float(config["technical_gate"]["minimum_probe_draw_std"]) and reload_ok)
        marker = {
            "schema_version": "p12f3l2-fourier-flow-technical-canary-v1" if update == int(config["training"]["technical_canary_updates"]) else "p12f3l2-fourier-flow-trained-v1",
            "created_utc": utc_now(), "pass": passed, "update": update,
            "mean_training_loss": loss_sum/max(loss_count,1),
            "last_preclip_gradient_norm": last_gradient,
            "mean_probe_draw_std": draw_std, "probe_all_finite": bool(torch.isfinite(draws).all()),
            "checkpoint_reload_exact": reload_ok, "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path), "frozen_digest": frozen_digest,
            "probe_metadata": metadata, "ph001_opened": False,
        }
        atomic_json(canary_path if update == int(config["training"]["technical_canary_updates"]) else terminal_path, marker)
        if not passed: raise RuntimeError("P12-F3-L2 technical gate failed")
        if pause_path.exists(): pause_path.unlink()
        print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    finally:
        store.close(); loader.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3-L2 requires a CUDA allocation")
    config = load_config(args.config)
    if args.stage == "fit-whitening": fit_whitening(config, args)
    else: train(config, args)


if __name__ == "__main__":
    main()
