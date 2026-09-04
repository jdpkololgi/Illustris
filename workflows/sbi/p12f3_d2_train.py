#!/usr/bin/env python3
"""Train the bounded P12-F3-D2 diffusion funnel on training phases only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.sbi.p12f3_conditional_models import (
    ALL_PATCH_CHANNELS,
    fourier_v_pair,
    low_mode_target,
    proxy_condition,
    standardized_low_field,
)
from workflows.sbi.p12f3_d2_contract import (
    CANARY_SCHEMA,
    CHECKPOINT_SCHEMA,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    RUN_SCHEMA,
    SELECTION_SCHEMA,
    TRAINED_SCHEMA,
    arm_config,
    digest,
    git_revision,
    load_d2_config,
    split_internal_refs,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import (
    D2ConditionalFourierVDenoiser,
    clone_model_state,
    configure_d2_determinism,
    load_model_state_copy,
    parameter_count,
    sample_fourier_d2_batched,
    update_ema_state,
)
from workflows.sbi.p12f3_fourier_modes import (
    build_fourier_layout,
    pack_fourier_components,
    whiten_components,
)
from workflows.sbi.p12f3_train_conditional_generative import (
    load_location_scale,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import (
    epoch_references,
    load_g1_model,
    target_tensor,
)


D2_SUPPORT_CHANNELS = ("support_random",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--arm",
        choices=("modern_base4", "modern_base8", "modern_base8_attention"),
        required=True,
    )
    parser.add_argument("--stage", choices=("canary", "science"), required=True)
    parser.add_argument("--seed-role", choices=("primary", "replication"), default="primary")
    parser.add_argument("--selection-marker", type=Path)
    parser.add_argument("--capacity-selection-marker", type=Path)
    parser.add_argument("--confirmation-marker", type=Path)
    parser.add_argument("--second-seed-license", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stop-after-presentations",
        type=int,
        help="allocation checkpoint boundary; must be divisible by accumulation and within the frozen stage cap",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_model(config: dict, arm: str) -> D2ConditionalFourierVDenoiser:
    architecture = arm_config(config, arm)
    return D2ConditionalFourierVDenoiser(
        condition_channels=int(config["matched_contract"]["condition_channels_count"]),
        base=int(architecture["base"]),
        time_channels=int(architecture["time_channels"]),
        coarse_attention=bool(architecture["coarse_attention"]),
        attention_heads=int(architecture["attention_heads"]),
    )


def build_d2_example(
    *, loader, store, g1_model, location_model, scaler, phase, core_id,
    conditional_config, f3_parent, device, whitening,
):
    """Exact F3-L2d example plus mask-only ``support_random`` metadata."""
    patch = loader.field_adapter(phase).extract(
        core_id,
        int(f3_parent["patch"]["conditioning_halo_voxels"]),
        ALL_PATCH_CHANNELS,
        alignment_voxels=int(f3_parent["patch"]["alignment_voxels"]),
    )
    support_patch = loader.field_adapter(phase).extract(
        core_id,
        int(f3_parent["patch"]["conditioning_halo_voxels"]),
        D2_SUPPORT_CHANNELS,
        alignment_voxels=int(f3_parent["patch"]["alignment_voxels"]),
    )
    scalar_geometry = ("core_id", "fold", "cap")
    array_geometry = (
        "context_start",
        "context_stop",
        "core_start",
        "core_stop",
        "authoritative_parent_id",
        "authoritative_frac_index_local",
    )
    if any(getattr(patch, name) != getattr(support_patch, name) for name in scalar_geometry):
        raise RuntimeError("D2 support_random metadata changed patch identity")
    if any(
        not np.array_equal(np.asarray(getattr(patch, name)), np.asarray(getattr(support_patch, name)))
        for name in array_geometry
    ) or patch.core_slice != support_patch.core_slice:
        raise RuntimeError("D2 support_random metadata is not geometry-identical")
    condition, g1_mean, _ = proxy_condition(
        patch, loader.field_normalization, g1_model, device=device, arm="proxy7"
    )
    target_data = store.extract(phase, patch)
    target = target_tensor(target_data["delta"], scaler, device)
    layout = build_fourier_layout(
        tuple(target.shape[-3:]),
        voxel_mpc_h=float(conditional_config["target"]["voxel_mpc_h"]),
        band_edges_h_mpc=tuple(
            float(value) for value in conditional_config["target"]["band_edges_h_mpc"]
        ),
    )
    target_low = low_mode_target(target - g1_mean, layout)
    with torch.inference_mode():
        location, log_scale = location_model(condition)
        standard_field = standardized_low_field(target_low, location, log_scale, layout)
        vector = whiten_components(
            pack_fourier_components(standard_field, layout), whitening, layout
        )
    if tuple(support_patch.channel_names) != D2_SUPPORT_CHANNELS:
        raise RuntimeError("D2 support_random was not isolated as mask-only metadata")
    support = torch.from_numpy(
        np.asarray(support_patch.values[0], dtype=np.float32)[None, None]
    ).to(device)
    if (
        support.shape[-3:] != condition.shape[-3:]
        or not torch.all(torch.isfinite(support))
        or not torch.all((support == 0) | (support == 1))
        or not torch.any(support)
    ):
        raise RuntimeError("D2 exact support_random metadata is invalid")
    return (
        condition,
        vector,
        layout,
        location,
        log_scale,
        patch,
        support.bool(),
        target_low,
    )


def _numpy_rng_state() -> tuple:
    return np.random.get_state()


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    optimizer_update: int,
    examples_seen: int,
    frozen_digest: str,
    arm: str,
    seed: int,
    loss_sum: float,
    loss_count: int,
) -> dict:
    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(item) for item in value)
        return value

    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "arm": arm,
        "seed": int(seed),
        "model": cpu_copy(model.state_dict()),
        "ema_model": cpu_copy(ema_state),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_update": int(optimizer_update),
        "examples_seen": int(examples_seen),
        "frozen_digest": frozen_digest,
        "loss_sum": float(
            loss_sum.detach().cpu() if torch.is_tensor(loss_sum) else loss_sum
        ),
        "loss_count": int(loss_count),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": _numpy_rng_state(),
        "ph006_used_for_fit": False,
        "ph001_opened": False,
    }


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def _diagnostic_ref_split(
    internal: dict[str, list[int]],
    phases: tuple[str, ...],
    *,
    selection_count: int,
    confirmation_count: int,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Compatibility wrapper around the frozen randomized phase-balanced split."""
    return split_internal_refs(
        internal,
        phases,
        seed=seed,
        selection_count=selection_count,
        confirmation_count=confirmation_count,
    )


@torch.inference_mode()
def internal_sample_diagnostics(
    model: torch.nn.Module,
    refs: list[tuple[str, int]],
    *,
    loader,
    store,
    g1_model,
    location_model,
    scaler: dict,
    d2_config: dict,
    conditional_config: dict,
    f3_parent: dict,
    device: str,
    whitening: dict,
    seed: int,
) -> dict:
    """Training-phase-only proper score and spread/error diagnostic."""
    model.eval()
    energy: list[float] = []
    crps: list[float] = []
    denoising: list[float] = []
    core_keys: list[str] = []
    predicted_variance = np.zeros(2, dtype=np.float64)
    squared_error = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    draws = int(d2_config["funnel"]["internal_sample_draws"])
    steps = int(d2_config["funnel"]["internal_sample_network_evaluations"])
    groups_seen: np.ndarray | None = None
    for ordinal, (phase, core_id) in enumerate(refs):
        condition, target, layout, _, _, _, support, _ = build_d2_example(
            loader=loader,
            store=store,
            g1_model=g1_model,
            location_model=location_model,
            scaler=scaler,
            phase=phase,
            core_id=core_id,
            conditional_config=conditional_config,
            f3_parent=f3_parent,
            device=device,
            whitening=whitening,
        )
        pair_generator = torch.Generator(device=device).manual_seed(
            seed + 17_000_000 + 1009 * ordinal
        )
        state, time_value, desired = fourier_v_pair(target, generator=pair_generator)
        predicted = model(
            state,
            time_value,
            condition,
            layout=layout,
            whitening=whitening,
            support_mask=support,
        )
        component_band = layout.component_group // 2
        band_losses = []
        for band in range(2):
            mask = torch.as_tensor(component_band == band, device=device)
            band_losses.append(torch.mean(torch.square(predicted[:, mask] - desired[:, mask])))
        denoising.append(float(torch.stack(band_losses).mean().cpu()))

        sample_generator = torch.Generator(device=device).manual_seed(
            seed + 29_000_000 + 1009 * ordinal
        )
        draw_batch = int(d2_config["funnel"]["internal_sample_draw_batch"])
        fields = sample_fourier_d2_batched(
            model,
            condition,
            layout=layout,
            whitening=whitening,
            draws=draws,
            draw_batch=draw_batch,
            steps=steps,
            generator=sample_generator,
            eta=0.0,
            support_mask=support,
        )
        vectors = whiten_components(
            pack_fourier_components(fields[:, None], layout), whitening, layout
        )
        scale = float(np.sqrt(layout.components))
        truth_distance = torch.linalg.vector_norm(vectors - target, dim=1).mean() / scale
        pair_distance = torch.cdist(vectors, vectors).mean() / scale
        energy.append(float((truth_distance - 0.5 * pair_distance).cpu()))
        marginal = torch.abs(vectors - target).mean() - 0.5 * torch.abs(
            vectors[:, None] - vectors[None, :]
        ).mean()
        crps.append(float(marginal.cpu()))
        core_keys.append(f"{phase}:{core_id}")
        sample_mean = vectors.mean(dim=0)
        groups_seen = component_band
        for band in range(2):
            mask = torch.as_tensor(component_band == band, device=device)
            predicted_variance[band] += float(vectors[:, mask].var(dim=0, unbiased=True).sum().cpu())
            squared_error[band] += float(torch.square(sample_mean[mask] - target[0, mask]).sum().cpu())
            counts[band] += int(mask.sum().item())
    if groups_seen is None or np.any(counts <= 0):
        raise RuntimeError("D2 internal sample diagnostic had no Fourier components")
    variance_ratio = predicted_variance / np.maximum(squared_error, 1.0e-12)
    result = {
        "cores": len(refs),
        "draws_per_core": draws,
        "network_evaluations": steps,
        "energy_score": float(np.mean(energy)),
        "energy_standard_error": float(
            np.std(energy, ddof=1) / np.sqrt(len(energy)) if len(energy) > 1 else 0.0
        ),
        "marginal_crps": float(np.mean(crps)),
        "denoising_loss": float(np.mean(denoising)),
        "band_spread_to_squared_error_ratio": variance_ratio.tolist(),
        "maximum_absolute_log_band_variance_ratio": float(
            np.max(np.abs(np.log(np.maximum(variance_ratio, 1.0e-12))))
        ),
        "core_keys": core_keys,
        "per_core_energy_score": energy,
        "per_core_marginal_crps": crps,
    }
    if not all(
        np.isfinite(value)
        for value in (
            result["energy_score"],
            result["marginal_crps"],
            result["denoising_loss"],
            result["maximum_absolute_log_band_variance_ratio"],
        )
    ):
        raise FloatingPointError("non-finite D2 internal sample diagnostic")
    return result


def _read_diagnostic_updates(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(row["optimizer_update"]) for row in _read_jsonl_rows(path)}


def _read_diagnostic_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = _read_jsonl_rows(path)
    updates = [int(row["optimizer_update"]) for row in rows]
    if len(updates) != len(set(updates)):
        raise RuntimeError("D2 internal diagnostics contain duplicate milestones")
    return rows


def _read_jsonl_rows(path: Path) -> list[dict]:
    """Read JSONL, dropping only a malformed unterminated final fragment."""
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            is_final_unterminated = index == len(lines) - 1 and not text.endswith("\n")
            if is_final_unterminated:
                break
            raise RuntimeError(f"interior D2 JSONL corruption: {path}")
    return rows


def milestone_checkpoint_path(output: Path, presentations: int) -> Path:
    return output / "milestones" / f"presentations_{int(presentations):05d}.pt"


def validate_stop_position(stop: int, optimizer_update: int, stage_target: int) -> None:
    """Permit terminal-marker recovery, but reject every other no-progress run."""
    terminal_recovery = stop == optimizer_update == stage_target
    if stop < optimizer_update or stop > stage_target or (
        stop == optimizer_update and not terminal_recovery
    ):
        raise ValueError("D2 stop update is outside the frozen stage budget")


def select_earliest_within_one_se(
    rows: list[dict], allowed_presentations: set[int], config: dict
) -> dict:
    """Freeze checkpoint age and raw/EMA weights on the 128-core split only.

    A candidate is within one paired standard error of the lowest-energy
    candidate when mean(E_candidate - E_best) <= SE(E_candidate - E_best).
    The earliest feasible age wins; energy breaks a same-age raw/EMA tie.
    """
    candidates: list[dict] = []
    for row in rows:
        presentations = int(row["presentations"])
        if presentations not in allowed_presentations:
            continue
        if row.get("confirmation") is not None:
            raise RuntimeError("D2 milestone selection must not inspect confirmation cores")
        for weight in ("raw", "ema"):
            diagnostic = row["selection"][weight]
            per_core = np.asarray(diagnostic["per_core_energy_score"], dtype=np.float64)
            if (
                len(per_core) != 128
                or not np.all(np.isfinite(per_core))
                or not all(
                    np.isfinite(float(diagnostic[name]))
                    for name in (
                        "marginal_crps",
                        "denoising_loss",
                        "maximum_absolute_log_band_variance_ratio",
                    )
                )
            ):
                raise RuntimeError("D2 milestone diagnostic is incomplete or non-finite")
            candidates.append(
                {
                    "presentations": presentations,
                    "optimizer_update": int(row["optimizer_update"]),
                    "weights": weight,
                    "energy_score": float(np.mean(per_core)),
                    "per_core_energy": per_core,
                    "marginal_crps": float(diagnostic["marginal_crps"]),
                    "denoising_loss": float(diagnostic["denoising_loss"]),
                    "variance_error": float(
                        diagnostic["maximum_absolute_log_band_variance_ratio"]
                    ),
                }
            )
    expected = sorted(allowed_presentations)
    present = sorted({candidate["presentations"] for candidate in candidates})
    if present != expected or len(candidates) != 2 * len(expected):
        raise RuntimeError("D2 milestone ladder is incomplete")
    best = min(candidates, key=lambda item: item["energy_score"])
    feasible: list[dict] = []
    comparisons = []
    for candidate in candidates:
        difference = candidate["per_core_energy"] - best["per_core_energy"]
        mean = float(np.mean(difference))
        standard_error = float(
            np.std(difference, ddof=1) / np.sqrt(len(difference))
            if len(difference) > 1
            else 0.0
        )
        within = bool(mean <= standard_error + 1.0e-12)
        crps_guard = bool(
            candidate["marginal_crps"]
            <= best["marginal_crps"]
            * (1.0 + float(config["funnel"]["internal_loss_relative_regression_maximum"]))
        )
        loss_guard = bool(
            candidate["denoising_loss"]
            <= best["denoising_loss"]
            * (1.0 + float(config["funnel"]["internal_loss_relative_regression_maximum"]))
        )
        variance_guard = bool(
            candidate["variance_error"]
            <= best["variance_error"]
            + float(config["funnel"]["internal_variance_ratio_log_regression_maximum"])
        )
        feasible_candidate = bool(within and crps_guard and loss_guard and variance_guard)
        comparisons.append(
            {
                "presentations": candidate["presentations"],
                "weights": candidate["weights"],
                "energy_score": candidate["energy_score"],
                "mean_energy_minus_best": mean,
                "paired_standard_error": standard_error,
                "within_one_standard_error": within,
                "marginal_crps_guard": crps_guard,
                "denoising_loss_guard": loss_guard,
                "band_variance_guard": variance_guard,
                "feasible": feasible_candidate,
            }
        )
        if feasible_candidate:
            feasible.append(candidate)
    selected = min(
        feasible,
        key=lambda item: (
            item["presentations"],
            item["energy_score"],
            0 if item["weights"] == "raw" else 1,
        ),
    )
    return {
        "rule": "earliest_paired_within_one_standard_error_of_best_energy",
        "selection_cores": 128,
        "best_presentations": best["presentations"],
        "best_weights": best["weights"],
        "best_energy_score": best["energy_score"],
        "selected_presentations": selected["presentations"],
        "selected_optimizer_update": selected["optimizer_update"],
        "selected_weights": selected["weights"],
        "selected_energy_score": selected["energy_score"],
        "comparisons": comparisons,
        "confirmation_used": False,
    }


def _append_jsonl(path: Path, payload: dict) -> None:
    """Atomically extend the small D2 trace; never leave a torn JSON line."""
    previous = path.read_text() if path.exists() else ""
    if previous and not previous.endswith("\n"):
        raise RuntimeError(f"D2 JSONL trace is already truncated: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(previous)
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _truncate_jsonl_to_update(path: Path, optimizer_update: int) -> None:
    """Discard uncheckpointed/duplicate rows after state-exact checkpoint restore."""
    if not path.exists():
        return
    kept: dict[int, dict] = {}
    for row in _read_jsonl_rows(path):
        update = int(row["optimizer_update"])
        if update <= optimizer_update:
            kept[update] = row
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        for update in sorted(kept):
            stream.write(json.dumps(kept[update], sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_selection(
    path: Path, arm: str, *, contract_path: Path, contract_marker: dict
) -> dict:
    marker = json.loads(path.read_text())
    frozen_inputs = marker.get("frozen_inputs", {})
    if (
        marker.get("schema_version") != SELECTION_SCHEMA
        or not marker.get("pass")
        or marker.get("stage") != "final"
        or marker.get("selected_arm") != arm
        or marker.get("ph006_used_for_selection")
        or marker.get("ph001_opened")
        or marker.get("contract_digest") != contract_marker["frozen_digest"]
        or frozen_inputs.get("contract") != str(contract_path.resolve())
        or frozen_inputs.get("contract_sha256") != sha256(contract_path)
        or marker.get("frozen_inputs_digest") != digest(frozen_inputs)
        or any(
            key.endswith("_sha256")
            and key != "contract_sha256"
            and frozen_inputs[key]
            != sha256(Path(frozen_inputs[key.removesuffix("_sha256")]))
            for key in frozen_inputs
        )
    ):
        raise PermissionError("D2 science run lacks a valid final funnel selection")
    return marker


def _load_attention_license(
    path: Path | None, *, contract_path: Path, contract_marker: dict
) -> dict:
    if path is None:
        raise PermissionError("D2 attention canary requires the frozen capacity selection")
    marker = json.loads(path.read_text())
    frozen_inputs = marker.get("frozen_inputs", {})
    if (
        marker.get("schema_version") != SELECTION_SCHEMA
        or marker.get("stage") != "capacity"
        or not marker.get("pass")
        or not marker.get("attention_licensed")
        or marker.get("selected_arm") != "modern_base8"
        or marker.get("ph006_used_for_selection")
        or marker.get("ph001_opened")
        or marker.get("contract_digest") != contract_marker["frozen_digest"]
        or frozen_inputs.get("contract") != str(contract_path.resolve())
        or frozen_inputs.get("contract_sha256") != sha256(contract_path)
    ):
        raise PermissionError("D2 attention arm was not licensed by the capacity gate")
    return marker


def _load_confirmation(
    path: Path | None,
    arm: str,
    *,
    contract_path: Path,
    contract_marker: dict,
    selection_marker: Path,
) -> dict:
    if path is None:
        raise PermissionError("D2 science continuation requires one-open confirmation")
    marker = json.loads(path.read_text())
    frozen = marker.get("frozen_inputs", {})
    if (
        marker.get("schema_version") != "p12f3-d2-internal-confirmation-v1"
        or not marker.get("pass")
        or not marker.get("internal_confirmation_opened")
        or marker.get("selected_arm") != arm
        or marker.get("ph006_used_for_selection")
        or marker.get("ph001_opened")
        or frozen.get("contract_digest") != contract_marker["frozen_digest"]
        or frozen.get("contract_sha256") != sha256(contract_path)
        or frozen.get("final_selection_sha256") != sha256(selection_marker)
        or frozen.get("final_selection") != str(selection_marker.resolve())
        or marker.get("frozen_digest") != digest(frozen)
    ):
        raise PermissionError("D2 science continuation lacks passing frozen confirmation")
    return marker


def _validate_replication_license(
    path: Path | None,
    arm: str,
    *,
    contract_path: Path,
    contract_marker: dict,
    selection_marker: Path,
    confirmation_marker: Path,
) -> dict:
    if path is None:
        raise PermissionError("D2 replication requires a post-ph006 license marker")
    marker = json.loads(path.read_text())
    if (
        marker.get("schema_version") != "p12f3-d2-second-seed-license-v1"
        or not marker.get("licensed")
        or marker.get("selected_arm") != arm
        or marker.get("ph001_opened")
        or marker.get("contract_digest") != contract_marker["frozen_digest"]
        or marker.get("contract_sha256") != sha256(contract_path)
        or marker.get("final_selection") != str(selection_marker.resolve())
        or marker.get("final_selection_sha256") != sha256(selection_marker)
        or marker.get("internal_confirmation")
        != str(confirmation_marker.resolve())
        or marker.get("internal_confirmation_sha256")
        != sha256(confirmation_marker)
        or marker.get("seed42_decision_sha256")
        != sha256(Path(marker.get("seed42_decision", "")))
    ):
        raise PermissionError("unsafe D2 second-seed license")
    frozen = {
        key: marker[key]
        for key in (
            "contract_digest",
            "contract_sha256",
            "final_selection",
            "final_selection_sha256",
            "internal_confirmation",
            "internal_confirmation_sha256",
            "seed42_decision",
            "seed42_decision_sha256",
            "selected_arm",
            "selected_presentations",
            "selected_weights",
            "ph001_opened",
        )
    }
    decision = json.loads(Path(marker["seed42_decision"]).read_text())
    if (
        marker.get("frozen_digest") != digest(frozen)
        or decision.get("schema_version") != "p12f3-d2-ph006-seed-decision-v1"
        or not decision.get("seed_pass")
        or decision.get("seed_role") != "primary"
        or int(decision.get("seed", -1)) != 42
        or decision.get("selected_arm") != arm
        or int(decision.get("selected_presentations", -1))
        != int(marker.get("selected_presentations", -1))
        or decision.get("selected_weights") != marker.get("selected_weights")
    ):
        raise PermissionError("D2 second-seed licence provenance changed")
    return marker


def _validate_existing_terminal(
    path: Path,
    *,
    schema: str,
    arm: str,
    seed: int,
    contract_digest: str,
    run_digest: str,
    config: dict,
    seed_role: str,
    gate_inputs: dict,
    replication_license: dict | None,
) -> dict:
    """Fail closed before treating an existing terminal marker as idempotent."""
    marker = json.loads(path.read_text())
    checkpoint = Path(marker.get("checkpoint", ""))
    continuation = Path(marker.get("continuation_checkpoint", ""))
    diagnostics = Path(marker.get("milestone_diagnostics", ""))
    expected_confirmation = schema == TRAINED_SCHEMA
    accumulation = int(config["funnel"]["gradient_accumulation_steps"])
    expected_presentations = int(
        config["funnel"][
            "science_total_presentations"
            if expected_confirmation
            else "canary_presentations"
        ]
    )
    selected_presentations = int(marker.get("selected_presentations", -1))
    selected_weights = marker.get("selected_weights")
    if (
        marker.get("schema_version") != schema
        or not marker.get("pass")
        or marker.get("arm") != arm
        or int(marker.get("seed", -1)) != seed
        or marker.get("seed_role") != seed_role
        or int(marker.get("presentations", -1)) != expected_presentations
        or int(marker.get("examples_seen", -1)) != expected_presentations
        or int(marker.get("optimizer_updates", -1))
        != expected_presentations // accumulation
        or marker.get("frozen_digest") != run_digest
        or selected_weights not in ("raw", "ema")
        or selected_presentations
        not in set(map(int, config["funnel"]["internal_sample_milestone_presentations"]))
        or marker.get("ph001_opened")
        or marker.get("ph006_used_for_fit")
        or marker.get("ph006_used_for_selection")
        or bool(marker.get("internal_confirmation_opened")) != expected_confirmation
        or marker.get("checkpoint_sha256") != sha256(checkpoint)
        or marker.get("continuation_checkpoint_sha256") != sha256(continuation)
        or marker.get("milestone_diagnostics_sha256") != sha256(diagnostics)
    ):
        raise RuntimeError(f"unsafe existing D2 terminal marker: {path}")
    manifest = json.loads((path.parent / "run_manifest.json").read_text())
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("frozen_digest") != marker.get("frozen_digest")
        or manifest.get("frozen", {}).get("d2_contract_digest") != contract_digest
        or manifest.get("ph001_opened")
    ):
        raise RuntimeError(f"existing D2 terminal binding changed: {path}")
    selected_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    continuation_state = torch.load(
        continuation, map_location="cpu", weights_only=False
    )
    for state, state_presentations in (
        (selected_state, selected_presentations),
        (continuation_state, expected_presentations),
    ):
        if (
            state.get("schema_version") != CHECKPOINT_SCHEMA
            or state.get("frozen_digest") != run_digest
            or state.get("arm") != arm
            or int(state.get("seed", -1)) != seed
            or int(state.get("examples_seen", -1)) != state_presentations
            or int(state.get("optimizer_update", -1))
            != state_presentations // accumulation
            or state.get("ph001_opened")
            or state.get("ph006_used_for_fit")
        ):
            raise RuntimeError(f"existing D2 terminal checkpoint changed: {path}")
    milestone_selection = marker.get("milestone_selection", {})
    if (
        int(milestone_selection.get("selected_presentations", -1))
        != selected_presentations
        or milestone_selection.get("selected_weights") != selected_weights
        or int(milestone_selection.get("selected_optimizer_update", -1))
        != selected_presentations // accumulation
        or checkpoint.resolve()
        != milestone_checkpoint_path(path.parent, selected_presentations).resolve()
    ):
        raise RuntimeError(f"existing D2 terminal milestone freeze changed: {path}")
    rows = _read_diagnostic_rows(diagnostics)
    allowed = (
        {int(config["funnel"]["canary_presentations"])}
        if not expected_confirmation
        else set(map(int, config["funnel"]["internal_sample_milestone_presentations"]))
    )
    if seed_role == "replication":
        if (
            replication_license is None
            or selected_presentations
            != int(replication_license.get("selected_presentations", -1))
            or selected_weights != replication_license.get("selected_weights")
        ):
            raise RuntimeError(f"existing D2 replication freeze changed: {path}")
    else:
        recomputed = select_earliest_within_one_se(rows, allowed, config)
        if (
            selected_presentations != int(recomputed["selected_presentations"])
            or selected_weights != recomputed["selected_weights"]
        ):
            raise RuntimeError(f"existing D2 checkpoint selection changed: {path}")
    if expected_confirmation:
        authorization_path = path.parent / "D2_SCIENCE_CONTINUATION_AUTHORIZED.json"
        authorization = json.loads(authorization_path.read_text())
        expected_authorization_frozen = {
            "run_frozen_digest": run_digest,
            "gate_inputs": gate_inputs,
            "seed_role": seed_role,
            "ph006_used_for_fit": False,
            "ph001_opened": False,
        }
        if (
            authorization.get("schema_version")
            != "p12f3-d2-science-continuation-v1"
            or not authorization.get("pass")
            or authorization.get("frozen") != expected_authorization_frozen
            or authorization.get("frozen_digest")
            != digest(expected_authorization_frozen)
            or marker.get("internal_confirmation")
            != gate_inputs["internal_confirmation"]["path"]
            or marker.get("internal_confirmation_sha256")
            != gate_inputs["internal_confirmation"]["sha256"]
            or marker.get("science_authorization")
            != str(authorization_path.resolve())
            or marker.get("science_authorization_sha256")
            != sha256(authorization_path)
        ):
            raise RuntimeError(f"existing D2 science authorization changed: {path}")
    return marker


def _checkpoint_now(
    path: Path,
    *,
    model,
    ema_state,
    optimizer,
    optimizer_update,
    examples_seen,
    run_digest,
    arm,
    seed,
    loss_sum,
    loss_count,
) -> None:
    atomic_checkpoint(
        path,
        checkpoint_payload(
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            optimizer_update=optimizer_update,
            examples_seen=examples_seen,
            frozen_digest=run_digest,
            arm=arm,
            seed=seed,
            loss_sum=loss_sum,
            loss_count=loss_count,
        ),
    )


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract_marker, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract_marker, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D2 training requires CUDA")
    roundtrip_path = args.output_root / "D2_TRANSFORM_ROUNDTRIP.json"
    if not roundtrip_path.exists():
        raise PermissionError(
            "D2 training requires the training-only conditional-transform round-trip audit"
        )
    roundtrip = json.loads(roundtrip_path.read_text())
    if (
        roundtrip.get("schema_version") != "p12f3-d2-transform-roundtrip-v1"
        or not roundtrip.get("pass")
        or not roundtrip.get("technical_pass")
        or roundtrip.get("frozen", {}).get("contract_digest")
        != contract_marker["frozen_digest"]
        or roundtrip.get("ph006_used")
        or roundtrip.get("ph001_opened")
    ):
        raise RuntimeError("unsafe D2 conditional-transform round-trip audit")
    gpu_smoke_path = args.output_root / "D2_GPU_SMOKE.json"
    if not gpu_smoke_path.exists():
        raise PermissionError("D2 training requires the one-GPU replay/memory smoke")
    gpu_smoke = json.loads(gpu_smoke_path.read_text())
    if (
        gpu_smoke.get("schema_version") != "p12f3-d2-gpu-smoke-v1"
        or not gpu_smoke.get("pass")
        or gpu_smoke.get("frozen", {}).get("contract_digest")
        != contract_marker["frozen_digest"]
        or gpu_smoke.get("frozen", {}).get("transform_roundtrip_sha256")
        != sha256(roundtrip_path)
        or gpu_smoke.get("ph006_used")
        or gpu_smoke.get("ph001_opened")
    ):
        raise RuntimeError("unsafe D2 one-GPU replay/memory smoke")
    # The D2 wrapper adds exact support_random as mask-only metadata.
    from workflows.sbi.p12f3_train_conditional_generative import load_config as load_conditional

    conditional, f3_parent, _ = load_conditional(
        Path(config["sources"]["parent_config"])
        if Path(config["sources"]["parent_config"]).is_absolute()
        else Path(__file__).resolve().parents[2] / config["sources"]["parent_config"]
    )

    frozen = contract_marker["frozen"]
    phases = tuple(frozen["training_phases"])
    selected = frozen["selected_core_ids"]
    training = frozen["training_core_ids"]
    internal = frozen["internal_validation_core_ids"]
    _, _, opened_phases, _, _, loader, store, opened_selected = _open_common(f3_parent)
    if list(opened_phases) != list(phases) or opened_selected != selected:
        store.close()
        loader.close()
        raise RuntimeError("D2 runtime core contract changed after freeze")

    seed = int(
        config["funnel"][
            "seed" if args.seed_role == "primary" else "replication_seed"
        ]
    )
    gate_inputs: dict[str, dict[str, str]] = {}
    replication_license: dict | None = None
    if args.stage == "science":
        if args.selection_marker is None:
            raise PermissionError("D2 science stage requires the frozen final selection")
        _load_selection(
            args.selection_marker,
            args.arm,
            contract_path=contract_path,
            contract_marker=contract_marker,
        )
        gate_inputs["final_selection"] = {
            "path": str(args.selection_marker.resolve()),
            "sha256": sha256(args.selection_marker),
        }
        _load_confirmation(
            args.confirmation_marker,
            args.arm,
            contract_path=contract_path,
            contract_marker=contract_marker,
            selection_marker=args.selection_marker,
        )
        gate_inputs["internal_confirmation"] = {
            "path": str(args.confirmation_marker.resolve()),
            "sha256": sha256(args.confirmation_marker),
        }
    elif args.selection_marker is not None:
        raise ValueError("canary runs may not consume a final selection marker")
    elif args.confirmation_marker is not None:
        raise ValueError("canary runs may not consume internal confirmation")
    if args.seed_role == "replication":
        if args.selection_marker is None:
            raise PermissionError("D2 replication requires the final selection marker")
        replication_license = _validate_replication_license(
            args.second_seed_license,
            args.arm,
            contract_path=contract_path,
            contract_marker=contract_marker,
            selection_marker=args.selection_marker,
            confirmation_marker=args.confirmation_marker,
        )
        gate_inputs["second_seed_license"] = {
            "path": str(args.second_seed_license.resolve()),
            "sha256": sha256(args.second_seed_license),
        }
    if args.arm == "modern_base8_attention" and args.stage == "canary":
        _load_attention_license(
            args.capacity_selection_marker,
            contract_path=contract_path,
            contract_marker=contract_marker,
        )
        gate_inputs["capacity_selection"] = {
            "path": str(args.capacity_selection_marker.resolve()),
            "sha256": sha256(args.capacity_selection_marker),
        }
    elif args.capacity_selection_marker is not None:
        raise ValueError("capacity selection marker is consumed only by the attention canary")

    output = args.output_root / "training" / args.arm / f"seed{seed}_v1"
    output.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(output / ".run.lock", purpose="P12-F3-D2 training")
    checkpoint_path = output / "checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    canary_path = output / "D2_CANARY_COMPLETE.json"
    trained_path = output / "D2_TRAINED.json"
    run_frozen = {
        "d2_contract": str(contract_path.resolve()),
        "d2_contract_sha256": sha256(contract_path),
        "d2_contract_digest": contract_marker["frozen_digest"],
        "deterministic_runtime": deterministic_runtime,
        "transform_roundtrip": str(roundtrip_path.resolve()),
        "transform_roundtrip_sha256": sha256(roundtrip_path),
        "gpu_smoke": str(gpu_smoke_path.resolve()),
        "gpu_smoke_sha256": sha256(gpu_smoke_path),
        "arm": args.arm,
        "arm_config": arm_config(config, args.arm),
        "seed": seed,
        "seed_role": args.seed_role,
        "training_core_ids": training,
        "internal_validation_core_ids": internal,
        "condition_channels": config["matched_contract"]["condition_channels"],
        "target": config["matched_contract"]["target"],
        "gradient_accumulation_steps": int(
            config["funnel"]["gradient_accumulation_steps"]
        ),
        "canary_presentations": int(config["funnel"]["canary_presentations"]),
        "maximum_presentations": int(config["funnel"]["science_total_presentations"]),
        "ph006_used_for_fit": False,
        "ph001_opened": False,
    }
    run_digest = digest(run_frozen)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("frozen_digest") != run_digest or manifest.get("ph001_opened"):
            store.close()
            loader.close()
            raise RuntimeError("D2 resume contract changed")
    elif any(path.name != ".run.lock" for path in output.iterdir()):
        store.close()
        loader.close()
        raise RuntimeError("non-empty D2 training output has no valid run manifest")
    else:
        atomic_json(
            manifest_path,
            {
                "schema_version": RUN_SCHEMA,
                "created_utc": utc_now(),
                "git_revision_at_launch": git_revision(),
                "pass": True,
                "frozen": run_frozen,
                "frozen_digest": run_digest,
                "truth_files_read": [f"{phase} training delta_R7" for phase in phases],
                "ph006_used_for_fit": False,
                "ph001_opened": False,
            },
        )
    if args.stage == "science":
        authorization_path = output / "D2_SCIENCE_CONTINUATION_AUTHORIZED.json"
        authorization_frozen = {
            "run_frozen_digest": run_digest,
            "gate_inputs": gate_inputs,
            "seed_role": args.seed_role,
            "ph006_used_for_fit": False,
            "ph001_opened": False,
        }
        authorization = {
            "schema_version": "p12f3-d2-science-continuation-v1",
            "created_utc": utc_now(),
            "pass": True,
            "frozen": authorization_frozen,
            "frozen_digest": digest(authorization_frozen),
            "ph006_used_for_fit": False,
            "ph001_opened": False,
        }
        if authorization_path.exists():
            existing = json.loads(authorization_path.read_text())
            if (
                existing.get("schema_version") != authorization["schema_version"]
                or not existing.get("pass")
                or existing.get("frozen") != authorization_frozen
                or existing.get("frozen_digest") != authorization["frozen_digest"]
                or existing.get("ph001_opened")
            ):
                raise RuntimeError("D2 science continuation authorization changed")
        else:
            if trained_path.exists():
                raise RuntimeError(
                    "D2 trained marker exists without its frozen continuation authorization"
                )
            atomic_json(authorization_path, authorization)

    terminal_path = canary_path if args.stage == "canary" else trained_path
    if terminal_path.exists():
        marker = _validate_existing_terminal(
            terminal_path,
            schema=CANARY_SCHEMA if args.stage == "canary" else TRAINED_SCHEMA,
            arm=args.arm,
            seed=seed,
            contract_digest=contract_marker["frozen_digest"],
            run_digest=run_digest,
            config=config,
            seed_role=args.seed_role,
            gate_inputs=gate_inputs,
            replication_license=replication_license,
        )
        print(json.dumps(marker, indent=2), flush=True)
        store.close()
        loader.close()
        run_lock.close()
        return

    conditional_root = Path(config["sources"]["conditional_output_root"])
    location_args = SimpleNamespace(
        output_root=conditional_root,
        gaussian_arm=config["sources"]["conditional_gaussian_arm"],
        gaussian_run=config["sources"]["conditional_gaussian_run"],
    )
    location_model, _, _, _ = load_location_scale(
        location_args, conditional, args.device
    )
    whitening_path = Path(config["sources"]["conditional_whitening"])
    whitening_marker = json.loads(whitening_path.read_text())
    if (
        whitening_marker.get("schema_version") != "p12f3-conditional-whitening-v1"
        or not whitening_marker.get("pass")
        or whitening_marker.get("validation_phase_used_for_fit")
        or whitening_marker.get("ph001_opened")
    ):
        store.close()
        loader.close()
        raise RuntimeError("unsafe D2 conditional whitening")
    whitening = whitening_marker["whitening"]
    g1_model, scaler = load_g1_model(f3_parent, args.device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(config, args.arm).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["funnel"]["learning_rate"]),
        weight_decay=float(config["funnel"]["weight_decay"]),
    )
    ema_state = clone_model_state(model)
    optimizer_update = 0
    examples_seen = 0
    loss_sum = torch.zeros((), device=args.device, dtype=torch.float64)
    loss_count = 0
    milestone_candidates = sorted(
        (output / "milestones").glob("presentations_*.pt")
        if (output / "milestones").exists()
        else (),
        key=lambda path: int(path.stem.removeprefix("presentations_")),
    )
    resume_path = checkpoint_path
    if milestone_candidates:
        newest_milestone = milestone_candidates[-1]
        if (
            not checkpoint_path.exists()
            or int(newest_milestone.stem.removeprefix("presentations_"))
            > int(
                torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )["examples_seen"]
            )
        ):
            resume_path = newest_milestone
    if resume_path.exists():
        if not args.resume:
            store.close()
            loader.close()
            raise RuntimeError("D2 checkpoint exists; explicit --resume is required")
        state = torch.load(resume_path, map_location=args.device, weights_only=False)
        if (
            state.get("schema_version") != CHECKPOINT_SCHEMA
            or state.get("frozen_digest") != run_digest
            or state.get("arm") != args.arm
            or int(state.get("seed", -1)) != seed
            or state.get("ph001_opened")
            or state.get("ph006_used_for_fit")
        ):
            store.close()
            loader.close()
            raise RuntimeError("unsafe D2 checkpoint")
        model.load_state_dict(state["model"], strict=True)
        ema_state = {
            name: value.to(device=args.device).clone()
            for name, value in state["ema_model"].items()
        }
        optimizer.load_state_dict(state["optimizer"])
        optimizer_update = int(state["optimizer_update"])
        examples_seen = int(state["examples_seen"])
        if examples_seen != optimizer_update * int(
            config["funnel"]["gradient_accumulation_steps"]
        ):
            store.close()
            loader.close()
            raise RuntimeError("D2 checkpoint presentation/update counters disagree")
        loss_sum = torch.tensor(
            float(state["loss_sum"]), device=args.device, dtype=torch.float64
        )
        loss_count = int(state["loss_count"])
        restore_checkpoint_rng(state)
    else:
        atomic_checkpoint(
            checkpoint_path,
            checkpoint_payload(
                model=model,
                ema_state=ema_state,
                optimizer=optimizer,
                optimizer_update=0,
                examples_seen=0,
                frozen_digest=run_digest,
                arm=args.arm,
                seed=seed,
                loss_sum=0.0,
                loss_count=0,
            ),
        )

    accumulation = int(config["funnel"]["gradient_accumulation_steps"])
    stage_presentations = int(
        config["funnel"][
            "canary_presentations"
            if args.stage == "canary"
            else "science_total_presentations"
        ]
    )
    if stage_presentations % accumulation:
        store.close()
        loader.close()
        raise RuntimeError("D2 stage presentations do not align with gradient accumulation")
    stage_target = stage_presentations // accumulation
    if args.stop_after_presentations is not None:
        if args.stop_after_presentations % accumulation:
            store.close()
            loader.close()
            raise ValueError("D2 stop presentations must align with gradient accumulation")
        stop = int(args.stop_after_presentations) // accumulation
    else:
        stop = stage_target
    try:
        validate_stop_position(stop, optimizer_update, stage_target)
    except ValueError:
        store.close()
        loader.close()
        raise
    if args.stage == "science" and args.seed_role == "primary" and examples_seen < int(
        config["funnel"]["canary_presentations"]
    ):
        store.close()
        loader.close()
        raise RuntimeError("D2 science stage cannot bypass its completed canary")

    refs_per_epoch = sum(len(training[phase]) for phase in phases)
    max_examples = int(config["funnel"]["science_total_presentations"])
    effective_passes = max_examples / refs_per_epoch
    if effective_passes > float(config["funnel"]["maximum_effective_training_passes"]):
        store.close()
        loader.close()
        raise RuntimeError("D2 runtime effective-pass cap changed")
    selection_refs = [
        (str(row["phase"]), int(row["core_id"]))
        for row in frozen["internal_selection_refs"]
    ]
    confirmation_refs = [
        (str(row["phase"]), int(row["core_id"]))
        for row in frozen["internal_confirmation_refs"]
    ]
    if (
        len(selection_refs) != int(config["funnel"]["internal_selection_cores"])
        or len(confirmation_refs)
        != int(config["funnel"]["internal_confirmation_cores"])
        or set(selection_refs) & set(confirmation_refs)
    ):
        raise RuntimeError("D2 frozen internal split is invalid")
    milestone_presentations = set(
        int(value)
        for value in config["funnel"]["internal_sample_milestone_presentations"]
    )
    if any(value % accumulation for value in milestone_presentations):
        raise RuntimeError("D2 diagnostic presentation is not optimizer-step aligned")
    milestones = {value // accumulation for value in milestone_presentations}
    diagnostic_path = output / "internal_diagnostics.jsonl"
    loss_path = output / "loss_trace.jsonl"
    _truncate_jsonl_to_update(loss_path, optimizer_update)
    _truncate_jsonl_to_update(diagnostic_path, optimizer_update)
    completed_diagnostics = _read_diagnostic_updates(diagnostic_path)
    started = time.monotonic()
    latest_diagnostics: dict | None = None

    def freeze_milestone_and_evaluate_selection(update: int) -> dict:
        """Persist an exact state and evaluate only the still-open 128-core split."""
        presentations = int(update * accumulation)
        if update not in milestones:
            raise RuntimeError("attempted to freeze an unregistered D2 milestone")
        milestone_path = milestone_checkpoint_path(output, presentations)
        milestone_path.parent.mkdir(parents=True, exist_ok=True)
        if milestone_path.exists():
            saved = torch.load(milestone_path, map_location="cpu", weights_only=False)
            if (
                saved.get("schema_version") != CHECKPOINT_SCHEMA
                or saved.get("frozen_digest") != run_digest
                or int(saved.get("optimizer_update", -1)) != update
                or int(saved.get("examples_seen", -1)) != presentations
                or saved.get("arm") != args.arm
                or int(saved.get("seed", -1)) != seed
            ):
                raise RuntimeError("D2 milestone checkpoint changed")
        else:
            atomic_checkpoint(
                milestone_path,
                checkpoint_payload(
                    model=model,
                    ema_state=ema_state,
                    optimizer=optimizer,
                    optimizer_update=update,
                    examples_seen=presentations,
                    frozen_digest=run_digest,
                    arm=args.arm,
                    seed=seed,
                    loss_sum=loss_sum,
                    loss_count=loss_count,
                ),
            )
        existing = {
            int(row["optimizer_update"]): row for row in _read_diagnostic_rows(diagnostic_path)
        }
        if update in existing:
            return existing[update]
        # Sampling diagnostics must not advance the persisted training RNG.
        # Checkpoints serialize the full continuation state.  Post-update GPU
        # replayability is a separate, tolerance-based preflight on the frozen
        # deterministic topology; universal bitwise identity is not claimed.
        devices = [torch.cuda.current_device()] if args.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            raw = internal_sample_diagnostics(
                model,
                selection_refs,
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                d2_config=config,
                conditional_config=conditional,
                f3_parent=f3_parent,
                device=args.device,
                whitening=whitening,
                seed=seed,
            )
            ema_model = build_model(config, args.arm).to(args.device)
            load_model_state_copy(ema_model, ema_state)
            ema = internal_sample_diagnostics(
                ema_model,
                selection_refs,
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                d2_config=config,
                conditional_config=conditional,
                f3_parent=f3_parent,
                device=args.device,
                whitening=whitening,
                seed=seed,
            )
        row = {
            "optimizer_update": update,
            "examples_seen": presentations,
            "presentations": presentations,
            "selection": {"raw": raw, "ema": ema},
            "confirmation": None,
            "milestone_checkpoint": str(milestone_path.resolve()),
            "milestone_checkpoint_sha256": sha256(milestone_path),
            "selection_uses_ph006": False,
            "ph001_opened": False,
        }
        _append_jsonl(diagnostic_path, row)
        completed_diagnostics.add(update)
        del ema_model
        model.train()
        return row

    try:
        # If an allocation ended during a long milestone diagnostic, resume it
        # before taking another optimization step so no milestone is skipped.
        if optimizer_update in milestones and optimizer_update not in completed_diagnostics:
            latest_diagnostics = freeze_milestone_and_evaluate_selection(optimizer_update)
        while optimizer_update < stop:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            update_loss = torch.zeros((), device=args.device, dtype=torch.float64)
            last_phase = ""
            last_core = -1
            for _ in range(accumulation):
                epoch = examples_seen // refs_per_epoch
                ordinal = examples_seen % refs_per_epoch
                last_phase, last_core = epoch_references(
                    training, phases, seed=seed, epoch=epoch
                )[ordinal]
                condition, target, layout, _, _, _, support, _ = build_d2_example(
                    loader=loader,
                    store=store,
                    g1_model=g1_model,
                    location_model=location_model,
                    scaler=scaler,
                    phase=last_phase,
                    core_id=last_core,
                    conditional_config=conditional,
                    f3_parent=f3_parent,
                    device=args.device,
                    whitening=whitening,
                )
                state, time_value, desired = fourier_v_pair(target)
                predicted = model(
                    state,
                    time_value,
                    condition,
                    layout=layout,
                    whitening=whitening,
                    support_mask=support,
                )
                component_band = layout.component_group // 2
                band_losses = []
                for band in range(2):
                    mask = torch.as_tensor(component_band == band, device=args.device)
                    band_losses.append(
                        torch.mean(torch.square(predicted[:, mask] - desired[:, mask]))
                    )
                micro_loss = torch.stack(band_losses).mean()
                if not torch.isfinite(micro_loss):
                    raise FloatingPointError("non-finite D2 microbatch loss")
                (micro_loss / accumulation).backward()
                update_loss += micro_loss.detach().to(torch.float64) / accumulation
                examples_seen += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["funnel"]["gradient_clip"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer_update += 1
            update_ema_state(
                ema_state,
                model,
                decay=float(config["diffusion"]["ema_decay"]),
                update=optimizer_update,
            )
            loss_sum += update_loss
            loss_count += 1
            should_log = (
                optimizer_update
                % int(config["funnel"]["loss_log_every_optimizer_updates"])
                == 0
                or optimizer_update == stop
            )
            should_checkpoint = (
                optimizer_update
                % int(config["funnel"]["checkpoint_every_optimizer_updates"])
                == 0
                or optimizer_update == stop
            )
            if should_log or should_checkpoint or optimizer_update in milestones:
                finite_parameters = torch.stack(
                    [
                        torch.isfinite(parameter).all()
                        for parameter in model.parameters()
                    ]
                ).all()
                if not bool(finite_parameters.detach().cpu()):
                    raise FloatingPointError("non-finite D2 parameter")
            if should_log:
                _append_jsonl(
                    loss_path,
                    {
                        "optimizer_update": optimizer_update,
                        "examples_seen": examples_seen,
                        "effective_passes": examples_seen / refs_per_epoch,
                        "loss": float(update_loss.detach().cpu()),
                        "mean_loss": float((loss_sum / loss_count).detach().cpu()),
                        "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                        "phase": last_phase,
                        "core_id": last_core,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            if should_checkpoint:
                atomic_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(
                        model=model,
                        ema_state=ema_state,
                        optimizer=optimizer,
                        optimizer_update=optimizer_update,
                        examples_seen=examples_seen,
                        frozen_digest=run_digest,
                        arm=args.arm,
                        seed=seed,
                        loss_sum=loss_sum,
                        loss_count=loss_count,
                    ),
                )
            if optimizer_update in milestones and optimizer_update not in completed_diagnostics:
                latest_diagnostics = freeze_milestone_and_evaluate_selection(
                    optimizer_update
                )
            if time.monotonic() - started >= args.max_wall_seconds and optimizer_update < stop:
                _checkpoint_now(
                    checkpoint_path,
                    model=model,
                    ema_state=ema_state,
                    optimizer=optimizer,
                    optimizer_update=optimizer_update,
                    examples_seen=examples_seen,
                    run_digest=run_digest,
                    arm=args.arm,
                    seed=seed,
                    loss_sum=loss_sum,
                    loss_count=loss_count,
                )
                atomic_json(
                    output / "PAUSED.json",
                    {
                        "schema_version": "p12f3-d2-paused-v1",
                        "optimizer_update": optimizer_update,
                        "examples_seen": examples_seen,
                        "frozen_digest": run_digest,
                        "pass": True,
                        "ph006_used_for_fit": False,
                        "ph001_opened": False,
                    },
                )
                raise SystemExit(75)

        if stop < stage_target:
            _checkpoint_now(
                checkpoint_path,
                model=model,
                ema_state=ema_state,
                optimizer=optimizer,
                optimizer_update=optimizer_update,
                examples_seen=examples_seen,
                run_digest=run_digest,
                arm=args.arm,
                seed=seed,
                loss_sum=loss_sum,
                loss_count=loss_count,
            )
            atomic_json(
                output / "PAUSED.json",
                {
                    "schema_version": "p12f3-d2-paused-v1",
                    "optimizer_update": optimizer_update,
                    "examples_seen": examples_seen,
                    "frozen_digest": run_digest,
                    "pass": True,
                    "ph006_used_for_fit": False,
                    "ph001_opened": False,
                },
            )
            return
        rows = _read_diagnostic_rows(diagnostic_path)
        terminal_matches = [
            row for row in rows if int(row["optimizer_update"]) == stage_target
        ]
        if len(terminal_matches) != 1:
            raise RuntimeError("D2 terminal internal diagnostic is missing or duplicated")
        latest_diagnostics = terminal_matches[0]
        allowed_presentations = (
            {int(config["funnel"]["canary_presentations"])}
            if args.stage == "canary"
            else set(int(value) for value in milestone_presentations)
        )
        if args.seed_role == "replication":
            if replication_license is None:
                raise RuntimeError("D2 replication lost its frozen seed-42 license")
            selected_presentations = int(replication_license["selected_presentations"])
            selected_weights = str(replication_license["selected_weights"])
            if (
                selected_presentations not in allowed_presentations
                or selected_weights not in ("raw", "ema")
            ):
                raise RuntimeError("D2 replication license changed checkpoint choice")
            milestone_selection = {
                "rule": "checkpoint_age_and_weights_frozen_from_passing_seed42",
                "selected_presentations": selected_presentations,
                "selected_optimizer_update": selected_presentations // accumulation,
                "selected_weights": selected_weights,
                "second_seed_license": str(args.second_seed_license.resolve()),
                "second_seed_license_sha256": sha256(args.second_seed_license),
                "confirmation_used": False,
            }
        else:
            milestone_selection = select_earliest_within_one_se(
                rows, allowed_presentations, config
            )
        selected_checkpoint = milestone_checkpoint_path(
            output, milestone_selection["selected_presentations"]
        )
        selected_state = torch.load(
            selected_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            selected_state.get("frozen_digest") != run_digest
            or int(selected_state.get("examples_seen", -1))
            != int(milestone_selection["selected_presentations"])
        ):
            raise RuntimeError("D2 selected milestone checkpoint changed")
        terminal_schema = CANARY_SCHEMA if args.stage == "canary" else TRAINED_SCHEMA
        marker_path = canary_path if args.stage == "canary" else trained_path
        immutable_continuation = (
            selected_checkpoint if args.stage == "canary" else checkpoint_path
        )
        marker = {
            "schema_version": terminal_schema,
            "created_utc": utc_now(),
            "pass": True,
            "arm": args.arm,
            "seed": seed,
            "seed_role": args.seed_role,
            "optimizer_updates": optimizer_update,
            "examples_seen": examples_seen,
            "presentations": examples_seen,
            "effective_passes": examples_seen / refs_per_epoch,
            "parameters": parameter_count(model),
            "mean_training_loss": float(
                (loss_sum / max(loss_count, 1)).detach().cpu()
            ),
            "internal_diagnostics": latest_diagnostics,
            "milestone_selection": milestone_selection,
            "milestone_diagnostics": str(diagnostic_path.resolve()),
            "milestone_diagnostics_sha256": sha256(diagnostic_path),
            "selected_weights": milestone_selection["selected_weights"],
            "selected_presentations": milestone_selection["selected_presentations"],
            "checkpoint": str(selected_checkpoint.resolve()),
            "checkpoint_sha256": sha256(selected_checkpoint),
            "continuation_checkpoint": str(immutable_continuation.resolve()),
            "continuation_checkpoint_sha256": sha256(immutable_continuation),
            "frozen_digest": run_digest,
            "ph006_used_for_fit": False,
            "ph006_used_for_selection": False,
            "internal_confirmation_opened": args.stage == "science",
            "internal_confirmation_pending": False,
            "internal_confirmation": None
            if args.stage != "science"
            else gate_inputs["internal_confirmation"]["path"],
            "internal_confirmation_sha256": None
            if args.stage != "science"
            else gate_inputs["internal_confirmation"]["sha256"],
            "science_authorization": None
            if args.stage != "science"
            else str(authorization_path.resolve()),
            "science_authorization_sha256": None
            if args.stage != "science"
            else sha256(authorization_path),
            "ph001_opened": False,
        }
        atomic_json(marker_path, marker)
        print(json.dumps(marker, indent=2), flush=True)
    finally:
        store.close()
        loader.close()
        run_lock.close()


if __name__ == "__main__":
    main()
