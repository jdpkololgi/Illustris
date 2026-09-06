#!/usr/bin/env python3
"""Open the D2 127-core arm-comparison confirmation split exactly once.

The 128-core split has already frozen the sequential A1-vs-A0 and, when
licensed, A2-vs-A1 decisions and each arm's raw/EMA choice.  This script repeats
those *paired contrasts* on the disjoint 127 cores with common sampling seeds.
It can close D2, but cannot switch to a runner-up or otherwise tune the funnel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import (
    configure_d2_determinism,
    load_model_state_copy,
)
from workflows.sbi.p12f3_d2_select import compare, load_canary
from workflows.sbi.p12f3_d2_train import build_model, internal_sample_diagnostics
from workflows.sbi.p12f3_train_conditional_generative import (
    load_config as load_conditional,
    load_location_scale,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model


SCHEMA = "p12f3-d2-internal-confirmation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--selection-marker", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _pseudo_marker(canary: dict, diagnostic: dict) -> dict:
    weight = str(canary["selected_weights"])
    return {
        "selected_weights": weight,
        "milestone_selection": {"selected_weights": weight},
        "internal_diagnostics": {
            "selection": {weight: diagnostic},
            "confirmation": None,
        },
    }


def confirmation_consistency(
    selection_comparison: dict, confirmation_comparison: dict
) -> dict:
    expected = bool(selection_comparison["eligible"])
    observed = bool(confirmation_comparison["eligible"])
    return {
        "selection_eligible": expected,
        "confirmation_eligible": observed,
        "decision_reproduced": expected == observed,
        "pass": expected == observed,
    }


def _evaluate_canary(
    canary: dict,
    refs,
    *,
    config,
    conditional,
    f3_parent,
    loader,
    store,
    g1_model,
    location_model,
    scaler,
    whitening,
    device,
    seed,
) -> dict:
    checkpoint = Path(canary["checkpoint"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = str(canary["selected_weights"])
    model = build_model(config, str(canary["arm"])).to(device)
    load_model_state_copy(
        model, state["model"] if weights == "raw" else state["ema_model"]
    )
    model.eval().requires_grad_(False)
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        result = internal_sample_diagnostics(
            model,
            refs,
            loader=loader,
            store=store,
            g1_model=g1_model,
            location_model=location_model,
            scaler=scaler,
            d2_config=config,
            conditional_config=conditional,
            f3_parent=f3_parent,
            device=device,
            whitening=whitening,
            seed=seed,
        )
    del model
    return result


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    configure_d2_determinism(config["reproducibility"], args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D2 internal confirmation requires CUDA")
    selection_path = args.selection_marker or args.output_root / "D2_FINAL_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("schema_version") != "p12f3-d2-funnel-selection-v1"
        or not selection.get("pass")
        or selection.get("stage") != "final"
        or selection.get("contract_digest") != contract["frozen_digest"]
        or selection.get("ph006_used_for_selection")
        or selection.get("ph001_opened")
    ):
        raise RuntimeError("unsafe D2 final selection")
    seed = int(config["funnel"]["seed"])
    arm_paths = {
        arm: args.output_root
        / "training"
        / arm
        / f"seed{seed}_v1"
        / "D2_CANARY_COMPLETE.json"
        for arm in ("modern_base4", "modern_base8", "modern_base8_attention")
    }
    canaries = {
        arm: load_canary(
            path,
            arm=arm,
            seed=seed,
            contract_digest=contract["frozen_digest"],
        )
        for arm, path in arm_paths.items()
        if path.exists()
    }
    if set(canaries) < {"modern_base4", "modern_base8"}:
        raise RuntimeError("D2 confirmation lacks frozen A0/A1 canaries")
    expects_attention = bool(selection.get("attention_licensed"))
    if expects_attention != ("modern_base8_attention" in canaries):
        raise RuntimeError("D2 A2 execution does not match its frozen license")
    frozen_inputs = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "contract_digest": contract["frozen_digest"],
        "final_selection": str(selection_path.resolve()),
        "final_selection_sha256": sha256(selection_path),
        "canaries": {
            arm: {
                "path": str(arm_paths[arm].resolve()),
                "sha256": sha256(arm_paths[arm]),
                "checkpoint": marker["checkpoint"],
                "checkpoint_sha256": marker["checkpoint_sha256"],
                "selected_weights": marker["selected_weights"],
            }
            for arm, marker in canaries.items()
        },
        "policy": config["funnel"]["internal_confirmation_policy"],
        "ph001_opened": False,
    }
    frozen_digest = digest(frozen_inputs)
    output = args.output_root / "D2_INTERNAL_CONFIRMATION.json"
    if output.exists():
        existing = json.loads(output.read_text())
        contrast_pass = bool(
            existing.get("paired_contrasts")
            and all(
                bool(row.get("pass"))
                for row in existing.get("paired_contrasts", {}).values()
            )
        )
        if (
            existing.get("schema_version") != SCHEMA
            or existing.get("frozen_inputs") != frozen_inputs
            or existing.get("frozen_digest") != frozen_digest
            or existing.get("ph001_opened")
            or existing.get("ph006_used_for_selection")
            or not existing.get("internal_confirmation_opened")
            or existing.get("selected_arm") != selection.get("selected_arm")
            or bool(existing.get("pass")) != contrast_pass
            or set(existing.get("arm_diagnostics", {})) != set(canaries)
            or any(
                int(row.get("cores", -1))
                != int(config["funnel"]["internal_confirmation_cores"])
                for row in existing.get("arm_diagnostics", {}).values()
            )
        ):
            raise RuntimeError("D2 internal confirmation was already opened under another freeze")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    lock = acquire_run_lock(
        args.output_root / ".confirmation.lock",
        purpose="P12-F3-D2 one-open paired confirmation",
    )
    repo_root = Path(__file__).resolve().parents[2]
    parent_path = Path(config["sources"]["parent_config"])
    if not parent_path.is_absolute():
        parent_path = repo_root / parent_path
    conditional, f3_parent, _ = load_conditional(parent_path)
    _, _, phases, _, _, loader, store, selected = _open_common(f3_parent)
    try:
        if (
            list(phases) != contract["frozen"]["training_phases"]
            or selected != contract["frozen"]["selected_core_ids"]
        ):
            raise RuntimeError("D2 confirmation runtime data contract changed")
        location_args = SimpleNamespace(
            output_root=Path(config["sources"]["conditional_output_root"]),
            gaussian_arm=config["sources"]["conditional_gaussian_arm"],
            gaussian_run=config["sources"]["conditional_gaussian_run"],
        )
        location_model, _, _, _ = load_location_scale(
            location_args, conditional, args.device
        )
        whitening_marker = json.loads(
            Path(config["sources"]["conditional_whitening"]).read_text()
        )
        if (
            not whitening_marker.get("pass")
            or whitening_marker.get("validation_phase_used_for_fit")
            or whitening_marker.get("ph001_opened")
        ):
            raise RuntimeError("unsafe D2 confirmation whitening")
        whitening = whitening_marker["whitening"]
        g1_model, scaler = load_g1_model(f3_parent, args.device)
        refs = [
            (str(row["phase"]), int(row["core_id"]))
            for row in contract["frozen"]["internal_confirmation_refs"]
        ]
        if len(refs) != int(config["funnel"]["internal_confirmation_cores"]):
            raise RuntimeError("D2 internal confirmation split changed")
        diagnostics = {}
        common_seed = seed + 1_000_000
        for arm, canary in canaries.items():
            diagnostics[arm] = _evaluate_canary(
                canary,
                refs,
                config=config,
                conditional=conditional,
                f3_parent=f3_parent,
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                whitening=whitening,
                device=args.device,
                seed=common_seed,
            )

        def contrast(candidate: str, reference: str, selection_row: dict, threshold: float) -> dict:
            confirmed = compare(
                _pseudo_marker(canaries[candidate], diagnostics[candidate]),
                _pseudo_marker(canaries[reference], diagnostics[reference]),
                config,
                threshold,
            )
            consistency = confirmation_consistency(selection_row, confirmed)
            return {
                "candidate": candidate,
                "reference": reference,
                "selection": selection_row,
                "confirmation": confirmed,
                "consistency": consistency,
                "pass": bool(consistency["pass"]),
            }

        contrasts = {
            "capacity": contrast(
                "modern_base8",
                "modern_base4",
                selection["capacity_comparison"],
                float(config["funnel"]["capacity_energy_relative_improvement_required"]),
            )
        }
        if expects_attention:
            contrasts["attention"] = contrast(
                "modern_base8_attention",
                "modern_base8",
                selection["attention_comparison"],
                float(config["funnel"]["attention_energy_relative_improvement_required"]),
            )
        passed = bool(all(row["pass"] for row in contrasts.values()))
        marker = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "pass": passed,
            "selected_arm": selection["selected_arm"],
            "arm_diagnostics": diagnostics,
            "paired_contrasts": contrasts,
            "failure_action": None if passed else "close_D2_without_runner_up",
            "frozen_inputs": frozen_inputs,
            "frozen_digest": frozen_digest,
            "internal_confirmation_opened": True,
            "ph006_used_for_selection": False,
            "truth_files_read": ["training-phase internal-validation delta_R7 only"],
            "ph001_opened": False,
        }
        atomic_json(output, marker)
        print(json.dumps(marker, indent=2, sort_keys=True))
    finally:
        store.close()
        loader.close()
        lock.close()


if __name__ == "__main__":
    main()
