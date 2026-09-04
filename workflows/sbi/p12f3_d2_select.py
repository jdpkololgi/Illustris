#!/usr/bin/env python3
"""Select D2 capacity/attention using only the frozen internal train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_d2_contract import (
    CANARY_SCHEMA,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    SELECTION_SCHEMA,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--stage", choices=("capacity", "final"), required=True)
    return parser.parse_args()


def _canary_path(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / "training" / arm / f"seed{seed}_v1" / "D2_CANARY_COMPLETE.json"


def load_canary(path: Path, *, arm: str, seed: int, contract_digest: str) -> dict:
    marker = json.loads(path.read_text())
    checkpoint = Path(marker.get("checkpoint", ""))
    if (
        marker.get("schema_version") != CANARY_SCHEMA
        or not marker.get("pass")
        or marker.get("arm") != arm
        or int(marker.get("seed", -1)) != seed
        or int(marker.get("examples_seen", -1)) != 2_500
        or marker.get("ph006_used_for_fit")
        or marker.get("ph006_used_for_selection")
        or marker.get("ph001_opened")
        or marker.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise RuntimeError(f"unsafe D2 canary {path}")
    # The run-level digest is nested in each marker; its manifest binds it to the
    # experiment contract.  Check that binding rather than conflating two digests.
    manifest_path = path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != "p12f3-d2-run-v1"
        or manifest.get("frozen_digest") != marker.get("frozen_digest")
        or manifest.get("frozen", {}).get("d2_contract_digest") != contract_digest
        or manifest.get("ph001_opened")
    ):
        raise RuntimeError(f"D2 canary run binding changed: {path}")
    diagnostics = marker.get("internal_diagnostics", {})
    if (
        diagnostics.get("presentations") != 2_500
        or diagnostics.get("confirmation") is not None
        or marker.get("selected_presentations") != 2_500
        or marker.get("selected_weights") not in ("raw", "ema")
        or marker.get("internal_confirmation_opened")
    ):
        raise RuntimeError("D2 canary violated the selection-only split contract")
    return marker


def paired_energy(candidate: dict, reference: dict) -> dict:
    candidate_values = np.asarray(candidate["per_core_energy_score"], dtype=np.float64)
    reference_values = np.asarray(reference["per_core_energy_score"], dtype=np.float64)
    if candidate.get("core_keys") != reference.get("core_keys") or candidate_values.shape != reference_values.shape:
        raise RuntimeError("D2 paired internal core identities changed")
    difference = reference_values - candidate_values
    mean = float(np.mean(difference))
    standard_error = float(
        np.std(difference, ddof=1) / np.sqrt(len(difference))
        if len(difference) > 1
        else 0.0
    )
    reference_mean = float(np.mean(reference_values))
    return {
        "cores": int(len(difference)),
        "mean_reference_minus_candidate": mean,
        "standard_error": standard_error,
        "relative_improvement": mean / max(abs(reference_mean), 1.0e-12),
        "one_standard_error_lower": mean - standard_error,
    }


def _frozen_weight(marker: dict) -> tuple[str, dict]:
    weight = marker.get("selected_weights")
    if weight not in ("raw", "ema"):
        raise RuntimeError("D2 canary did not freeze raw/EMA on selection cores")
    diagnostics = marker["internal_diagnostics"]
    if diagnostics.get("confirmation") is not None:
        raise RuntimeError("D2 architecture selection may not inspect confirmation cores")
    if marker.get("milestone_selection", {}).get("selected_weights") != weight:
        raise RuntimeError("D2 canary raw/EMA freeze is inconsistent")
    return weight, diagnostics["selection"][weight]


def compare(candidate_marker: dict, reference_marker: dict, config: dict, threshold: float) -> dict:
    candidate_weight, candidate = _frozen_weight(candidate_marker)
    reference_weight, reference = _frozen_weight(reference_marker)
    paired = paired_energy(candidate, reference)
    crps_regression = (
        candidate["marginal_crps"] / max(reference["marginal_crps"], 1.0e-12) - 1.0
    )
    loss_regression = (
        candidate["denoising_loss"] / max(reference["denoising_loss"], 1.0e-12) - 1.0
    )
    variance_regression = (
        candidate["maximum_absolute_log_band_variance_ratio"]
        - reference["maximum_absolute_log_band_variance_ratio"]
    )
    eligible = bool(
        paired["relative_improvement"] >= threshold
        and paired["one_standard_error_lower"] > 0.0
        and crps_regression
        <= float(config["funnel"]["internal_loss_relative_regression_maximum"])
        and loss_regression
        <= float(config["funnel"]["internal_loss_relative_regression_maximum"])
        and variance_regression
        <= float(config["funnel"]["internal_variance_ratio_log_regression_maximum"])
    )
    return {
        "eligible": eligible,
        "selection": {
            "candidate_weight": candidate_weight,
            "reference_weight": reference_weight,
            "paired_energy": paired,
            "paired_interval_convention": "mean_minus_one_standard_error_strictly_positive",
            "marginal_crps_relative_regression": float(crps_regression),
            "denoising_loss_relative_regression": float(loss_regression),
            "log_variance_calibration_regression": float(variance_regression),
            "eligible": eligible,
        },
        "confirmation_used": False,
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    seed = int(config["funnel"]["seed"])
    output_name = (
        "D2_CAPACITY_SELECTION.json" if args.stage == "capacity" else "D2_FINAL_SELECTION.json"
    )
    output_path = args.output_root / output_name

    base4_path = _canary_path(args.output_root, "modern_base4", seed)
    base8_path = _canary_path(args.output_root, "modern_base8", seed)
    base4 = load_canary(
        base4_path, arm="modern_base4", seed=seed, contract_digest=contract["frozen_digest"]
    )
    base8 = load_canary(
        base8_path, arm="modern_base8", seed=seed, contract_digest=contract["frozen_digest"]
    )
    capacity = compare(
        base8,
        base4,
        config,
        float(config["funnel"]["capacity_energy_relative_improvement_required"]),
    )
    capacity_arm = "modern_base8" if capacity["eligible"] else "modern_base4"
    frozen_inputs = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "base4_canary": str(base4_path.resolve()),
        "base4_canary_sha256": sha256(base4_path),
        "base8_canary": str(base8_path.resolve()),
        "base8_canary_sha256": sha256(base8_path),
    }
    if args.stage == "capacity":
        selected_arm = capacity_arm
        attention_licensed = capacity_arm == "modern_base8"
        attention = None
    else:
        capacity_path = args.output_root / "D2_CAPACITY_SELECTION.json"
        capacity_marker = json.loads(capacity_path.read_text())
        if (
            capacity_marker.get("schema_version") != SELECTION_SCHEMA
            or not capacity_marker.get("pass")
            or capacity_marker.get("stage") != "capacity"
            or capacity_marker.get("contract_digest") != contract["frozen_digest"]
            or capacity_marker.get("selected_arm") != capacity_arm
            or capacity_marker.get("capacity_comparison") != capacity
            or bool(capacity_marker.get("attention_licensed"))
            != (capacity_arm == "modern_base8")
            or capacity_marker.get("ph006_used_for_selection")
            or capacity_marker.get("ph001_opened")
            or capacity_marker.get("frozen_inputs_digest") != digest(frozen_inputs)
        ):
            raise RuntimeError("D2 final selection does not match capacity freeze")
        frozen_inputs["capacity_selection"] = str(capacity_path.resolve())
        frozen_inputs["capacity_selection_sha256"] = sha256(capacity_path)
        if capacity_marker.get("attention_licensed"):
            attention_path = _canary_path(
                args.output_root, "modern_base8_attention", seed
            )
            attention_marker = load_canary(
                attention_path,
                arm="modern_base8_attention",
                seed=seed,
                contract_digest=contract["frozen_digest"],
            )
            attention = compare(
                attention_marker,
                base8,
                config,
                float(config["funnel"]["attention_energy_relative_improvement_required"]),
            )
            selected_arm = (
                "modern_base8_attention" if attention["eligible"] else "modern_base8"
            )
            frozen_inputs["attention_canary"] = str(attention_path.resolve())
            frozen_inputs["attention_canary_sha256"] = sha256(attention_path)
        else:
            attention = {"eligible": False, "not_run_reason": "base8_capacity_gate_failed"}
            selected_arm = "modern_base4"
        attention_licensed = bool(capacity_marker.get("attention_licensed"))

    marker = {
        "schema_version": SELECTION_SCHEMA,
        "created_utc": utc_now(),
        "pass": True,
        "contract_digest": contract["frozen_digest"],
        "stage": args.stage,
        "selected_arm": selected_arm,
        "attention_licensed": attention_licensed,
        "capacity_comparison": capacity,
        "attention_comparison": attention,
        "selection_unit": "training_phase_internal_128_selection_cores_only",
        "frozen_inputs": frozen_inputs,
        "frozen_inputs_digest": digest(frozen_inputs),
        "ph006_used_for_selection": False,
        "truth_files_read": ["training-phase internal-validation delta_R7 only"],
        "ph001_opened": False,
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        immutable = (
            "schema_version",
            "pass",
            "contract_digest",
            "stage",
            "selected_arm",
            "attention_licensed",
            "capacity_comparison",
            "attention_comparison",
            "selection_unit",
            "frozen_inputs",
            "frozen_inputs_digest",
            "ph006_used_for_selection",
            "truth_files_read",
            "ph001_opened",
        )
        if any(existing.get(key) != marker.get(key) for key in immutable):
            raise RuntimeError("existing D2 selection marker does not match recomputed freeze")
        marker = existing
    else:
        atomic_json(output_path, marker)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
