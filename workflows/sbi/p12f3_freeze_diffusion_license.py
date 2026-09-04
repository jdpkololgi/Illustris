#!/usr/bin/env python3
"""Freeze the registered go/no-go decision for the P12-F3 diffusion arm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-contract", type=Path, required=True)
    parser.add_argument("--observable-autopsy", type=Path, required=True)
    parser.add_argument("--gaussian-selection", type=Path, required=True)
    parser.add_argument("--flow-loss", type=Path, required=True)
    parser.add_argument("--flow-report", type=Path, required=True)
    parser.add_argument("--flow-shear", type=Path, required=True)
    parser.add_argument("--flow-visual", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def read_safe(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("ph001_opened"):
        raise RuntimeError(f"blind phase was opened in {path}")
    return value


def loss_plateau(trace_path: Path, *, window: int, maximum_improvement: float) -> dict:
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if not rows or int(rows[-1]["update"]) != 10_000:
        raise RuntimeError("F3-L2c loss trace is not complete at 10000 updates")
    previous = np.asarray(
        [row["loss"] for row in rows if 10_000 - 2 * window < row["update"] <= 10_000 - window],
        dtype=np.float64,
    )
    final = np.asarray(
        [row["loss"] for row in rows if 10_000 - window < row["update"] <= 10_000],
        dtype=np.float64,
    )
    if len(previous) < 10 or len(final) < 10:
        raise RuntimeError("insufficient loss observations for plateau gate")
    previous_mean = float(previous.mean())
    final_mean = float(final.mean())
    fractional_improvement = (previous_mean - final_mean) / max(abs(previous_mean), 1e-12)
    return {
        "window_updates": int(window),
        "previous_mean_loss": previous_mean,
        "final_mean_loss": final_mean,
        "fractional_improvement": float(fractional_improvement),
        "maximum_fractional_improvement": float(maximum_improvement),
        "pass": bool(fractional_improvement <= maximum_improvement),
    }


def paired_energy(candidate: dict, reference: dict, *, repeats: int, seed: int) -> dict:
    left = {int(row["core_id"]): float(row["energy"]) for row in candidate["per_core_proper_scores"]}
    right = {int(row["core_id"]): float(row["energy"]) for row in reference["per_core_proper_scores"]}
    if set(left) != set(right) or len(left) != 256:
        raise RuntimeError("paired energy cores do not match")
    core = np.asarray(sorted(left), dtype=np.int64)
    difference = np.asarray([left[value] - right[value] for value in core], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = difference[rng.integers(0, len(difference), size=(repeats, len(difference)))].mean(axis=1)
    quantile = np.quantile(bootstrap, (0.025, 0.5, 0.975))
    return {
        "candidate_minus_reference_mean": float(difference.mean()),
        "q025_q50_q975": quantile.tolist(),
        "cores": int(len(core)),
        "bootstrap_repeats": int(repeats),
        "pass": bool(quantile[-1] < 0.0),
    }


def science_gates(
    report: dict, shear: dict, visual: dict, reference: dict, contract: dict, *, seed: int
) -> dict:
    limits = contract["science_gates"]
    ratios = np.asarray(visual["posterior_to_truth_power"][:2], dtype=np.float64)
    proper_names = ("energy", "coarse_energy", "marginal_crps", "variogram_p0p5")
    proper_fractional = {
        name: float(report["proper_scores"][name] / reference["proper_scores"][name] - 1.0)
        for name in proper_names
    }
    paired = paired_energy(
        report,
        reference,
        repeats=int(limits["primary_energy_paired_bootstrap_repeats"]),
        seed=seed,
    )
    gates = {
        "low_band_power": {
            "values": ratios.tolist(),
            "limit": float(limits["low_band_power_ratio_absolute_tolerance"]),
            "pass": bool(np.all(np.abs(ratios - 1.0) <= limits["low_band_power_ratio_absolute_tolerance"])),
        },
        "ordered_eigenvalue_tarp": {
            "value": float(report["tarp"]["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"]),
            "limit": float(limits["ordered_eigenvalue_tarp_maximum"]),
        },
        "eigengap_tarp": {
            "value": float(report["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"]),
            "limit": float(limits["eigengap_tarp_maximum"]),
        },
        "five_shear_tarp": {
            "value": float(shear["joint_tarp_blocked"]["full_max_abs_ecp_minus_alpha"]),
            "limit": float(limits["five_shear_tarp_maximum"]),
        },
        "global_coverage": {
            "value": float(max(report["global_coverage_error"].values())),
            "limit": float(limits["global_coverage_error_maximum"]),
        },
        "conditional_coverage": {
            "value": float(report["maximum_conditional_coverage_error"]),
            "limit": float(limits["conditional_coverage_error_maximum"]),
        },
        "proper_score_non_worsening": {
            "fractional_changes": proper_fractional,
            "limit": float(limits["proper_score_worsening_maximum"]),
            "pass": bool(max(proper_fractional.values()) <= limits["proper_score_worsening_maximum"]),
        },
        "primary_energy_paired_improvement": paired,
    }
    for name in (
        "ordered_eigenvalue_tarp", "eigengap_tarp", "five_shear_tarp",
        "global_coverage", "conditional_coverage",
    ):
        gates[name]["pass"] = bool(gates[name]["value"] <= gates[name]["limit"])
    gates["all_pass"] = bool(all(row["pass"] for row in gates.values() if isinstance(row, dict)))
    return gates


def main() -> None:
    args = parse_args()
    contract = read_safe(args.decision_contract)
    if contract.get("schema_version") != "p12f3-conditional-decision-v1":
        raise RuntimeError("wrong conditional-decision contract")
    parent = Path(contract["parent_config"])
    if not parent.is_absolute():
        parent = REPO_ROOT / parent
    if sha256(parent) != contract["parent_config_sha256"]:
        raise RuntimeError("conditional parent config changed")
    observable = read_safe(args.observable_autopsy)
    selection = read_safe(args.gaussian_selection)
    report = read_safe(args.flow_report)
    shear = read_safe(args.flow_shear)
    visual = read_safe(args.flow_visual)
    reference = read_safe(args.reference_report)
    plateau_contract = contract["diffusion_license"]["f3l2c_internal_plateau"]
    plateau = loss_plateau(
        args.flow_loss,
        window=int(plateau_contract["window_updates"]),
        maximum_improvement=float(plateau_contract["maximum_fractional_loss_improvement_final_over_previous"]),
    )
    gates = science_gates(report, shear, visual, reference, contract, seed=args.seed)
    license_gates = {
        "observable_proxy_signal": bool(observable["diagnosis"]["observable_proxy_signal"]),
        "conditional_gaussian_headroom": bool(
            selection.get("pass") and selection.get("aligned_proxy_gate_pass")
            and selection.get("selected_arm") == "proxy7"
        ),
        "f3l2c_internal_plateau": bool(plateau["pass"]),
        "remaining_non_gaussian_or_sampler_limitation": bool(not gates["all_pass"]),
    }
    licensed = bool(all(license_gates.values()))
    inputs = (
        args.decision_contract, args.observable_autopsy, args.gaussian_selection,
        args.flow_loss, args.flow_report, args.flow_shear, args.flow_visual,
        args.reference_report,
    )
    payload = {
        "schema_version": "p12f3-diffusion-license-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "licensed": licensed,
        "license_gates": license_gates,
        "plateau": plateau,
        "f3l2c_science_gates": gates,
        "interpretation": contract["diffusion_license"]["interpretation"],
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
