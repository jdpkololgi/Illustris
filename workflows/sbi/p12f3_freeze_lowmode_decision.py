#!/usr/bin/env python3
"""Freeze the registered P12-F3 low-mode decision from matched ph006 reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import paired_core_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_lowmode_decision_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _report_identity(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        report.get("phase"),
        report.get("cores"),
        report.get("draws"),
        report.get("conditioning_contract_sha256"),
        report.get("target_scaler_sha256"),
        tuple(int(row["core_id"]) for row in report["per_core_proper_scores"]),
    )


def _proper_by_core(report: dict[str, Any], score: str) -> np.ndarray:
    rows = sorted(report["per_core_proper_scores"], key=lambda row: int(row["core_id"]))
    return np.asarray([float(row[score]) for row in rows], dtype=np.float64)


def build_decision(
    config: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    visual: dict[str, Any],
) -> dict[str, Any]:
    expected = set(config["expected_methods"])
    if set(reports) != expected or set(visual["methods"]) != expected:
        raise RuntimeError("P12-F3 decision requires the exact four registered methods")
    if visual.get("ph001_opened") or visual.get("phase") != "ph006":
        raise PermissionError("visual audit is not sealed ph006 evidence")

    identity = _report_identity(next(iter(reports.values())))
    for method, report in reports.items():
        if report.get("schema_version") != "p12f-common-evaluation-report-v1":
            raise RuntimeError(f"unsupported report schema for {method}")
        if report.get("method") != method or _report_identity(report) != identity:
            raise RuntimeError(f"unmatched report identity for {method}")
        if report.get("ph001_opened") or report.get("truth_files_read") != ["ph006 density/T-web"]:
            raise PermissionError(f"invalid phase provenance for {method}")
        closure = report.get("physics_closure", {})
        if not closure.get("all_finite") or not closure.get("all_ordered"):
            raise RuntimeError(f"physics closure failed for {method}")
    if identity[0] != "ph006" or identity[1] != config["expected_cores"] or identity[2] != config["expected_draws"]:
        raise RuntimeError("panel size, phase, or draw count differs from decision contract")

    gates = config["gates"]
    wide = reports["hybrid_wide_h24"]
    local = reports["hybrid_local_h8"]
    g1 = reports["g1_wide_h24"]
    visual_wide = visual["methods"]["hybrid_wide_h24"]
    visual_local = visual["methods"]["hybrid_local_h8"]
    visual_g1 = visual["methods"]["g1_wide_h24"]

    band_rows = []
    band_passes = []
    for band in range(int(config["registered_low_bands"])):
        wide_error = abs(float(visual_wide["posterior_to_truth_power"][band]) - 1.0)
        local_error = abs(float(visual_local["posterior_to_truth_power"][band]) - 1.0)
        g1_error = abs(float(visual_g1["posterior_to_truth_power"][band]) - 1.0)
        improvement = (local_error - wide_error) / max(local_error, 1e-12)
        passed = (
            wide_error < local_error
            and wide_error < g1_error
            and improvement >= float(gates["low_band_error_improvement_vs_local"])
        )
        band_passes.append(passed)
        band_rows.append(
            {
                "band_index": band,
                "g1_ratio": float(visual_g1["posterior_to_truth_power"][band]),
                "local_ratio": float(visual_local["posterior_to_truth_power"][band]),
                "wide_ratio": float(visual_wide["posterior_to_truth_power"][band]),
                "fractional_error_improvement_vs_local": float(improvement),
                "pass": bool(passed),
            }
        )

    eigen_worsening = (
        float(visual_wide["eigen_tarp"]["maximum_deviation"])
        - float(visual_g1["eigen_tarp"]["maximum_deviation"])
    )
    tarp_coverage = {
        "eigengap_tarp": {
            "value": float(visual_wide["gap_tarp"]["maximum_deviation"]),
            "limit": float(gates["eigengap_tarp_maximum_deviation"]),
        },
        "ordered_eigen_tarp_worsening_vs_g1": {
            "value": float(eigen_worsening),
            "limit": float(gates["ordered_eigen_tarp_absolute_worsening"]),
        },
        "global_coverage_error": {
            "value": max(float(value) for value in wide["global_coverage_error"].values()),
            "limit": float(gates["global_coverage_error"]),
        },
        "conditional_coverage_error": {
            "value": float(wide["maximum_conditional_coverage_error"]),
            "limit": float(gates["conditional_coverage_error"]),
        },
    }
    for row in tarp_coverage.values():
        row["pass"] = bool(row["value"] <= row["limit"])

    proper = {}
    for score in ("energy", "coarse_energy", "variogram_p0p5", "marginal_crps"):
        candidate_value = float(wide["proper_scores"][score])
        reference_value = float(g1["proper_scores"][score])
        worsening = (candidate_value - reference_value) / max(abs(reference_value), 1e-12)
        proper[score] = {
            "g1": reference_value,
            "wide_hybrid": candidate_value,
            "fractional_worsening": float(worsening),
            "pass": bool(worsening <= float(gates["proper_score_fractional_worsening"])),
        }

    bootstrap = {}
    for comparator, reference in (("g1_wide_h24", g1), ("hybrid_local_h8", local)):
        result = paired_core_bootstrap(
            _proper_by_core(wide, "energy"),
            _proper_by_core(reference, "energy"),
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]),
        )
        result["pass"] = bool(result["interval95"][0] > 0.0)
        bootstrap[comparator] = result

    gate_groups = {
        "low_band_power": bool(all(band_passes)),
        "tarp_and_coverage": bool(all(row["pass"] for row in tarp_coverage.values())),
        "proper_scores": bool(all(row["pass"] for row in proper.values())),
        "paired_primary_score": bool(all(row["pass"] for row in bootstrap.values())),
    }
    passed = bool(all(gate_groups.values()))
    return {
        "schema_version": "p12f3-lowmode-decision-v1",
        "pass": passed,
        "decision": "advance_to_f3h" if passed else "stop_f3l_do_not_launch_f3h",
        "scientific_interpretation": (
            "The learned wide low-mode field improves eigengap calibration, but it does not "
            "jointly reproduce the low-band amplitudes, ordered-eigenvalue dependence, "
            "conditional coverage, and field proper scores required for promotion."
        ),
        "gate_groups": gate_groups,
        "low_band_power": band_rows,
        "tarp_and_coverage": tarp_coverage,
        "proper_scores": proper,
        "paired_energy_bootstrap": bootstrap,
        "matched_identity": {
            "phase": identity[0],
            "cores": identity[1],
            "draws": identity[2],
            "conditioning_contract_sha256": identity[3],
            "target_scaler_sha256": identity[4],
            "core_id": list(identity[5]),
        },
        "ph001_opened": False,
        "truth_files_read": ["ph006 density/T-web"],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f3-lowmode-decision-v1":
        raise RuntimeError("unsupported P12-F3 decision config")
    reports: dict[str, dict[str, Any]] = {}
    artifacts = {}
    for path in args.report:
        payload = json.loads(path.read_text())
        method = str(payload["method"])
        if method in reports:
            raise RuntimeError(f"duplicate report for {method}")
        reports[method] = payload
        artifacts[method] = {"path": str(path.resolve()), "sha256": sha256(path)}
    visual = json.loads(args.visual_audit.read_text())
    decision = build_decision(config, reports, visual)
    decision.update(
        {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "visual_audit": str(args.visual_audit.resolve()),
            "visual_audit_sha256": sha256(args.visual_audit),
            "report_artifacts": artifacts,
        }
    )
    atomic_json(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
