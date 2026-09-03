#!/usr/bin/env python3
"""Freeze the one-shot F3-L2 anti-gaming decision from matched ph006 evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import paired_core_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3l2_fourier_decision_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--shear-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def report_identity(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        report.get("phase"),
        int(report.get("cores", -1)),
        int(report.get("draws", -1)),
        tuple(int(row["core_id"]) for row in report["per_core_proper_scores"]),
    )


def proper_by_core(report: dict[str, Any], score: str) -> np.ndarray:
    rows = sorted(report["per_core_proper_scores"], key=lambda row: int(row["core_id"]))
    return np.asarray([float(row[score]) for row in rows], dtype=np.float64)


def build_decision(
    config: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    shear: dict[str, dict[str, Any]],
    visual: dict[str, Any],
) -> dict[str, Any]:
    expected = set(config["expected_methods"])
    if set(reports) != expected or set(shear) != expected or set(visual["methods"]) != expected:
        raise RuntimeError("F3-L2 requires the exact registered method triplet")
    identity = report_identity(reports["g1_wide_h24"])
    shear_identity = (
        int(shear["g1_wide_h24"].get("galaxies", -1)),
        shear["g1_wide_h24"].get("core_id_sha256"),
    )
    for method in expected:
        report = reports[method]
        shear_row = shear[method]
        if (
            report.get("schema_version") != "p12f-common-evaluation-report-v1"
            or report.get("method") != method
            or report_identity(report) != identity
            or report.get("ph001_opened")
            or not report.get("physics_closure", {}).get("all_finite")
            or not report.get("physics_closure", {}).get("all_ordered")
        ):
            raise RuntimeError(f"unsafe or unmatched common report for {method}")
        if (
            shear_row.get("schema_version") != "p12f3l2-lowmode-shear-audit-v1"
            or shear_row.get("method") != method
            or shear_row.get("phase") != "ph006"
            or shear_row.get("ph001_opened")
            or int(shear_row.get("cores", -1)) != identity[1]
            or int(shear_row.get("draws", -1)) != identity[2]
            or (
                int(shear_row.get("galaxies", -1)),
                shear_row.get("core_id_sha256"),
            ) != shear_identity
        ):
            raise RuntimeError(f"unsafe or unmatched shear report for {method}")
    if identity[:3] != ("ph006", int(config["expected_cores"]), int(config["expected_draws"])):
        raise RuntimeError("F3-L2 panel identity differs from the frozen gate")
    if visual.get("phase") != "ph006" or visual.get("ph001_opened"):
        raise PermissionError("F3-L2 visual audit is not sealed ph006 evidence")

    gates = config["gates"]
    flow_visual = visual["methods"]["fourier_flow_h24"]
    g1_visual = visual["methods"]["g1_wide_h24"]
    power = []
    for index in range(int(config["registered_low_bands"])):
        value = float(flow_visual["posterior_to_truth_power"][index])
        reference = float(g1_visual["posterior_to_truth_power"][index])
        error = abs(value - 1.0)
        reference_error = abs(reference - 1.0)
        improvement = (reference_error - error) / max(reference_error, 1e-12)
        passed = (
            error <= float(gates["low_band_power_ratio_absolute_tolerance"])
            and improvement >= float(gates["low_band_error_improvement_over_g1"])
        )
        power.append({
            "band": index + 1,
            "g1_ratio": reference,
            "flow_ratio": value,
            "flow_absolute_error": error,
            "fractional_error_improvement_over_g1": improvement,
            "pass": bool(passed),
        })

    flow = reports["fourier_flow_h24"]
    flow_shear = shear["fourier_flow_h24"]
    scalar = {
        "ordered_eigen_tarp": {
            "value": float(
                flow["tarp"]["ordered_eigenvalues"]
                ["full_max_abs_ecp_minus_alpha"]
            ),
            "limit": float(gates["joint_ordered_eigen_tarp_maximum"]),
        },
        "eigengap_tarp": {
            "value": float(
                flow["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"]
            ),
            "limit": float(gates["joint_eigengap_tarp_maximum"]),
        },
        "global_coverage": {
            "value": max(float(v) for v in flow["global_coverage_error"].values()),
            "limit": float(gates["global_coverage_error_maximum"]),
        },
        "conditional_coverage": {
            "value": float(flow["maximum_conditional_coverage_error"]),
            "limit": float(gates["conditional_coverage_error_maximum"]),
        },
        "shear_marginal_coverage": {
            "value": float(flow_shear["maximum_marginal_coverage_error"]),
            "limit": float(gates["traceless_shear_coverage_error_maximum"]),
        },
        "shear_joint_tarp": {
            "value": float(
                flow_shear["joint_tarp_blocked"]
                ["full_max_abs_ecp_minus_alpha"]
            ),
            "limit": float(gates["traceless_shear_joint_tarp_maximum"]),
        },
    }
    for row in scalar.values():
        row["pass"] = bool(row["value"] <= row["limit"])

    g1 = reports["g1_wide_h24"]
    proper = {}
    for score in ("energy", "coarse_energy", "variogram_p0p5", "marginal_crps"):
        candidate = float(flow["proper_scores"][score])
        reference = float(g1["proper_scores"][score])
        worsening = (candidate - reference) / max(abs(reference), 1e-12)
        proper[score] = {
            "g1": reference,
            "flow": candidate,
            "fractional_worsening": worsening,
            "pass": bool(worsening <= float(gates["proper_score_worsening_maximum"])),
        }
    primary_bootstrap = paired_core_bootstrap(
        proper_by_core(flow, "energy"),
        proper_by_core(g1, "energy"),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["bootstrap_seed"]),
    )

    groups = {
        "low_band_power": bool(all(row["pass"] for row in power)),
        "joint_and_coverage": bool(all(row["pass"] for row in scalar.values())),
        "proper_scores": bool(all(row["pass"] for row in proper.values())),
    }
    passed = bool(all(groups.values()))
    power_pass = groups["low_band_power"]
    dependence_pass = bool(
        scalar["ordered_eigen_tarp"]["pass"]
        and scalar["eigengap_tarp"]["pass"]
        and scalar["shear_joint_tarp"]["pass"]
    )
    if passed:
        diagnosis = "direct Fourier flow passes; license one matched Fourier diffusion comparison"
    elif power_pass and not dependence_pass:
        diagnosis = "band amplitudes repaired but tidal dependence remains wrong; stop hierarchical G1 repair"
    else:
        diagnosis = "direct Fourier low-mode posterior fails the simultaneous anti-gaming gate"
    return {
        "schema_version": "p12f3l2-fourier-decision-v1",
        "pass": passed,
        "decision": "advance_to_matched_fourier_diffusion" if passed else "stop_hierarchical_g1_repair",
        "diagnosis": diagnosis,
        "gate_groups": groups,
        "low_band_power": power,
        "joint_and_coverage": scalar,
        "proper_scores": proper,
        "paired_primary_energy_bootstrap": primary_bootstrap,
        "matched_identity": {
            "phase": identity[0], "cores": identity[1], "draws": identity[2],
            "core_id": list(identity[3]),
        },
        "ph001_opened": False,
        "truth_files_read": ["ph006 density/T-web"],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f3l2-fourier-decision-v1":
        raise RuntimeError("unsupported F3-L2 decision config")
    reports, shear, artifacts = {}, {}, {"reports": {}, "shear": {}}
    for path in args.report:
        row = json.loads(path.read_text()); method = str(row["method"])
        reports[method] = row; artifacts["reports"][method] = {"path": str(path.resolve()), "sha256": sha256(path)}
    for path in args.shear_report:
        row = json.loads(path.read_text()); method = str(row["method"])
        shear[method] = row; artifacts["shear"][method] = {"path": str(path.resolve()), "sha256": sha256(path)}
    visual = json.loads(args.visual_audit.read_text())
    decision = build_decision(config, reports, shear, visual)
    decision.update({
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "visual_audit": str(args.visual_audit.resolve()), "visual_audit_sha256": sha256(args.visual_audit),
        "artifacts": artifacts,
    })
    atomic_json(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
