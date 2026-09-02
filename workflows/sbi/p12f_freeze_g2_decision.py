#!/usr/bin/env python3
"""Freeze the bounded P12-F v2 G2 finalist/no-finalist decision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--proper-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.g2_report.read_text())
    proper = json.loads(args.proper_scores.read_text())
    if (
        report.get("method") != "gaussian_shell_correlated_g2"
        or report.get("phase") != "ph006"
        or report.get("ph001_opened")
        or proper.get("ph001_opened")
    ):
        raise RuntimeError("G2 decision received unsafe or mismatched evidence")
    thresholds = config["selection_gates"]
    nested = report["nested_draw_reports"]["256"]
    tarp = nested["tarp"]
    tarp_maximum = max(
        float(tarp[name]["maximum_deviation"])
        for name in ("ordered_eigenvalues", "eigengaps")
    )
    seed_summary = tarp["reference_seed_maxima"]
    seed_p90 = max(float(seed_summary["ordered_p90"]), float(seed_summary["eigengap_p90"]))
    scalar_reports = [nested["voxel"]]
    scalar_reports.extend(nested["ordered_eigenvalues"].values())
    scalar_reports.extend(nested["eigengaps"].values())
    global_coverage = max(
        float(row["coverage"][level]["absolute_error"])
        for row in scalar_reports
        for level in ("0.68", "0.90")
    )
    conditional = report["conditional_reports_256_draws"]
    conditional_scalar = max(
        float(row["coverage"][level]["absolute_error"])
        for variable in (
            "random_response",
            "boundary_distance",
            "tracer_density",
            "true_environment",
        )
        for row in conditional[variable].values()
        for level in ("0.68", "0.90")
    )
    conditional_shell_tarp = max(
        float(row[key]["maximum_deviation"])
        for row in conditional["shell"].values()
        for key in ("ordered_eigenvalue_tarp", "eigengap_tarp")
    )
    physics = report["physics_closure"]
    gates = {
        "finite_and_physics": bool(
            physics["all_finite"]
            and physics["all_ordered"]
            and not physics["additional_gaussian_smoothing"]
            and float(physics["maximum_trace_max_abs"]) <= 1e-5
        ),
        "tarp": tarp_maximum <= float(thresholds["tarp"]),
        "tarp_reference_seed_p90": seed_p90
        <= float(thresholds["tarp_reference_seed_p90"]),
        "global_coverage": global_coverage
        <= float(thresholds["global_coverage"]),
        "conditional_coverage": max(conditional_scalar, conditional_shell_tarp)
        <= float(thresholds["conditional_coverage"]),
        "proper_scores": bool(proper["gates"]["pass"]),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": (
            "p12f-v2-field-finalist-v1" if passed else "p12f-v2-no-field-finalist-v1"
        ),
        "created_utc": utc_now(),
        "decision": "promote_g2_local_patch_challenger" if passed else "no_field_finalist",
        "candidate_pass": passed,
        "pass": True,
        "gates": gates,
        "diagnostics": {
            "tarp_maximum_deviation": tarp_maximum,
            "tarp_reference_seed_p90": seed_p90,
            "global_coverage_maximum_error": global_coverage,
            "conditional_scalar_maximum_error": conditional_scalar,
            "conditional_shell_tarp_maximum": conditional_shell_tarp,
        },
        "config_sha256": sha256(args.config),
        "g2_report_sha256": sha256(args.g2_report),
        "proper_scores_sha256": sha256(args.proper_scores),
        "scope": "local patch posterior only",
        "full_cap_coherence_established": False,
        "failure_action": "proceed with P12-A alone",
        "truth_files_read": ["ph006 evaluation evidence"],
        "ph001_opened": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
