#!/usr/bin/env python3
"""Freeze matched ph006 P12-F method selection or a no-finalist decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import freeze_method_selection
from workflows.sbi.p12f_common_evaluator import attach_g1_comparison


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
EXPECTED = {
    "gaussian_independent_g0",
    "gaussian_correlated_g1",
    "rectified_flow_f1b",
    "score_diffusion_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def matched_identity(report: dict) -> tuple:
    return (
        report.get("phase"),
        report.get("cores"),
        report.get("draws"),
        report.get("config_sha256"),
        report.get("panel_sha256"),
        report.get("conditioning_contract_sha256"),
        report.get("target_scaler_sha256"),
        tuple(
            int(row["core_id"])
            for row in report.get("per_core_proper_scores", [])
        ),
    )


def require_complete_tarp(report: dict) -> None:
    """Refuse method selection when a required TARP diagnostic did not run."""
    diagnostics = report.get("tarp", {})
    required = ("ordered_eigenvalues", "eigengaps")
    missing = [
        name
        for name in required
        if not isinstance(diagnostics.get(name), dict)
        or diagnostics[name].get("available") is not True
    ]
    if missing:
        raise RuntimeError(
            f"{report.get('method', 'unknown')} lacks required TARP diagnostics: "
            + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f-matched-challengers-v1":
        raise RuntimeError("unsupported P12-F matched comparison config")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F blind role changed")
    reports = {}
    report_paths = {}
    for path in args.report:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version")
            != "p12f-common-evaluation-report-v1"
            or payload.get("phase") != "ph006"
            or payload.get("ph001_opened")
            or payload.get("truth_files_read") != ["ph006 density/T-web"]
        ):
            raise RuntimeError(f"report is not a frozen ph006 evaluation: {path}")
        method = str(payload["method"])
        require_complete_tarp(payload)
        if method in reports:
            raise RuntimeError(f"duplicate P12-F report for {method}")
        reports[method] = payload
        report_paths[method] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    if set(reports) != EXPECTED:
        raise RuntimeError(
            f"method selection requires exactly {sorted(EXPECTED)}, got {sorted(reports)}"
        )
    identity = matched_identity(next(iter(reports.values())))
    for method, report in reports.items():
        if matched_identity(report) != identity:
            raise RuntimeError(f"{method} does not share exact evaluation rows/contracts")
    reference = reports["gaussian_correlated_g1"]
    for method, report in reports.items():
        if method != "gaussian_correlated_g1":
            attach_g1_comparison(report, reference, seed=42)
    thresholds = {
        "tarp": float(config["selection_gates"]["tarp"]),
        "global_coverage": float(config["selection_gates"]["global_coverage"]),
        "conditional_coverage": float(
            config["selection_gates"]["conditional_coverage"]
        ),
        "joint_improvement": float(
            config["selection_gates"]["joint_improvement"]
        ),
        "other_score_worsening": float(
            config["selection_gates"]["other_score_worsening"]
        ),
    }
    marker = freeze_method_selection(reports, thresholds=thresholds)
    marker.update(
        {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "report_artifacts": report_paths,
            "matched_identity": {
                "phase": identity[0],
                "cores": identity[1],
                "draws": identity[2],
                "config_sha256": identity[3],
                "panel_sha256": identity[4],
                "conditioning_contract_sha256": identity[5],
                "target_scaler_sha256": identity[6],
                "core_id": list(identity[7]),
            },
            "blind_policy": (
                "freeze P12-A and any field-finalist ph001 predictions before one "
                "shared truth opening; no ph001 truth was read here"
            ),
            "truth_files_read": ["ph006 density/T-web"],
            "ph001_opened": False,
            "open_count": 0,
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    name = (
        "P12F_NO_FIELD_FINALIST.json"
        if marker["field_finalist"] is None
        else "P12F_METHOD_SELECTION_FROZEN.json"
    )
    atomic_json(args.output_root / name, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
