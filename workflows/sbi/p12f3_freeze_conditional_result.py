#!/usr/bin/env python3
"""Freeze the matched F3-L2c/F3-L2d result after the diffusion gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_freeze_diffusion_license import read_safe, science_gates


CANDIDATES = ("flow", "diffusion")


def parse_keyed(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in output:
            raise ValueError("candidate inputs must be unique KEY=PATH values")
        output[key] = Path(path)
    if set(output) != set(CANDIDATES):
        raise RuntimeError(f"expected candidate keys {CANDIDATES}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-contract", type=Path, required=True)
    parser.add_argument("--diffusion-license", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--shear", action="append", required=True)
    parser.add_argument("--visual", action="append", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = read_safe(args.decision_contract)
    license_report = read_safe(args.diffusion_license)
    if (
        contract.get("schema_version") != "p12f3-conditional-decision-v1"
        or license_report.get("schema_version") != "p12f3-diffusion-license-v1"
        or not license_report.get("licensed")
    ):
        raise RuntimeError("conditional diffusion was not validly licensed")
    report_paths = parse_keyed(args.report)
    shear_paths = parse_keyed(args.shear)
    visual_paths = parse_keyed(args.visual)
    reports = {key: read_safe(path) for key, path in report_paths.items()}
    shears = {key: read_safe(path) for key, path in shear_paths.items()}
    visuals = {key: read_safe(path) for key, path in visual_paths.items()}
    reference = read_safe(args.reference_report)
    results = {
        key: science_gates(
            reports[key], shears[key], visuals[key], reference, contract,
            seed=args.seed + index,
        )
        for index, key in enumerate(CANDIDATES)
    }
    passing = [key for key in CANDIDATES if results[key]["all_pass"]]
    if passing:
        status = "replication_required"
        selected = passing[0] if len(passing) == 1 else "lowest_composite_after_replication"
        interpretation = (
            "At least one seed-42 candidate passed every registered gate. A second "
            "seed is required before promotion or architecture selection."
        )
    else:
        status = "no_conditional_width_finalist"
        selected = None
        interpretation = (
            "Neither target-identical conditional generative objective passed the "
            "simultaneous ph006 calibration and proper-score gates. Retain F3-L2b as "
            "the most informative field-research checkpoint, not as a calibrated "
            "production field posterior."
        )
    inputs = (
        args.decision_contract, args.diffusion_license, args.reference_report,
        *report_paths.values(), *shear_paths.values(), *visual_paths.values(),
    )
    payload = {
        "schema_version": "p12f3-conditional-result-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selected_candidate": selected,
        "seed42_results": results,
        "interpretation": interpretation,
        "second_seed_required": bool(passing),
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
