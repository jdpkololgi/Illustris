#!/usr/bin/env python3
"""Build the P8 two-rotation decision table without hiding shell failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, SHELL_NAMES


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
SCREEN_ROTATIONS = (0, 2)
HOME_WEDGE_ANCHORS = {"G-PATCH": 0.804, "U-PATCH": 0.876, "F-PATCH": 0.841}


def report_path(root: Path, model: str, rotation: int, seed: int) -> Path:
    if model == "CLASSICAL-CIC":
        return root / "classical" / f"rotation_{rotation}" / "cic_report.json"
    slug = model[0].lower() + "_patch"
    return root / slug / f"rotation_{rotation}" / f"seed_{seed}" / "best_validation_report.json"


def summarize_model(root: Path, model: str, rotations, seed: int) -> dict:
    paths = [report_path(root, model, rotation, seed) for rotation in rotations]
    present = [path.exists() for path in paths]
    if not all(present):
        if model == "F-PATCH":
            preflight = root / "f_patch" / "resource_preflight.json"
            if preflight.exists():
                row = json.loads(preflight.read_text())
                if row["decision"] == "NO_GO_FROZEN_V2_A_RESOURCE_INFEASIBLE":
                    return {
                        "model": model,
                        "complete": False,
                        "status": "preflight_no_go",
                        "reason": row["decision"],
                        "resource_preflight": str(preflight),
                        "screen_not_run": True,
                    }
        return {
            "model": model,
            "complete": False,
            "present_rotations": [int(r) for r, ok in zip(rotations, present) if ok],
            "missing": [str(path) for path, ok in zip(paths, present) if not ok],
        }
    reports = [json.loads(path.read_text()) for path in paths]
    key = "train_affine" if model == "CLASSICAL-CIC" else None
    if key:
        reports = [report[key] for report in reports]
    shell_values = {
        name: [report["per_shell"][name]["lambda1"]["r2"] for report in reports]
        for name in SHELL_NAMES
    }
    per_shell = {
        name: {
            "mean_r2_lambda1": float(np.mean(values)),
            "by_rotation": [float(value) for value in values],
        }
        for name, values in shell_values.items()
    }
    four_shell = [report["primary_macro_r2_lambda1"] for report in reports]
    first_three = [
        float(np.mean([report["per_shell"][name]["lambda1"]["r2"] for name in SHELL_NAMES[:3]]))
        for report in reports
    ]
    result = {
        "model": model,
        "complete": True,
        "rotations": [int(v) for v in rotations],
        "seed": None if model == "CLASSICAL-CIC" else int(seed),
        "four_shell_primary_mean": float(np.mean(four_shell)),
        "four_shell_primary_by_rotation": [float(value) for value in four_shell],
        "first_three_shell_diagnostic_mean": float(np.mean(first_three)),
        "first_three_shell_diagnostic_by_rotation": first_three,
        "worst_mean_shell_r2_lambda1": float(
            min(row["mean_r2_lambda1"] for row in per_shell.values())
        ),
        "per_shell": per_shell,
        "reports": [str(path) for path in paths],
    }
    if model in HOME_WEDGE_ANCHORS:
        result["historical_home_wedge_anchor_r2_lambda1"] = HOME_WEDGE_ANCHORS[model]
        result["contextual_source_to_transfer_gap"] = float(
            HOME_WEDGE_ANCHORS[model] - result["four_shell_primary_mean"]
        )
        result["source_gap_warning"] = (
            "context only: historical dense-wedge random-node score is not a matched fold"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotations", type=int, nargs="+", default=SCREEN_ROTATIONS)
    args = parser.parse_args()
    models = [
        summarize_model(args.p8_root, model, args.rotations, args.seed)
        for model in ("CLASSICAL-CIC", "G-PATCH", "U-PATCH", "F-PATCH")
    ]
    complete = [row for row in models if row["complete"]]
    learned = [row for row in complete if row["model"] != "CLASSICAL-CIC"]
    classical = next((row for row in complete if row["model"] == "CLASSICAL-CIC"), None)
    resolved_learned = [
        row for row in models
        if row["model"] != "CLASSICAL-CIC"
        and (row["complete"] or row.get("status") == "preflight_no_go")
    ]
    macro_leader = max(complete, key=lambda row: row["four_shell_primary_mean"]) if complete else None
    supported_leader = max(
        complete, key=lambda row: row["first_three_shell_diagnostic_mean"]
    ) if complete else None
    gate = {
        "status": "UNDETERMINED",
        "reason": "all learned screens and strong matched classical rows are not complete",
    }
    # CIC alone is explicitly not strong enough to certify a learned victory.  It
    # is allowed to expose a failure, but exact DTFE/another validated strong row
    # is required before declaring the adoption gate passed.
    if classical is not None and len(resolved_learned) == 3:
        gate = {
            "status": "PENDING_STRONG_CLASSICAL",
            "reason": (
                "every learned branch is resolved by a two-rotation screen or a "
                "pre-registered resource NO-GO, but exact DTFE or another validated "
                "strong classical row is still required"
            ),
            "no_macro_only_win": True,
        }
    comparisons = {}
    if classical is not None:
        for row in learned:
            four_delta = (
                row["four_shell_primary_mean"] - classical["four_shell_primary_mean"]
            )
            supported_delta = (
                row["first_three_shell_diagnostic_mean"]
                - classical["first_three_shell_diagnostic_mean"]
            )
            macro_only = four_delta > 0 and supported_delta < 0
            comparisons[row["model"]] = {
                "four_shell_delta_vs_cic": float(four_delta),
                "first_three_shell_delta_vs_cic": float(supported_delta),
                "macro_only_apparent_win": bool(macro_only),
                "interpretation": (
                    "not a classical win: positive four-shell delta is driven by "
                    "CIC collapse in shell 4"
                    if macro_only else
                    "requires the per-shell table and strong-classical gate"
                ),
            }
    payload = {
        "schema_version": 1,
        "stage": "P8 one-seed two-rotation screen summary",
        "rotations": [int(v) for v in args.rotations],
        "seed": int(args.seed),
        "models": models,
        "four_shell_macro_leader": None if macro_leader is None else macro_leader["model"],
        "first_three_shell_diagnostic_leader": (
            None if supported_leader is None else supported_leader["model"]
        ),
        "scientific_interpretation": (
            "The four-shell macro remains the registered primary, but it cannot by itself "
            "establish a learned win when a classical method collapses only in shell 4."
        ),
        "comparisons_to_cic": comparisons,
        "learned_adoption_gate": gate,
        "p10_required_for_production_transfer": True,
    }
    output = args.p8_root / "screen_summary.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
