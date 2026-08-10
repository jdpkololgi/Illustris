#!/usr/bin/env python3
"""Write the terminal, machine-readable P8 ph000 development decision.

This is deliberately a metadata-only closeout.  It does not evaluate predictions,
retrain a model, or reinterpret a missing run as a failure.  The manifest records the
branches that were completed, the branches stopped without execution, and the frozen
handoff to independent-phase P10 validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_EVIDENCE = {
    "u_patch_rotation0": "docs/evidence/p8/u_patch_rotation0_diagnostics.json",
    "u_patch_rotation2": "docs/evidence/p8/u_patch_rotation2_diagnostics.json",
    "u_cic_closeout": "docs/evidence/p8/ucic_v2_closeout.json",
    "multitracer_closeout": "docs/evidence/p8/multitracer_mt4_decision.json",
    "density_rotation0_closeout": "docs/evidence/p8/density_first_rotation0_closeout.json",
    "density_darkai_like_rescore": "docs/evidence/p8/density_d0_darkai_like_rescore.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def build_closeout(repo: Path) -> dict:
    evidence = {}
    for name, relative in REQUIRED_EVIDENCE.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"required P8 evidence is missing: {path}")
        evidence[name] = {
            "path": relative,
            "sha256": sha256(path),
        }

    darkai = json.loads(
        (repo / REQUIRED_EVIDENCE["density_darkai_like_rescore"]).read_text()
    )
    d0 = json.loads(
        (repo / REQUIRED_EVIDENCE["density_rotation0_closeout"]).read_text()
    )
    mt4 = json.loads(
        (repo / REQUIRED_EVIDENCE["multitracer_closeout"]).read_text()
    )
    ucic = json.loads((repo / REQUIRED_EVIDENCE["u_cic_closeout"]).read_text())

    return {
        "schema_version": "p8-final-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repo),
        "status": "P8_COMPLETE_PH000_DETERMINISTIC_DEVELOPMENT_FROZEN",
        "scope": (
            "ph000 blocked-patch deterministic development; no independent-phase or "
            "production-DESI claim"
        ),
        "scientific_decision": {
            "deterministic_learned_handoff": "U-PATCH-BRIGHT_REFERENCE",
            "classical_handoff": "CLASSICAL-CIC",
            "production_vac": "NOT_AUTHORIZED_P10_FRESH_PHASE_REMAINS_BLOCKING",
            "graphnet": "RUNNER_UP_NOT_PROMOTED",
            "f_patch_v2_a": "NO_GO_FROZEN_RESOURCE_INFEASIBLE",
            "u_cic_residual": ucic["decision"],
            "multitracer_information": mt4["decision"]["multitracer_information"],
            "multitracer_encoder": mt4["decision"]["current_encoder_adoption"],
            "density_primary_point_estimator": d0["decision"][
                "primary_point_estimator"
            ],
            "density_rotation0_field_tensor": "RETAIN_EXPERIMENTAL_SECONDARY_EVIDENCE",
        },
        "frozen_reference_metrics": {
            "u_patch_bright_macro_r2_lambda1": {
                "rotation_0": 0.5070046680843436,
                "rotation_2": 0.5197376839763646,
            },
            "u_patch_bright_first_three_shell_r2_lambda1": {
                "rotation_0": 0.5608930253062123,
                "rotation_2": 0.5860,
            },
            "density_d0_rotation0_raw_deployable": {
                "macro_r2_lambda1": d0["tidal"]["z_observed_deployable"]
                ["raw_physical"]["macro_r2_lambda1"],
                "first_three_shell_r2_lambda1": d0["tidal"]
                ["z_observed_deployable"]["raw_physical"]
                ["first_three_shell_macro_r2_lambda1"],
                "per_shell_r2_lambda1": d0["tidal"]["z_observed_deployable"]
                ["raw_physical"]["per_shell_r2_lambda1"],
            },
        },
        "darkai_like_diagnostic": {
            "selected_equal_volume_cells": darkai["grid_cell_classes"]
            ["predicted_fft"]["selected_cells"],
            "mode_weighted_spectra": darkai["spectra"]["mode_weighted_summary"],
            "sign_threshold_balanced_accuracy": darkai["grid_cell_classes"]
            ["darkai_sign_threshold"]["balanced_accuracy"],
            "sign_threshold_exact_accuracy": darkai["grid_cell_classes"]
            ["darkai_sign_threshold"]["exact_cell_accuracy"],
            "sign_threshold_recall": darkai["grid_cell_classes"]
            ["darkai_sign_threshold"]["recall"],
            "interpretation": (
                "NGC/z<0.4/equal-volume scoring improves the D0 bulk-field summary, "
                "but leaves substantial amplitude suppression and does not close the "
                "gap to the external DarkAI headline."
            ),
        },
        "closed_without_run": {
            "density_d0_rotation2": {
                "status": "NOT_RUN_SUPERSEDED_BY_PH000_FREEZE",
                "reason": (
                    "D0 failed primary point adoption; the DarkAI-like rescore showed "
                    "only partial protocol recovery. Replicating a secondary-only ph000 "
                    "branch has lower value than independent-phase validation."
                ),
            },
            "density_d1_auxiliary": {
                "status": "NOT_RUN_SUPERSEDED_BY_PH000_FREEZE",
                "reason": (
                    "Do not tune a downstream loss on repeatedly inspected ph000 folds "
                    "after the density objective failed primary adoption."
                ),
            },
            "five_fold_three_seed_expansion": {
                "status": "NOT_RUN_DEFERRED_TO_P10_INDEPENDENT_PHASES",
                "reason": (
                    "Spend replication compute on phase-level transfer rather than "
                    "further adaptive optimization of ph000."
                ),
            },
        },
        "next_gate": {
            "stage": "P10",
            "action": (
                "Build/audit independent-phase products, train the frozen U-PATCH "
                "reference with the registered response contract, retain CIC as the "
                "classical anchor, select on ph006, and open ph001 once."
            ),
            "posterior_estimation": "DEFERRED_TO_P12_AFTER_DETERMINISTIC_TRANSFER",
        },
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tracked-output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    payload = build_closeout(repo)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for output in (args.tracked_output, args.runtime_output):
        output = output if output.is_absolute() else repo / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    runtime_marker = args.runtime_output.parent / "P8_COMPLETE"
    runtime_marker.write_text(payload["status"] + "\n")


if __name__ == "__main__":
    main()
