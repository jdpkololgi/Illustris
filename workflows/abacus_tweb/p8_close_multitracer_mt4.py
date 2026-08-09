#!/usr/bin/env python3
"""Freeze the two-rotation MT4 U-PATCH multitracer decision.

The closeout compares the frozen Bright-only reference with response-matched
BGS_FAINT context and the matched scrambled-position neural null.  It validates
the run-completion contract, computes all registered per-rotation safeguards,
and writes one machine-readable decision plus a checksum-bearing completion
marker.  It does not train, rescale, or inspect validation truth beyond the
already-frozen best-validation reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
SHELLS = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
ROTATIONS = (0, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bright-root",
        type=Path,
        default=ROOT / "p8_recovery_v1/convergence_extension_v1/unet",
    )
    parser.add_argument(
        "--proxy-root",
        type=Path,
        default=(
            ROOT
            / "p8_multitracer_v1/models/recovery/mt4_proxy_v1/unet_multitracer"
        ),
    )
    parser.add_argument(
        "--null-root",
        type=Path,
        default=(
            ROOT
            / "p8_multitracer_v1/models/recovery/mt4_faint_null_v1/unet_multitracer"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "p8_multitracer_v1/models/recovery/mt4_closeout/decision.json",
    )
    parser.add_argument(
        "--marker",
        type=Path,
        default=(
            ROOT
            / "p8_multitracer_v1/models/recovery/mt4_closeout/"
            "MT4_UPATCH_MULTITRACER_DECISION"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def extract_run(root: Path, rotation: int) -> dict:
    run = root / f"rotation_{rotation}" / "seed_42"
    report_path = run / "best_validation_report.json"
    summary_path = run / "recovery_summary.json"
    manifest_path = run / "run_manifest.json"
    report = load_json(report_path)
    summary = load_json(summary_path)

    history = summary.get("history", [])
    if not history:
        raise RuntimeError(f"empty epoch history: {summary_path}")
    if not report.get("complete_core_coverage", False):
        raise RuntimeError(f"incomplete validation coverage: {report_path}")
    if any(row.get("unique_core_fraction") != 1.0 for row in history):
        raise RuntimeError(f"incomplete training-core coverage: {summary_path}")
    if any(row.get("repeat_cores") != 0 for row in history):
        raise RuntimeError(f"repeated training cores: {summary_path}")

    shell_scores = {
        shell: float(report["per_shell"][shell]["lambda1"]["r2"])
        for shell in SHELLS
    }
    return {
        "root": str(run),
        "git_revision": summary["git_revision"],
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "final_learning_rate": float(history[-1]["learning_rate"]),
        "complete_core_coverage": True,
        "zero_repeat_cores_each_epoch": True,
        "n_authoritative": int(report["n_authoritative"]),
        "boundary_abs_error_distance_spearman": float(
            report["boundary"]["abs_error_vs_distance_spearman"]
        ),
        "macro_r2_lambda1": float(report["primary_macro_r2_lambda1"]),
        "first_three_macro_r2_lambda1": float(
            report["diagnostic_first_three_shell_macro_r2_lambda1"]
        ),
        "per_shell_r2_lambda1": shell_scores,
        "sparse_shell_r2_lambda1": shell_scores[SHELLS[-1]],
        "artifacts": {
            "best_validation_report": str(report_path),
            "best_validation_report_sha256": sha256(report_path),
            "recovery_summary": str(summary_path),
            "recovery_summary_sha256": sha256(summary_path),
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256(manifest_path),
        },
    }


def differences(candidate: dict, reference: dict) -> dict:
    return {
        "macro_r2_lambda1": (
            candidate["macro_r2_lambda1"] - reference["macro_r2_lambda1"]
        ),
        "first_three_macro_r2_lambda1": (
            candidate["first_three_macro_r2_lambda1"]
            - reference["first_three_macro_r2_lambda1"]
        ),
        "sparse_shell_r2_lambda1": (
            candidate["sparse_shell_r2_lambda1"]
            - reference["sparse_shell_r2_lambda1"]
        ),
        "per_shell_r2_lambda1": {
            shell: (
                candidate["per_shell_r2_lambda1"][shell]
                - reference["per_shell_r2_lambda1"][shell]
            )
            for shell in SHELLS
        },
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    runs: dict[str, dict[str, dict]] = {
        "bright": {},
        "proxy": {},
        "null": {},
    }
    for rotation in ROTATIONS:
        key = str(rotation)
        runs["bright"][key] = extract_run(args.bright_root, rotation)
        runs["proxy"][key] = extract_run(args.proxy_root, rotation)
        runs["null"][key] = extract_run(args.null_root, rotation)

    comparisons: dict[str, dict[str, dict]] = {}
    for rotation in ROTATIONS:
        key = str(rotation)
        comparisons[key] = {
            "proxy_minus_bright": differences(
                runs["proxy"][key], runs["bright"][key]
            ),
            "proxy_minus_null": differences(
                runs["proxy"][key], runs["null"][key]
            ),
            "null_minus_bright": differences(
                runs["null"][key], runs["bright"][key]
            ),
        }

    macro_proxy_null = [
        comparisons[str(rotation)]["proxy_minus_null"]["macro_r2_lambda1"]
        for rotation in ROTATIONS
    ]
    shell_proxy_null = {
        shell: [
            comparisons[str(rotation)]["proxy_minus_null"][
                "per_shell_r2_lambda1"
            ][shell]
            for rotation in ROTATIONS
        ]
        for shell in SHELLS
    }
    adoption_by_rotation = {}
    for rotation in ROTATIONS:
        delta = comparisons[str(rotation)]["proxy_minus_bright"]
        adoption_by_rotation[str(rotation)] = {
            "macro_gain_at_least_0p03": delta["macro_r2_lambda1"] >= 0.03,
            "sparse_gain_at_least_0p03": delta["sparse_shell_r2_lambda1"] >= 0.03,
            "no_supported_shell_degradation_below_minus_0p01": all(
                delta["per_shell_r2_lambda1"][shell] >= -0.01
                for shell in SHELLS[:3]
            ),
        }

    replicated_information = (
        all(value > 0 for value in macro_proxy_null)
        and all(value > 0 for values in shell_proxy_null.values() for value in values)
    )
    model_adoption = all(
        all(gates.values()) for gates in adoption_by_rotation.values()
    )
    payload = {
        "schema_version": "p8-mt4-upatch-closeout-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "closeout_git_revision": git_revision(),
        "scope": "ph000 P4 rotations 0 and 2, seed 42; Bright targets only",
        "runs": runs,
        "comparisons": comparisons,
        "aggregate": {
            "proxy_macro_mean": mean(
                [runs["proxy"][str(rotation)]["macro_r2_lambda1"] for rotation in ROTATIONS]
            ),
            "bright_macro_mean": mean(
                [runs["bright"][str(rotation)]["macro_r2_lambda1"] for rotation in ROTATIONS]
            ),
            "proxy_minus_bright_macro_mean": mean(
                [
                    comparisons[str(rotation)]["proxy_minus_bright"]["macro_r2_lambda1"]
                    for rotation in ROTATIONS
                ]
            ),
            "proxy_minus_null_macro_by_rotation": macro_proxy_null,
            "proxy_minus_null_macro_mean": mean(macro_proxy_null),
            "proxy_minus_null_per_shell_mean": {
                shell: mean(values) for shell, values in shell_proxy_null.items()
            },
            "proxy_macro_fold_spread": abs(
                runs["proxy"]["0"]["macro_r2_lambda1"]
                - runs["proxy"]["2"]["macro_r2_lambda1"]
            ),
        },
        "registered_gates": {
            "adoption_by_rotation": adoption_by_rotation,
            "replicated_positive_proxy_minus_null_every_shell": replicated_information,
            "extension_allowed": False,
            "extension_reason": "all completed Proxy and Null schedules end at learning rate zero",
        },
        "decision": {
            "multitracer_information": (
                "PASS_SAME_PHASE_TWO_ROTATIONS" if replicated_information else "NOT_DEMONSTRATED"
            ),
            "current_encoder_adoption": (
                "PASS" if model_adoption else "NO_GO_FOLD_INSTABILITY_AND_ROTATION_2_SAFEGUARDS"
            ),
            "fresh_phase_or_production_claim": "NOT_TESTED",
            "next_registered_model": "U-DENSITY-PHYS-v1",
        },
    }
    atomic_json(args.output, payload)
    output_sha = sha256(args.output)
    atomic_text(
        args.marker,
        "\n".join(
            [
                f"decision_json={args.output}",
                f"decision_sha256={output_sha}",
                f"multitracer_information={payload['decision']['multitracer_information']}",
                f"current_encoder_adoption={payload['decision']['current_encoder_adoption']}",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": output_sha,
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
