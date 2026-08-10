#!/usr/bin/env python3
"""Write the registered rotation-0 P8.9 density-first decision artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
TRAINING = ROOT / "p8_density_phys_v1/d0_runs/rotation_0/seed_42/scientific_v1/training_summary.json"
STITCHED = ROOT / "p8_density_phys_v1/d0_stitched/rotation_0/seed_42/stitched_field_manifest.json"
EVALUATION = ROOT / "p8_density_phys_v1/d0_evaluation/rotation_0/seed_42/field_downstream_metrics.json"
CONTEXT = ROOT / "p8_density_phys_v1/d0_learned_context/rotation_0/seed_42/learned_context_report.json"
DOWNSTREAM = ROOT / "p8_density_phys_v1/d0_downstream/rotation_0/seed_42/downstream_run_manifest.json"
U_REFERENCE = ROOT / "p8_recovery_v1/convergence_extension_v1/unet/rotation_0/seed_42/best_validation_report.json"
OUTPUT = ROOT / "p8_density_phys_v1/d0_decision/rotation_0/seed_42"
TRACKED = REPO_ROOT / "docs/evidence/p8/density_first_rotation0_closeout.json"
SHELL_NAMES = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--stitched", type=Path, default=STITCHED)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--context", type=Path, default=CONTEXT)
    parser.add_argument("--downstream", type=Path, default=DOWNSTREAM)
    parser.add_argument("--u-reference", type=Path, default=U_REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tracked-output", type=Path, default=TRACKED)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def coordinate_summary(row: dict) -> dict:
    result = {}
    for name in ("raw_physical", "train_fold_affine_diagnostic"):
        metric = row[name]
        classification = metric["pooled"]["classification"]
        result[name] = {
            "macro_r2_lambda1": float(metric["primary_macro_r2_lambda1"]),
            "first_three_shell_macro_r2_lambda1": float(
                metric["diagnostic_first_three_shell_macro_r2_lambda1"]
            ),
            "pooled_r2_lambda1": float(metric["pooled"]["lambda1"]["r2"]),
            "per_shell_r2_lambda1": [
                float(metric["per_shell"][shell]["lambda1"]["r2"])
                for shell in SHELL_NAMES
            ],
            "balanced_accuracy": float(classification["balanced_accuracy"]),
            "macro_f1": float(classification["macro_f1"]),
            "void_recall": float(classification["void_recall"]),
            "knot_recall": float(classification["knot_recall"]),
            "spatial_block_interval_p16_p50_p84": [
                float(metric["spatial_block_interval"][key])
                for key in ("p16", "p50", "p84")
            ],
        }
    result["train_fold_affine_coefficients"] = row["affine"]
    return result


def finite_spectrum_summary(spectrum: dict) -> dict:
    k = np.asarray(spectrum["k_centres_h_mpc"], dtype=np.float64)
    count = np.asarray(spectrum["mode_count"], dtype=np.int64)
    correlation = np.asarray(spectrum["cross_correlation_r"], dtype=np.float64)
    transfer = np.asarray(spectrum["cross_transfer"], dtype=np.float64)
    output = {}
    for label, low, high in (
        ("k_0p02_to_0p08", 0.02, 0.08),
        ("k_0p08_to_0p20", 0.08, 0.20),
    ):
        selected = (
            (k >= low) & (k < high) & (count >= 100)
            & np.isfinite(correlation) & np.isfinite(transfer)
        )
        output[label] = {
            "bins": int(selected.sum()),
            "mode_weighted_cross_correlation_r": float(
                np.average(correlation[selected], weights=count[selected])
            ),
            "mode_weighted_cross_transfer": float(
                np.average(transfer[selected], weights=count[selected])
            ),
        }
    return output


def build_decision(training: dict, stitched: dict, evaluation: dict,
                   context: dict, downstream: dict, u_reference: dict) -> dict:
    if downstream.get("status") != "PASS":
        raise RuntimeError("P8.9 downstream manifest is not PASS")
    field = evaluation["field_metrics"]
    coordinates = evaluation["tidal"]["coordinates"]
    deployable = coordinates["z_observed_deployable"]
    raw = deployable["raw_physical"]
    affine = deployable["train_fold_affine_diagnostic"]
    raw_shell = np.asarray([
        raw["per_shell"][name]["lambda1"]["r2"] for name in SHELL_NAMES
    ], dtype=np.float64)
    reference_shell = np.asarray([
        u_reference["per_shell"][name]["lambda1"]["r2"] for name in SHELL_NAMES
    ], dtype=np.float64)
    shell_delta = raw_shell - reference_shell
    raw_gap = float(
        raw["primary_macro_r2_lambda1"]
        - u_reference["primary_macro_r2_lambda1"]
    )
    component_r2 = {
        name: float(row["r2"])
        for name, row in evaluation["tidal"]
        ["predicted_vs_windowed_true_tensor_components_z_cosmo"].items()
    }
    orientation = evaluation["tidal"]["orientation_z_cosmo"]["bins"]
    raw_class = raw["pooled"]["classification"]
    reference_class = u_reference["pooled"]["classification"]
    tails = field["overall"]["tails"]
    tail_suppression = bool(
        tails["3.0"]["count_ratio_prediction_to_truth"] < 0.5
        or tails["-0.5"]["count_ratio_prediction_to_truth"] < 0.5
    )
    field_credible = bool(
        field["macro_shell_r2_delta_r7"] > 0.6
        and finite_spectrum_summary(evaluation["spectra"]["pooled_caps"])
        ["k_0p02_to_0p08"]["mode_weighted_cross_correlation_r"] > 0.8
    )
    unique_tensor_class_benefit = bool(
        min(component_r2.values()) > 0.8
        and raw_class["balanced_accuracy"] > reference_class["balanced_accuracy"]
        and raw_class["knot_recall"] > reference_class["knot_recall"]
        and raw_class["void_recall"] > reference_class["void_recall"]
    )
    numerical_continuation = bool(raw_gap >= -0.03)
    shell_continuation = bool(
        np.any(shell_delta > 0.0) and np.min(shell_delta[:3]) >= -0.01
    )
    rotation2_go = bool(
        numerical_continuation or shell_continuation or unique_tensor_class_benefit
    )

    return {
        "schema_version": "p8-density-first-rotation0-closeout-v1",
        "status": "PASS",
        "model": "U-DENSITY-PHYS-v1",
        "scope": "same-phase ph000 rotation-0 development evidence only",
        "training": {
            "epochs_completed": int(training["epochs_completed"]),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_macro_shell_r2_delta_r7": float(
                training["best_macro_shell_r2_delta_r7"]
            ),
        },
        "stitching": {
            "status": stitched["status"],
            "checkpoint_epoch": int(stitched["checkpoint_epoch"]),
            "support_coverage": stitched["support_coverage"],
            "trained_patch_parity": stitched["trained_patch_parity"],
        },
        "field": {
            "overall_r2_delta_r7": float(field["overall"]["r2"]),
            "macro_shell_r2_delta_r7": float(field["macro_shell_r2_delta_r7"]),
            "per_shell_r2_delta_r7": [
                float(field["by_shell"][name]["r2"]) for name in SHELL_NAMES
            ],
            "prediction_to_truth_std": float(
                field["overall"]["prediction_std"] / field["overall"]["truth_std"]
            ),
            "tail_count_ratio_prediction_to_truth": {
                threshold: float(row["count_ratio_prediction_to_truth"])
                for threshold, row in tails.items()
            },
            "spectrum": finite_spectrum_summary(evaluation["spectra"]["pooled_caps"]),
        },
        "tidal": {
            "z_cosmo_oracle": coordinate_summary(coordinates["z_cosmo_oracle"]),
            "z_observed_deployable": coordinate_summary(deployable),
            "tensor_component_r2": component_r2,
            "tensor_component_r2_min_max": [
                float(min(component_r2.values())), float(max(component_r2.values()))
            ],
            "orientation_smallest_eigengap": orientation["0"],
            "orientation_largest_eigengap": orientation["3"],
        },
        "learned_context": {
            "anchors": int(context["anchors"]["n"]),
            "eigenvalue_rmse_over_reference_std": {
                radius: float(row["overall"]["eigenvalues"]["rmse_over_reference_std"])
                for radius, row in context["radii"].items()
            },
            "traceless_shear_rmse_over_reference_std": {
                radius: float(
                    row["overall"]["traceless_shear_eigenvalues"]
                    ["rmse_over_reference_std"]
                )
                for radius, row in context["radii"].items()
            },
        },
        "matched_u_patch_rotation0": {
            "macro_r2_lambda1": float(u_reference["primary_macro_r2_lambda1"]),
            "first_three_shell_macro_r2_lambda1": float(
                u_reference["diagnostic_first_three_shell_macro_r2_lambda1"]
            ),
            "per_shell_r2_lambda1": reference_shell.tolist(),
            "pooled_r2_lambda1": float(u_reference["pooled"]["lambda1"]["r2"]),
            "balanced_accuracy": float(reference_class["balanced_accuracy"]),
            "macro_f1": float(reference_class["macro_f1"]),
            "void_recall": float(reference_class["void_recall"]),
            "knot_recall": float(reference_class["knot_recall"]),
        },
        "registered_gates": {
            "raw_deployable_macro_delta_from_u_patch": raw_gap,
            "within_0p03_macro": numerical_continuation,
            "raw_deployable_per_shell_delta_from_u_patch": shell_delta.tolist(),
            "shell_gain_without_supported_shell_loss_gt_0p01": shell_continuation,
            "unique_tensor_and_raw_class_benefit": unique_tensor_class_benefit,
            "credible_field_information": field_credible,
            "strong_tail_suppression": tail_suppression,
            "d1_auxiliary_trigger_met": bool(
                field_credible and (tail_suppression or raw_gap < 0.0)
            ),
        },
        "decision": {
            "primary_point_estimator": "NO_PROMOTION_ROTATION0_RAW_PHYSICAL_BELOW_U_PATCH",
            "affine_row": "DIAGNOSTIC_ONLY_NEVER_SUBSTITUTE_FOR_RAW_PHYSICAL",
            "secondary_field_tensor_candidate": (
                "RETAIN_ROTATION2_GO" if rotation2_go else "CLOSE"
            ),
            "rotation2_continuation": (
                "GO_UNIQUE_TENSOR_AND_RAW_CLASS_BENEFIT"
                if unique_tensor_class_benefit
                else ("GO_NUMERICAL_GATE" if rotation2_go else "NO_GO")
            ),
            "d1_auxiliary": (
                "TRIGGER_MET_REGISTER_ONE_FIXED_AUXILIARY_BEFORE_TRAINING"
                if field_credible and (tail_suppression or raw_gap < 0.0)
                else "CLOSED"
            ),
            "production_vac": "NOT_AUTHORIZED_P10_FRESH_PHASE_REMAINS_BLOCKING",
        },
    }


def main() -> None:
    args = parse_args()
    inputs = {
        "training": args.training,
        "stitched": args.stitched,
        "evaluation": args.evaluation,
        "context": args.context,
        "downstream": args.downstream,
        "u_reference": args.u_reference,
    }
    decision = build_decision(*(load(path) for path in inputs.values()))
    decision["created_utc"] = datetime.now(timezone.utc).isoformat()
    decision["git_revision"] = git_sha()
    decision["inputs"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in inputs.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = args.output / "density_first_baseline_decision.json"
    atomic_json(runtime, decision)
    atomic_json(args.tracked_output, decision)
    (args.output / "DENSITY_FIRST_BASELINE_DECISION").write_text(
        decision["decision"]["rotation2_continuation"] + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
