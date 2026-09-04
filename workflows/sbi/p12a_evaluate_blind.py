#!/usr/bin/env python3
"""Immutable P12-A evaluation after the single ph001 opening.

The evaluator cannot create/open truth and cannot fit a calibration map.  It
requires the canonical opened marker, a pre-open evaluation contract, an
already-authorized truth package, and the content-addressed posterior shards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_production_contract import BLIND_SCHEMA, OPEN_SCHEMA
from workflows.sbi.p12a_blind_evaluation_contract import SCHEMA as CONTRACT_SCHEMA
from workflows.sbi.p12a_open_blind import (
    AUTHORIZATION_SCHEMA,
    TRUTH_COMPLETE_SCHEMA,
    validate_truth_complete,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    randomized_ranks,
    rank_cdf_maximum_deviation,
)


RESULT_SCHEMA = "p12a-ph001-blind-evaluation-v1"
PROPER_SCORE_SCHEMA = "p12a-ph001-proper-score-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _artifact_paths(opened: dict) -> dict[Path, str]:
    result = {}
    for record in opened.get("truth_files_read", []):
        path = Path(record.get("path", "")).resolve()
        result[path] = record.get("sha256", "")
    return result


def validate_evaluation_implementation(contract: dict) -> None:
    expected = {
        "blind_evaluator": Path(__file__).resolve(),
        "proper_score_evaluator": Path(__file__).with_name("p12a_blind_proper_score.py").resolve(),
        "one_open_guard": Path(__file__).with_name("p12a_open_blind.py").resolve(),
    }
    records = contract.get("evaluation_implementation", {})
    if set(records) != set(expected):
        raise RuntimeError("blind evaluation implementation inventory is not frozen")
    for name, path in expected.items():
        record = records[name]
        if (
            Path(record.get("path", "")).resolve() != path
            or record.get("sha256") != sha256(path)
            or ("bytes" in record and int(record["bytes"]) != path.stat().st_size)
        ):
            raise RuntimeError(f"blind evaluation implementation changed: {name}")


def validate_proper_score_report(
    report: dict,
    *,
    frozen_path: Path,
    opened_path: Path,
    contract_path: Path,
    truth_manifest_path: Path,
) -> None:
    if (
        report.get("schema_version") != PROPER_SCORE_SCHEMA
        or report.get("phase") != "ph001"
        or report.get("open_count") != 1
        or report.get("sealed_phase_opened") is not True
        or report.get("post_open_refit_performed") is not False
        or report.get("post_open_tuning_allowed") is not False
    ):
        raise PermissionError("ph001 proper-score report is not frozen evaluation-only evidence")
    for key, path in (
        ("frozen_predictions", frozen_path),
        ("opened_marker", opened_path),
        ("evaluation_contract", contract_path),
        ("truth_manifest", truth_manifest_path),
    ):
        record = report.get(key, {})
        if (
            Path(record.get("path", "")).resolve() != path.resolve()
            or record.get("sha256") != sha256(path)
        ):
            raise RuntimeError(f"proper-score report does not bind {key}")


def validate_open_state(
    *, frozen_path: Path, opened_path: Path, contract_path: Path, truth_manifest_path: Path
) -> tuple[dict, dict, dict, dict]:
    frozen_path = frozen_path.resolve()
    opened_path = opened_path.resolve()
    contract_path = contract_path.resolve()
    truth_manifest_path = truth_manifest_path.resolve()
    frozen = json.loads(frozen_path.read_text())
    opened = json.loads(opened_path.read_text())
    contract = json.loads(contract_path.read_text())
    truth_manifest = json.loads(truth_manifest_path.read_text())
    if frozen.get("schema_version") != BLIND_SCHEMA or not frozen.get("pass"):
        raise PermissionError("blind predictions are not frozen")
    if frozen.get("open_count", 0) != 0 or frozen.get("truth_files_read", []) != []:
        raise PermissionError("frozen prediction marker is not truth-free")
    if (
        opened.get("schema_version") != OPEN_SCHEMA
        or opened.get("phase") != "ph001"
        or opened.get("state") != "blind_truth_opened"
        or opened.get("open_count") != 1
        or opened.get("sealed_phase_opened") is not True
        or opened.get("truth_materialization_complete") is not True
        or opened.get("post_open_refit_allowed") is not False
        or opened.get("post_open_tuning_allowed") is not False
        or opened.get("pass") is not True
    ):
        raise PermissionError("canonical ph001 opening transition is absent")
    if opened_path != frozen_path.parent / "P12_BLIND_OPENED.json":
        raise PermissionError("opened marker is not at its canonical path")
    reference = opened.get("frozen_predictions_reference", {})
    if (
        Path(reference.get("path", "")).resolve() != frozen_path
        or reference.get("sha256") != sha256(frozen_path)
    ):
        raise RuntimeError("opened marker does not bind the frozen predictions")
    if contract.get("schema_version") != CONTRACT_SCHEMA or not contract.get("pass"):
        raise RuntimeError("blind evaluation contract is not frozen")
    if (
        contract.get("phase") != "ph001"
        or contract.get("open_count") != 0
        or contract.get("sealed_phase_opened")
        or contract.get("post_open_refit_allowed") is not False
    ):
        raise PermissionError("evaluation contract was not frozen before opening")
    validate_evaluation_implementation(contract)
    contract_reference = opened.get("evaluation_contract_reference", {})
    if (
        Path(contract_reference.get("path", "")).resolve() != contract_path
        or contract_reference.get("sha256") != sha256(contract_path)
    ):
        raise RuntimeError("opened marker does not bind the frozen evaluation contract")
    authorization_reference = opened.get("authorization_reference", {})
    authorization_path = Path(authorization_reference.get("path", "")).resolve()
    if (
        authorization_reference.get("sha256") != sha256(authorization_path)
        or json.loads(authorization_path.read_text()).get("schema_version")
        != AUTHORIZATION_SCHEMA
    ):
        raise RuntimeError("opened marker does not bind a valid one-open authorization")
    truth_manifest = validate_truth_complete(
        truth_complete_path=truth_manifest_path,
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=contract_path,
    )
    if truth_manifest.get("schema_version") != TRUTH_COMPLETE_SCHEMA:
        raise RuntimeError("ph001 truth completion schema changed")
    truth_reference = opened.get("truth_complete_reference", {})
    if (
        Path(truth_reference.get("path", "")).resolve() != truth_manifest_path
        or truth_reference.get("sha256") != sha256(truth_manifest_path)
    ):
        raise RuntimeError("opened marker does not bind the truth-completion marker")
    opened_artifacts = _artifact_paths(opened)
    if opened_artifacts.get(truth_manifest_path) != sha256(truth_manifest_path):
        raise PermissionError("truth manifest was not registered by the one-open transition")
    array_spec = truth_manifest.get("array", {})
    array_path = Path(array_spec.get("path", "")).resolve()
    if (
        "ph001" not in str(array_path).lower()
        or not array_path.is_file()
        or sha256(array_path) != array_spec.get("sha256")
        or opened_artifacts.get(array_path) != array_spec.get("sha256")
    ):
        raise PermissionError("truth array is stale or absent from the opened marker")
    return frozen, opened, contract, truth_manifest


def _prediction_manifest_by_schema(frozen: dict) -> dict[str, dict]:
    result = {}
    for record in frozen.get("prediction_manifests", []):
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("prediction manifest changed after blind freeze")
        payload = json.loads(path.read_text())
        result[payload["schema_version"]] = payload
    return result


def load_prediction_arrays(frozen: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifests = _prediction_manifest_by_schema(frozen)
    complete = manifests.get("p12a-blind-export-complete-v1")
    if complete is None:
        raise RuntimeError("frozen marker lacks complete P12-A export")
    summary_parts: dict[str, list[np.ndarray]] = {}
    audit_parent, audit_draws = [], []
    for record in complete.get("shards", []):
        marker_path = Path(record["path"])
        if sha256(marker_path) != record["sha256"]:
            raise RuntimeError("posterior shard marker changed after blind freeze")
        marker = json.loads(marker_path.read_text())
        summary_path = Path(marker["summary"])
        audit_path = Path(marker["audit_draws"])
        if sha256(summary_path) != marker["summary_sha256"] or sha256(audit_path) != marker["audit_draws_sha256"]:
            raise RuntimeError("posterior shard artifact changed after blind freeze")
        with np.load(summary_path, mmap_mode="r") as archive:
            for name in archive.files:
                summary_parts.setdefault(name, []).append(np.asarray(archive[name]))
        with np.load(audit_path, mmap_mode="r") as archive:
            audit_parent.append(np.asarray(archive["parent_node_id"], dtype=np.int64))
            audit_draws.append(np.asarray(archive["eigenvalue_draws"], dtype=np.float32))
    summary = {name: np.concatenate(parts) for name, parts in summary_parts.items()}
    audit = {
        "parent_node_id": np.concatenate(audit_parent),
        "eigenvalue_draws": np.concatenate(audit_draws),
    }
    if len(np.unique(summary["parent_node_id"])) != len(summary["parent_node_id"]):
        raise RuntimeError("posterior summaries contain duplicate parents")
    if len(np.unique(audit["parent_node_id"])) != len(audit["parent_node_id"]):
        raise RuntimeError("audit draws contain duplicate parents")
    return summary, audit


def weighted_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.sum(np.square(truth - prediction))
    total = np.sum(np.square(truth - truth.mean()))
    return float(1.0 - residual / total) if total > 0 else float("nan")


def interval_coverage(truth: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.mean((truth >= low) & (truth <= high), axis=0)


def _quartile(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return np.searchsorted(edges, values, side="right").astype(np.int8)


def _conditional_coverage(summary: dict[str, np.ndarray], truth: np.ndarray) -> dict:
    strata = {
        "shell": np.asarray(summary["shell"], dtype=np.int8),
        "ntilde_quartile": _quartile(summary["ntilde_mpc3"]),
        "boundary_distance_quartile": _quartile(summary["distance_to_support_boundary_mpc"]),
    }
    result, maximum = {}, 0.0
    for name, labels in strata.items():
        rows = {}
        for value in np.unique(labels):
            chosen = labels == value
            c68 = interval_coverage(truth[chosen], summary["eigenvalue_q16"][chosen], summary["eigenvalue_q84"][chosen])
            c90 = interval_coverage(truth[chosen], summary["eigenvalue_q05"][chosen], summary["eigenvalue_q95"][chosen])
            error = max(float(np.max(np.abs(c68 - 0.68))), float(np.max(np.abs(c90 - 0.90))))
            maximum = max(maximum, error)
            rows[str(int(value))] = {"rows": int(np.count_nonzero(chosen)), "coverage68": c68.tolist(), "coverage90": c90.tolist(), "maximum_absolute_error": error}
        result[name] = rows
    return {"strata": result, "maximum_absolute_error": maximum}


def _brier(summary: dict[str, np.ndarray], truth: np.ndarray, contract: dict) -> dict:
    threshold = float(contract["class_threshold"])
    true_class = np.sum(truth > threshold, axis=1)
    one_hot = np.eye(4, dtype=np.float64)[true_class]
    probability = np.asarray(summary["web_class_probability"], dtype=np.float64)
    model = float(np.mean(np.sum(np.square(probability - one_hot), axis=1)))
    climatology = np.empty_like(probability)
    shell = np.asarray(summary["shell"], dtype=np.int8)
    for value in range(4):
        climatology[shell == value] = contract["shell_class_climatology"][str(value)]["probability_void_sheet_filament_knot"]
    baseline = float(np.mean(np.sum(np.square(climatology - one_hot), axis=1)))
    skill = float(1.0 - model / baseline)
    return {"model": model, "training_shell_climatology": baseline, "skill": skill}


def _tarp_and_ranks(audit_draws: np.ndarray, audit_truth: np.ndarray, seed: int) -> dict:
    from workflows.sbi.p12f_dependency_rescue_evaluator import tarp_curve

    samples = np.transpose(np.asarray(audit_draws, dtype=np.float64), (1, 0, 2))
    gap_samples = np.diff(samples, axis=2)
    gap_truth = np.diff(audit_truth, axis=1)
    eigen_tarp = tarp_curve(samples, audit_truth, seed=seed)
    gap_tarp = tarp_curve(gap_samples, gap_truth, seed=seed + 1)
    rank = randomized_ranks(samples, audit_truth, seed=seed + 2)
    component = [rank_cdf_maximum_deviation(rank[:, index]) for index in range(3)]
    return {
        "joint_eigenvalue_tarp": eigen_tarp,
        "joint_eigengap_tarp": gap_tarp,
        "physical_rank_cdf_maximum_by_eigenvalue": component,
        "physical_rank_cdf_maximum": float(max(component)),
    }


def evaluate_arrays(
    *, summary: dict[str, np.ndarray], audit: dict[str, np.ndarray], truth_parent: np.ndarray,
    truth_eigenvalues: np.ndarray, contract: dict, proper_score: dict, seed: int = 42,
) -> dict:
    parent = np.asarray(summary["parent_node_id"], dtype=np.int64)
    truth_parent = np.asarray(truth_parent, dtype=np.int64)
    if not np.array_equal(parent, truth_parent):
        raise RuntimeError("truth package does not exactly match frozen supported parents")
    truth = np.asarray(truth_eigenvalues, dtype=np.float64)
    if truth.shape != (len(parent), 3) or not np.all(np.isfinite(truth)) or np.any(np.diff(truth, axis=1) < 0):
        raise RuntimeError("blind eigenvalue truth is invalid")
    order = np.argsort(parent)
    audit_position_in_order = np.searchsorted(parent[order], audit["parent_node_id"])
    if (
        np.any(audit_position_in_order >= len(parent))
        or not np.array_equal(
            parent[order][audit_position_in_order], audit["parent_node_id"]
        )
    ):
        raise RuntimeError("audit draws do not align to blind truth")
    audit_position = order[audit_position_in_order]
    global68 = interval_coverage(truth, summary["eigenvalue_q16"], summary["eigenvalue_q84"])
    global90 = interval_coverage(truth, summary["eigenvalue_q05"], summary["eigenvalue_q95"])
    global_error = max(float(np.max(np.abs(global68 - 0.68))), float(np.max(np.abs(global90 - 0.90))))
    conditional = _conditional_coverage(summary, truth)
    dependence = _tarp_and_ranks(audit["eigenvalue_draws"], truth[audit_position], seed)
    mean_r2 = [weighted_r2(truth[:, i], summary["eigenvalue_mean"][:, i]) for i in range(3)]
    base_r2 = [weighted_r2(truth[:, i], summary["base_prediction_eigenvalues"][:, i]) for i in range(3)]
    brier = _brier(summary, truth, contract)
    gates = contract["gates"]
    decisions = {
        "joint_eigenvalue_tarp": dependence["joint_eigenvalue_tarp"]["maximum_deviation"] <= gates["joint_eigenvalue_tarp_maximum"],
        "joint_eigengap_tarp": dependence["joint_eigengap_tarp"]["maximum_deviation"] <= gates["joint_eigengap_tarp_maximum"],
        "physical_rank_cdf": dependence["physical_rank_cdf_maximum"] <= gates["physical_rank_cdf_maximum"],
        "global_coverage": global_error <= gates["global_coverage_absolute_error_maximum"],
        "conditional_coverage": conditional["maximum_absolute_error"] <= gates["conditional_coverage_absolute_error_maximum"],
        "posterior_mean_accuracy": mean_r2[0] - base_r2[0] >= gates["posterior_mean_lambda1_r2_delta_minimum"],
        "web_class_brier_skill": brier["skill"] > gates["multiclass_brier_skill_minimum"],
        "proper_score": proper_score.get("ci95", [float("-inf")])[0] > gates["fmpe_minus_gaussian_log_score_ci95_lower_minimum"],
    }
    return {
        "rows": int(len(parent)),
        "audit_rows": int(len(audit_position)),
        "coverage68": global68.tolist(),
        "coverage90": global90.tolist(),
        "global_coverage_maximum_absolute_error": global_error,
        "conditional_coverage": conditional,
        "dependence": dependence,
        "posterior_mean_r2": mean_r2,
        "base_unet_r2": base_r2,
        "posterior_minus_base_lambda1_r2": float(mean_r2[0] - base_r2[0]),
        "web_class_brier": brier,
        "fmpe_minus_gaussian_physical_log_score": proper_score,
        "gates": decisions,
        "pass": bool(all(decisions.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--proper-score-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite blind result: {args.output}")
    frozen, opened, contract, truth_manifest = validate_open_state(
        frozen_path=args.frozen_predictions,
        opened_path=args.opened_marker,
        contract_path=args.evaluation_contract,
        truth_manifest_path=args.truth_manifest,
    )
    proper_score = json.loads(args.proper_score_report.read_text())
    validate_proper_score_report(
        proper_score,
        frozen_path=args.frozen_predictions,
        opened_path=args.opened_marker,
        contract_path=args.evaluation_contract,
        truth_manifest_path=args.truth_manifest,
    )
    summary, audit = load_prediction_arrays(frozen)
    truth_path = Path(truth_manifest["array"]["path"])
    with np.load(truth_path, mmap_mode="r") as truth:
        result = evaluate_arrays(
            summary=summary,
            audit=audit,
            truth_parent=truth["parent_node_id"],
            truth_eigenvalues=truth["eigenvalues"],
            contract=contract,
            proper_score=proper_score,
            seed=args.seed,
        )
    report: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "estimand": "per-galaxy joint ordered tidal-eigenvalue posterior conditional on H_fid",
        "not_a_coherent_field_posterior": True,
        "frozen_predictions": {"path": str(args.frozen_predictions.resolve()), "sha256": sha256(args.frozen_predictions)},
        "opened_marker": {"path": str(args.opened_marker.resolve()), "sha256": sha256(args.opened_marker)},
        "evaluation_contract": {"path": str(args.evaluation_contract.resolve()), "sha256": sha256(args.evaluation_contract)},
        "truth_manifest": {"path": str(args.truth_manifest.resolve()), "sha256": sha256(args.truth_manifest)},
        "proper_score_report": {"path": str(args.proper_score_report.resolve()), "sha256": sha256(args.proper_score_report)},
        "post_open_refit_performed": False,
        "post_open_tuning_allowed": False,
        "truth_files_read": opened["truth_files_read"],
        "open_count": 1,
        "sealed_phase_opened": True,
        **result,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
