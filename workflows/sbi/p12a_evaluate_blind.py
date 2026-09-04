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

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12_production_contract import (
    BLIND_SCHEMA,
    OPEN_SCHEMA,
    QUALITY_BITS,
)
from workflows.sbi.p12a_blind_evaluation_contract import (
    IMPLEMENTATION_FILES,
    SCHEMA as CONTRACT_SCHEMA,
)
from workflows.sbi.p12a_blind_energy_score import (
    gaussian_samples,
    joint_energy_score,
    clustered_mean_interval,
)
from workflows.sbi.p12a_immutable_io import write_json_exclusive
from workflows.sbi.p12a_open_blind import (
    AUTHORIZATION_SCHEMA,
    TRUTH_COMPLETE_SCHEMA,
    validate_evaluation_contract,
    validate_truth_complete,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    randomized_ranks,
    rank_cdf_maximum_deviation,
)


RESULT_SCHEMA = "p12a-ph001-blind-evaluation-v2"
PROPER_SCORE_SCHEMA = "p12a-ph001-joint-energy-score-v1"


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
    expected = {name: path.resolve() for name, path in IMPLEMENTATION_FILES.items()}
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
    contract: dict | None = None,
    audit_parent: np.ndarray | None = None,
    audit_core: np.ndarray | None = None,
    audit_draws: np.ndarray | None = None,
    audit_truth: np.ndarray | None = None,
    gaussian_base: np.ndarray | None = None,
    audit_shell: np.ndarray | None = None,
    audit_cap: np.ndarray | None = None,
    expected_truth_files: list[dict] | None = None,
) -> dict:
    if (
        report.get("schema_version") != PROPER_SCORE_SCHEMA
        or report.get("phase") != "ph001"
        or report.get("open_count") != 1
        or report.get("sealed_phase_opened") is not True
        or report.get("post_open_refit_performed") is not False
        or report.get("post_open_tuning_allowed") is not False
        or report.get("unnormalized_fmpe_log_score_used") is not False
        or report.get("lower_is_better") is not True
        or report.get("comparison") != "gaussian_minus_fmpe; positive favours FMPE"
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
    if any(
        value is None
        for value in (
            contract,
            audit_parent,
            audit_core,
            audit_draws,
            audit_truth,
            gaussian_base,
            audit_shell,
            audit_cap,
        )
    ):
        raise RuntimeError("deep energy-score validation requires raw draws and identities")
    protocol = contract["evaluation_protocol"]
    expected_output = Path(contract["canonical_outputs"]["energy_score_report"]).resolve()
    if Path(report.get("score_array", {}).get("path", "")).resolve() != Path(
        contract["canonical_outputs"]["energy_score_array"]
    ).resolve():
        raise RuntimeError("energy-score array is not at its frozen canonical path")
    if (
        int(report.get("rows", -1)) != int(protocol["audit_rows"])
        or int(report.get("posterior_draws", -1)) != int(protocol["posterior_draws"])
        or int(report.get("energy_pairing_offset", -1))
        != int(protocol["energy_pairing_offset"])
        or int(report.get("gaussian_draw_seed", -1))
        != int(protocol["gaussian_draw_seed"])
        or report.get("gaussian_ordering_transform")
        != protocol["gaussian_ordering_transform"]
        or int(report.get("bootstrap_repetitions", -1))
        != int(protocol["bootstrap_repetitions"])
        or int(report.get("bootstrap_seed", -1)) != int(protocol["bootstrap_seed"])
        or report.get("bootstrap_unit") != protocol["bootstrap_unit"]
    ):
        raise RuntimeError("energy-score report protocol differs from the frozen contract")
    if expected_truth_files is not None and report.get("truth_files_read") != expected_truth_files:
        raise RuntimeError("energy-score report truth provenance differs from opened state")
    score_record = report["score_array"]
    score_path = Path(score_record["path"])
    if (
        not score_path.is_file()
        or score_record.get("sha256") != sha256(score_path)
        or int(score_record.get("bytes", -1)) != score_path.stat().st_size
    ):
        raise RuntimeError("energy-score sidecar is absent or stale")
    with np.load(score_path, mmap_mode="r") as archive:
        required = {
            "parent_node_id",
            "core_id",
            "fmpe_joint_energy_score",
            "gaussian_joint_energy_score",
            "gaussian_minus_fmpe",
        }
        if not required.issubset(archive.files):
            raise RuntimeError("energy-score sidecar schema is incomplete")
        parent = np.asarray(archive["parent_node_id"], dtype=np.int64)
        core = np.asarray(archive["core_id"], dtype=np.int64)
        fmpe = np.asarray(archive["fmpe_joint_energy_score"], dtype=np.float64)
        gaussian = np.asarray(archive["gaussian_joint_energy_score"], dtype=np.float64)
        difference = np.asarray(archive["gaussian_minus_fmpe"], dtype=np.float64)
    if not np.array_equal(parent, np.asarray(audit_parent, dtype=np.int64)):
        raise RuntimeError("energy-score sidecar parent order differs from frozen audit")
    if not np.array_equal(core, np.asarray(audit_core, dtype=np.int64)):
        raise RuntimeError("energy-score sidecar cores differ from frozen audit")
    if not np.array_equal(difference, gaussian - fmpe):
        raise RuntimeError("energy-score sidecar arithmetic is invalid")
    gaussian_path = Path(contract["gaussian_baseline"]["path"])
    if contract["gaussian_baseline"].get("sha256") != sha256(gaussian_path):
        raise RuntimeError("Gaussian baseline changed before score replay")
    gaussian_contract = json.loads(gaussian_path.read_text())
    replay_gaussian_draws = gaussian_samples(
        base=np.asarray(gaussian_base),
        shell=np.asarray(audit_shell),
        cap=np.asarray(audit_cap),
        gaussian=gaussian_contract,
        draws=int(protocol["posterior_draws"]),
        seed=int(protocol["gaussian_draw_seed"]),
    )
    replay_fmpe = joint_energy_score(
        np.asarray(audit_draws),
        np.asarray(audit_truth),
        pairing_offset=int(protocol["energy_pairing_offset"]),
    )
    replay_gaussian = joint_energy_score(
        replay_gaussian_draws,
        np.asarray(audit_truth),
        pairing_offset=int(protocol["energy_pairing_offset"]),
    )
    if not np.array_equal(fmpe, replay_fmpe) or not np.array_equal(
        gaussian, replay_gaussian
    ):
        raise RuntimeError("energy-score sidecar failed full raw-draw recomputation")
    recomputed = clustered_mean_interval(
        difference,
        core,
        repeats=int(protocol["bootstrap_repetitions"]),
        seed=int(protocol["bootstrap_seed"]),
    )
    for key in ("mean", "spatial_blocks"):
        if not np.allclose(report.get(key), recomputed[key], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"energy-score report failed recomputation: {key}")
    if not np.allclose(report.get("ci95"), recomputed["ci95"], rtol=0.0, atol=1e-12):
        raise RuntimeError("energy-score interval failed frozen core-bootstrap replay")
    if not np.isclose(
        report.get("fmpe_mean_joint_energy_score", np.nan), fmpe.mean(), rtol=0.0, atol=1e-12
    ) or not np.isclose(
        report.get("gaussian_mean_joint_energy_score", np.nan),
        gaussian.mean(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("energy-score component means failed recomputation")
    expected_pass = bool(
        recomputed["ci95"][0]
        > contract["gates"]["gaussian_minus_fmpe_energy_score_ci95_lower_minimum"]
    )
    if report.get("pass") is not expected_pass:
        raise RuntimeError("energy-score pass flag is not implied by the frozen gate")
    return {**recomputed, "pass": expected_pass, "report_path": str(expected_output)}


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
    # This checks the complete frozen gate/estimand/source inventory and exact
    # Python/conda fingerprint, rather than only the evaluator source hashes.
    validate_evaluation_contract(contract_path)
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
        deep_artifacts=False,
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
    if (
        audit["eigenvalue_draws"].ndim != 3
        or audit["eigenvalue_draws"].shape[2] != 3
        or not np.all(np.isfinite(audit["eigenvalue_draws"]))
        or np.any(
            np.diff(audit["eigenvalue_draws"], axis=2)
            < -32.0 * np.finfo(np.float32).eps
        )
    ):
        raise RuntimeError("audit posterior draws are invalid or unordered")
    return summary, audit


def audit_core_ids(frozen: dict, audit_parent: np.ndarray) -> np.ndarray:
    manifests = _prediction_manifest_by_schema(frozen)
    context_path = Path(manifests["p12a-blind-base-context-v1"]["array"])
    with np.load(context_path, mmap_mode="r") as context:
        reference = np.asarray(context["parent_node_id"], dtype=np.int64)
        order = np.argsort(reference)
        position = np.searchsorted(reference[order], audit_parent)
        if np.any(position >= len(reference)) or not np.array_equal(
            reference[order][position], audit_parent
        ):
            raise RuntimeError("frozen audit parents are absent from blind context")
        return np.asarray(context["core_id"][order[position]], dtype=np.int64)


def load_classical_predictions(
    frozen: dict, reference_parent: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    manifests = _prediction_manifest_by_schema(frozen)
    result: dict[str, dict[str, np.ndarray]] = {}
    for estimator in ("cic", "dtfe"):
        marker = manifests.get(f"p12-blind-{estimator}-prediction-v1")
        if marker is None or marker.get("estimator") != estimator:
            raise RuntimeError(f"frozen marker lacks {estimator} prediction")
        path = Path(marker["prediction"])
        if marker.get("prediction_sha256") != sha256(path):
            raise RuntimeError(f"frozen {estimator} prediction changed")
        with np.load(path, mmap_mode="r") as archive:
            parent = np.asarray(archive["parent_node_id"], dtype=np.int64)
            if not np.array_equal(parent, reference_parent):
                raise RuntimeError(f"{estimator} prediction order differs from P12-A")
            raw = np.asarray(archive["raw_eigenvalues"], dtype=np.float32)
            affine = np.asarray(
                archive["train_affine_ordered_eigenvalues"], dtype=np.float32
            )
            for label, values in (("raw", raw), ("train-affine", affine)):
                if (
                    values.shape != (len(reference_parent), 3)
                    or not np.all(np.isfinite(values))
                    or np.any(
                        np.diff(values, axis=1)
                        < -32.0 * np.finfo(np.float32).eps
                    )
                ):
                    raise RuntimeError(
                        f"frozen {estimator} {label} predictions are invalid or unordered"
                    )
            result[estimator] = {
                "raw_eigenvalues": raw,
                "train_affine_ordered_eigenvalues": affine,
            }
    return result


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
        "redshift_shell": np.asarray(summary["shell"], dtype=np.int8),
        "radial_selection_density_quartile": _quartile(summary["ntilde_mpc3"]),
        "random_support_boundary_distance_quartile": _quartile(
            summary["distance_to_support_boundary_mpc"]
        ),
    }
    result, nonsparse_maximum = {}, 0.0
    sparse_error = float("nan")
    for name, labels in strata.items():
        rows = {}
        for value in np.unique(labels):
            chosen = labels == value
            c68 = interval_coverage(truth[chosen], summary["eigenvalue_q16"][chosen], summary["eigenvalue_q84"][chosen])
            c90 = interval_coverage(truth[chosen], summary["eigenvalue_q05"][chosen], summary["eigenvalue_q95"][chosen])
            error = max(float(np.max(np.abs(c68 - 0.68))), float(np.max(np.abs(c90 - 0.90))))
            if name == "redshift_shell" and int(value) == 3:
                sparse_error = error
            else:
                nonsparse_maximum = max(nonsparse_maximum, error)
            rows[str(int(value))] = {"rows": int(np.count_nonzero(chosen)), "coverage68": c68.tolist(), "coverage90": c90.tolist(), "maximum_absolute_error": error}
        result[name] = rows
    if not np.isfinite(sparse_error):
        raise RuntimeError("blind evaluation lacks the registered sparse redshift shell")
    return {
        "strata": result,
        "nonsparse_maximum_absolute_error": nonsparse_maximum,
        "sparse_shell_maximum_absolute_error": sparse_error,
        "maximum_absolute_error": max(nonsparse_maximum, sparse_error),
    }


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


def _tarp_and_ranks(
    audit_draws: np.ndarray,
    audit_truth: np.ndarray,
    seed: int,
    *,
    tarp_repetitions: int,
    eigengap_seed_offset: int,
    rank_seed_offset: int,
    rank_repetitions: int,
) -> dict:
    from workflows.sbi.p12f_dependency_rescue_evaluator import tarp_curve

    samples = np.transpose(np.asarray(audit_draws, dtype=np.float64), (1, 0, 2))
    gap_samples = np.diff(samples, axis=2)
    gap_truth = np.diff(audit_truth, axis=1)
    if tarp_repetitions <= 0 or rank_repetitions != 1:
        raise RuntimeError("frozen TARP/rank repetition contract is invalid")
    eigen_replicates = [
        tarp_curve(samples, audit_truth, seed=seed + repeat)
        for repeat in range(tarp_repetitions)
    ]
    gap_replicates = [
        tarp_curve(
            gap_samples,
            gap_truth,
            seed=seed + int(eigengap_seed_offset) + repeat,
        )
        for repeat in range(tarp_repetitions)
    ]

    def summarize_tarp(replicates: list[dict], first_seed: int) -> dict:
        maximum = np.asarray(
            [item["maximum_deviation"] for item in replicates], dtype=np.float64
        )
        return {
            **replicates[0],
            "primary_seed": int(first_seed),
            "seed_repetitions": int(len(replicates)),
            "replicate_seeds": [
                int(first_seed + repeat) for repeat in range(len(replicates))
            ],
            "replicate_maximum_deviation": maximum.tolist(),
            "replicate_p90_maximum_deviation": float(np.quantile(maximum, 0.90)),
            "gate_aggregation": "p90 of fixed-seed replicate maximum deviations",
        }

    eigen_tarp = summarize_tarp(eigen_replicates, seed)
    gap_tarp = summarize_tarp(
        gap_replicates, seed + int(eigengap_seed_offset)
    )
    rank = randomized_ranks(samples, audit_truth, seed=seed + int(rank_seed_offset))
    component = [rank_cdf_maximum_deviation(rank[:, index]) for index in range(3)]
    return {
        "joint_eigenvalue_tarp": eigen_tarp,
        "joint_eigengap_tarp": gap_tarp,
        "physical_rank_cdf_maximum_by_eigenvalue": component,
        "physical_rank_cdf_maximum": float(max(component)),
        "seeds": {
            "joint_eigenvalue_tarp": int(seed),
            "joint_eigengap_tarp": int(seed + eigengap_seed_offset),
            "component_randomized_rank": int(seed + rank_seed_offset),
        },
        "rank_repetitions": int(rank_repetitions),
    }


def point_diagnostics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rows = []
    for index in range(3):
        target = truth[:, index]
        estimate = prediction[:, index]
        centered = target - target.mean()
        slope = float(np.dot(centered, estimate - estimate.mean()) / np.dot(centered, centered))
        correlation = float(np.corrcoef(target, estimate)[0, 1])
        rows.append(
            {
                "r2": weighted_r2(target, estimate),
                "slope": slope,
                "variance_ratio": float(np.var(estimate) / np.var(target)),
                "pearson_r": correlation,
            }
        )
    return {"lambda1_lambda2_lambda3": rows}


def evaluate_arrays(
    *, summary: dict[str, np.ndarray], audit: dict[str, np.ndarray], truth_parent: np.ndarray,
    truth_eigenvalues: np.ndarray, contract: dict, proper_score: dict,
    classical: dict[str, dict[str, np.ndarray]] | None = None,
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
    protocol = contract["evaluation_protocol"]
    dependence = _tarp_and_ranks(
        audit["eigenvalue_draws"],
        truth[audit_position],
        int(protocol["tarp_seed"]),
        tarp_repetitions=int(protocol["tarp_repetitions"]),
        eigengap_seed_offset=int(protocol["eigengap_tarp_seed_offset"]),
        rank_seed_offset=int(protocol["rank_seed_offset"]),
        rank_repetitions=int(protocol["rank_repetitions"]),
    )
    mean_r2 = [weighted_r2(truth[:, i], summary["eigenvalue_mean"][:, i]) for i in range(3)]
    base_r2 = [weighted_r2(truth[:, i], summary["base_prediction_eigenvalues"][:, i]) for i in range(3)]
    brier = _brier(summary, truth, contract)
    gates = contract["gates"]
    decisions = {
        "joint_eigenvalue_tarp": dependence["joint_eigenvalue_tarp"]["replicate_p90_maximum_deviation"] <= gates["joint_eigenvalue_tarp_maximum"],
        "joint_eigengap_tarp": dependence["joint_eigengap_tarp"]["replicate_p90_maximum_deviation"] <= gates["joint_eigengap_tarp_maximum"],
        "physical_rank_cdf": dependence["physical_rank_cdf_maximum"] <= gates["physical_rank_cdf_maximum"],
        "global_coverage": global_error <= gates["global_coverage_absolute_error_maximum"],
        "nonsparse_conditional_coverage": conditional["nonsparse_maximum_absolute_error"] <= gates["nonsparse_conditional_coverage_absolute_error_maximum"],
        "sparse_shell_release": conditional["sparse_shell_maximum_absolute_error"] <= gates["sparse_shell_release_absolute_error_maximum"],
        "posterior_mean_accuracy": mean_r2[0] - base_r2[0] >= gates["posterior_mean_lambda1_r2_delta_minimum"],
        "web_class_brier_skill": brier["skill"] > gates["multiclass_brier_skill_minimum"],
        "proper_score": bool(proper_score.get("pass"))
        and proper_score.get("ci95", [float("-inf")])[0]
        > gates["gaussian_minus_fmpe_energy_score_ci95_lower_minimum"],
    }
    simultaneous_pass = bool(all(decisions.values()))
    if not simultaneous_pass:
        release_status = "blocked"
    elif (
        conditional["sparse_shell_maximum_absolute_error"]
        > gates["sparse_shell_green_absolute_error_maximum"]
    ):
        release_status = "amber"
    else:
        release_status = "green"
    if "quality_bitmask" not in summary:
        raise RuntimeError("posterior summaries lack the frozen quality bitmask")
    sparse = np.asarray(summary["shell"], dtype=np.int8) == 3
    sparse_quality_bit = np.uint16(QUALITY_BITS["sparse_shell_z_ge_0p45"])
    sparse_flag_complete = bool(
        np.all(
            (
                np.asarray(summary["quality_bitmask"], dtype=np.uint16)[sparse]
                & sparse_quality_bit
            )
            != 0
        )
    )
    if not sparse_flag_complete:
        release_status = "blocked"
        simultaneous_pass = False
        decisions["sparse_shell_quality_flag"] = False
    else:
        decisions["sparse_shell_quality_flag"] = sparse_flag_complete
    classical_report = {}
    if classical is not None:
        for estimator, predictions in classical.items():
            classical_report[estimator] = {
                "raw": point_diagnostics(truth, predictions["raw_eigenvalues"]),
                "train_affine_ordered": point_diagnostics(
                    truth, predictions["train_affine_ordered_eigenvalues"]
                ),
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
        "gaussian_minus_fmpe_joint_energy_score": proper_score,
        "classical_deterministic": classical_report,
        "gates": decisions,
        "release_status": release_status,
        "full_footprint_claim_allowed": simultaneous_pass,
        "sparse_shell_quality_flag_complete": sparse_flag_complete,
        "pass": simultaneous_pass,
    }


def recompute_evaluation_report(
    *,
    frozen_path: Path,
    opened_path: Path,
    contract_path: Path,
    truth_manifest_path: Path,
    proper_score_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute every published blind result from frozen arrays and truth."""

    frozen_path = frozen_path.resolve()
    opened_path = opened_path.resolve()
    contract_path = contract_path.resolve()
    truth_manifest_path = truth_manifest_path.resolve()
    proper_score_path = proper_score_path.resolve()
    output_path = output_path.resolve()
    frozen, opened, contract, truth_manifest = validate_open_state(
        frozen_path=frozen_path,
        opened_path=opened_path,
        contract_path=contract_path,
        truth_manifest_path=truth_manifest_path,
    )
    expected_score = Path(contract["canonical_outputs"]["energy_score_report"]).resolve()
    expected_output = Path(contract["canonical_outputs"]["evaluation_report"]).resolve()
    if proper_score_path != expected_score:
        raise PermissionError("energy-score report is not at its frozen canonical path")
    if output_path != expected_output:
        raise PermissionError("blind evaluation output is not at its frozen canonical path")
    summary, audit = load_prediction_arrays(frozen)
    if (
        len(audit["parent_node_id"])
        != int(contract["evaluation_protocol"]["audit_rows"])
        or audit["eigenvalue_draws"].shape[1]
        != int(contract["evaluation_protocol"]["posterior_draws"])
    ):
        raise RuntimeError("frozen audit set differs from evaluation contract")
    audit_core = audit_core_ids(frozen, audit["parent_node_id"])
    classical = load_classical_predictions(frozen, summary["parent_node_id"])
    truth_path = Path(truth_manifest["array"]["path"])
    with np.load(truth_path, mmap_mode="r") as truth:
        truth_parent = np.asarray(truth["parent_node_id"], dtype=np.int64)
        parent = np.asarray(summary["parent_node_id"], dtype=np.int64)
        if not np.array_equal(truth_parent, parent):
            raise RuntimeError("truth package differs from frozen posterior rows")
        order = np.argsort(parent)
        position = np.searchsorted(parent[order], audit["parent_node_id"])
        if np.any(position >= len(parent)) or not np.array_equal(
            parent[order][position], audit["parent_node_id"]
        ):
            raise RuntimeError("audit parents are absent from the compact truth")
        audit_position = order[position]
        proper_score = json.loads(proper_score_path.read_text())
        proper_recomputed = validate_proper_score_report(
            proper_score,
            frozen_path=frozen_path,
            opened_path=opened_path,
            contract_path=contract_path,
            truth_manifest_path=truth_manifest_path,
            contract=contract,
            audit_parent=audit["parent_node_id"],
            audit_core=audit_core,
            audit_draws=audit["eigenvalue_draws"],
            audit_truth=np.asarray(truth["eigenvalues"][audit_position]),
            gaussian_base=summary["base_prediction_eigenvalues"][audit_position],
            audit_shell=summary["shell"][audit_position],
            audit_cap=summary["cap"][audit_position],
            expected_truth_files=opened["truth_files_read"],
        )
        result = evaluate_arrays(
            summary=summary,
            audit=audit,
            truth_parent=truth_parent,
            truth_eigenvalues=truth["eigenvalues"],
            contract=contract,
            proper_score=proper_recomputed,
            classical=classical,
        )
    report: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "estimand": "per-galaxy joint ordered tidal-eigenvalue posterior conditional on H_fid",
        "not_a_coherent_field_posterior": True,
        "frozen_predictions": {"path": str(frozen_path), "sha256": sha256(frozen_path)},
        "opened_marker": {"path": str(opened_path), "sha256": sha256(opened_path)},
        "evaluation_contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
        "truth_manifest": {"path": str(truth_manifest_path), "sha256": sha256(truth_manifest_path)},
        "proper_score_report": {"path": str(proper_score_path), "sha256": sha256(proper_score_path)},
        "post_open_refit_performed": False,
        "post_open_tuning_allowed": False,
        "truth_files_read": opened["truth_files_read"],
        "open_count": 1,
        "sealed_phase_opened": True,
        **result,
    }
    return report


def validate_evaluation_report(
    report_path: Path, *, evaluation_contract_path: Path
) -> dict[str, Any]:
    """Deep-replay an existing report; metadata/schema checks alone are insufficient."""

    report_path = report_path.resolve()
    evaluation_contract_path = evaluation_contract_path.resolve()
    existing = json.loads(report_path.read_text())
    if existing.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("blind evaluation report schema changed")
    references: dict[str, Path] = {}
    for key in (
        "frozen_predictions",
        "opened_marker",
        "evaluation_contract",
        "truth_manifest",
        "proper_score_report",
    ):
        record = existing.get(key, {})
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or record.get("sha256") != sha256(path):
            raise RuntimeError(f"blind evaluation report does not bind {key}")
        references[key] = path
    if references["evaluation_contract"] != evaluation_contract_path:
        raise RuntimeError("blind evaluation report binds a different contract")
    replay = recompute_evaluation_report(
        frozen_path=references["frozen_predictions"],
        opened_path=references["opened_marker"],
        contract_path=evaluation_contract_path,
        truth_manifest_path=references["truth_manifest"],
        proper_score_path=references["proper_score_report"],
        output_path=report_path,
    )
    replay["created_utc"] = existing.get("created_utc")
    replay["git_revision"] = existing.get("git_revision")
    if existing != replay:
        raise RuntimeError("existing blind evaluation differs from full frozen replay")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--proper-score-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        existing_payload = json.loads(args.output.read_text())
        for key, supplied in (
            ("frozen_predictions", args.frozen_predictions),
            ("opened_marker", args.opened_marker),
            ("evaluation_contract", args.evaluation_contract),
            ("truth_manifest", args.truth_manifest),
            ("proper_score_report", args.proper_score_report),
        ):
            if Path(str(existing_payload.get(key, {}).get("path", ""))).resolve() != supplied.resolve():
                raise PermissionError(f"existing evaluation is bound to a different {key}")
        existing = validate_evaluation_report(
            args.output, evaluation_contract_path=args.evaluation_contract
        )
        print(json.dumps(existing, indent=2), flush=True)
        return
    report = recompute_evaluation_report(
        frozen_path=args.frozen_predictions,
        opened_path=args.opened_marker,
        contract_path=args.evaluation_contract,
        truth_manifest_path=args.truth_manifest,
        proper_score_path=args.proper_score_report,
        output_path=args.output,
    )
    write_json_exclusive(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
