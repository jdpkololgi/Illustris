#!/usr/bin/env python3
"""Fail-closed, two-phase transition for the single P12 ph001 opening.

The opening is deliberately split into two immutable state changes:

``authorize``
    Revalidates every frozen prediction artifact and the pre-registered
    evaluation contract, then consumes the one-open budget by creating
    ``P12_BLIND_OPEN_AUTHORIZED.json`` with ``open_count=1``.  This happens
    *before* any ph001 truth may be read.

``finalize``
    Runs only after an authorized truth builder has written the immutable
    ``P12A_PH001_TRUTH_COMPLETE.json`` marker.  It re-hashes that marker and
    every truth artifact and creates ``P12_BLIND_OPENED.json``.

Both state files are created with ``O_EXCL``.  A failed or interrupted truth
build therefore cannot silently obtain a second opening.  Truth builders must
call :func:`validate_open_authorization` before opening ph001 truth and must
copy its ``open_count=1`` and content-addressed contract references into their
completion marker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12_production_contract import (
    BLIND_SCHEMA,
    OPEN_SCHEMA,
    assert_truth_free_payload,
    freeze_blind_predictions,
)
from workflows.sbi.p12a_blind_evaluation_contract import (
    CANONICAL_OUTPUTS,
    CONDITIONAL_STRATA,
    COVARIATE_DEFINITIONS,
    EVALUATION_PROTOCOL,
    FORBIDDEN_ARTIFACTS,
    GATES,
    IMPLEMENTATION_FILES,
    RELEASE_POLICY,
    SCIENTIFIC_CONTRACT_FILES,
    SCHEMA as CONTRACT_SCHEMA,
    TRUTH_CONSTRUCTION_CONTRACT,
    TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES,
    TRUTH_FREE_IDENTITY_INPUTS,
    validate_nested_truth_free_identities,
    validate_runtime_fingerprint,
    truth_source_contract,
)
from workflows.sbi.p12a_immutable_io import write_json_exclusive
from workflows.sbi.p12a_blind_preauthorization import validate_frozen_audit_export


AUTHORIZATION_SCHEMA = "p12-blind-open-authorized-v1"
TRUTH_COMPLETE_SCHEMA = "p12a-ph001-truth-complete-v1"
AUTHORIZATION_FILENAME = "P12_BLIND_OPEN_AUTHORIZED.json"
TRUTH_COMPLETE_FILENAME = "P12A_PH001_TRUTH_COMPLETE.json"
OPEN_FILENAME = "P12_BLIND_OPENED.json"
AUTHORIZATION_TOKEN = "OPEN_PH001_ONCE"
EVALUATION_IMPLEMENTATION_FILES = IMPLEMENTATION_FILES
TRUTH_PROVENANCE_SCOPE = (
    "content-addressed manifest/inventory roots; each Particle A/B payload member is "
    "enumerated with registered POSIX checksum and byte size, while each halo payload "
    "member is bound by its registered POSIX checksum when present and otherwise by "
    "direct SHA256, in the bound source inventories"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _record_matches(record: Mapping[str, Any], path: Path) -> bool:
    path = path.resolve()
    return (
        Path(str(record.get("path", ""))).resolve() == path
        and record.get("sha256") == sha256(path)
        and ("bytes" not in record or int(record["bytes"]) == path.stat().st_size)
    )


def _canonical_path(frozen_path: Path, filename: str) -> Path:
    return frozen_path.resolve().parent / filename


def _source_record(frozen: Mapping[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    record = frozen.get(key, {})
    path = Path(str(record.get("path", ""))).resolve()
    if not _record_matches(record, path):
        raise RuntimeError(f"frozen source is stale: {key}")
    return path, dict(record)


def validate_frozen_predictions(path: Path, *, deep: bool = True) -> dict[str, Any]:
    """Validate the marker and, by default, replay the full freeze validator."""

    path = path.resolve()
    frozen = _load_json(path)
    assert_truth_free_payload(frozen)
    if frozen.get("schema_version") != BLIND_SCHEMA or frozen.get("pass") is not True:
        raise RuntimeError("P12 blind predictions are not frozen")
    candidate_path, candidate_record = _source_record(frozen, "p12a_candidate")
    selection_path, selection_record = _source_record(frozen, "p12f_selection")
    deterministic_path, deterministic_record = _source_record(
        frozen, "p10_deterministic_contract"
    )
    manifest_records = frozen.get("prediction_manifests", [])
    if len(manifest_records) != 4:
        raise RuntimeError("frozen prediction manifest inventory changed")
    manifest_paths: list[Path] = []
    for record in manifest_records:
        manifest_path = Path(str(record.get("path", ""))).resolve()
        if not _record_matches(record, manifest_path):
            raise RuntimeError("frozen prediction manifest is stale")
        manifest_paths.append(manifest_path)

    if deep:
        replay = freeze_blind_predictions(
            candidate_marker=candidate_path,
            method_selection_marker=selection_path,
            prediction_manifests=manifest_paths,
            deterministic_contract=deterministic_path,
        )
        for key, expected in (
            ("p12a_candidate", candidate_record),
            ("p12f_selection", selection_record),
            ("p10_deterministic_contract", deterministic_record),
        ):
            observed = replay.get(key, {})
            if (
                Path(str(observed.get("path", ""))).resolve()
                != Path(str(expected["path"])).resolve()
                or observed.get("sha256") != expected.get("sha256")
            ):
                raise RuntimeError(f"deep revalidation changed frozen source: {key}")
        expected_manifests = sorted(
            (str(Path(str(item["path"])).resolve()), item["sha256"])
            for item in manifest_records
        )
        replay_manifests = sorted(
            (str(Path(str(item["path"])).resolve()), item["sha256"])
            for item in replay.get("prediction_manifests", [])
        )
        if replay_manifests != expected_manifests:
            raise RuntimeError("deep prediction revalidation changed the frozen inventory")
        candidate = _load_json(candidate_path)
        for name, item in candidate.get("artifacts", {}).items():
            artifact_path = Path(str(item.get("path", ""))).resolve()
            if not _record_matches(item, artifact_path):
                raise RuntimeError(f"P12-A candidate artifact changed: {name}")
        checkpoint_path = Path(
            str(candidate.get("artifacts", {}).get("checkpoint", {}).get("path", ""))
        ).resolve()
        # This is a truth-free CPU construction smoke: authorization is refused
        # if the stored estimator can no longer be reconstructed by the frozen
        # runtime.  No posterior fitting or ph001 target access occurs.
        from workflows.sbi.p12a_blind_inference import reconstruct_fmpe

        posterior, checkpoint = reconstruct_fmpe(checkpoint_path, "cpu")
        if posterior is None or not isinstance(checkpoint, dict):
            raise RuntimeError("frozen FMPE checkpoint reconstruction failed")
        del posterior, checkpoint
        audit_validation = validate_frozen_audit_export(frozen, candidate)
        # The validator already checks exact identities.  These fixed-size
        # assertions also make the authorization evidence fail closed if its
        # return schema is weakened in a later refactor.
        if (
            audit_validation["audit_rows"] != 50_000
            or audit_validation["posterior_draws"] != 512
            or audit_validation["summary_rows"] <= 0
            or audit_validation["shards"] <= 0
        ):
            raise RuntimeError("deep P12-A audit validation returned invalid counts")
        # This is an in-memory authorization receipt only; the frozen marker on
        # disk is never modified.  The caller copies the counts into the
        # exclusive authorization marker so later guards can require proof that
        # the deep path, rather than a shallow unit-test path, was used.
        frozen = dict(frozen)
        frozen["_authorization_deep_validation"] = audit_validation
    return frozen


def validate_evaluation_contract(path: Path) -> dict[str, Any]:
    contract = _load_json(path.resolve())
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("phase") != "ph001"
        or contract.get("pass") is not True
        or contract.get("open_count") != 0
        or bool(contract.get("sealed_phase_opened"))
        or contract.get("post_open_refit_allowed") is not False
    ):
        raise PermissionError("P12-A blind evaluation contract is not pre-open/frozen")
    truth_paths = json.dumps(contract.get("truth_files_read", []), sort_keys=True).lower()
    if "ph001" in truth_paths:
        raise PermissionError("evaluation contract records ph001 truth access")
    if (
        contract.get("gates") != GATES
        or contract.get("conditional_strata") != CONDITIONAL_STRATA
        or contract.get("covariate_definitions") != COVARIATE_DEFINITIONS
        or contract.get("release_policy") != RELEASE_POLICY
        or contract.get("evaluation_protocol") != EVALUATION_PROTOCOL
        or contract.get("canonical_outputs") != CANONICAL_OUTPUTS
    ):
        raise RuntimeError("blind evaluation gates or conditional strata changed")
    if (
        float(contract.get("class_threshold", float("nan"))) != 0.2
        or contract.get("bootstrap_unit") != "authoritative core"
        or contract.get("primary_proper_score")
        != (
            "physical joint energy score on frozen 50k x 512 production draws; "
            "positive Gaussian-minus-FMPE difference favours FMPE"
        )
        or contract.get("unnormalized_fmpe_log_score_is_decisive") is not False
    ):
        raise RuntimeError("blind evaluation estimand changed")
    for key in ("candidate", "gaussian_baseline", "dataset_marker", "training_sample"):
        record = contract.get(key, {})
        record_path = Path(str(record.get("path", ""))).resolve()
        if "ph001" in str(record_path).lower() or not _record_matches(record, record_path):
            raise RuntimeError(f"blind evaluation contract source changed: {key}")
    candidate = _load_json(Path(contract["candidate"]["path"]))
    if candidate.get("strict_calibration_pass_marker_present") is not False:
        raise RuntimeError("P12-A candidate no longer records the strict calibration failure")
    expected_artifacts = contract.get("candidate_artifacts", {})
    if set(expected_artifacts) != set(candidate.get("artifacts", {})):
        raise RuntimeError("blind evaluation contract candidate-artifact inventory changed")
    for name, record in expected_artifacts.items():
        artifact_path = Path(str(record.get("path", ""))).resolve()
        if not _record_matches(record, artifact_path):
            raise RuntimeError(f"blind candidate artifact changed: {name}")
    if contract.get("truth_files_read") != [contract["training_sample"]["path"]]:
        raise RuntimeError("evaluation contract truth provenance is not training-only")
    climatology = contract.get("shell_class_climatology", {})
    if set(climatology) != {"0", "1", "2", "3"}:
        raise RuntimeError("evaluation contract shell climatology is incomplete")
    for shell in climatology.values():
        probability = np.asarray(
            shell.get("probability_void_sheet_filament_knot", []), dtype=np.float64
        )
        if probability.shape != (4,) or np.any(probability < 0) or not np.isclose(
            probability.sum(), 1.0
        ):
            raise RuntimeError("evaluation contract shell climatology is invalid")
    records = contract.get("evaluation_implementation", {})
    if set(records) != set(EVALUATION_IMPLEMENTATION_FILES):
        raise RuntimeError("blind evaluation implementation inventory is not frozen")
    for name, implementation_path in EVALUATION_IMPLEMENTATION_FILES.items():
        if not _record_matches(records[name], implementation_path):
            raise RuntimeError(f"blind evaluation implementation changed: {name}")
    scientific_records = contract.get("scientific_contract_sources", {})
    if set(scientific_records) != set(SCIENTIFIC_CONTRACT_FILES):
        raise RuntimeError("blind scientific-contract source inventory is not frozen")
    for name, scientific_path in SCIENTIFIC_CONTRACT_FILES.items():
        if not _record_matches(scientific_records[name], scientific_path):
            raise RuntimeError(f"blind scientific-contract source changed: {name}")
    expected_absent = {
        name: str(path.resolve()) for name, path in FORBIDDEN_ARTIFACTS.items()
    }
    if contract.get("forbidden_artifacts_absent") != expected_absent:
        raise RuntimeError("blind forbidden-artifact contract changed")
    for name, path in FORBIDDEN_ARTIFACTS.items():
        if path.exists():
            raise RuntimeError(f"forbidden post-fit artifact appeared: {name}: {path}")
    truth_records = contract.get("truth_construction_implementation", {})
    if set(truth_records) != set(TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES):
        raise RuntimeError("blind truth-construction implementation inventory is not frozen")
    for name, implementation_path in TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES.items():
        if not _record_matches(truth_records[name], implementation_path):
            raise RuntimeError(f"blind truth-construction implementation changed: {name}")
    identity_records = contract.get("truth_free_identity_inputs", {})
    if set(identity_records) != set(TRUTH_FREE_IDENTITY_INPUTS):
        raise RuntimeError("blind truth-free identity inventory is not frozen")
    for name, identity_path in TRUTH_FREE_IDENTITY_INPUTS.items():
        if not _record_matches(identity_records[name], identity_path):
            raise RuntimeError(f"blind truth-free identity input changed: {name}")
    validate_nested_truth_free_identities()
    if contract.get("truth_source_contract") != truth_source_contract(
        TRUTH_FREE_IDENTITY_INPUTS["phase_registry"]
    ):
        raise RuntimeError("blind truth-source location contract changed")
    validate_runtime_fingerprint(contract.get("runtime", {}))
    if contract.get("truth_construction_contract") != TRUTH_CONSTRUCTION_CONTRACT:
        raise RuntimeError("blind truth-construction physics or isolation contract changed")
    return contract


def build_authorization_marker(
    *,
    frozen_path: Path,
    evaluation_contract_path: Path,
    explicit_authorization: str,
    deep: bool = True,
) -> dict[str, Any]:
    if explicit_authorization != AUTHORIZATION_TOKEN:
        raise PermissionError("explicit one-open authorization token is absent")
    frozen = validate_frozen_predictions(frozen_path, deep=deep)
    validate_evaluation_contract(evaluation_contract_path)
    deep_validation = frozen.pop("_authorization_deep_validation", None)
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "state": "truth_access_authorized",
        "frozen_predictions_reference": _record(frozen_path),
        "evaluation_contract_reference": _record(evaluation_contract_path),
        "deep_prediction_revalidation": {
            "performed": bool(deep),
            "prediction_manifest_count": len(frozen["prediction_manifests"]),
            **(deep_validation or {}),
            "pass": True,
        },
        "truth_files_read": [],
        "truth_materialization_complete": False,
        "open_count": 1,
        "sealed_phase_opened": True,
        "evaluation_only": True,
        "post_open_refit_allowed": False,
        "post_open_tuning_allowed": False,
        "pass": True,
    }


def validate_open_authorization(
    *,
    authorization_path: Path,
    frozen_path: Path | None = None,
    evaluation_contract_path: Path | None = None,
) -> dict[str, Any]:
    """Guard to call immediately before any truth-bearing ph001 read."""

    authorization_path = authorization_path.resolve()
    marker = _load_json(authorization_path)
    deep_validation = marker.get("deep_prediction_revalidation", {})
    if (
        marker.get("schema_version") != AUTHORIZATION_SCHEMA
        or marker.get("phase") != "ph001"
        or marker.get("state") != "truth_access_authorized"
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("truth_materialization_complete") is not False
        or marker.get("post_open_refit_allowed") is not False
        or marker.get("post_open_tuning_allowed") is not False
        or marker.get("pass") is not True
        or marker.get("truth_files_read") != []
        or deep_validation.get("performed") is not True
        or deep_validation.get("pass") is not True
        or int(deep_validation.get("prediction_manifest_count", -1)) != 4
        or int(deep_validation.get("summary_rows", -1))
        != TRUTH_CONSTRUCTION_CONTRACT["expected_supported_context_rows"]
        or int(deep_validation.get("audit_rows", -1))
        != EVALUATION_PROTOCOL["audit_rows"]
        or int(deep_validation.get("posterior_draws", -1))
        != EVALUATION_PROTOCOL["posterior_draws"]
        or int(deep_validation.get("shards", -1)) != 4
    ):
        raise PermissionError("canonical ph001 truth authorization is invalid")
    frozen_reference = marker.get("frozen_predictions_reference", {})
    contract_reference = marker.get("evaluation_contract_reference", {})
    bound_frozen = Path(str(frozen_reference.get("path", ""))).resolve()
    bound_contract = Path(str(contract_reference.get("path", ""))).resolve()
    if frozen_path is not None and bound_frozen != frozen_path.resolve():
        raise PermissionError("authorization is bound to a different prediction freeze")
    if evaluation_contract_path is not None and bound_contract != evaluation_contract_path.resolve():
        raise PermissionError("authorization is bound to a different evaluation contract")
    if not _record_matches(frozen_reference, bound_frozen):
        raise RuntimeError("frozen predictions changed after truth authorization")
    if not _record_matches(contract_reference, bound_contract):
        raise RuntimeError("evaluation contract changed after truth authorization")
    # The contract hash alone is insufficient if a frozen builder or evaluator
    # is edited in place after authorization.  Revalidate every nested source
    # hash before each truth-bearing stage.
    validate_evaluation_contract(bound_contract)
    canonical = bound_frozen.parent / AUTHORIZATION_FILENAME
    if authorization_path != canonical:
        raise PermissionError(f"truth authorization must use canonical path {canonical}")
    return marker


def build_truth_complete_marker(
    *,
    authorization_path: Path,
    truth_artifacts: list[Path],
    truth_input_manifests: list[Path] | None = None,
    truth_array: Path,
    rows: int,
) -> dict[str, Any]:
    """Helper for the future authorized truth builder's terminal marker.

    Calling this helper is truth-bearing and is therefore valid only after the
    exclusive authorization marker exists.  It does not interpret the science
    arrays; the dedicated builder/evaluator performs those checks.
    """

    authorization = validate_open_authorization(authorization_path=authorization_path)
    if rows <= 0:
        raise RuntimeError("truth completion must contain supported ph001 rows")
    unique_paths = {path.resolve() for path in [*truth_artifacts, truth_array]}
    if not unique_paths:
        raise RuntimeError("truth completion contains no artifacts")
    for path in unique_paths:
        if "ph001" not in str(path).lower():
            raise PermissionError("every truth artifact must identify ph001")
    artifacts = [_record(path) for path in sorted(unique_paths, key=str)]
    input_paths = sorted(
        {path.resolve() for path in (truth_input_manifests or [])}, key=str
    )
    inputs = [_record(path) for path in input_paths]
    array_record = _record(truth_array)
    return {
        "schema_version": TRUTH_COMPLETE_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "state": "truth_materialization_complete",
        "authorization_reference": _record(authorization_path),
        "frozen_predictions_reference": authorization["frozen_predictions_reference"],
        "evaluation_contract_reference": authorization["evaluation_contract_reference"],
        "array": array_record,
        "truth_artifacts": artifacts,
        "generated_truth_artifacts": artifacts,
        "truth_input_manifests": inputs,
        "truth_read_provenance_scope": TRUTH_PROVENANCE_SCOPE,
        "rows": int(rows),
        "truth_files_read": [*inputs, *artifacts],
        "truth_materialization_complete": True,
        "fit_performed": False,
        "phase_registry_mutation_performed": False,
        "open_count": 1,
        "sealed_phase_opened": True,
        "post_open_refit_allowed": False,
        "post_open_tuning_allowed": False,
        "pass": True,
    }


def validate_truth_complete(
    *,
    truth_complete_path: Path,
    authorization_path: Path,
    frozen_path: Path,
    evaluation_contract_path: Path,
    deep_artifacts: bool = True,
) -> dict[str, Any]:
    authorization = validate_open_authorization(
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )
    marker = _load_json(truth_complete_path.resolve())
    if (
        marker.get("schema_version") != TRUTH_COMPLETE_SCHEMA
        or marker.get("phase") != "ph001"
        or marker.get("state") != "truth_materialization_complete"
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("truth_materialization_complete") is not True
        or marker.get("fit_performed") is not False
        or marker.get("phase_registry_mutation_performed") is not False
        or marker.get("post_open_refit_allowed") is not False
        or marker.get("post_open_tuning_allowed") is not False
        or marker.get("pass") is not True
        or int(marker.get("rows", 0)) <= 0
        or marker.get("truth_read_provenance_scope") != TRUTH_PROVENANCE_SCOPE
    ):
        raise PermissionError("authorized ph001 truth completion is invalid")
    canonical = frozen_path.resolve().parent / TRUTH_COMPLETE_FILENAME
    if truth_complete_path.resolve() != canonical:
        raise PermissionError(f"truth completion must use canonical path {canonical}")
    references = (
        ("authorization_reference", authorization_path.resolve(), sha256(authorization_path)),
        ("frozen_predictions_reference", frozen_path.resolve(), sha256(frozen_path)),
        (
            "evaluation_contract_reference",
            evaluation_contract_path.resolve(),
            sha256(evaluation_contract_path),
        ),
    )
    for key, expected_path, expected_hash in references:
        record = marker.get(key, {})
        if (
            Path(str(record.get("path", ""))).resolve() != expected_path
            or record.get("sha256") != expected_hash
        ):
            raise RuntimeError(f"truth completion does not bind {key}")
    if marker["frozen_predictions_reference"] != authorization["frozen_predictions_reference"]:
        raise RuntimeError("truth completion changed the authorized prediction reference")
    if marker["evaluation_contract_reference"] != authorization["evaluation_contract_reference"]:
        raise RuntimeError("truth completion changed the authorized evaluation contract")
    artifacts = marker.get("truth_artifacts", [])
    inputs = marker.get("truth_input_manifests", [])
    if (
        not artifacts
        or marker.get("generated_truth_artifacts") != artifacts
        or marker.get("truth_files_read") != [*inputs, *artifacts]
    ):
        raise RuntimeError("truth completion artifact inventory is absent or inconsistent")
    for record in inputs:
        path = Path(str(record.get("path", ""))).resolve()
        if (
            not path.is_file()
            or not isinstance(record.get("sha256"), str)
            or (deep_artifacts and not _record_matches(record, path))
        ):
            raise RuntimeError("truth completion contains a stale input manifest")
    artifact_by_path: dict[Path, str] = {}
    for record in artifacts:
        path = Path(str(record.get("path", ""))).resolve()
        if (
            "ph001" not in str(path).lower()
            or not path.is_file()
            or not isinstance(record.get("sha256"), str)
            or (deep_artifacts and not _record_matches(record, path))
        ):
            raise RuntimeError("truth completion contains a stale or non-ph001 artifact")
        artifact_by_path[path] = str(record["sha256"])
    array = marker.get("array", {})
    array_path = Path(str(array.get("path", ""))).resolve()
    # The compact array is the only truth payload read by post-open scoring and
    # is therefore always re-hashed, even in the lightweight chain-claim path.
    # ``deep_artifacts=False`` skips only the large, already-finalized density and
    # T-web intermediates; the finalize transition validates all of those once.
    if not _record_matches(array, array_path) or artifact_by_path.get(array_path) != array.get("sha256"):
        raise RuntimeError("truth completion does not register its canonical array")
    frozen = _load_json(frozen_path)
    context_manifest = None
    for record in frozen.get("prediction_manifests", []):
        path = Path(str(record.get("path", ""))).resolve()
        if not _record_matches(record, path):
            raise RuntimeError("frozen prediction manifest changed before truth finalization")
        payload = _load_json(path)
        if payload.get("schema_version") == "p12a-blind-base-context-v1":
            context_manifest = payload
    if context_manifest is None:
        raise RuntimeError("frozen predictions lack the authoritative blind context")
    context_path = Path(str(context_manifest.get("array", ""))).resolve()
    if context_manifest.get("array_sha256") != sha256(context_path):
        raise RuntimeError("authoritative blind context changed before truth finalization")
    with np.load(context_path, mmap_mode="r") as context, np.load(
        array_path, mmap_mode="r"
    ) as truth:
        if not {"parent_node_id", "eigenvalues"}.issubset(truth.files):
            raise RuntimeError("truth completion array lacks parent IDs or eigenvalues")
        reference_parent = np.asarray(context["parent_node_id"], dtype=np.int64)
        truth_parent = np.asarray(truth["parent_node_id"], dtype=np.int64)
        eigenvalues = np.asarray(truth["eigenvalues"], dtype=np.float64)
        if (
            int(marker["rows"]) != len(reference_parent)
            or not np.array_equal(truth_parent, reference_parent)
        ):
            raise RuntimeError("truth completion does not exactly cover frozen supported parents")
        if (
            eigenvalues.shape != (len(reference_parent), 3)
            or not np.all(np.isfinite(eigenvalues))
            or np.any(np.diff(eigenvalues, axis=1) < 0.0)
        ):
            raise RuntimeError("truth completion contains invalid ordered eigenvalues")
    return marker


def build_opened_marker(
    *,
    frozen_path: Path,
    evaluation_contract_path: Path,
    authorization_path: Path,
    truth_complete_path: Path,
) -> dict[str, Any]:
    truth_complete = validate_truth_complete(
        truth_complete_path=truth_complete_path,
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )
    truth_files = [_record(truth_complete_path)]
    truth_files.extend(dict(record) for record in truth_complete["truth_files_read"])
    return {
        "schema_version": OPEN_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "state": "blind_truth_opened",
        "authorization_reference": _record(authorization_path),
        "frozen_predictions_reference": _record(frozen_path),
        "evaluation_contract_reference": _record(evaluation_contract_path),
        "truth_complete_reference": _record(truth_complete_path),
        "truth_files_read": truth_files,
        "truth_materialization_complete": True,
        "open_count": 1,
        "sealed_phase_opened": True,
        "evaluation_only": True,
        "phase_registry_mutation_performed": False,
        "post_open_refit_allowed": False,
        "post_open_tuning_allowed": False,
        "pass": True,
    }


def validate_opened_marker(
    *,
    opened_path: Path,
    frozen_path: Path,
    evaluation_contract_path: Path,
    authorization_path: Path,
    truth_complete_path: Path,
) -> dict[str, Any]:
    """Validate an existing immutable final transition for safe job replay."""

    opened_path = opened_path.resolve()
    canonical = _canonical_path(frozen_path, OPEN_FILENAME)
    if opened_path != canonical:
        raise PermissionError(f"opened marker must use canonical path {canonical}")
    marker = _load_json(opened_path)
    truth_complete = validate_truth_complete(
        truth_complete_path=truth_complete_path,
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )
    expected_truth_files = [_record(truth_complete_path)]
    expected_truth_files.extend(
        dict(record) for record in truth_complete["truth_files_read"]
    )
    if (
        marker.get("schema_version") != OPEN_SCHEMA
        or marker.get("phase") != "ph001"
        or marker.get("state") != "blind_truth_opened"
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("truth_materialization_complete") is not True
        or marker.get("evaluation_only") is not True
        or marker.get("phase_registry_mutation_performed") is not False
        or marker.get("post_open_refit_allowed") is not False
        or marker.get("post_open_tuning_allowed") is not False
        or marker.get("pass") is not True
        or marker.get("authorization_reference") != _record(authorization_path)
        or marker.get("frozen_predictions_reference") != _record(frozen_path)
        or marker.get("evaluation_contract_reference")
        != _record(evaluation_contract_path)
        or marker.get("truth_complete_reference") != _record(truth_complete_path)
        or marker.get("truth_files_read") != expected_truth_files
    ):
        raise RuntimeError("existing opened marker differs from frozen one-open state")
    return marker


def _authorize(args: argparse.Namespace) -> dict[str, Any]:
    canonical = _canonical_path(args.frozen_predictions, AUTHORIZATION_FILENAME)
    if args.output.resolve() != canonical:
        raise PermissionError(f"authorization marker must use canonical path {canonical}")
    if args.output.exists():
        return validate_open_authorization(
            authorization_path=args.output,
            frozen_path=args.frozen_predictions,
            evaluation_contract_path=args.evaluation_contract,
        )
    marker = build_authorization_marker(
        frozen_path=args.frozen_predictions,
        evaluation_contract_path=args.evaluation_contract,
        explicit_authorization=args.authorization,
        deep=True,
    )
    write_json_exclusive(args.output, marker)
    return marker


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    canonical = _canonical_path(args.frozen_predictions, OPEN_FILENAME)
    if args.output.resolve() != canonical:
        raise PermissionError(f"opened marker must use canonical path {canonical}")
    if args.output.exists():
        return validate_opened_marker(
            opened_path=args.output,
            frozen_path=args.frozen_predictions,
            evaluation_contract_path=args.evaluation_contract,
            authorization_path=args.authorization_marker,
            truth_complete_path=args.truth_complete,
        )
    marker = build_opened_marker(
        frozen_path=args.frozen_predictions,
        evaluation_contract_path=args.evaluation_contract,
        authorization_path=args.authorization_marker,
        truth_complete_path=args.truth_complete,
    )
    write_json_exclusive(args.output, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize", help="consume the one-open budget")
    authorize.add_argument("--frozen-predictions", type=Path, required=True)
    authorize.add_argument("--evaluation-contract", type=Path, required=True)
    authorize.add_argument("--authorization", required=True)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.set_defaults(handler=_authorize)
    finalize = subparsers.add_parser("finalize", help="freeze completed truth artifacts")
    finalize.add_argument("--frozen-predictions", type=Path, required=True)
    finalize.add_argument("--evaluation-contract", type=Path, required=True)
    finalize.add_argument("--authorization-marker", type=Path, required=True)
    finalize.add_argument("--truth-complete", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(handler=_finalize)
    args = parser.parse_args()
    marker = args.handler(args)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
