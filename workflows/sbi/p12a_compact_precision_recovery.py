#!/usr/bin/env python3
"""Authorized precision-only recovery of the failed P12-A compact-truth join.

The pre-open truth builder remains byte-for-byte unchanged.  This narrowly
scoped recovery accepts only the already diagnosed single float32 threshold
ambiguity, keeps the native CACTUS CWEB label and stored eigenvalues unchanged,
and records the exception plus its implementation in the terminal truth
provenance.  It cannot fit, score, recalibrate, or change an acceptance gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from workflows.abacus_tweb.p10_target_contract import stored_class_consistency
from workflows.sbi import p12a_authorized_truth as frozen


SCHEMA = "p12a-compact-precision-exception-v1"
CLAIM_SCHEMA = "p12a-compact-precision-resume-claim-v1"
JOB_SCHEMA = "p12a-compact-precision-resume-job-v1"
SUBMISSION_SCHEMA = "p12a-compact-precision-resume-submitted-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = (
    REPO_ROOT
    / "docs/evidence/p12/p12a_blind_opening_20260905/"
    "P12A_COMPACT_CLOSURE_DIAGNOSTIC_57935350.json"
)
EXCEPTION = frozen.BLIND_ROOT / "P12A_COMPACT_PRECISION_EXCEPTION.json"
RECOVERY_SLURM = Path(__file__).with_name(
    "submit_p12a_ph001_compact_precision_recovery.slurm"
)
SUBMISSION_SCRIPT = Path(__file__).with_name(
    "submit_p12a_ph001_precision_recovery_chain.sh"
)
TARGET_CONTRACT = REPO_ROOT / "workflows/abacus_tweb/p10_target_contract.py"
RESUME_CLAIM = frozen.BLIND_ROOT / "P12A_COMPACT_PRECISION_RECOVERY_CLAIM.json"
RESUME_SUBMITTED = (
    frozen.BLIND_ROOT / "P12A_COMPACT_PRECISION_RECOVERY_SUBMITTED.json"
)
RESUME_RECORD_ROOT = frozen.BLIND_ROOT / "chain_submissions/precision_recovery"
RESUME_JOBS = ("compact_precision_recovery", "postopen_dispatch")
FAILED_COMPACT_JOB = "57928446"
BLOCKED_DISPATCHER_JOB = "57928546"
USER_AUTHORIZATION = (
    "2026-09-05 user request: Please continue; in response to the explicit "
    "precision-only exception request, retaining predictions, eigenvalues, "
    "thresholds, calibration gates and open_count=1"
)


class PrecisionRecoveryError(RuntimeError):
    """The bounded recovery contract was violated."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _reference_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and left.get("sha256") == right.get("sha256")
        and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
    )


def validate_diagnostic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    closure = payload.get("closure", {})
    if (
        payload.get("schema_version") != "p12a-compact-closure-diagnostic-v1"
        or payload.get("open_count") != 1
        or payload.get("identity_join_exact") is not True
        or payload.get("posterior_scores_computed") is not False
        or payload.get("predictions_modified") is not False
        or payload.get("truth_outputs_modified") is not False
        or closure.get("rows") != frozen.EXPECTED_CONTEXT_ROWS
        or closure.get("source_dtype") != "float32"
        or closure.get("finite") is not True
        or closure.get("ordered") is not True
        or closure.get("source_values_preserved_by_float32") is not True
        or closure.get("float32_comparison_class_mismatches") != 1
        or closure.get("float64_comparison_class_mismatches") != 0
        or closure.get("rows_at_rounded_threshold") != 1
        or closure.get("all_float32_mismatches_explained_by_threshold_rounding")
        is not True
    ):
        raise PrecisionRecoveryError("diagnostic is not the authorized one-row precision case")
    return payload


def validate_diagnostic() -> dict[str, Any]:
    frozen.authorization_context()
    payload = validate_diagnostic_payload(_load(DIAGNOSTIC))
    if (
        not _reference_equal(payload.get("authorization", {}), frozen.record(frozen.AUTHORIZATION))
        or not _reference_equal(
            payload.get("annotation_stage", {}),
            frozen.record(frozen.stage_marker_path(frozen.TRUTH_ROOT, "annotation")),
        )
        or not frozen.record_matches(payload.get("diagnostic_implementation", {}))
    ):
        raise PrecisionRecoveryError("diagnostic provenance changed")
    return payload


def _frozen_builder_records() -> dict[str, Any]:
    # This call revalidates every pre-open source hash.  The recovery therefore
    # cannot conceal an edit to the frozen builder or evaluation contract.
    frozen.authorization_context()
    contract = _load(frozen.EVALUATION_CONTRACT)
    records = contract.get("truth_construction_implementation", {})
    expected = {
        "authorized_truth_wrapper": frozen.record(Path(frozen.__file__)),
        "compact_truth_slurm": frozen.record(
            Path(frozen.__file__).with_name("submit_p12a_ph001_compact_truth.slurm")
        ),
    }
    for key, value in expected.items():
        if not _reference_equal(records.get(key, {}), value):
            raise PrecisionRecoveryError(f"frozen truth source changed: {key}")
    return expected


def _recovery_sources() -> dict[str, Any]:
    return {
        "recovery_implementation": frozen.record(Path(__file__)),
        "recovery_slurm": frozen.record(RECOVERY_SLURM),
        "submission_script": frozen.record(SUBMISSION_SCRIPT),
        "precision_aware_target_contract": frozen.record(TARGET_CONTRACT),
    }


def authorize_exception() -> dict[str, Any]:
    diagnostic = validate_diagnostic()
    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": frozen.git_revision(),
        "phase": frozen.PHASE,
        "pass": True,
        "open_count": 1,
        "sealed_phase_opened": True,
        "user_authorization": USER_AUTHORIZATION,
        "scope": "one-row float32 threshold-validation recovery only",
        "scientific_threshold": frozen.TARGET["web_threshold"],
        "native_cweb_retained": True,
        "stored_eigenvalues_retained": True,
        "allowed_boundary_ambiguity_rows": 1,
        "posterior_predictions_modified": False,
        "posterior_scores_inspected": False,
        "fit_or_recalibration_allowed": False,
        "acceptance_gate_change_allowed": False,
        "evaluation_contract_rewrite_allowed": False,
        "original_failed_compact_job": FAILED_COMPACT_JOB,
        "blocked_dispatcher_to_supersede": BLOCKED_DISPATCHER_JOB,
        "authorization_reference": frozen.record(frozen.AUTHORIZATION),
        "evaluation_contract_reference": frozen.record(frozen.EVALUATION_CONTRACT),
        "diagnostic_reference": frozen.record(DIAGNOSTIC),
        "diagnostic_closure": diagnostic["closure"],
        "frozen_builder_sources": _frozen_builder_records(),
        "recovery_sources": _recovery_sources(),
    }
    frozen.write_json_exclusive(EXCEPTION, payload)
    return validate_exception()


def validate_exception() -> dict[str, Any]:
    validate_diagnostic()
    original = _frozen_builder_records()
    payload = _load(EXCEPTION)
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("phase") != frozen.PHASE
        or payload.get("pass") is not True
        or payload.get("open_count") != 1
        or payload.get("sealed_phase_opened") is not True
        or payload.get("user_authorization") != USER_AUTHORIZATION
        or payload.get("scope") != "one-row float32 threshold-validation recovery only"
        or payload.get("scientific_threshold") != frozen.TARGET["web_threshold"]
        or payload.get("native_cweb_retained") is not True
        or payload.get("stored_eigenvalues_retained") is not True
        or payload.get("allowed_boundary_ambiguity_rows") != 1
        or payload.get("posterior_predictions_modified") is not False
        or payload.get("posterior_scores_inspected") is not False
        or payload.get("fit_or_recalibration_allowed") is not False
        or payload.get("acceptance_gate_change_allowed") is not False
        or payload.get("evaluation_contract_rewrite_allowed") is not False
        or payload.get("original_failed_compact_job") != FAILED_COMPACT_JOB
        or payload.get("blocked_dispatcher_to_supersede") != BLOCKED_DISPATCHER_JOB
        or payload.get("diagnostic_closure") != _load(DIAGNOSTIC)["closure"]
        or not _reference_equal(
            payload.get("authorization_reference", {}), frozen.record(frozen.AUTHORIZATION)
        )
        or not _reference_equal(
            payload.get("evaluation_contract_reference", {}),
            frozen.record(frozen.EVALUATION_CONTRACT),
        )
        or not _reference_equal(payload.get("diagnostic_reference", {}), frozen.record(DIAGNOSTIC))
        or payload.get("frozen_builder_sources") != original
        or payload.get("recovery_sources") != _recovery_sources()
    ):
        raise PrecisionRecoveryError("precision exception marker changed")
    return payload


def precision_aware_join(
    *,
    context_parent: np.ndarray,
    canonical_parent: np.ndarray,
    canonical_targetid: np.ndarray,
    annotated_targetid: np.ndarray,
    annotated_eigenvalues: np.ndarray,
    annotated_cweb: np.ndarray,
    expected_boundary_ambiguities: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact identity join allowing only diagnosed float32 threshold ambiguity."""
    context_parent = np.asarray(context_parent, dtype=np.int64)
    canonical_parent = np.asarray(canonical_parent, dtype=np.int64)
    canonical_targetid = np.asarray(canonical_targetid, dtype=np.int64)
    if len(np.unique(context_parent)) != len(context_parent):
        raise PrecisionRecoveryError("frozen context parent IDs are not unique")
    if not np.array_equal(canonical_parent, np.arange(len(canonical_parent), dtype=np.int64)):
        raise PrecisionRecoveryError("P1 canonical parent IDs are not identity aligned")
    if np.any(context_parent < 0) or np.any(context_parent >= len(canonical_parent)):
        raise PrecisionRecoveryError("frozen context parent lies outside P1")
    targetid = canonical_targetid[context_parent]
    annotated_targetid = np.asarray(annotated_targetid, dtype=np.int64)
    if not np.array_equal(annotated_targetid, np.arange(1, len(annotated_targetid) + 1)):
        raise PrecisionRecoveryError("annotated parent TARGETIDs are not sequential")
    row = targetid - 1
    if np.any(row < 0) or np.any(row >= len(annotated_targetid)):
        raise PrecisionRecoveryError("P1 TARGETID lies outside annotated parent")
    if not np.array_equal(annotated_targetid[row], targetid):
        raise PrecisionRecoveryError("compact truth TARGETID join is not exact")

    source_eigenvalues = np.asarray(annotated_eigenvalues)
    eigenvalues = source_eigenvalues.astype(np.float32, copy=False)[row]
    cweb = np.asarray(annotated_cweb, dtype=np.uint8)[row]
    if (
        source_eigenvalues.dtype != np.float32
        or eigenvalues.shape != (len(context_parent), 3)
        or not np.all(np.isfinite(eigenvalues))
        or np.any(np.diff(eigenvalues, axis=1) < 0.0)
    ):
        raise PrecisionRecoveryError("joined compact eigenvalues are invalid")
    check = stored_class_consistency(eigenvalues, cweb, threshold=frozen.TARGET["web_threshold"])
    boundary = int(check["boundary_ambiguous"].sum())
    nonboundary = int(check["nonboundary_mismatch"].sum())
    float64_expected = np.sum(
        eigenvalues.astype(np.float64) > float(frozen.TARGET["web_threshold"]), axis=1
    ).astype(np.uint8)
    float64_mismatch = int(np.count_nonzero(float64_expected != cweb))
    if (
        nonboundary != 0
        or boundary != int(expected_boundary_ambiguities)
        or float64_mismatch != 0
    ):
        raise PrecisionRecoveryError(
            "joined truth differs from the diagnosed one-row precision exception"
        )
    audit = {
        "precision_validator": "stored_class_consistency plus diagnosed float64 cross-check",
        "scientific_threshold": float(frozen.TARGET["web_threshold"]),
        "eigenvalue_storage_dtype": "float32",
        "native_cweb_retained": True,
        "stored_eigenvalues_retained": True,
        "boundary_ambiguity_rows": boundary,
        "nonboundary_class_mismatch_rows": nonboundary,
        "float64_crosscheck_mismatch_rows": float64_mismatch,
    }
    return targetid, eigenvalues, cweb, audit


def build_compact_truth(*, output_path: Path, smoke_only: bool = False) -> dict[str, Any]:
    import fitsio

    exception = validate_exception()
    truth_root = frozen.validate_truth_root(frozen.TRUTH_ROOT)
    frozen.guard_stage(stage="compact", truth_root=truth_root)
    if frozen.stage_marker_path(truth_root, "compact").exists():
        return frozen.validate_stage_marker(stage="compact", truth_root=truth_root)
    output = output_path.resolve()
    attempts_root = (truth_root / "attempts/compact").resolve()
    if attempts_root not in output.parents:
        raise PermissionError("recovered compact truth is outside a job-scoped attempt")
    context_marker, context_manifest_path, context_path = frozen._frozen_context()
    if int(context_marker.get("rows", -1)) != frozen.EXPECTED_CONTEXT_ROWS:
        raise PrecisionRecoveryError("frozen context row count changed")
    p1 = _load(frozen.P1_MANIFEST)
    if (
        p1.get("phase") != frozen.PHASE
        or p1.get("target_truth_present") is not False
        or p1.get("index_sha256") != frozen.sha256(frozen.P1_INDEX)
    ):
        raise PrecisionRecoveryError("truth-free P1 identity contract changed")
    with np.load(context_path, mmap_mode="r") as context, np.load(
        frozen.P1_INDEX, mmap_mode="r"
    ) as canonical:
        context_parent = np.asarray(context["parent_node_id"], dtype=np.int64)
        canonical_parent = np.asarray(canonical["parent_node_id"], dtype=np.int64)
        canonical_targetid = np.asarray(canonical["targetid"], dtype=np.int64)
    annotation = frozen.validate_stage_marker(stage="annotation", truth_root=truth_root)
    annotated_path = Path(annotation["artifacts"]["annotated_parent"]["path"])
    table = fitsio.read(
        str(annotated_path),
        columns=["TARGETID", "CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3"],
    )
    annotated_eigenvalues = np.column_stack(
        (table["LAMBDA1"], table["LAMBDA2"], table["LAMBDA3"])
    )
    targetid, eigenvalues, cweb, precision_audit = precision_aware_join(
        context_parent=context_parent,
        canonical_parent=canonical_parent,
        canonical_targetid=canonical_targetid,
        annotated_targetid=table["TARGETID"],
        annotated_eigenvalues=annotated_eigenvalues,
        annotated_cweb=table["CWEB"],
        expected_boundary_ambiguities=exception["allowed_boundary_ambiguity_rows"],
    )
    if smoke_only:
        return {
            "schema_version": "p12a-compact-precision-full-row-smoke-v1",
            "created_utc": frozen.utc_now(),
            "exception_reference": frozen.record(EXCEPTION),
            "rows": int(len(context_parent)),
            "identity_join_exact": True,
            "precision_recovery": precision_audit,
            "array_or_stage_marker_written": False,
            "posterior_scores_computed": False,
            "open_count": 1,
            "pass": True,
        }
    if output.exists():
        with np.load(output, mmap_mode="r") as existing:
            exact = (
                {"parent_node_id", "targetid", "eigenvalues", "cweb"}.issubset(existing.files)
                and np.array_equal(existing["parent_node_id"], context_parent)
                and np.array_equal(existing["targetid"], targetid)
                and np.array_equal(existing["eigenvalues"], eigenvalues)
                and np.array_equal(existing["cweb"], cweb)
            )
        if not exact:
            raise PrecisionRecoveryError("orphan recovery output differs from recomputation")
    else:
        frozen.write_npz_exclusive(
            output,
            compressed=False,
            parent_node_id=context_parent,
            targetid=targetid,
            eigenvalues=eigenvalues,
            cweb=cweb,
        )
    artifacts = {
        "compact_truth": frozen.record(output),
        "frozen_context_manifest": frozen.record(context_manifest_path),
        "frozen_context_array": frozen.record(context_path),
        "p1_manifest": frozen.record(frozen.P1_MANIFEST),
        "p1_canonical_index": frozen.record(frozen.P1_INDEX),
    }
    provenance = (
        context_path,
        frozen.P1_INDEX,
        annotated_path,
        EXCEPTION,
        DIAGNOSTIC,
        Path(__file__),
        RECOVERY_SLURM,
        SUBMISSION_SCRIPT,
        TARGET_CONTRACT,
    )
    return frozen.write_stage_marker(
        stage="compact",
        truth_root=truth_root,
        upstream=("annotation",),
        artifacts=artifacts,
        audit={
            "rows": int(len(context_parent)),
            "unique_parent_rows": int(len(np.unique(context_parent))),
            "exact_frozen_context_order": True,
            "finite_ordered_eigenvalues": True,
            "class_threshold_closure": True,
            "precision_exception_reference": frozen.record(EXCEPTION),
            "precision_diagnostic_reference": frozen.record(DIAGNOSTIC),
            "precision_recovery": precision_audit,
            "mean_eigenvalues": eigenvalues.mean(axis=0, dtype=np.float64).tolist(),
            "web_class_counts_void_sheet_filament_knot": np.bincount(
                cweb, minlength=4
            ).tolist(),
        },
        truth_input_manifests=provenance,
    )


def _safe_submission_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value):
        raise ValueError("unsafe recovery submission ID")
    return value


def claim_resume(*, submission_id: str, failed_job: str, blocked_dispatcher: str) -> dict[str, Any]:
    submission_id = _safe_submission_id(submission_id)
    exception = validate_exception()
    if failed_job != FAILED_COMPACT_JOB or blocked_dispatcher != BLOCKED_DISPATCHER_JOB:
        raise PrecisionRecoveryError("recovery does not supersede the diagnosed jobs")
    if frozen.stage_marker_path(frozen.TRUTH_ROOT, "compact").exists():
        raise PrecisionRecoveryError("compact stage already exists; no recovery submission allowed")
    payload = {
        "schema_version": CLAIM_SCHEMA,
        "created_utc": frozen.utc_now(),
        "submission_id": submission_id,
        "phase": frozen.PHASE,
        "pass": True,
        "state": "claimed_before_any_recovery_sbatch",
        "expected_jobs": list(RESUME_JOBS),
        "failed_compact_job": failed_job,
        "superseded_blocked_dispatcher_job": blocked_dispatcher,
        "exception_reference": frozen.record(EXCEPTION),
        "automatic_retry_allowed": False,
        "fit_or_recalibration_allowed": False,
        "open_count": exception["open_count"],
    }
    frozen.write_json_exclusive(RESUME_CLAIM, payload)
    return payload


def _validate_claim(submission_id: str) -> dict[str, Any]:
    validate_exception()
    claim = _load(RESUME_CLAIM)
    if (
        claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("submission_id") != _safe_submission_id(submission_id)
        or claim.get("phase") != frozen.PHASE
        or claim.get("pass") is not True
        or claim.get("state") != "claimed_before_any_recovery_sbatch"
        or claim.get("expected_jobs") != list(RESUME_JOBS)
        or claim.get("failed_compact_job") != FAILED_COMPACT_JOB
        or claim.get("superseded_blocked_dispatcher_job") != BLOCKED_DISPATCHER_JOB
        or claim.get("automatic_retry_allowed") is not False
        or claim.get("fit_or_recalibration_allowed") is not False
        or claim.get("open_count") != 1
        or not _reference_equal(claim.get("exception_reference", {}), frozen.record(EXCEPTION))
    ):
        raise PrecisionRecoveryError("recovery claim changed")
    return claim


def _job_marker(submission_id: str, job: str) -> Path:
    return RESUME_RECORD_ROOT / _safe_submission_id(submission_id) / f"{job}.json"


def record_resume_job(
    *, submission_id: str, job: str, job_id: str, dependency_job_id: str | None
) -> dict[str, Any]:
    _validate_claim(submission_id)
    if job not in RESUME_JOBS or not job_id.isdigit() or int(job_id) <= 0:
        raise ValueError("invalid recovery job record")
    index = RESUME_JOBS.index(job)
    previous_id = None
    if index:
        previous = _load(_job_marker(submission_id, RESUME_JOBS[index - 1]))
        previous_id = str(previous.get("slurm_job_id", ""))
        if previous.get("schema_version") != JOB_SCHEMA or not previous_id.isdigit():
            raise PrecisionRecoveryError("previous recovery job record is invalid")
    if dependency_job_id != previous_id:
        raise PrecisionRecoveryError("recovery job dependency is not sequential")
    payload = {
        "schema_version": JOB_SCHEMA,
        "created_utc": frozen.utc_now(),
        "submission_id": submission_id,
        "job": job,
        "job_index": index,
        "slurm_job_id": job_id,
        "dependency_slurm_job_id": dependency_job_id,
        "claim_reference": frozen.record(RESUME_CLAIM),
        "recorded_immediately_after_sbatch": True,
        "pass": True,
    }
    marker = _job_marker(submission_id, job)
    marker.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_json_exclusive(marker, payload)
    return payload


def record_resume_submission(*, submission_id: str) -> dict[str, Any]:
    _validate_claim(submission_id)
    previous = None
    jobs = {}
    for index, name in enumerate(RESUME_JOBS):
        path = _job_marker(submission_id, name)
        row = _load(path)
        if (
            row.get("schema_version") != JOB_SCHEMA
            or row.get("submission_id") != submission_id
            or row.get("job") != name
            or row.get("job_index") != index
            or row.get("dependency_slurm_job_id") != previous
            or row.get("pass") is not True
        ):
            raise PrecisionRecoveryError("recovery job ledger is invalid")
        previous = str(row["slurm_job_id"])
        jobs[name] = {"slurm_job_id": previous, "marker": frozen.record(path)}
    payload = {
        "schema_version": SUBMISSION_SCHEMA,
        "created_utc": frozen.utc_now(),
        "submission_id": submission_id,
        "phase": frozen.PHASE,
        "jobs": jobs,
        "dependency_policy": "strict afterok",
        "exception_reference": frozen.record(EXCEPTION),
        "automatic_retry_allowed": False,
        "open_count": 1,
        "pass": True,
    }
    frozen.write_json_exclusive(RESUME_SUBMITTED, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("authorize")
    sub.add_parser("validate")
    recover = sub.add_parser("recover")
    recover.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True,
                       help="Job-scoped intended array path, validated but not written")
    claim = sub.add_parser("claim-resume")
    claim.add_argument("--submission-id", required=True)
    claim.add_argument("--failed-job", required=True)
    claim.add_argument("--blocked-dispatcher", required=True)
    job = sub.add_parser("record-resume-job")
    job.add_argument("--submission-id", required=True)
    job.add_argument("--job", choices=RESUME_JOBS, required=True)
    job.add_argument("--job-id", required=True)
    job.add_argument("--dependency-job-id")
    submission = sub.add_parser("record-resume")
    submission.add_argument("--submission-id", required=True)
    args = parser.parse_args()
    if args.command == "authorize":
        result = authorize_exception()
    elif args.command == "validate":
        result = validate_exception()
    elif args.command == "recover":
        result = build_compact_truth(output_path=args.output)
    elif args.command == "smoke":
        result = build_compact_truth(output_path=args.output, smoke_only=True)
    elif args.command == "claim-resume":
        result = claim_resume(
            submission_id=args.submission_id,
            failed_job=args.failed_job,
            blocked_dispatcher=args.blocked_dispatcher,
        )
    elif args.command == "record-resume-job":
        result = record_resume_job(
            submission_id=args.submission_id,
            job=args.job,
            job_id=args.job_id,
            dependency_job_id=args.dependency_job_id,
        )
    else:
        result = record_resume_submission(submission_id=args.submission_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
