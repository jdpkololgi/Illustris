#!/usr/bin/env python3
"""Authorized, isolated construction of the single ph001 P12-A truth package.

This is a state/provenance wrapper around the already validated P10 physics
implementation.  It does not alter the phase registry and never writes truth
under the ordinary ``p10_multiphase/ph001`` product tree.  Every public command
first validates the canonical exclusive one-open authorization.

The expensive stages remain separate Slurm jobs:

1. restore ParticleSubsample B from HPSS into the isolated truth root;
2. build the complete 2048^3 10-percent A+B TSC count grid;
3. run the registered 16-rank CACTUS R=7 Mpc/h T-web solve;
4. annotate the existing truth-free BRIGHT parent by host-halo linkage;
5. join only the exact frozen supported P12-A context rows.

Attempt outputs are immutable and job-ID-scoped.  A canonical stage marker is
created with ``O_EXCL`` only after validation.  Re-entry is idempotent only when
that marker still binds the same authorization and unchanged artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_phase_assets import expand_phase, load_registry
from workflows.abacus_tweb.p10_run_tweb import (
    validate_density_input,
    validate_rank_outputs,
)
from workflows.abacus_tweb.p10_stage_particle_b import (
    parse_checksum_manifest,
    phase_staging_paths,
    verify_b_tree,
)
from workflows.sbi.p12a_open_blind import (
    TRUTH_COMPLETE_FILENAME,
    TRUTH_PROVENANCE_SCOPE,
    build_truth_complete_marker,
    validate_open_authorization,
    validate_truth_complete,
)
from workflows.sbi.p12a_immutable_io import write_json_exclusive, write_npz_exclusive


PHASE = "ph001"
STAGE_SCHEMA = "p12a-ph001-truth-stage-complete-v1"
STAGES = ("particle_b", "density", "tweb", "annotation", "compact")
STAGE_ARTIFACT_KEYS = {
    "particle_b": {"p10_b_stage_marker"},
    "density": {"density", "p10_density_manifest", "density_source_inventory"},
    "tweb": {"p10_tweb_complete", "rank_products"},
    "annotation": {
        "truth_free_blind_parent",
        "truth_free_blind_parent_marker",
        "annotated_parent",
        "halo_source_inventory",
    },
    "compact": {
        "compact_truth",
        "frozen_context_manifest",
        "frozen_context_array",
        "p1_manifest",
        "p1_canonical_index",
    },
}
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs/p10_phase_registry_v1.json"
P10_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
BLIND_ROOT = P10_ROOT / "blind_predictions/ph001"
TRUTH_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1")
FROZEN_PREDICTIONS = BLIND_ROOT / "P12_BLIND_PREDICTIONS_FROZEN.json"
AUTHORIZATION = BLIND_ROOT / "P12_BLIND_OPEN_AUTHORIZED.json"
EVALUATION_CONTRACT = REPO_ROOT / "docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json"
BLIND_PARENT = (
    P10_ROOT
    / "ph001/catalogues/blind_parent/ph001_bgs_bright_parent_linkage.fits"
)
BLIND_PARENT_MARKER = BLIND_PARENT.with_suffix(BLIND_PARENT.suffix + ".complete.json")
P1_MANIFEST = P10_ROOT / "ph001/p1_canonical/manifest.json"
P1_INDEX = P10_ROOT / "ph001/p1_canonical/canonical_index.npz"
HALO_INFO_ROOT = Path(
    "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/"
    "AbacusSummit_base_c000_ph001/halos/z0.200/halo_info"
)
TRUTH_CHAIN_CLAIM = BLIND_ROOT / "P12A_PH001_TRUTH_CHAIN_CLAIM.json"
TRUTH_CHAIN_SUBMITTED = BLIND_ROOT / "P12A_PH001_TRUTH_CHAIN_SUBMITTED.json"
POSTOPEN_CHAIN_CLAIM = BLIND_ROOT / "P12A_PH001_POSTOPEN_CHAIN_CLAIM.json"
POSTOPEN_CHAIN_SUBMITTED = BLIND_ROOT / "P12A_PH001_POSTOPEN_CHAIN_SUBMITTED.json"
TARGET = {
    "grid_size": 2048,
    "box_size_mpc_h": 2000.0,
    "mass_assignment": "TSC",
    "particle_a_fraction": 0.03,
    "particle_b_fraction": 0.07,
    "particle_total_fraction": 0.10,
    "tidal_smoothing_mpc_h": 7.0,
    "web_threshold": 0.2,
    "eigenvalue_order": "lambda1<=lambda2<=lambda3",
    "mpi_ranks": 16,
}
EXPECTED_CONTEXT_ROWS = 4_897_905
CHAIN_STATE = {
    "truth": {
        "claim": TRUTH_CHAIN_CLAIM,
        "submitted": TRUTH_CHAIN_SUBMITTED,
        "jobs": ("particle_b", "density", "tweb", "annotation", "compact"),
    },
    "postopen": {
        "claim": POSTOPEN_CHAIN_CLAIM,
        "submitted": POSTOPEN_CHAIN_SUBMITTED,
        "jobs": ("finalize", "energy_score", "evaluate", "plot"),
    },
}
CHAIN_SUBMISSION_ROOT = BLIND_ROOT / "chain_submissions"


class AuthorizedTruthError(RuntimeError):
    """The authorized truth-build contract was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def record_matches(item: Mapping[str, Any]) -> bool:
    path = Path(str(item.get("path", ""))).resolve()
    return (
        path.is_file()
        and item.get("sha256") == sha256(path)
        and ("bytes" not in item or int(item["bytes"]) == path.stat().st_size)
    )


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Adopt an exact crash-orphaned attempt manifest, never a differing one."""

    if path.exists():
        if json.loads(path.read_text()) != dict(payload):
            raise AuthorizedTruthError(f"existing attempt manifest differs: {path}")
        return
    write_json_exclusive(path, payload)


def _validate_submission_id(submission_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", submission_id):
        raise ValueError("chain submission ID contains unsafe characters")
    return submission_id


def _chain_claim(*, kind: str, submission_id: str) -> dict[str, Any]:
    state = CHAIN_STATE[kind]
    authorization_context()
    claim = json.loads(state["claim"].read_text())
    if (
        claim.get("schema_version") != "p12a-ph001-chain-claim-v1"
        or claim.get("phase") != PHASE
        or claim.get("kind") != kind
        or claim.get("submission_id") != submission_id
        or claim.get("state") != "claimed_before_any_sbatch"
        or claim.get("expected_jobs") != list(state["jobs"])
        or claim.get("duplicate_submission_allowed") is not False
        or claim.get("pass") is not True
        or claim.get("authorization_reference") != record(AUTHORIZATION)
    ):
        raise PermissionError("chain submission does not match its exclusive claim")
    return claim


def _chain_job_marker(kind: str, submission_id: str, job: str) -> Path:
    return CHAIN_SUBMISSION_ROOT / kind / submission_id / f"{job}.json"


def _validated_chain_job(
    *,
    kind: str,
    submission_id: str,
    job: str,
    expected_index: int,
    expected_dependency: str | None,
) -> dict[str, Any]:
    state = CHAIN_STATE[kind]
    marker_path = _chain_job_marker(kind, submission_id, job)
    marker = json.loads(marker_path.read_text())
    job_id = str(marker.get("slurm_job_id", ""))
    if (
        marker.get("schema_version") != "p12a-ph001-chain-job-v1"
        or marker.get("phase") != PHASE
        or marker.get("kind") != kind
        or marker.get("submission_id") != submission_id
        or marker.get("job") != job
        or int(marker.get("job_index", -1)) != expected_index
        or not job_id.isdigit()
        or int(job_id) <= 0
        or marker.get("dependency_slurm_job_id") != expected_dependency
        or marker.get("claim_reference") != record(state["claim"])
        or marker.get("recorded_immediately_after_sbatch") is not True
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("pass") is not True
    ):
        raise PermissionError(f"invalid recorded chain job: {job}")
    return marker


def claim_chain(*, kind: str, submission_id: str) -> dict[str, Any]:
    if kind not in CHAIN_STATE:
        raise ValueError("chain kind/submission ID is invalid")
    submission_id = _validate_submission_id(submission_id)
    authorization_context()
    if kind == "postopen":
        validate_truth_complete(
            truth_complete_path=FROZEN_PREDICTIONS.parent / TRUTH_COMPLETE_FILENAME,
            authorization_path=AUTHORIZATION,
            frozen_path=FROZEN_PREDICTIONS,
            evaluation_contract_path=EVALUATION_CONTRACT,
            deep_artifacts=False,
        )
    state = CHAIN_STATE[kind]
    payload = {
        "schema_version": "p12a-ph001-chain-claim-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": PHASE,
        "kind": kind,
        "submission_id": submission_id,
        "expected_jobs": list(state["jobs"]),
        "authorization_reference": record(AUTHORIZATION),
        "state": "claimed_before_any_sbatch",
        "duplicate_submission_allowed": False,
        "manual_reconciliation_required_after_partial_submission": True,
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    write_json_exclusive(state["claim"], payload)
    return payload


def record_chain_job(
    *,
    kind: str,
    submission_id: str,
    job: str,
    job_id: str,
    dependency_job_id: str | None,
) -> dict[str, Any]:
    if kind not in CHAIN_STATE:
        raise ValueError("chain kind is invalid")
    submission_id = _validate_submission_id(submission_id)
    state = CHAIN_STATE[kind]
    _chain_claim(kind=kind, submission_id=submission_id)
    if job not in state["jobs"] or not str(job_id).isdigit() or int(job_id) <= 0:
        raise ValueError("chain job name/ID is invalid")
    index = state["jobs"].index(job)
    previous_job_id: str | None = None
    predecessor_job_ids: set[str] = set()
    for previous_index, previous_name in enumerate(state["jobs"][:index]):
        previous = _validated_chain_job(
            kind=kind,
            submission_id=submission_id,
            job=previous_name,
            expected_index=previous_index,
            expected_dependency=previous_job_id,
        )
        previous_job_id = str(previous["slurm_job_id"])
        predecessor_job_ids.add(previous_job_id)
    if dependency_job_id != previous_job_id:
        raise ValueError("chain job dependency differs from recorded predecessor")
    if str(job_id) in predecessor_job_ids:
        raise ValueError("a Slurm job ID cannot represent two chain stages")
    payload = {
        "schema_version": "p12a-ph001-chain-job-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": PHASE,
        "kind": kind,
        "submission_id": submission_id,
        "job": job,
        "job_index": int(index),
        "slurm_job_id": str(job_id),
        "dependency_slurm_job_id": dependency_job_id,
        "claim_reference": record(state["claim"]),
        "recorded_immediately_after_sbatch": True,
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    marker = _chain_job_marker(kind, submission_id, job)
    write_json_exclusive(marker, payload)
    return payload


def record_chain_submission(
    *, kind: str, submission_id: str
) -> dict[str, Any]:
    if kind not in CHAIN_STATE:
        raise ValueError("chain kind is invalid")
    state = CHAIN_STATE[kind]
    submission_id = _validate_submission_id(submission_id)
    _chain_claim(kind=kind, submission_id=submission_id)
    jobs: dict[str, dict[str, Any]] = {}
    previous: str | None = None
    observed_job_ids: set[str] = set()
    for index, name in enumerate(state["jobs"]):
        marker_path = _chain_job_marker(kind, submission_id, name)
        marker = _validated_chain_job(
            kind=kind,
            submission_id=submission_id,
            job=name,
            expected_index=index,
            expected_dependency=previous,
        )
        previous = str(marker["slurm_job_id"])
        if previous in observed_job_ids:
            raise PermissionError("chain contains a reused Slurm job ID")
        observed_job_ids.add(previous)
        jobs[name] = {"marker": record(marker_path), "slurm_job_id": previous}
    payload = {
        "schema_version": "p12a-ph001-chain-submitted-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": PHASE,
        "kind": kind,
        "submission_id": submission_id,
        "claim_reference": record(state["claim"]),
        "jobs": jobs,
        "dependency_policy": "strict afterok in listed order",
        "automatic_resubmission_allowed": False,
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    write_json_exclusive(state["submitted"], payload)
    return payload


def validate_truth_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != TRUTH_ROOT.resolve():
        raise PermissionError(f"ph001 truth must remain isolated at {TRUTH_ROOT}")
    if P10_ROOT.joinpath("ph001").resolve() in resolved.parents:
        raise PermissionError("truth root overlaps the ordinary sealed phase product tree")
    return resolved


def authorization_context(
    *,
    authorization_path: Path = AUTHORIZATION,
    frozen_path: Path = FROZEN_PREDICTIONS,
    evaluation_contract_path: Path = EVALUATION_CONTRACT,
) -> dict[str, Any]:
    return validate_open_authorization(
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )


def stage_marker_path(truth_root: Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown authorized truth stage: {stage}")
    return truth_root.resolve() / "stages" / f"{stage.upper()}_COMPLETE.json"


def _same_reference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and left.get("sha256") == right.get("sha256")
    )


def validate_stage_marker(
    *,
    stage: str,
    truth_root: Path = TRUTH_ROOT,
    authorization_path: Path = AUTHORIZATION,
    frozen_path: Path = FROZEN_PREDICTIONS,
    evaluation_contract_path: Path = EVALUATION_CONTRACT,
    deep_artifacts: bool = True,
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization = authorization_context(
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )
    path = stage_marker_path(truth_root, stage)
    if not path.is_file():
        raise FileNotFoundError(path)
    marker = json.loads(path.read_text())
    if (
        marker.get("schema_version") != STAGE_SCHEMA
        or marker.get("phase") != PHASE
        or marker.get("stage") != stage
        or marker.get("truth_root") != str(truth_root)
        or marker.get("open_count") != 1
        or marker.get("sealed_phase_opened") is not True
        or marker.get("fit_performed") is not False
        or marker.get("phase_registry_mutation_performed") is not False
        or marker.get("post_open_refit_allowed") is not False
        or marker.get("post_open_tuning_allowed") is not False
        or marker.get("truth_read_provenance_scope") != TRUTH_PROVENANCE_SCOPE
        or marker.get("pass") is not True
    ):
        raise PermissionError(f"invalid authorized truth stage marker: {stage}")
    references = (
        ("authorization_reference", record(authorization_path)),
        ("frozen_predictions_reference", authorization["frozen_predictions_reference"]),
        ("evaluation_contract_reference", authorization["evaluation_contract_reference"]),
    )
    for key, expected in references:
        if not _same_reference(marker.get(key, {}), expected):
            raise AuthorizedTruthError(f"{stage} changed its {key}")
    expected_upstream_stage = (
        None if stage == STAGES[0] else STAGES[STAGES.index(stage) - 1]
    )
    upstream_records = marker.get("upstream_stages", [])
    if expected_upstream_stage is None:
        if upstream_records != []:
            raise AuthorizedTruthError("particle_b stage must not claim an upstream stage")
    elif (
        len(upstream_records) != 1
        or Path(str(upstream_records[0].get("path", ""))).resolve()
        != stage_marker_path(truth_root, expected_upstream_stage).resolve()
    ):
        raise AuthorizedTruthError(
            f"{stage} does not bind its exact upstream stage {expected_upstream_stage}"
        )
    for upstream in upstream_records:
        if not record_matches(upstream):
            raise AuthorizedTruthError(f"{stage} upstream stage changed")
    artifacts = marker.get("artifacts", {})
    if set(artifacts) != STAGE_ARTIFACT_KEYS[stage]:
        raise AuthorizedTruthError(f"{stage} artifact inventory changed")
    if marker.get("generated_artifacts") != artifacts:
        raise AuthorizedTruthError(f"{stage} generated-artifact inventory changed")
    truth_inputs = marker.get("truth_input_manifests", [])
    if marker.get("truth_files_read") != truth_inputs:
        raise AuthorizedTruthError(f"{stage} truth provenance differs from its inputs")
    for item in truth_inputs:
        if not record_matches(item):
            raise AuthorizedTruthError(f"{stage} truth input manifest changed")
    if deep_artifacts:
        for name, item in artifacts.items():
            if isinstance(item, dict) and "path" in item and not record_matches(item):
                raise AuthorizedTruthError(f"{stage} artifact changed: {name}")
            if isinstance(item, list):
                for member in item:
                    if not record_matches(member):
                        raise AuthorizedTruthError(f"{stage} artifact changed: {name}")
        if stage == "particle_b":
            particle_root = Path(str(marker.get("audit", {}).get("particle_root", "")))
            verification = verify_b_tree(
                particle_root,
                phase=PHASE,
                registry=load_registry(REGISTRY),
                checksums=True,
                asdf_headers=True,
            )
            if (
                verification.get("verified") is not True
                or verification.get("field_asdf_count")
                != marker["audit"].get("field_asdf_count")
                or verification.get("halo_asdf_count")
                != marker["audit"].get("halo_asdf_count")
                or verification.get("payload_bytes")
                != marker["audit"].get("payload_bytes")
            ):
                raise AuthorizedTruthError("particle_b staged payload changed")
    return marker


def stage_status(**kwargs: Any) -> bool:
    try:
        validate_stage_marker(**kwargs)
        return True
    except FileNotFoundError:
        # Authorization must still be valid even when the stage has not begun.
        authorization_context(
            authorization_path=kwargs.get("authorization_path", AUTHORIZATION),
            frozen_path=kwargs.get("frozen_path", FROZEN_PREDICTIONS),
            evaluation_contract_path=kwargs.get(
                "evaluation_contract_path", EVALUATION_CONTRACT
            ),
        )
        return False


def guard_stage(*, stage: str, truth_root: Path = TRUTH_ROOT) -> dict[str, Any]:
    """Validate authorization and the immediately required upstream stage."""

    truth_root = validate_truth_root(truth_root)
    authorization = authorization_context()
    index = STAGES.index(stage)
    upstream = None
    if index:
        upstream = STAGES[index - 1]
        validate_stage_marker(
            stage=upstream, truth_root=truth_root, deep_artifacts=True
        )
    return {
        "authorized": True,
        "stage": stage,
        "upstream_stage": upstream,
        "open_count": int(authorization["open_count"]),
    }


def _upstream_records(
    truth_root: Path,
    stages: tuple[str, ...],
    *,
    authorization_path: Path = AUTHORIZATION,
    frozen_path: Path = FROZEN_PREDICTIONS,
    evaluation_contract_path: Path = EVALUATION_CONTRACT,
) -> list[dict[str, Any]]:
    result = []
    for stage in stages:
        validate_stage_marker(
            stage=stage,
            truth_root=truth_root,
            authorization_path=authorization_path,
            frozen_path=frozen_path,
            evaluation_contract_path=evaluation_contract_path,
            deep_artifacts=False,
        )
        result.append(record(stage_marker_path(truth_root, stage)))
    return result


def write_stage_marker(
    *,
    stage: str,
    artifacts: dict[str, Any],
    audit: dict[str, Any],
    truth_input_manifests: tuple[Path, ...] = (),
    upstream: tuple[str, ...] = (),
    truth_root: Path = TRUTH_ROOT,
    authorization_path: Path = AUTHORIZATION,
    frozen_path: Path = FROZEN_PREDICTIONS,
    evaluation_contract_path: Path = EVALUATION_CONTRACT,
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    expected_upstream = (
        () if stage == STAGES[0] else (STAGES[STAGES.index(stage) - 1],)
    )
    if upstream != expected_upstream:
        raise AuthorizedTruthError(
            f"{stage} completion requires upstream={expected_upstream}, got {upstream}"
        )
    authorization = authorization_context(
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )
    path = stage_marker_path(truth_root, stage)
    if path.exists():
        existing = validate_stage_marker(
            stage=stage,
            truth_root=truth_root,
            authorization_path=authorization_path,
            frozen_path=frozen_path,
            evaluation_contract_path=evaluation_contract_path,
        )
        if existing.get("artifacts") != artifacts:
            raise AuthorizedTruthError(
                f"{stage} completion already binds different artifacts"
            )
        return existing
    marker = {
        "schema_version": STAGE_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": PHASE,
        "stage": stage,
        "truth_root": str(truth_root),
        "authorization_reference": record(authorization_path),
        "frozen_predictions_reference": authorization["frozen_predictions_reference"],
        "evaluation_contract_reference": authorization["evaluation_contract_reference"],
        "upstream_stages": _upstream_records(
            truth_root,
            upstream,
            authorization_path=authorization_path,
            frozen_path=frozen_path,
            evaluation_contract_path=evaluation_contract_path,
        ),
        "artifacts": artifacts,
        "generated_artifacts": artifacts,
        "audit": audit,
        "truth_input_manifests": [record(path) for path in truth_input_manifests],
        "truth_files_read": [record(path) for path in truth_input_manifests],
        "truth_read_provenance_scope": TRUTH_PROVENANCE_SCOPE,
        "fit_performed": False,
        "phase_registry_mutation_performed": False,
        "open_count": 1,
        "sealed_phase_opened": True,
        "post_open_refit_allowed": False,
        "post_open_tuning_allowed": False,
        "pass": True,
    }
    write_json_exclusive(path, marker)
    return validate_stage_marker(
        stage=stage,
        truth_root=truth_root,
        authorization_path=authorization_path,
        frozen_path=frozen_path,
        evaluation_contract_path=evaluation_contract_path,
    )


def complete_particle_b(
    *, staging_root: Path, truth_root: Path = TRUTH_ROOT
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    registry = load_registry(REGISTRY)
    staging_root = staging_root.resolve()
    attempts_root = (truth_root / "attempts/particle_b").resolve()
    if attempts_root not in staging_root.parents or staging_root.name != "staging":
        raise PermissionError("Particle-B staging root is not a job-scoped authorized attempt")
    _, particle_root, marker_path = phase_staging_paths(
        staging_root, PHASE
    )
    marker = json.loads(marker_path.read_text())
    registered_source = expand_phase(registry, PHASE)["particle_b"]
    if (
        marker.get("schema_version") != "p10-b-stage-complete-v1"
        or marker.get("phase") != PHASE
        or marker.get("registry_sha256") != sha256(REGISTRY)
        or marker.get("verification", {}).get("verified") is not True
        or marker.get("source") != registered_source
    ):
        raise AuthorizedTruthError("isolated ph001 Particle-B restore is invalid")
    tree_inventory = verify_b_tree(
        particle_root,
        phase=PHASE,
        registry=registry,
        checksums=True,
        asdf_headers=True,
    )
    if tree_inventory.get("verified") is not True:
        raise AuthorizedTruthError("isolated ph001 Particle-B tree failed verification")
    verification = marker["verification"]
    if (
        verification.get("verified") is not True
        or verification.get("checksums") is None
        or verification.get("asdf_headers") is None
    ):
        raise AuthorizedTruthError("P10 marker lacks full checksum/header verification")
    return write_stage_marker(
        stage="particle_b",
        truth_root=truth_root,
        artifacts={"p10_b_stage_marker": record(marker_path)},
        audit={
            "staging_root": str(staging_root),
            "particle_root": str(particle_root.resolve()),
            "field_asdf_count": tree_inventory["field_asdf_count"],
            "halo_asdf_count": tree_inventory["halo_asdf_count"],
            "payload_bytes": tree_inventory["payload_bytes"],
            "checksums": tree_inventory["checksums"],
            "asdf_headers": tree_inventory["asdf_headers"],
            "checksums_reverified_at_authorized_completion": True,
            "asdf_headers_reverified_at_authorized_completion": True,
            "registered_hpss_source": registered_source,
        },
        truth_input_manifests=(REGISTRY,),
    )


def complete_density(
    *, density_path: Path, density_manifest_path: Path, truth_root: Path = TRUTH_ROOT
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    if (
        truth_root not in density_path.resolve().parents
        or truth_root not in density_manifest_path.resolve().parents
    ):
        raise PermissionError("density output/manifest is outside the isolated truth root")
    _upstream_records(truth_root, ("particle_b",))
    registry = load_registry(REGISTRY)
    particle_stage = validate_stage_marker(
        stage="particle_b", truth_root=truth_root, deep_artifacts=False
    )
    p10_particle_marker = Path(
        particle_stage["artifacts"]["p10_b_stage_marker"]["path"]
    ).resolve()
    density_manifest = json.loads(density_manifest_path.read_text())
    expected_b_root = Path(particle_stage["audit"]["particle_root"]).resolve()
    expected_a_root = Path(
        expand_phase(registry, PHASE)["assets"]["snapshot_root"]
    ).resolve()
    if (
        density_manifest.get("registry_sha256") != sha256(REGISTRY)
        or density_manifest.get("target_contract") != registry["target_contract"]
        or Path(str(density_manifest.get("stage_marker", ""))).resolve()
        != p10_particle_marker
        or Path(
            str(density_manifest.get("inputs", {}).get("b_root", ""))
        ).resolve()
        != expected_b_root
        or Path(
            str(density_manifest.get("inputs", {}).get("a_root", ""))
        ).resolve()
        != expected_a_root
    ):
        raise AuthorizedTruthError(
            "density manifest does not bind the frozen registry and isolated Particle-B stage"
        )
    report = validate_density_input(
        density_path, density_manifest_path, registry, PHASE
    )
    directories = density_manifest.get("inputs", {}).get("directories", {})
    expected_directories = {
        "field_A": expected_a_root / "field_rv_A",
        "halo_A": expected_a_root / "halo_rv_A",
        "field_B": expected_b_root / "field_rv_B",
        "halo_B": expected_b_root / "halo_rv_B",
    }
    source_groups: dict[str, Any] = {}
    all_paths: list[Path] = []
    for name, expected_directory in expected_directories.items():
        item = directories.get(name, {})
        directory = Path(str(item.get("directory", ""))).resolve()
        checksum_path = Path(str(item.get("checksum_manifest", ""))).resolve()
        files = [Path(value).resolve() for value in item.get("files", [])]
        if (
            directory != expected_directory.resolve()
            or checksum_path.parent != directory
            or len(set(files)) != len(files)
            or any(path.parent != directory for path in files)
        ):
            raise AuthorizedTruthError(f"density source directory changed: {name}")
        checksums = parse_checksum_manifest(checksum_path)
        if set(checksums) != {path.name for path in files}:
            raise AuthorizedTruthError(f"density checksum inventory differs: {name}")
        records = []
        for path in files:
            crc, size = checksums[path.name]
            if not path.is_file() or path.stat().st_size != size:
                raise AuthorizedTruthError(f"density source size differs: {path}")
            records.append(
                {"path": str(path), "bytes": int(size), "posix_cksum_crc": int(crc)}
            )
        source_groups[name] = {
            "directory": str(directory),
            "checksum_manifest": record(checksum_path),
            "files": records,
        }
        all_paths.extend(files)
    processed = [
        Path(value).resolve()
        for value in density_manifest.get("build", {}).get("processed_files", [])
    ]
    if processed != all_paths:
        raise AuthorizedTruthError("density build did not read the exact registered A+B files")
    source_inventory_path = density_manifest_path.with_suffix(".sources.json")
    source_inventory = {
        "schema_version": "p12a-ph001-density-source-inventory-v1",
        "phase": PHASE,
        "registry": record(REGISTRY),
        "particle_a_root": str(expected_a_root),
        "particle_b_root": str(expected_b_root),
        "groups": source_groups,
        "unique_source_files_read": len(set(all_paths)),
        "provenance_hash_basis": "registered POSIX-cksum manifests plus exact byte sizes",
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    write_or_validate_json(source_inventory_path, source_inventory)
    return write_stage_marker(
        stage="density",
        truth_root=truth_root,
        upstream=("particle_b",),
        artifacts={
            "density": record(density_path),
            "p10_density_manifest": record(density_manifest_path),
            "density_source_inventory": record(source_inventory_path),
        },
        audit={**report, "target": TARGET},
        truth_input_manifests=(source_inventory_path, p10_particle_marker),
    )


def complete_tweb(*, tweb_dir: Path, truth_root: Path = TRUTH_ROOT) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    tweb_dir = tweb_dir.resolve()
    if truth_root not in tweb_dir.parents:
        raise PermissionError("T-web output is outside the isolated truth root")
    density = validate_stage_marker(stage="density", truth_root=truth_root)
    complete_path = tweb_dir / "TWEB_COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    density_record = density["artifacts"]["density"]
    density_manifest_record = density["artifacts"]["p10_density_manifest"]
    if (
        complete.get("schema_version") != "p10-tweb-complete-v1"
        or complete.get("phase") != PHASE
        or complete.get("registry_sha256") != sha256(REGISTRY)
        or complete.get("target_contract") != load_registry(REGISTRY)["target_contract"]
        or int(complete.get("mpi_size", -1)) != TARGET["mpi_ranks"]
        or Path(complete.get("density", {}).get("path", "")).resolve()
        != Path(density_record["path"]).resolve()
        or complete.get("density", {}).get("manifest_sha256")
        != density_manifest_record["sha256"]
    ):
        raise AuthorizedTruthError("P10 T-web completion does not bind authorized density")
    report = validate_rank_outputs(
        tweb_dir,
        expected_ranks=TARGET["mpi_ranks"],
        ngrid=TARGET["grid_size"],
        boxsize=TARGET["box_size_mpc_h"],
        threshold=TARGET["web_threshold"],
        rsmooth=TARGET["tidal_smoothing_mpc_h"],
    )
    rank_records = [record(Path(item["path"])) for item in report["records"]]
    return write_stage_marker(
        stage="tweb",
        truth_root=truth_root,
        upstream=("density",),
        artifacts={
            "p10_tweb_complete": record(complete_path),
            "rank_products": rank_records,
        },
        audit={
            "tweb_dir": str(tweb_dir),
            "rank_count": report["rank_count"],
            "x_coverage": report["x_coverage"],
            "total_bytes": report["total_bytes"],
            "target": TARGET,
        },
        truth_input_manifests=(
            Path(density_record["path"]),
            Path(density_manifest_record["path"]),
        ),
    )


def audit_annotated_parent(path: Path, *, chunk_size: int = 1_000_000) -> dict[str, Any]:
    import fitsio

    required = ("TARGETID", "FILE_NUM", "HALO_INDEX", "BOX_INDEX")
    target_columns = ("CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3")
    file_numbers: set[int] = set()
    with fitsio.FITS(str(BLIND_PARENT)) as source, fitsio.FITS(str(path)) as annotated:
        source_hdu, annotated_hdu = source[1], annotated[1]
        rows = int(source_hdu.get_nrows())
        if int(annotated_hdu.get_nrows()) != rows:
            raise AuthorizedTruthError("annotated parent row count changed")
        names = set(annotated_hdu.get_colnames())
        if not set((*required, *target_columns)).issubset(names):
            raise AuthorizedTruthError("annotated parent lacks linkage or T-web columns")
        count = np.zeros(4, dtype=np.int64)
        total = np.zeros(3, dtype=np.float64)
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            left = source_hdu[start:stop][list(required)]
            right = annotated_hdu[start:stop][list((*required, *target_columns))]
            for name in required:
                if not np.array_equal(left[name], right[name]):
                    raise AuthorizedTruthError(f"annotated parent linkage changed: {name}")
            file_numbers.update(int(value) for value in np.unique(left["FILE_NUM"]) if value >= 0)
            eigen = np.column_stack(
                (right["LAMBDA1"], right["LAMBDA2"], right["LAMBDA3"])
            ).astype(np.float64)
            if not np.all(np.isfinite(eigen)) or np.any(np.diff(eigen, axis=1) < 0.0):
                raise AuthorizedTruthError("annotated parent eigenvalues are invalid")
            expected_class = np.sum(eigen > TARGET["web_threshold"], axis=1)
            if not np.array_equal(expected_class.astype(np.uint8), right["CWEB"]):
                raise AuthorizedTruthError("annotated parent CWEB threshold is inconsistent")
            count += np.bincount(expected_class, minlength=4)
            total += eigen.sum(axis=0)
    return {
        "rows": rows,
        "finite_ordered_eigenvalues": True,
        "linkage_identity_exact": True,
        "web_class_counts_void_sheet_filament_knot": count.tolist(),
        "mean_eigenvalues": (total / rows).tolist(),
        "halo_file_numbers_read": sorted(file_numbers),
    }


def complete_annotation(
    *,
    annotated_parent_path: Path,
    halo_info_root: Path = HALO_INFO_ROOT,
    truth_root: Path = TRUTH_ROOT,
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    if truth_root not in annotated_parent_path.resolve().parents:
        raise PermissionError("annotated truth is outside the isolated truth root")
    _upstream_records(truth_root, ("tweb",))
    registry = load_registry(REGISTRY)
    expected_halo_root = (
        Path(expand_phase(registry, PHASE)["assets"]["snapshot_root"]) / "halo_info"
    ).resolve()
    halo_info_root = halo_info_root.resolve()
    if halo_info_root != expected_halo_root:
        raise PermissionError("annotation halo-info root differs from the frozen registry")
    parent_marker = json.loads(BLIND_PARENT_MARKER.read_text())
    if (
        parent_marker.get("phase") != PHASE
        or parent_marker.get("target_truth_present") is not False
        or Path(str(parent_marker.get("output", {}).get("path", ""))).resolve()
        != BLIND_PARENT.resolve()
        or parent_marker.get("output", {}).get("sha256") != sha256(BLIND_PARENT)
    ):
        raise AuthorizedTruthError("truth-free blind parent provenance changed")
    audit = audit_annotated_parent(annotated_parent_path)
    source_files = [
        halo_info_root / f"halo_info_{value:03d}.asdf"
        for value in audit["halo_file_numbers_read"]
    ]
    if not source_files or not all(path.is_file() for path in source_files):
        raise AuthorizedTruthError("annotation did not bind every halo_info source")
    checksum_path = halo_info_root / "checksums.crc32"
    source_records = []
    checksum_record = None
    if checksum_path.is_file():
        checksum_record = record(checksum_path)
        checksums = parse_checksum_manifest(checksum_path)
        for path in source_files:
            if path.name not in checksums:
                raise AuthorizedTruthError(f"halo checksum manifest lacks {path.name}")
            crc, size = checksums[path.name]
            if path.stat().st_size != size:
                raise AuthorizedTruthError(f"halo source size differs: {path}")
            source_records.append(
                {"path": str(path), "bytes": int(size), "posix_cksum_crc": int(crc)}
            )
        hash_basis = "registered POSIX-cksum manifest plus exact byte sizes"
    else:
        source_records = [record(path) for path in source_files]
        hash_basis = "direct SHA256 because no checksum manifest was present"
    source_inventory_path = annotated_parent_path.with_suffix(".halo_sources.json")
    source_inventory = {
        "schema_version": "p12a-ph001-halo-source-inventory-v1",
        "phase": PHASE,
        "registry": record(REGISTRY),
        "halo_info_root": str(halo_info_root),
        "halo_files_read": source_records,
        "checksum_manifest": checksum_record,
        "provenance_hash_basis": hash_basis,
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": True,
    }
    write_or_validate_json(source_inventory_path, source_inventory)
    tweb_stage = validate_stage_marker(
        stage="tweb", truth_root=truth_root, deep_artifacts=False
    )
    return write_stage_marker(
        stage="annotation",
        truth_root=truth_root,
        upstream=("tweb",),
        artifacts={
            "truth_free_blind_parent": record(BLIND_PARENT),
            "truth_free_blind_parent_marker": record(BLIND_PARENT_MARKER),
            "annotated_parent": record(annotated_parent_path),
            "halo_source_inventory": record(source_inventory_path),
        },
        audit={**audit, "target": TARGET, "halo_position_field": "x_com"},
        truth_input_manifests=(
            source_inventory_path,
            BLIND_PARENT,
            Path(tweb_stage["artifacts"]["p10_tweb_complete"]["path"]),
        ),
    )


def join_by_parent(
    *,
    context_parent: np.ndarray,
    canonical_parent: np.ndarray,
    canonical_targetid: np.ndarray,
    annotated_targetid: np.ndarray,
    annotated_eigenvalues: np.ndarray,
    annotated_cweb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure exact identity join used by the production compact-truth stage."""

    context_parent = np.asarray(context_parent, dtype=np.int64)
    canonical_parent = np.asarray(canonical_parent, dtype=np.int64)
    canonical_targetid = np.asarray(canonical_targetid, dtype=np.int64)
    if len(np.unique(context_parent)) != len(context_parent):
        raise AuthorizedTruthError("frozen context parent IDs are not unique")
    if not np.array_equal(canonical_parent, np.arange(len(canonical_parent), dtype=np.int64)):
        raise AuthorizedTruthError("P1 canonical parent IDs are not identity aligned")
    if np.any(context_parent < 0) or np.any(context_parent >= len(canonical_parent)):
        raise AuthorizedTruthError("frozen context parent lies outside P1")
    targetid = canonical_targetid[context_parent]
    annotated_targetid = np.asarray(annotated_targetid, dtype=np.int64)
    if not np.array_equal(
        annotated_targetid, np.arange(1, len(annotated_targetid) + 1, dtype=np.int64)
    ):
        raise AuthorizedTruthError("annotated parent TARGETIDs are not sequential")
    row = targetid - 1
    if np.any(row < 0) or np.any(row >= len(annotated_targetid)):
        raise AuthorizedTruthError("P1 TARGETID lies outside annotated parent")
    if not np.array_equal(annotated_targetid[row], targetid):
        raise AuthorizedTruthError("compact truth TARGETID join is not exact")
    eigenvalues = np.asarray(annotated_eigenvalues, dtype=np.float32)[row]
    cweb = np.asarray(annotated_cweb, dtype=np.uint8)[row]
    if (
        eigenvalues.shape != (len(context_parent), 3)
        or not np.all(np.isfinite(eigenvalues))
        or np.any(np.diff(eigenvalues, axis=1) < 0.0)
        or not np.array_equal(
            cweb, np.sum(eigenvalues > TARGET["web_threshold"], axis=1).astype(np.uint8)
        )
    ):
        raise AuthorizedTruthError("joined compact truth fails physical closure")
    return targetid, eigenvalues, cweb


def _frozen_context() -> tuple[dict[str, Any], Path, Path]:
    frozen = json.loads(FROZEN_PREDICTIONS.read_text())
    for item in frozen.get("prediction_manifests", []):
        path = Path(item["path"])
        if item.get("sha256") != sha256(path):
            raise AuthorizedTruthError("frozen prediction manifest changed")
        payload = json.loads(path.read_text())
        if payload.get("schema_version") == "p12a-blind-base-context-v1":
            array = Path(payload["array"])
            if payload.get("array_sha256") != sha256(array):
                raise AuthorizedTruthError("frozen P12-A context array changed")
            return payload, path, array
    raise AuthorizedTruthError("frozen marker lacks the P12-A context")


def build_compact_truth(
    *, output_path: Path, truth_root: Path = TRUTH_ROOT
) -> dict[str, Any]:
    import fitsio

    truth_root = validate_truth_root(truth_root)
    authorization_context()
    annotation = validate_stage_marker(stage="annotation", truth_root=truth_root)
    output = output_path.resolve()
    attempts_root = (truth_root / "attempts/compact").resolve()
    if attempts_root not in output.parents:
        raise PermissionError("compact truth output is not a job-scoped authorized attempt")
    marker_path = stage_marker_path(truth_root, "compact")
    if marker_path.exists():
        return validate_stage_marker(stage="compact", truth_root=truth_root)
    context_marker, context_manifest_path, context_path = _frozen_context()
    if int(context_marker.get("rows", -1)) != EXPECTED_CONTEXT_ROWS:
        raise AuthorizedTruthError(
            "frozen P12-A context does not have the registered supported-row count"
        )
    p1 = json.loads(P1_MANIFEST.read_text())
    if (
        p1.get("phase") != PHASE
        or p1.get("target_truth_present") is not False
        or p1.get("index_sha256") != sha256(P1_INDEX)
    ):
        raise AuthorizedTruthError("truth-free P1 identity contract changed")
    with np.load(context_path, mmap_mode="r") as context, np.load(
        P1_INDEX, mmap_mode="r"
    ) as canonical:
        context_parent = np.asarray(context["parent_node_id"], dtype=np.int64)
        canonical_parent = np.asarray(canonical["parent_node_id"], dtype=np.int64)
        canonical_targetid = np.asarray(canonical["targetid"], dtype=np.int64)
    annotated_path = Path(annotation["artifacts"]["annotated_parent"]["path"])
    table = fitsio.read(
        str(annotated_path), columns=["TARGETID", "CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3"]
    )
    annotated_eigenvalues = np.column_stack(
        (table["LAMBDA1"], table["LAMBDA2"], table["LAMBDA3"])
    )
    targetid, eigenvalues, cweb = join_by_parent(
        context_parent=context_parent,
        canonical_parent=canonical_parent,
        canonical_targetid=canonical_targetid,
        annotated_targetid=table["TARGETID"],
        annotated_eigenvalues=annotated_eigenvalues,
        annotated_cweb=table["CWEB"],
    )
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
            raise AuthorizedTruthError("orphan compact attempt differs from exact recomputation")
    else:
        write_npz_exclusive(
            output,
            compressed=False,
            parent_node_id=context_parent,
            targetid=targetid,
            eigenvalues=eigenvalues,
            cweb=cweb,
        )
    artifacts = {
        "compact_truth": record(output),
        "frozen_context_manifest": record(context_manifest_path),
        "frozen_context_array": record(context_path),
        "p1_manifest": record(P1_MANIFEST),
        "p1_canonical_index": record(P1_INDEX),
    }
    return write_stage_marker(
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
            "mean_eigenvalues": eigenvalues.mean(axis=0, dtype=np.float64).tolist(),
            "web_class_counts_void_sheet_filament_knot": np.bincount(
                cweb, minlength=4
            ).tolist(),
        },
        truth_input_manifests=(context_path, P1_INDEX, annotated_path),
    )


def complete_terminal_truth(*, truth_root: Path = TRUTH_ROOT) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    compact = validate_stage_marker(stage="compact", truth_root=truth_root)
    terminal = FROZEN_PREDICTIONS.parent / TRUTH_COMPLETE_FILENAME
    if terminal.exists():
        return validate_truth_complete(
            truth_complete_path=terminal,
            authorization_path=AUTHORIZATION,
            frozen_path=FROZEN_PREDICTIONS,
            evaluation_contract_path=EVALUATION_CONTRACT,
        )
    stage_paths: list[Path] = []
    terminal_artifacts: dict[Path, None] = {}
    terminal_inputs: dict[Path, None] = {}
    for stage in STAGES:
        stage_payload = validate_stage_marker(
            stage=stage, truth_root=truth_root, deep_artifacts=True
        )
        stage_path = stage_marker_path(truth_root, stage)
        stage_paths.append(stage_path)
        terminal_artifacts[stage_path.resolve()] = None
        for value in stage_payload["artifacts"].values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict) and "path" in item:
                    terminal_artifacts[Path(item["path"]).resolve()] = None
        for item in stage_payload.get("truth_input_manifests", []):
            terminal_inputs[Path(item["path"]).resolve()] = None
    compact_path = Path(compact["artifacts"]["compact_truth"]["path"])
    if int(compact["audit"]["rows"]) != EXPECTED_CONTEXT_ROWS:
        raise AuthorizedTruthError("compact truth row count changed before terminal freeze")
    payload = build_truth_complete_marker(
        authorization_path=AUTHORIZATION,
        truth_artifacts=list(terminal_artifacts),
        truth_input_manifests=list(terminal_inputs),
        truth_array=compact_path,
        rows=int(compact["audit"]["rows"]),
    )
    payload.update(
        {
            "truth_root": str(truth_root),
            "physics_contract": TARGET,
            "stage_manifests": [record(path) for path in stage_paths],
            "compact_stage_reference": record(stage_marker_path(truth_root, "compact")),
        }
    )
    write_json_exclusive(terminal, payload)
    return validate_truth_complete(
        truth_complete_path=terminal,
        authorization_path=AUTHORIZATION,
        frozen_path=FROZEN_PREDICTIONS,
        evaluation_contract_path=EVALUATION_CONTRACT,
    )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--truth-root", type=Path, default=TRUTH_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    guard = sub.add_parser("guard")
    _common_parser(guard)
    guard.add_argument("--stage", choices=STAGES, required=True)
    status = sub.add_parser("status")
    _common_parser(status)
    status.add_argument("--stage", choices=STAGES, required=True)
    artifact = sub.add_parser("artifact")
    _common_parser(artifact)
    artifact.add_argument("--stage", choices=STAGES, required=True)
    artifact.add_argument("--name", required=True)
    particle = sub.add_parser("complete-particle-b")
    _common_parser(particle)
    particle.add_argument("--staging-root", type=Path, required=True)
    density = sub.add_parser("complete-density")
    _common_parser(density)
    density.add_argument("--density", type=Path, required=True)
    density.add_argument("--manifest", type=Path, required=True)
    tweb = sub.add_parser("complete-tweb")
    _common_parser(tweb)
    tweb.add_argument("--tweb-dir", type=Path, required=True)
    annotation = sub.add_parser("complete-annotation")
    _common_parser(annotation)
    annotation.add_argument("--annotated-parent", type=Path, required=True)
    annotation.add_argument("--halo-info-root", type=Path, default=HALO_INFO_ROOT)
    compact = sub.add_parser("build-compact")
    _common_parser(compact)
    compact.add_argument("--output", type=Path, required=True)
    terminal = sub.add_parser("complete-terminal")
    _common_parser(terminal)
    chain_claim = sub.add_parser("claim-chain")
    chain_claim.add_argument("--kind", choices=tuple(CHAIN_STATE), required=True)
    chain_claim.add_argument("--submission-id", required=True)
    chain_record = sub.add_parser("record-chain")
    chain_record.add_argument("--kind", choices=tuple(CHAIN_STATE), required=True)
    chain_record.add_argument("--submission-id", required=True)
    chain_job = sub.add_parser("record-chain-job")
    chain_job.add_argument("--kind", choices=tuple(CHAIN_STATE), required=True)
    chain_job.add_argument("--submission-id", required=True)
    chain_job.add_argument("--job", required=True)
    chain_job.add_argument("--job-id", required=True)
    chain_job.add_argument("--dependency-job-id")
    args = parser.parse_args()

    if args.command == "guard":
        print(json.dumps(guard_stage(stage=args.stage, truth_root=args.truth_root)))
        return 0
    if args.command == "status":
        if stage_status(stage=args.stage, truth_root=args.truth_root):
            print(json.dumps({"stage": args.stage, "complete": True}))
            return 0
        print(json.dumps({"stage": args.stage, "complete": False}))
        return 3
    if args.command == "artifact":
        marker = validate_stage_marker(
            stage=args.stage, truth_root=args.truth_root, deep_artifacts=False
        )
        item = marker["artifacts"].get(args.name)
        if not isinstance(item, dict) or "path" not in item:
            raise KeyError(f"stage {args.stage} has no scalar artifact {args.name}")
        print(item["path"])
        return 0
    if args.command == "claim-chain":
        print(json.dumps(claim_chain(kind=args.kind, submission_id=args.submission_id)))
        return 0
    if args.command == "record-chain":
        print(
            json.dumps(
                record_chain_submission(
                    kind=args.kind, submission_id=args.submission_id
                )
            )
        )
        return 0
    if args.command == "record-chain-job":
        print(
            json.dumps(
                record_chain_job(
                    kind=args.kind,
                    submission_id=args.submission_id,
                    job=args.job,
                    job_id=args.job_id,
                    dependency_job_id=args.dependency_job_id,
                )
            )
        )
        return 0
    if args.command == "complete-particle-b":
        result = complete_particle_b(
            staging_root=args.staging_root, truth_root=args.truth_root
        )
    elif args.command == "complete-density":
        result = complete_density(
            density_path=args.density,
            density_manifest_path=args.manifest,
            truth_root=args.truth_root,
        )
    elif args.command == "complete-tweb":
        result = complete_tweb(tweb_dir=args.tweb_dir, truth_root=args.truth_root)
    elif args.command == "complete-annotation":
        result = complete_annotation(
            annotated_parent_path=args.annotated_parent,
            halo_info_root=args.halo_info_root,
            truth_root=args.truth_root,
        )
    elif args.command == "build-compact":
        result = build_compact_truth(
            output_path=args.output, truth_root=args.truth_root
        )
    elif args.command == "complete-terminal":
        result = complete_terminal_truth(truth_root=args.truth_root)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorizedTruthError, OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        raise SystemExit(2) from error
