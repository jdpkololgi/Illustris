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
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_phase_assets import load_registry
from workflows.abacus_tweb.p10_run_tweb import (
    validate_density_input,
    validate_rank_outputs,
)
from workflows.abacus_tweb.p10_stage_particle_b import (
    phase_staging_paths,
    verify_b_tree,
)
from workflows.sbi.p12a_open_blind import (
    TRUTH_COMPLETE_FILENAME,
    build_truth_complete_marker,
    validate_open_authorization,
    validate_truth_complete,
    write_json_exclusive,
)


PHASE = "ph001"
STAGE_SCHEMA = "p12a-ph001-truth-stage-complete-v1"
STAGES = ("particle_b", "density", "tweb", "annotation", "compact")
STAGE_ARTIFACT_KEYS = {
    "particle_b": {"p10_b_stage_marker"},
    "density": {"density", "p10_density_manifest"},
    "tweb": {"p10_tweb_complete", "rank_products"},
    "annotation": {
        "truth_free_blind_parent",
        "truth_free_blind_parent_marker",
        "annotated_parent",
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
        or marker.get("post_open_refit_allowed") is not False
        or marker.get("post_open_tuning_allowed") is not False
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
    truth_files = [
        item
        for value in artifacts.values()
        for item in (value if isinstance(value, list) else [value])
        if isinstance(item, dict) and "path" in item
    ]
    if marker.get("truth_files_read") != truth_files:
        raise AuthorizedTruthError(f"{stage} truth provenance differs from its artifacts")
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
        "audit": audit,
        "truth_files_read": [
            item
            for value in artifacts.values()
            for item in (value if isinstance(value, list) else [value])
            if isinstance(item, dict) and "path" in item
        ],
        "fit_performed": False,
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


def complete_particle_b(*, truth_root: Path = TRUTH_ROOT) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    registry = load_registry(REGISTRY)
    _, particle_root, marker_path = phase_staging_paths(
        truth_root / "particle_b", PHASE
    )
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("schema_version") != "p10-b-stage-complete-v1"
        or marker.get("phase") != PHASE
        or marker.get("registry_sha256") != sha256(REGISTRY)
        or marker.get("verification", {}).get("verified") is not True
    ):
        raise AuthorizedTruthError("isolated ph001 Particle-B restore is invalid")
    tree_inventory = verify_b_tree(
        particle_root,
        phase=PHASE,
        registry=registry,
        checksums=False,
        asdf_headers=False,
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
            "particle_root": str(particle_root.resolve()),
            "field_asdf_count": tree_inventory["field_asdf_count"],
            "halo_asdf_count": tree_inventory["halo_asdf_count"],
            "payload_bytes": tree_inventory["payload_bytes"],
            "checksums": verification["checksums"],
            "asdf_headers": verification["asdf_headers"],
            "checksums_reverified_at_authorized_completion": True,
            "asdf_headers_reverified_at_authorized_completion": True,
        },
    )


def complete_density(
    *, density_path: Path, density_manifest_path: Path, truth_root: Path = TRUTH_ROOT
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    if truth_root not in density_path.resolve().parents:
        raise PermissionError("density output is outside the isolated truth root")
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
    if (
        density_manifest.get("registry_sha256") != sha256(REGISTRY)
        or density_manifest.get("target_contract") != registry["target_contract"]
        or Path(str(density_manifest.get("stage_marker", ""))).resolve()
        != p10_particle_marker
        or Path(
            str(density_manifest.get("inputs", {}).get("b_root", ""))
        ).resolve()
        != expected_b_root
    ):
        raise AuthorizedTruthError(
            "density manifest does not bind the frozen registry and isolated Particle-B stage"
        )
    report = validate_density_input(
        density_path, density_manifest_path, registry, PHASE
    )
    return write_stage_marker(
        stage="density",
        truth_root=truth_root,
        upstream=("particle_b",),
        artifacts={
            "density": record(density_path),
            "p10_density_manifest": record(density_manifest_path),
        },
        audit={**report, "target": TARGET},
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
    )


def audit_annotated_parent(path: Path, *, chunk_size: int = 1_000_000) -> dict[str, Any]:
    import fitsio

    required = ("TARGETID", "FILE_NUM", "HALO_INDEX", "BOX_INDEX")
    target_columns = ("CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3")
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
    }


def complete_annotation(
    *, annotated_parent_path: Path, truth_root: Path = TRUTH_ROOT
) -> dict[str, Any]:
    truth_root = validate_truth_root(truth_root)
    authorization_context()
    if truth_root not in annotated_parent_path.resolve().parents:
        raise PermissionError("annotated truth is outside the isolated truth root")
    _upstream_records(truth_root, ("tweb",))
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
    return write_stage_marker(
        stage="annotation",
        truth_root=truth_root,
        upstream=("tweb",),
        artifacts={
            "truth_free_blind_parent": record(BLIND_PARENT),
            "truth_free_blind_parent_marker": record(BLIND_PARENT_MARKER),
            "annotated_parent": record(annotated_parent_path),
        },
        audit={**audit, "target": TARGET, "halo_position_field": "x_com"},
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


def build_compact_truth(*, truth_root: Path = TRUTH_ROOT) -> dict[str, Any]:
    import fitsio

    truth_root = validate_truth_root(truth_root)
    authorization_context()
    annotation = validate_stage_marker(stage="annotation", truth_root=truth_root)
    output = truth_root / "compact/ph001_p12a_truth.npz"
    marker_path = stage_marker_path(truth_root, "compact")
    if marker_path.exists():
        return validate_stage_marker(stage="compact", truth_root=truth_root)
    if output.exists():
        raise AuthorizedTruthError("unregistered compact truth exists; refusing overwrite")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            parent_node_id=context_parent,
            targetid=targetid,
            eigenvalues=eigenvalues,
            cweb=cweb,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
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
    compact_path = Path(compact["artifacts"]["compact_truth"]["path"])
    if int(compact["audit"]["rows"]) != EXPECTED_CONTEXT_ROWS:
        raise AuthorizedTruthError("compact truth row count changed before terminal freeze")
    payload = build_truth_complete_marker(
        authorization_path=AUTHORIZATION,
        truth_artifacts=list(terminal_artifacts),
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
    compact = sub.add_parser("build-compact")
    _common_parser(compact)
    terminal = sub.add_parser("complete-terminal")
    _common_parser(terminal)
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
    if args.command == "complete-particle-b":
        result = complete_particle_b(truth_root=args.truth_root)
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
            annotated_parent_path=args.annotated_parent, truth_root=args.truth_root
        )
    elif args.command == "build-compact":
        result = build_compact_truth(truth_root=args.truth_root)
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
