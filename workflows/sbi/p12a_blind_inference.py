#!/usr/bin/env python3
"""Truth-free P12-A blind inference without weakening the OOF exporter.

The first stage runs the frozen five-phase U-PATCH on a dedicated observed-only
ph001 field adapter and writes a seven-feature context. The second reconstructs the
frozen FMPE exactly and writes deterministic 512-draw posterior summaries. No target
array, T-web product, density truth or truth-bearing phase contract is accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    increments_to_eigenvalues,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_train_patch_recovery import torch_load
from workflows.sbi.p12_export_unet_summaries import ntilde_at_rows
from workflows.sbi.p12_prepare_base_response_dataset import sample_random_support_distance
from workflows.sbi.p12_production_contract import (
    P12A_SCHEMA,
    QUALITY_BITS,
    assert_truth_free_payload,
    deterministic_audit_subset,
    posterior_summaries,
    quality_bitmask,
)
from workflows.sbi.p12_train_base_response_fmpe import (
    sample_posterior,
    theta_to_eigenvalues,
)


FORBIDDEN_ARRAY_TOKENS = ("truth", "target", "tweb", "cweb")
REQUIRED_CONTEXT_ARRAYS = (
    "parent_node_id",
    "core_id",
    "base_prediction_eigenvalues",
    "redshift",
    "ntilde_mpc3",
    "cap",
    "shell",
    "distance_to_support_boundary_mpc",
    "support_random",
    "context",
    "audit_selected",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def reject_truth_bearing_names(names: tuple[str, ...] | list[str]) -> None:
    for name in names:
        lowered = str(name).lower()
        # TARGETID is an observed catalogue row identifier, not a supervised
        # target.  Permit only this exact spelling; broader target-bearing
        # arrays remain forbidden below.
        if lowered == "targetid":
            continue
        if any(token in lowered for token in FORBIDDEN_ARRAY_TOKENS):
            raise PermissionError(f"blind input exposes forbidden array name: {name}")


def validate_blind_checkpoint(checkpoint: dict, *, latent_channels: int = 32) -> None:
    if checkpoint.get("schema_version") != "p10-arm-a-best-v1":
        raise RuntimeError("unsupported U-PATCH checkpoint schema")
    if checkpoint.get("model") != "unet":
        raise RuntimeError("blind base context requires U-PATCH")
    if checkpoint.get("validation_phase") != "ph006":
        raise RuntimeError("production U-PATCH was not selected on ph006")
    expected = ("ph000", "ph002", "ph003", "ph004", "ph005")
    if tuple(checkpoint.get("training_phases", ())) != expected:
        raise RuntimeError("U-PATCH training phases do not match the frozen five-phase fit")
    if "ph001" in checkpoint.get("training_phases", ()):
        raise RuntimeError("blind phase appears in U-PATCH training")
    weight = checkpoint["state_dict"].get("unet.output.weight")
    if weight is None or int(weight.shape[0]) != int(latent_channels):
        raise RuntimeError("U-PATCH latent width mismatch")


def validate_observed_assignment(assignment: Any) -> None:
    names = tuple(assignment.files)
    reject_truth_bearing_names(names)
    required = {"parent_node_id", "supervised_eligible", "cap", "shell"}
    if not required.issubset(names):
        raise RuntimeError(f"observed assignment is missing {sorted(required - set(names))}")


def validate_blind_adapter_inputs(
    manifest: dict,
    *,
    adapter_root: Path,
    assignment_path: Path,
    points_path: Path,
) -> None:
    """Bind the observed-only adapter to the exact ph001 rows and field manifest."""
    if not manifest.get("pass") or "ph001" not in str(adapter_root).lower():
        raise PermissionError("blind field adapter is not an approved ph001 input")
    if not set(unet_impl.CHANNELS).issubset(manifest.get("channel_order", ())):
        raise RuntimeError("blind field adapter does not expose the frozen R0 channels")
    expected_assignment = Path(manifest.get("p4_active_assignment", ""))
    expected_points = Path(manifest.get("points", ""))
    p3_manifest = Path(manifest.get("p3_manifest", ""))
    if (
        expected_assignment.resolve() != assignment_path.resolve()
        or expected_points.resolve() != points_path.resolve()
    ):
        raise RuntimeError("blind adapter row identity differs from the supplied inputs")
    if (
        not p3_manifest.is_file()
        or sha256(p3_manifest) != manifest.get("p3_manifest_sha256")
        or sha256(assignment_path) != manifest.get("p4_active_assignment_sha256")
    ):
        raise RuntimeError("blind adapter provenance hash mismatch")
    referenced = (expected_assignment, expected_points, p3_manifest) + tuple(
        Path(record.get("field_path", ""))
        for record in manifest.get("caps", {}).values()
    )
    if any("ph001" not in str(path).lower() for path in referenced):
        raise PermissionError("blind adapter references a non-ph001 source")
    if any(
        any(token in str(path).lower() for token in FORBIDDEN_ARRAY_TOKENS)
        for path in referenced
    ):
        raise PermissionError("blind adapter references a truth-bearing source")


def validate_blind_selection_manifest(selection: dict) -> None:
    expected = ("ph000", "ph002", "ph003", "ph004", "ph005")
    if (
        selection.get("pass") is not True
        or tuple(selection.get("fit_phases", ())) != expected
        or "ph001" not in selection.get("application_phases", ())
        or "ph001" in selection.get("fit_phases", ())
        or selection.get("gates", {}).get("no_validation_or_blind_fit") is not True
    ):
        raise PermissionError("radial selection manifest is not frozen blind-safe")


def validate_blind_phase_contract(
    contract: dict,
    *,
    phase_contract_path: Path,
    assignment_path: Path,
    redshift_path: Path,
) -> None:
    """Bind observed redshifts to the sealed ph001 loader contract."""
    if (
        contract.get("schema_version") != "p10-phase-loader-contract-v1"
        or contract.get("phase") != "ph001"
        or contract.get("role") != "sealed_blind_test"
        or contract.get("pass") is not True
        or contract.get("target") is not None
        or contract.get("truth_present") is not False
        or contract.get("gates", {}).get("truth_sealed_if_blind") is not True
    ):
        raise PermissionError("ph001 loader contract is not sealed and truth-free")
    registered_assignment = Path(contract.get("inputs", {}).get("assignment", ""))
    if (
        registered_assignment.resolve() != assignment_path.resolve()
        or contract["inputs"].get("assignment_sha256") != sha256(assignment_path)
    ):
        raise RuntimeError("ph001 redshift/assignment contract identity mismatch")
    expected_redshift = phase_contract_path.parent / "parent_redshift.npy"
    if expected_redshift.resolve() != redshift_path.resolve():
        raise RuntimeError("blind redshift is not the registered parent-aligned array")
    if any(token in str(redshift_path).lower() for token in FORBIDDEN_ARRAY_TOKENS):
        raise PermissionError("blind redshift path appears truth-bearing")


def _parent_lookup(parent: np.ndarray, rows: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.int64)
    if len(np.unique(parent)) != len(parent) or np.any(parent < 0):
        raise RuntimeError("assignment parent identifiers are invalid")
    size = int(max(parent.max(initial=-1), np.max(rows, initial=-1)) + 1)
    lookup = np.full(size, -1, dtype=np.int64)
    lookup[parent] = np.arange(len(parent), dtype=np.int64)
    if np.any(rows >= len(lookup)) or np.any(lookup[rows] < 0):
        raise RuntimeError("authoritative parent lacks observed assignment row")
    return lookup[rows]


def export_blind_unet_context(
    *,
    adapter_root: Path,
    assignment_path: Path,
    points_path: Path,
    redshift_path: Path,
    phase_contract_path: Path,
    response_field_manifest_path: Path,
    selection_manifest_path: Path,
    candidate_marker_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
    base: int = 24,
    latent_channels: int = 32,
    audit_rows: int = 50_000,
    seed: int = 42,
) -> dict:
    """Export only observed ph001 covariates and U-PATCH base predictions."""
    candidate = json.loads(candidate_marker_path.read_text())
    assert_truth_free_payload(candidate)
    if candidate.get("schema_version") != P12A_SCHEMA or not candidate.get("pass"):
        raise RuntimeError("P12-A production candidate is not frozen")
    base_encoder = candidate.get("base_encoder", {})
    base_checkpoint_record = base_encoder.get("checkpoint", {})
    if (
        base_checkpoint_record.get("sha256") != sha256(checkpoint_path)
        or base_encoder.get("selected_epoch") != 20
        or base_encoder.get("response_aware_encoder") is not False
    ):
        raise RuntimeError("blind U-PATCH differs from the P12-A base encoder")
    checkpoint = torch_load(checkpoint_path, device)
    validate_blind_checkpoint(checkpoint, latent_channels=latent_channels)
    adapter = CanonicalFieldPatchAdapter(adapter_root, selection_manifest=None, rotation=None)
    if adapter.manifest.get("ph001_opened") or adapter.manifest.get("phase") not in (
        None,
        "ph001",
    ):
        raise PermissionError("blind field adapter provenance is invalid")
    validate_blind_adapter_inputs(
        adapter.manifest,
        adapter_root=adapter_root,
        assignment_path=assignment_path,
        points_path=points_path,
    )
    assignment = np.load(assignment_path, mmap_mode="r")
    validate_observed_assignment(assignment)
    points = np.load(points_path, mmap_mode="r")
    if points.ndim != 2 or points.shape[1] < 4:
        raise RuntimeError("canonical observed points are invalid")
    phase_contract = json.loads(phase_contract_path.read_text())
    validate_blind_phase_contract(
        phase_contract,
        phase_contract_path=phase_contract_path,
        assignment_path=assignment_path,
        redshift_path=redshift_path,
    )
    parent_redshift = np.load(redshift_path, mmap_mode="r")
    if parent_redshift.shape != (len(points),) or not np.all(
        np.isfinite(parent_redshift)
    ):
        raise RuntimeError("blind parent redshift array is invalid")
    response_manifest = json.loads(response_field_manifest_path.read_text())
    if (
        response_manifest.get("phase") != "ph001"
        or response_manifest.get("pass") is not True
        or response_manifest.get("ph001_opened")
        or response_manifest.get("truth_files_read")
    ):
        raise PermissionError("response manifest is not truth-free")
    selection = json.loads(selection_manifest_path.read_text())
    validate_blind_selection_manifest(selection)
    model = unet_impl.UPatch(base=base, latent_channels=latent_channels).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    parent_parts: list[np.ndarray] = []
    core_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for core_id in range(len(adapter.core_cap)):
            patch = adapter.extract(
                core_id,
                unet_impl.HALO_VOXELS,
                unet_impl.CHANNELS,
                alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
            )
            values, coordinates = unet_impl.model_inputs(
                patch, checkpoint["normalization"], device
            )
            latent = model.sample_latent(values, coordinates)
            scaled = model.head(latent)
            prediction = increments_to_eigenvalues(
                unscale_increments(scaled.cpu().numpy(), checkpoint["scaler"])
            ).astype(np.float32)
            patch_parent = np.asarray(patch.authoritative_parent_id, dtype=np.int64)
            parent_parts.append(patch_parent)
            core_parts.append(np.full(len(patch_parent), core_id, dtype=np.int64))
            prediction_parts.append(prediction)
    adapter.close()
    parent = np.concatenate(parent_parts)
    core = np.concatenate(core_parts)
    prediction = np.concatenate(prediction_parts)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("blind authoritative parents are duplicated")
    authoritative = np.asarray(assignment["supervised_eligible"], dtype=bool)
    expected = np.asarray(assignment["parent_node_id"][authoritative], dtype=np.int64)
    if not np.array_equal(np.sort(parent), np.sort(expected)):
        raise RuntimeError("blind U-PATCH parent set is not complete")
    row = _parent_lookup(np.asarray(assignment["parent_node_id"]), parent)
    cap = np.asarray(assignment["cap"][row], dtype=np.uint8)
    shell = np.asarray(assignment["shell"][row], dtype=np.int8)
    if np.any(parent >= len(points)):
        raise RuntimeError("blind parent lies outside canonical points")
    redshift = np.asarray(parent_redshift[parent], dtype=np.float32)
    if not np.array_equal(np.asarray(points[parent, 3], dtype=np.uint8), cap):
        raise RuntimeError("assignment/point cap mismatch")
    boundary, support = sample_random_support_distance(response_manifest, points, parent)
    supported = np.asarray(support, dtype=bool)
    if not np.any(supported):
        raise RuntimeError("blind response supports no authoritative galaxy")
    parent = parent[supported]
    core = core[supported]
    prediction = prediction[supported]
    redshift = redshift[supported]
    cap = cap[supported]
    shell = shell[supported]
    boundary = np.asarray(boundary[supported], dtype=np.float32)
    ntilde = ntilde_at_rows(selection, cap, redshift)
    context = np.column_stack(
        (
            prediction,
            redshift,
            np.log(np.maximum(ntilde, np.float32(1e-12))),
            cap.astype(np.float32),
            np.log1p(boundary),
        )
    ).astype(np.float32)
    if context.shape[1] != 7 or not np.all(np.isfinite(context)):
        raise RuntimeError("blind P12-A context is invalid")
    audit_index = deterministic_audit_subset(
        parent, shell, cap, boundary, maximum=audit_rows, seed=seed
    )
    audit_selected = np.zeros(len(parent), dtype=bool)
    audit_selected[audit_index] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        parent_node_id=parent,
        core_id=core,
        base_prediction_eigenvalues=prediction,
        redshift=redshift,
        ntilde_mpc3=ntilde,
        cap=cap,
        shell=shell,
        distance_to_support_boundary_mpc=boundary,
        support_random=np.ones(len(parent), dtype=bool),
        context=context,
        audit_selected=audit_selected,
    )
    manifest = {
        "schema_version": "p12a-blind-base-context-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "phase": "ph001",
        "rows": int(len(parent)),
        "unsupported_rows_omitted": int(np.count_nonzero(~supported)),
        "conditioning_features": [
            "base_lambda1",
            "base_lambda2",
            "base_lambda3",
            "redshift",
            "log_ntilde_mpc3",
            "cap_ngc",
            "log1p_random_support_boundary_distance_mpc",
        ],
        "array": str(output_path),
        "array_sha256": sha256(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "candidate": str(candidate_marker_path),
        "candidate_sha256": sha256(candidate_marker_path),
        "adapter_manifest": str(adapter_root / "adapter_manifest.json"),
        "adapter_manifest_sha256": sha256(adapter_root / "adapter_manifest.json"),
        "assignment": str(assignment_path),
        "assignment_sha256": sha256(assignment_path),
        "redshift": str(redshift_path),
        "redshift_sha256": sha256(redshift_path),
        "phase_contract": str(phase_contract_path),
        "phase_contract_sha256": sha256(phase_contract_path),
        "response_manifest": str(response_field_manifest_path),
        "response_manifest_sha256": sha256(response_field_manifest_path),
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    atomic_json(output_path.with_suffix(".json"), manifest)
    assignment.close()
    return manifest


def validate_context_archive(archive: Any) -> None:
    names = tuple(archive.files)
    reject_truth_bearing_names(names)
    missing = set(REQUIRED_CONTEXT_ARRAYS) - set(names)
    if missing:
        raise RuntimeError(f"blind context is missing arrays: {sorted(missing)}")
    lengths = {name: len(archive[name]) for name in REQUIRED_CONTEXT_ARRAYS}
    if len(set(lengths.values())) != 1:
        raise RuntimeError("blind context arrays are not row aligned")
    if not np.all(np.asarray(archive["support_random"], dtype=bool)):
        raise RuntimeError("M=0 rows are forbidden from P12-A output")
    parent = np.asarray(archive["parent_node_id"], dtype=np.int64)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("blind context parent identifiers are not unique")
    core = np.asarray(archive["core_id"], dtype=np.int64)
    if np.any(core < 0) or np.any(np.diff(core) < 0):
        raise RuntimeError("blind context core identifiers are invalid or unordered")
    context = np.asarray(archive["context"], dtype=np.float32)
    if context.shape != (len(parent), 7) or not np.all(np.isfinite(context)):
        raise RuntimeError("blind context has invalid seven-feature rows")


def reconstruct_fmpe(checkpoint_path: Path, device: str) -> tuple[Any, dict]:
    """Reconstruct the exact frozen SBI FMPE from architecture and state dictionary."""
    checkpoint = torch_load(checkpoint_path, device)
    if checkpoint.get("schema_version") != "p12a-fmpe-estimator-v1":
        raise RuntimeError("unsupported P12-A FMPE checkpoint")
    from sbi.inference import FMPE
    from sbi.neural_nets import posterior_flow_nn
    from sbi.utils import BoxUniform

    low = torch.as_tensor(checkpoint["prior_low"], dtype=torch.float32, device=device)
    high = torch.as_tensor(checkpoint["prior_high"], dtype=torch.float32, device=device)
    prior = BoxUniform(low=low, high=high)
    builder = posterior_flow_nn(
        model="mlp",
        hidden_features=int(checkpoint["hidden_features"]),
        num_layers=int(checkpoint["num_layers"]),
        z_score_theta="none",
        z_score_x="none",
    )
    estimator = builder(
        torch.zeros((2, 3), dtype=torch.float32, device=device),
        torch.zeros((2, 7), dtype=torch.float32, device=device),
    )
    estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    estimator.eval()
    inference = FMPE(prior=prior, density_estimator=builder, device=device)
    posterior = inference.build_posterior(estimator)
    # SBI's construction path may normalize/check a user prior on CPU even when
    # the estimator and posterior sampler live on CUDA.  `sample_batched()` then
    # checks CUDA candidates against CPU Uniform bounds and fails before drawing.
    # Rebind the posterior-owned prior explicitly; this is an in-place device
    # move and does not alter the frozen prior limits.
    # Move the complete posterior rather than only its prior. Otherwise the
    # ODE sampler remains on the build-time CPU device and produces CPU
    # candidates which are then checked against CUDA prior bounds.
    posterior.to(device)
    # torch.distributions exposes `support` as a lazy property. SBI's prior
    # validation can materialize and cache that Interval before BoxUniform.to()
    # rebuilds the base distribution, leaving stale CPU bounds even though
    # `base_dist.low/high` are now CUDA tensors.
    posterior.prior.__dict__.pop("support", None)
    prior_devices = {
        posterior.prior.base_dist.low.device.type,
        posterior.prior.base_dist.high.device.type,
    }
    support_constraint = posterior.prior.support.base_constraint
    support_devices = {
        support_constraint.lower_bound.device.type,
        support_constraint.upper_bound.device.type,
    }
    expected_device = torch.device(device).type
    if prior_devices != {expected_device} or support_devices != {expected_device}:
        raise RuntimeError(
            "P12-A posterior prior/support remained on "
            f"{sorted(prior_devices | support_devices)}, expected {expected_device}"
        )
    return posterior, checkpoint


def posterior_inference_shard(
    *,
    candidate_marker_path: Path,
    context_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    start: int,
    stop: int,
    draws: int,
    seed: int,
    device: str,
    sample_chunk: int,
    quality_thresholds_path: Path,
) -> dict:
    archive = np.load(context_path, mmap_mode="r")
    validate_context_archive(archive)
    if not (0 <= start < stop <= len(archive["parent_node_id"])):
        raise ValueError("invalid blind shard interval")
    quality = json.loads(quality_thresholds_path.read_text())
    if quality.get("schema_version") != "p12a-production-quality-thresholds-v1":
        raise RuntimeError("unsupported P12-A quality-threshold contract")
    if quality.get("pass") is not True or quality.get("ph001_opened"):
        raise PermissionError("P12-A quality thresholds are not frozen and blind-safe")
    candidate = json.loads(candidate_marker_path.read_text())
    assert_truth_free_payload(candidate)
    if candidate.get("schema_version") != P12A_SCHEMA or not candidate.get("pass"):
        raise RuntimeError("P12-A production candidate is not frozen")
    artifacts = candidate.get("artifacts", {})
    if (
        artifacts.get("checkpoint", {}).get("sha256") != sha256(checkpoint_path)
        or artifacts.get("quality_thresholds", {}).get("sha256")
        != sha256(quality_thresholds_path)
    ):
        raise RuntimeError("blind posterior artifacts differ from the frozen candidate")
    response_contract = quality["response_covariate"]
    if response_contract.get("name") != "log_ntilde_mpc3" or response_contract.get("context_index") != 4:
        raise RuntimeError("P12-A response covariate contract mismatch")
    response_training_range = (
        float(response_contract["training_minimum"]),
        float(response_contract["training_maximum"]),
    )
    prior_width_threshold = np.asarray(
        quality["prior_dominated_width"]["threshold_by_ordered_eigenvalue"], dtype=np.float64
    )
    boundary_contract = quality["boundary_distance"]
    posterior, checkpoint = reconstruct_fmpe(checkpoint_path, device)
    context = np.asarray(archive["context"][start:stop], dtype=np.float32)
    scaled_context = (
        (context - np.asarray(checkpoint["context_mean"], dtype=np.float32))
        / np.asarray(checkpoint["context_std"], dtype=np.float32)
    ).astype(np.float32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scaled = sample_posterior(posterior, scaled_context, draws, sample_chunk, device)
    theta = scaled * np.asarray(checkpoint["theta_std"], dtype=np.float32) + np.asarray(
        checkpoint["theta_mean"], dtype=np.float32
    )
    eigen = theta_to_eigenvalues(theta).astype(np.float32)
    summary = posterior_summaries(eigen)
    width = summary["eigenvalue_q84"] - summary["eigenvalue_q16"]
    bits = quality_bitmask(
        redshift=archive["redshift"][start:stop],
        boundary_distance_mpc_h=archive["distance_to_support_boundary_mpc"][start:stop],
        response_covariate=context[:, 4],
        posterior_width=width,
        response_training_range=response_training_range,
        prior_width_threshold=prior_width_threshold,
        boundary_r_mpc=float(boundary_contract["threshold_r_mpc"]),
        boundary_2r_mpc=float(boundary_contract["threshold_2r_mpc"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "parent_node_id": np.asarray(archive["parent_node_id"][start:stop], dtype=np.int64),
        "core_id": np.asarray(archive["core_id"][start:stop], dtype=np.int64),
        "base_prediction_eigenvalues": np.asarray(
            archive["base_prediction_eigenvalues"][start:stop], dtype=np.float32
        ),
        "redshift": np.asarray(archive["redshift"][start:stop], dtype=np.float32),
        "ntilde_mpc3": np.asarray(archive["ntilde_mpc3"][start:stop], dtype=np.float32),
        "cap": np.asarray(archive["cap"][start:stop], dtype=np.uint8),
        "shell": np.asarray(archive["shell"][start:stop], dtype=np.int8),
        "support_random": np.ones(stop - start, dtype=bool),
        "distance_to_support_boundary_mpc": np.asarray(
            archive["distance_to_support_boundary_mpc"][start:stop], dtype=np.float32
        ),
        "quality_bitmask": bits,
        **summary,
    }
    np.savez_compressed(output_path, **arrays)
    audit = np.asarray(archive["audit_selected"][start:stop], dtype=bool)
    audit_path = output_path.with_name(output_path.stem + "_audit_draws.npz")
    np.savez_compressed(
        audit_path,
        parent_node_id=arrays["parent_node_id"][audit],
        eigenvalue_draws=eigen[audit],
    )
    marker = {
        "schema_version": "p12a-blind-posterior-shard-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "start": int(start),
        "stop": int(stop),
        "rows": int(stop - start),
        "draws": int(draws),
        "seed": int(seed),
        "summary": str(output_path),
        "summary_sha256": sha256(output_path),
        "audit_draws": str(audit_path),
        "audit_draws_sha256": sha256(audit_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "context": str(context_path),
        "context_sha256": sha256(context_path),
        "quality_thresholds": str(quality_thresholds_path),
        "quality_thresholds_sha256": sha256(quality_thresholds_path),
        "candidate": str(candidate_marker_path),
        "candidate_sha256": sha256(candidate_marker_path),
        "quality_bits": QUALITY_BITS,
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    atomic_json(output_path.with_suffix(".json"), marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    context = subparsers.add_parser("context")
    context.add_argument("--adapter-root", type=Path, required=True)
    context.add_argument("--assignment", type=Path, required=True)
    context.add_argument("--points", type=Path, required=True)
    context.add_argument("--redshift", type=Path, required=True)
    context.add_argument("--phase-contract", type=Path, required=True)
    context.add_argument("--response-field-manifest", type=Path, required=True)
    context.add_argument("--selection-manifest", type=Path, required=True)
    context.add_argument("--candidate", type=Path, required=True)
    context.add_argument("--checkpoint", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--device", default="cuda")
    sample = subparsers.add_parser("sample")
    sample.add_argument("--candidate", type=Path, required=True)
    sample.add_argument("--context", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--start", type=int, required=True)
    sample.add_argument("--stop", type=int, required=True)
    sample.add_argument("--draws", type=int, default=512)
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--sample-chunk", type=int, default=2048)
    sample.add_argument("--quality-thresholds", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "context":
        result = export_blind_unet_context(
            adapter_root=args.adapter_root,
            assignment_path=args.assignment,
            points_path=args.points,
            redshift_path=args.redshift,
            phase_contract_path=args.phase_contract,
            response_field_manifest_path=args.response_field_manifest,
            selection_manifest_path=args.selection_manifest,
            candidate_marker_path=args.candidate,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device,
        )
    else:
        result = posterior_inference_shard(
            candidate_marker_path=args.candidate,
            context_path=args.context,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            start=args.start,
            stop=args.stop,
            draws=args.draws,
            seed=args.seed,
            device=args.device,
            sample_chunk=args.sample_chunk,
            quality_thresholds_path=args.quality_thresholds,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
