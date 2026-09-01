#!/usr/bin/env python3
"""Run the bounded P11 paired-degradation U-PATCH JEPA comparison.

The deployable student always consumes the canonical ``V_final`` BRIGHT field.
For the JEPA arm only, an EMA teacher consumes the phase/core-aligned ``V_dense``
BRIGHT field.  The teacher is stop-gradient and has no tidal target head.  Every
arm uses the same P11 phase-balanced patch order, deterministic supported-core
block masks, target weights, and optimizer-update budget.  ``ph001`` is never
constructed or opened.

The command is exactly resumable at a patch cursor.  Exit code 75 means that an
atomic checkpoint was written and another interactive allocation should resume
the run.  Periodic fixed-ph006 latent exports make representation alignment and
collapse inspectable without using those diagnostics as training targets.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb import p11_factorial_training as p11_impl
from workflows.abacus_tweb.p11_jepa_latent_diagnostics import (
    DEFAULT_THRESHOLDS,
    GATE_VERSION,
    REGISTERED_TRAJECTORY_STEPS,
    load_latent_snapshot,
    save_latent_snapshot,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    acquire_run_lock,
    atomic_json,
    evaluate_complete_phase,
    increments_to_eigenvalues,
    sha256,
    unscale_increments,
)
from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    append_jsonl,
    improved,
    reconcile_loss_trace,
    rewrite_jsonl,
    should_stop,
)
from workflows.abacus_tweb.p8_train_patch_recovery import (
    atomic_torch_save,
    git_revision,
    torch_load,
)
from workflows.abacus_tweb.p10_train_arm_a import prepare_phase_runtime
from workflows.abacus_tweb.p10_training_contract import (
    PatchRef,
    epoch_hash,
    phase_equal_patch_objective,
    resume_state,
    validate_resume_state,
)


DEFAULT_CONTRACT = REPO_ROOT / "configs/p11_paired_degrade_jepa_v1.json"
DEFAULT_OUTPUT = p11_impl.DEFAULT_ROOT / "training/paired_degrade_jepa_v1"
DEFAULT_FINAL_CONTRACT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r1_random"
)
CONTINUE_EXIT_CODE = 75
ARMS = ("supervised_masked", "masked_reconstruction", "response_only", "jepa")
ALIGNED_LAYERS = ("latent", "bottleneck")
MINIMUM_SUPPORT = 1.0e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(path: Path) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema_version") != "p11-paired-degrade-jepa-contract-v1":
        raise RuntimeError("unsupported P11 JEPA contract")
    split = contract["phase_split"]
    if tuple(split["training"]) != p11_impl.P11_TRAINING_PHASES:
        raise RuntimeError("JEPA/P11 training phase mismatch")
    if split["validation_and_selection"] != p11_impl.P11_VALIDATION_PHASE:
        raise RuntimeError("JEPA/P11 validation phase mismatch")
    if split["sealed_blind_test"] != p11_impl.P11_SEALED_PHASE:
        raise RuntimeError("JEPA/P11 sealed phase mismatch")
    if set(contract["matched_arms"]) != set(ARMS):
        raise RuntimeError("JEPA matched-arm contract is incomplete")
    guards = contract["scientific_guards"]
    if guards.get("ph001_may_be_opened") or guards.get("jepa_is_posterior"):
        raise RuntimeError("unsafe P11 scientific guards")
    if guards.get("exact_latent_equality_required"):
        raise RuntimeError("exact cross-view latent equality is not physically admissible")
    weights = contract["architecture"]["aligned_layers"]
    if tuple(weights) != ALIGNED_LAYERS or not np.isclose(sum(weights.values()), 1.0):
        raise RuntimeError("aligned-layer weights must be frozen and sum to one")
    diagnostics = contract["diagnostics"]
    if tuple(diagnostics["registered_latent_trajectory_steps"]) != (
        REGISTERED_TRAJECTORY_STEPS
    ):
        raise RuntimeError("latent diagnostic trajectory must remain 0/250/500")
    if int(diagnostics["latent_export_every_updates"]) != 250:
        raise RuntimeError("latent export cadence must retain the registered step 250")
    registered_gate = dict(diagnostics["registered_gate"])
    if registered_gate.pop("version", None) != GATE_VERSION:
        raise RuntimeError("latent diagnostic gate version changed")
    if registered_gate != DEFAULT_THRESHOLDS:
        raise RuntimeError("latent diagnostic thresholds differ from the frozen gate")
    return contract


def source_contract(contract_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        contract_path.resolve(),
        REPO_ROOT / "workflows/abacus_tweb/p11_factorial_training.py",
        REPO_ROOT / "workflows/abacus_tweb/p10_training_contract.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_train_unet_patch.py",
        REPO_ROOT / "workflows/abacus_tweb/p8_deterministic_common.py",
        REPO_ROOT / "workflows/abacus_tweb/p11_jepa_latent_diagnostics.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def aggregate_file_contract(paths: dict[str, Path]) -> dict:
    """Hash every required frozen artifact and then hash the canonical inventory."""
    records = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"required P11 data-contract artifact is absent: {path}")
        records[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "p11-jepa-frozen-data-contract-v1",
        "files": records,
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_dense_adapter_marker(
    marker: dict,
    *,
    training: tuple[str, ...],
    validation: str,
    sealed: str,
) -> None:
    """Fail closed unless the dense view has the registered P11 response semantics."""
    expected_gates = {
        "finite_nonzero_normalization",
        "finite_positive_curves",
        "ph001_not_opened",
        "ph006_application_only",
        "shell_closure_below_10pct",
        "training_phases_only",
    }
    if marker.get("schema_version") != "p11-dense-response-adapter-v1":
        raise RuntimeError("unsupported P11 V_dense response-adapter schema")
    if marker.get("view") != "V_dense" or marker.get("tracer") != "BGS_BRIGHT":
        raise RuntimeError("P11 dense adapter must describe the BGS_BRIGHT V_dense view")
    if tuple(marker.get("training_phases", ())) != training:
        raise RuntimeError("P11 dense-adapter training phases violate the frozen split")
    if marker.get("validation_phase") != validation:
        raise RuntimeError("P11 dense-adapter validation phase violates the frozen split")
    if marker.get("sealed_phase") != sealed:
        raise RuntimeError("P11 dense-adapter blind phase violates the frozen split")
    if (
        not marker.get("pass")
        or marker.get("sealed_phase_opened", True)
        or marker.get("truth_or_targets_read", True)
    ):
        raise RuntimeError("P11 V_dense response adapter is failing or blind/target contaminated")
    gates = marker.get("gates", {})
    if not expected_gates.issubset(gates) or not all(gates[name] is True for name in expected_gates):
        raise RuntimeError("P11 V_dense response-adapter gates are incomplete or failing")
    if tuple(marker.get("channel_order", ())) != (
        "counts",
        "exposure_apodized",
        "log_count_ratio",
    ):
        raise RuntimeError("P11 V_dense raw channel contract has changed")
    if tuple(marker.get("model_mapping", ())) != (
        "zscored_log1p_counts",
        "clipped_expm1_log_count_ratio",
        "common_random_support_exposure",
    ):
        raise RuntimeError("P11 V_dense model-channel mapping has changed")
    response = marker.get("response_contract", {})
    if (
        response.get("support") != "P3b-R exposure_apodized_random"
        or response.get("angular_response")
        != "P3b-R angular_response enters mu, not the third channel"
        or response.get("mu")
        != "ntilde_dense(z) * voxel_volume * angular_response * support_exposure"
    ):
        raise RuntimeError("P11 V_dense no longer uses the registered common-random response")


def validate_r1_ready_manifest(
    ready: dict,
    *,
    root: Path,
    training: tuple[str, ...],
    validation: str,
    sealed: str,
) -> None:
    """Require a self-consistent, root-local P3b-R R1 readiness inventory."""
    root = Path(root)
    if (
        ready.get("schema_version") != "p3br-r1-training-loader-ready-v1"
        or ready.get("view")
        != "R1 BRIGHT counts plus random-derived response, capacity matched to R0"
        or ready.get("field_channels")
        != ["counts", "exposure_apodized", "log_count_ratio"]
        or ready.get("ph001_opened", True)
        or ready.get("ph001_product_built", True)
        or not ready.get("pass")
    ):
        raise RuntimeError("P11 V_final must use the passed P3b-R R1 response contract")
    roles = ready.get("roles", {})
    if (
        not set(training).issubset(set(roles.get("training", ())))
        or roles.get("validation_and_selection") != validation
        or roles.get("sealed_blind_test") != sealed
    ):
        raise RuntimeError("P3b-R R1 readiness roles violate the frozen P11 split")

    inventory_path = root / "adapter_inventory.json"
    if Path(str(ready.get("adapter_inventory", ""))) != inventory_path:
        raise RuntimeError("P3b-R R1 readiness points outside its contract-root inventory")
    if ready.get("adapter_inventory_sha256") != sha256(inventory_path):
        raise RuntimeError("P3b-R R1 readiness inventory hash is stale")
    inventory = json.loads(inventory_path.read_text())
    if (
        inventory.get("schema_version") != "p3br-r1-adapter-inventory-v1"
        or not inventory.get("pass")
        or inventory.get("ph001_product_built", True)
    ):
        raise RuntimeError("P3b-R R1 adapter inventory is failing or unsafe")

    adapters = ready.get("adapters", {})
    inventory_phases = inventory.get("phases", {})
    for phase in training + (validation,):
        expected = root / "adapters" / phase / "field" / "adapter_manifest.json"
        ready_record = adapters.get(phase, {})
        inventory_record = inventory_phases.get(phase, {})
        if (
            Path(str(ready_record.get("path", ""))) != expected
            or Path(str(inventory_record.get("field_manifest", ""))) != expected
            or not ready_record.get("pass")
        ):
            raise RuntimeError(f"{phase} R1 adapter pointer is absent or outside contract root")
        current_hash = sha256(expected)
        if (
            ready_record.get("sha256") != current_hash
            or inventory_record.get("field_manifest_sha256") != current_hash
        ):
            raise RuntimeError(f"{phase} R1 adapter hash is stale")

    transform_path = root / "transforms" / "field" / "field_transform.json"
    if (
        Path(str(ready.get("field_transform", ""))) != transform_path
        or ready.get("field_transform_sha256") != sha256(transform_path)
    ):
        raise RuntimeError("P3b-R R1 field-transform pointer or hash is stale")


def frozen_data_contract(args: argparse.Namespace, contract: dict) -> dict:
    """Digest all small artifacts that determine examples, weights and transforms."""
    split = contract["phase_split"]
    training = tuple(split["training"])
    validation = str(split["validation_and_selection"])
    ready_path = args.contract_root / "TRAINING_LOADER_READY.json"
    ready = json.loads(ready_path.read_text())
    validate_r1_ready_manifest(
        ready,
        root=args.contract_root,
        training=training,
        validation=validation,
        sealed=str(split["sealed_blind_test"]),
    )
    dense_marker_path = args.adapter_contract / "P11_DENSE_RESPONSE_ADAPTER_READY.json"
    dense_marker = json.loads(dense_marker_path.read_text())
    validate_dense_adapter_marker(
        dense_marker,
        training=training,
        validation=validation,
        sealed=str(split["sealed_blind_test"]),
    )
    paths: dict[str, Path] = {
        "training_loader_ready": ready_path,
        "final_adapter_inventory": args.contract_root / "adapter_inventory.json",
        "target_scaler": args.contract_root / "transforms/target_scaler.json",
        "final_field_transform": args.contract_root / "transforms/field/field_transform.json",
        "dense_response_adapter": dense_marker_path,
        "dense_cosmology_selection_manifest": Path(
            dense_marker["p10_selection_manifest"]
        ),
        "factorial_products": args.factorial_root / "FACTORIAL_VIEW_PRODUCTS_READY.json",
    }
    for phase in training + (validation,):
        phase_root = args.contract_root / "phases" / phase
        paths[f"{phase}_phase_contract"] = phase_root / "phase_contract.json"
        paths[f"{phase}_active_row_weight"] = phase_root / "active_row_weight.npy"
        paths[f"{phase}_field_adapter_manifest"] = (
            args.contract_root / "adapters" / phase / "field/adapter_manifest.json"
        )
        factorial_counts = (
            args.factorial_root / phase / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json"
        )
        paths[f"{phase}_factorial_counts"] = factorial_counts
        factorial_record = json.loads(factorial_counts.read_text())
        response_manifest = Path(factorial_record["response_manifest"])
        paths[f"{phase}_common_random_response_manifest"] = response_manifest
        if phase in training:
            paths[f"{phase}_training_core_id"] = phase_root / "training_core_id.npy"
            paths[f"{phase}_training_core_weight"] = phase_root / "training_core_weight.npy"
        else:
            paths[f"{phase}_validation_core_id"] = phase_root / "validation_core_id.npy"
    return aggregate_file_contract(paths)


def validate_pair(final_patch, dense_patch) -> None:
    scalar_fields = ("core_id", "fold", "cap")
    for name in scalar_fields:
        if int(getattr(final_patch, name)) != int(getattr(dense_patch, name)):
            raise RuntimeError(f"paired view differs in {name}")
    array_fields = (
        "context_start",
        "context_stop",
        "core_start",
        "core_stop",
        "authoritative_parent_id",
        "authoritative_frac_index_local",
    )
    for name in array_fields:
        left = np.asarray(getattr(final_patch, name))
        right = np.asarray(getattr(dense_patch, name))
        if left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"paired view differs in {name}")
    if final_patch.core_slice != dense_patch.core_slice:
        raise RuntimeError("paired view differs in core slice")
    if tuple(final_patch.values.shape[1:]) != tuple(dense_patch.values.shape[1:]):
        raise RuntimeError("paired view differs in context lattice shape")


def supported_core_mask(final_patch, dense_patch) -> np.ndarray:
    """Common registered M=1 proxy restricted to the authoritative voxel core.

    Both paired adapters must independently produce the same binary validity map
    under the frozen ``exposure_apodized > 1e-4`` rule.  Intersecting mismatched
    maps would hide a response-contract defect, so parity is asserted first.
    """
    validate_pair(final_patch, dense_patch)
    final_at = {name: i for i, name in enumerate(final_patch.channel_names)}
    dense_at = {name: i for i, name in enumerate(dense_patch.channel_names)}
    name = "exposure_apodized"
    final_support = np.asarray(final_patch.values[final_at[name]]) > MINIMUM_SUPPORT
    dense_support = np.asarray(dense_patch.values[dense_at[name]]) > MINIMUM_SUPPORT
    if not np.array_equal(final_support, dense_support):
        mismatch = int(np.count_nonzero(final_support != dense_support))
        raise RuntimeError(
            f"paired final/dense support parity failed at {mismatch} voxels "
            f"under exposure_apodized>{MINIMUM_SUPPORT:g}"
        )
    core = np.zeros_like(final_support, dtype=bool)
    core[final_patch.core_slice] = True
    return core & final_support


def _valid_cuboid_starts(eligible: np.ndarray, block_voxels: int) -> np.ndarray:
    """All starts whose complete B^3 cuboid lies inside ``eligible``."""
    eligible = np.asarray(eligible, dtype=np.int64)
    block = int(block_voxels)
    if eligible.ndim != 3 or block <= 0 or np.any(np.asarray(eligible.shape) < block):
        return np.empty((0, 3), dtype=np.int64)
    integral = np.pad(eligible, ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    b = block
    sums = (
        integral[b:, b:, b:]
        - integral[:-b, b:, b:]
        - integral[b:, :-b, b:]
        - integral[b:, b:, :-b]
        + integral[:-b, :-b, b:]
        + integral[:-b, b:, :-b]
        + integral[b:, :-b, :-b]
        - integral[:-b, :-b, :-b]
    )
    return np.argwhere(sums == block ** 3).astype(np.int64)


def deterministic_block_mask(
    eligible: np.ndarray,
    *,
    seed: int,
    epoch: int,
    phase_index: int,
    core_id: int,
    block_voxels: int,
    blocks: int,
) -> np.ndarray:
    """Select exactly ``blocks`` disjoint, fully supported cuboids; never fall back."""
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.ndim != 3 or not np.any(eligible):
        raise ValueError("eligible mask must contain supported 3-D core voxels")
    if block_voxels <= 0 or blocks <= 0:
        raise ValueError("invalid mask geometry")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, epoch, phase_index, core_id, 11011])
    )
    starts = _valid_cuboid_starts(eligible, int(block_voxels))
    if len(starts) < int(blocks):
        raise RuntimeError(
            f"only {len(starts)} complete supported cuboids exist; require {blocks}"
        )
    result = np.zeros_like(eligible, dtype=bool)
    accepted = 0
    for index in rng.permutation(len(starts)):
        start = starts[int(index)]
        stop = start + int(block_voxels)
        selection = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))
        if np.any(result[selection]):
            continue
        result[selection] = True
        accepted += 1
        if accepted == int(blocks):
            break
    expected = int(blocks) * int(block_voxels) ** 3
    if accepted != int(blocks):
        raise RuntimeError(
            f"only {accepted} non-overlapping supported cuboids could be selected; "
            f"require {blocks}"
        )
    if np.any(result & ~eligible) or int(result.sum()) != expected:
        raise RuntimeError("constructed an invalid JEPA target mask")
    return result


def resize_mask(mask: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[0] != 1:
        raise ValueError("mask must have shape (1,nx,ny,nz)")
    return (
        F.interpolate(mask[:, None].float(), size=spatial_shape, mode="nearest")[:, 0]
        > 0.5
    )


def masked_vectors(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    resized = resize_mask(mask, tuple(values.shape[2:]))
    vectors = values.permute(0, 2, 3, 4, 1)[resized]
    if vectors.ndim != 2 or len(vectors) < 2:
        raise RuntimeError("JEPA target mask has insufficient latent vectors")
    return vectors


def spread_covariance_loss(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style anti-collapse losses over target-region student latents."""
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError("spread loss requires at least two latent vectors")
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1.0e-4)
    spread = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / float(max(len(vectors) - 1, 1))
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_penalty = off_diagonal.square().sum() / float(vectors.shape[1])
    return spread, covariance_penalty


def alignment_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    pred = masked_vectors(prediction, mask)
    truth = masked_vectors(target.detach(), mask)
    pred = F.layer_norm(pred, (pred.shape[1],))
    truth = F.layer_norm(truth, (truth.shape[1],))
    return F.smooth_l1_loss(pred, truth)


def representation_statistics(
    student: np.ndarray,
    teacher: np.ndarray,
    predicted: np.ndarray | None,
    response_only: np.ndarray | None = None,
) -> dict:
    """Small, projection-free diagnostics for exported paired representations."""
    student = np.asarray(student, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("paired student/teacher arrays must share shape (N,D)")
    if predicted is not None:
        predicted = np.asarray(predicted, dtype=np.float64)
        if predicted.shape != student.shape:
            raise ValueError("trained predictor array must share shape (N,D)")
    if response_only is not None:
        response_only = np.asarray(response_only, dtype=np.float64)
        if response_only.shape != student.shape:
            raise ValueError("response-only latent must share shape (N,D)")

    def cosine(left, right):
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return float(np.mean(np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)))

    centered = student - student.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False) ** 2
    total_singular = float(singular.sum())
    if total_singular <= 1e-20:
        effective_rank = 0.0
    else:
        probability = singular / total_singular
        entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        effective_rank = float(np.exp(entropy))
    return {
        "rows": int(len(student)),
        "dimensions": int(student.shape[1]),
        "student_teacher_cosine": cosine(student, teacher),
        "predicted_teacher_cosine": (
            cosine(predicted, teacher) if predicted is not None else None
        ),
        "student_to_teacher_norm_ratio": float(
            np.mean(np.linalg.norm(student, axis=1))
            / max(float(np.mean(np.linalg.norm(teacher, axis=1))), 1e-12)
        ),
        "student_channel_std_mean": float(std.mean()),
        "student_channel_std_min": float(std.min()),
        "student_effective_rank": effective_rank,
        "student_collapse_fraction_std_lt_0p05": float(np.mean(std < 0.05)),
        "response_only_student_cosine": (
            cosine(response_only, student) if response_only is not None else None
        ),
    }


def unet_features(unet: unet_impl.UNet3D, values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return registered U-PATCH output and bottleneck without changing its layers."""
    e0 = unet.enc0(values)
    e1 = unet.enc1(unet.pool(e0))
    e2 = unet.enc2(unet.pool(e1))
    bottleneck = unet.bottleneck(unet.pool(e2))
    d2 = unet.dec2(torch.cat((unet.up(bottleneck, e2), e2), dim=1))
    d1 = unet.dec1(torch.cat((unet.up(d2, e1), e1), dim=1))
    d0 = unet.dec0(torch.cat((unet.up(d1, e0), e0), dim=1))
    return {"latent": unet.output(d0), "bottleneck": bottleneck}


def sample_feature(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        values, points, mode="bilinear", align_corners=True, padding_mode="border"
    )[0, :, 0, 0].T


class PointwisePredictor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.SiLU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PairedDegradeJEPA(nn.Module):
    """Deployable U-PATCH student plus training-only EMA teacher/predictors."""
    def __init__(self, *, base: int, latent_channels: int, head_width: int):
        super().__init__()
        self.student = unet_impl.UPatch(
            base=base, latent_channels=latent_channels, head_width=head_width
        )
        self.teacher = copy.deepcopy(self.student.unet)
        self.teacher.requires_grad_(False)
        self.predictors = nn.ModuleDict(
            {
                "latent": PointwisePredictor(latent_channels),
                "bottleneck": PointwisePredictor(base * 4),
            }
        )
        self.reconstruction = nn.Conv3d(latent_channels, 2, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        # The target encoder is updated only by EMA and never uses train-time state.
        self.teacher.eval()
        return self

    def encode_student(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return unet_features(self.student.unet, values)

    @torch.no_grad()
    def encode_teacher(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        return unet_features(self.teacher, values)

    def predict_targets(self, features: dict[str, torch.Tensor], points: torch.Tensor) -> torch.Tensor:
        return self.student.head(sample_feature(features["latent"], points))

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for target, online in zip(self.teacher.parameters(), self.student.unet.parameters()):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(self.teacher.buffers(), self.student.unet.buffers()):
            target.copy_(online)


@dataclass
class PairedInputs:
    final_patch: object
    dense_patch: object
    final_values: torch.Tensor
    dense_values: torch.Tensor
    points: torch.Tensor
    target_mask: torch.Tensor
    eligible_voxels: int
    target_voxels: int


def paired_inputs(
    loader: p11_impl.P11DensePhaseBalancedLoader,
    phase: str,
    core_id: int,
    *,
    final_normalization: dict,
    dense_normalization: dict,
    device: str,
    mask_spec: dict,
    seed: int,
    epoch: int,
    phase_index: int,
) -> PairedInputs:
    dense_adapter = loader.field_adapter(phase)
    final_patch = dense_adapter.base.extract(
        core_id,
        unet_impl.HALO_VOXELS,
        unet_impl.CHANNELS,
        alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
    )
    dense_patch = dense_adapter.extract(
        core_id,
        unet_impl.HALO_VOXELS,
        unet_impl.CHANNELS,
        alignment_voxels=unet_impl.ALIGNMENT_VOXELS,
    )
    validate_pair(final_patch, dense_patch)
    final_values, points = unet_impl.model_inputs(final_patch, final_normalization, device)
    dense_values, dense_points = unet_impl.model_inputs(dense_patch, dense_normalization, device)
    if not torch.equal(points, dense_points):
        raise RuntimeError("paired views produced different galaxy sampling coordinates")
    eligible = supported_core_mask(final_patch, dense_patch)
    mask = deterministic_block_mask(
        eligible,
        seed=seed,
        epoch=epoch,
        phase_index=phase_index,
        core_id=core_id,
        block_voxels=int(mask_spec["block_voxels"]),
        blocks=int(mask_spec["blocks"]),
    )
    target_voxels = int(np.count_nonzero(mask))
    eligible_voxels = int(np.count_nonzero(eligible))
    fraction = target_voxels / eligible_voxels
    low, high = (
        float(value) for value in mask_spec["registered_mask_fraction_range"]
    )
    expected = int(mask_spec["blocks"]) * int(mask_spec["block_voxels"]) ** 3
    if target_voxels != expected:
        raise RuntimeError(
            f"JEPA mask contains {target_voxels} voxels; registered geometry requires "
            f"exactly {expected}"
        )
    if not (np.isfinite(fraction) and fraction > 0 and low <= fraction <= high):
        raise RuntimeError(
            f"JEPA mask fraction {fraction:.6g} lies outside registered "
            f"[{low:.6g}, {high:.6g}]"
        )
    return PairedInputs(
        final_patch=final_patch,
        dense_patch=dense_patch,
        final_values=final_values,
        dense_values=dense_values,
        points=points,
        target_mask=torch.from_numpy(mask[None]).to(device),
        eligible_voxels=eligible_voxels,
        target_voxels=target_voxels,
    )


def run_real_view_parity_gate(
    *,
    loader: p11_impl.P11DensePhaseBalancedLoader,
    phases: tuple[str, ...],
    validation_phase: str,
    validation_core: np.ndarray,
    final_normalization: dict,
    dense_normalization: dict,
    device: str,
    mask_spec: dict,
    seed: int,
    data_contract: dict,
    output: Path,
) -> dict:
    """Exercise one real paired patch per visible phase before any optimizer step."""
    marker = output / "P11_REAL_VIEW_PARITY_GATE.json"
    if marker.exists():
        cached = json.loads(marker.read_text())
        if (
            cached.get("pass")
            and cached.get("data_contract_aggregate_sha256")
            == data_contract["aggregate_sha256"]
            and not cached.get("sealed_phase_opened")
        ):
            return cached
        raise RuntimeError("stale or failing P11 real-view parity marker")
    low, high = (float(value) for value in mask_spec["registered_mask_fraction_range"])
    records = []
    for phase_index, phase in enumerate(phases + (validation_phase,)):
        if phase == validation_phase:
            candidates = validation_core
        else:
            candidates = np.load(
                loader.root / "phases" / phase / "training_core_id.npy", mmap_mode="r"
            )
        core_id = int(np.sort(np.asarray(candidates, dtype=np.int64))[0])
        paired = paired_inputs(
            loader,
            phase,
            core_id,
            final_normalization=final_normalization,
            dense_normalization=dense_normalization,
            device=device,
            mask_spec=mask_spec,
            seed=seed,
            epoch=1,
            phase_index=phase_index,
        )
        fraction = paired.target_voxels / paired.eligible_voxels
        expected = int(mask_spec["blocks"]) * int(mask_spec["block_voxels"]) ** 3
        gates = {
            "paired_geometry_exact": True,
            "paired_parent_order_exact": True,
            "support_threshold_parity_exact": True,
            "exact_registered_cuboid_voxels": paired.target_voxels == expected,
            "mask_fraction_registered": low <= fraction <= high,
        }
        records.append(
            {
                "phase": phase,
                "core_id": core_id,
                "fold": int(paired.final_patch.fold),
                "eligible_voxels": paired.eligible_voxels,
                "target_voxels": paired.target_voxels,
                "mask_fraction": fraction,
                "gates": gates,
                "pass": bool(all(gates.values())),
            }
        )
    report = {
        "schema_version": "p11-real-view-pair-parity-v1",
        "created_utc": utc_now(),
        "data_contract_aggregate_sha256": data_contract["aggregate_sha256"],
        "support_semantics": f"exposure_apodized>{MINIMUM_SUPPORT:g} as registered M=1 target proxy",
        "records": records,
        "sealed_phase_opened": False,
        "pass": bool(all(row["pass"] for row in records)),
    }
    atomic_json(marker, report)
    if not report["pass"]:
        raise RuntimeError("P11 real-view pair parity gate failed")
    return report


def masked_student_values(values: torch.Tensor, mask: torch.Tensor, *, response_only: bool) -> torch.Tensor:
    """Mask model-order signal channels (counts, density proxy), retaining exposure.

    ``FieldPatch.channel_names`` uses ``(counts, exposure, log_count_ratio)``,
    but :func:`unet_impl.model_inputs` deliberately remaps this to the registered
    U-PATCH tensor order ``(counts, density_proxy, exposure)``.  Consequently
    model channels 0/1 are cosmological signal and channel 2 is response.
    """
    result = values.clone()
    if response_only:
        result[:, :2] = 0.0
    else:
        result[:, :2] = result[:, :2].masked_fill(mask[:, None], 0.0)
    return result


def arm_auxiliary_losses(
    *,
    arm: str,
    model: PairedDegradeJEPA,
    student_features: dict[str, torch.Tensor],
    dense_values: torch.Tensor,
    unmasked_final_values: torch.Tensor,
    target_mask: torch.Tensor,
    layer_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    zero = student_features["latent"].sum() * 0.0
    losses = {"alignment": zero, "spread": zero, "covariance": zero, "reconstruction": zero}
    if arm == "jepa":
        teacher_features = model.encode_teacher(dense_values)
        alignment = zero
        spread = zero
        covariance = zero
        for layer, layer_weight in layer_weights.items():
            predicted = model.predictors[layer](student_features[layer])
            alignment = alignment + float(layer_weight) * alignment_loss(
                predicted, teacher_features[layer], target_mask
            )
            vectors = masked_vectors(student_features[layer], target_mask)
            layer_spread, layer_covariance = spread_covariance_loss(vectors)
            spread = spread + float(layer_weight) * layer_spread
            covariance = covariance + float(layer_weight) * layer_covariance
        losses.update(alignment=alignment, spread=spread, covariance=covariance)
    elif arm == "masked_reconstruction":
        reconstruction = model.reconstruction(student_features["latent"])
        selected = resize_mask(target_mask, tuple(reconstruction.shape[2:]))
        prediction = reconstruction.permute(0, 2, 3, 4, 1)[selected]
        target = unmasked_final_values[:, :2].permute(0, 2, 3, 4, 1)[selected]
        losses["reconstruction"] = F.smooth_l1_loss(prediction, target)
    elif arm not in ("supervised_masked", "response_only"):
        raise ValueError(f"unknown matched arm: {arm}")
    return losses


def phase_scaled_patch_loss(value: torch.Tensor, ref: PatchRef, phase_core_count: int) -> torch.Tensor:
    return phase_equal_patch_objective(
        value,
        phase_weight_denominator=float(phase_core_count),
        phase_objective_scale=ref.phase_objective_scale,
    )


def module_parameters_finite(module: nn.Module) -> bool:
    return bool(
        all(torch.all(torch.isfinite(parameter)).item() for parameter in module.parameters())
    )


def technical_canary_gates(
    *,
    finite_pre_parameters: bool,
    finite_post_parameters: bool,
    finite_loss: bool,
    gradient_norm: float,
    mask_fraction: float,
    registered_mask_fraction_range: tuple[float, float],
    checkpoint_reload_valid: bool,
    latent_snapshot_valid: bool,
) -> dict:
    low, high = (float(value) for value in registered_mask_fraction_range)
    gates = {
        "finite_pre_parameters": bool(finite_pre_parameters),
        "finite_post_parameters": bool(finite_post_parameters),
        "finite_loss": bool(finite_loss),
        "finite_gradient_norm": bool(np.isfinite(gradient_norm)),
        "nonzero_registered_mask_fraction": bool(
            np.isfinite(mask_fraction) and mask_fraction > 0 and low <= mask_fraction <= high
        ),
        "checkpoint_reload_valid": bool(checkpoint_reload_valid),
        "final_latent_snapshot_valid": bool(latent_snapshot_valid),
    }
    return {"gates": gates, "pass": bool(all(gates.values()))}


def jsonable_args(args: argparse.Namespace) -> dict:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def frozen_execution(
    args: argparse.Namespace, contract_sha256: str, data_contract_sha256: str
) -> dict:
    return {
        "arm": args.arm,
        "seed": int(args.seed),
        "contract_sha256": contract_sha256,
        "data_contract_aggregate_sha256": data_contract_sha256,
        "contract_root": str(args.contract_root),
        "factorial_root": str(args.factorial_root),
        "adapter_contract": str(args.adapter_contract),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_FINAL_CONTRACT)
    parser.add_argument("--factorial-root", type=Path, default=p11_impl.DEFAULT_ROOT)
    parser.add_argument("--adapter-contract", type=Path, default=p11_impl.DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="canary_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--loss-log-every", type=int, default=25)
    parser.add_argument("--latent-export-every", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--validation-reserve-seconds", type=float, default=1200.0)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint_every", "loss_log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.latent_export_every is not None and args.latent_export_every != 250:
        parser.error("--latent-export-every is frozen to 250 updates")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.validation_reserve_seconds < 0:
        parser.error("--validation-reserve-seconds must be non-negative")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        parser.error("--stop-after-updates must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P11 JEPA requires a CUDA interactive allocation")
    contract = load_contract(args.contract)
    contract_hash = sha256(args.contract)
    data_contract = frozen_data_contract(args, contract)
    optimization = contract["optimization"]
    architecture = contract["architecture"]
    objective = contract["objective"]
    mask_spec = contract["masking"]
    diagnostics = contract["diagnostics"]
    latent_export_every = int(
        diagnostics["latent_export_every_updates"]
        if args.latent_export_every is None
        else args.latent_export_every
    )
    sources = source_contract(args.contract)
    frozen = frozen_execution(args, contract_hash, data_contract["aggregate_sha256"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = args.output_root / args.run_name / args.arm / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(output / ".run.lock", purpose="P11 paired-degrade JEPA")
    checkpoint_path = output / "p11_jepa_checkpoint.pt"
    complete_marker = output / "P11_MATCHED_ARM_COMPLETE.json"
    if complete_marker.exists():
        print(complete_marker.read_text(), flush=True)
        lock.close()
        return
    resume = bool(args.auto_resume and checkpoint_path.exists())
    if any(path.name != ".run.lock" for path in output.iterdir()) and not resume:
        raise RuntimeError(f"non-empty P11 output requires --auto-resume: {output}")
    data_contract_path = output / "FROZEN_DATA_CONTRACT.json"
    if resume:
        if not data_contract_path.is_file():
            raise RuntimeError("resume is missing the frozen P11 data-contract digest")
        if json.loads(data_contract_path.read_text()) != data_contract:
            raise RuntimeError("P11 data-contract artifacts changed since launch")
    else:
        atomic_json(data_contract_path, data_contract)

    loader = p11_impl.P11DensePhaseBalancedLoader(
        args.contract_root,
        factorial_root=args.factorial_root,
        adapter_contract=args.adapter_contract,
    )
    phases = tuple(loader.training_phases)
    validation_phase = loader.validation_phase
    if loader.manifest.get("schema_version") != "p3br-r1-training-loader-ready-v1":
        raise RuntimeError("P11 student V_final is not backed by the frozen P3b-R R1 loader")
    if tuple(contract["phase_split"]["training"]) != phases:
        raise RuntimeError("runtime phases differ from frozen JEPA contract")
    if loader.blind_phase in phases + (validation_phase,):
        raise RuntimeError("sealed phase entered a visible P11 role")
    scaler = loader.target_scaler
    dense_normalization = loader.field_normalization
    # V_final uses the stored P3b-R channels and its frozen R1 normalization;
    # V_dense has a separately fitted ntilde/normalization but shares the same M.
    final_transform = json.loads(
        (args.contract_root / "transforms/field/field_transform.json").read_text()
    )
    final_normalization = final_transform["normalization"]
    runtime = {
        phase: prepare_phase_runtime(loader, phase, scaler, training=True)
        for phase in phases
    }
    validation_runtime = prepare_phase_runtime(loader, validation_phase, scaler, training=False)
    validation_refs = loader.validation_refs()
    validation_core = np.asarray([ref.core_id for ref in validation_refs], dtype=np.int64)
    validation_assignment_path = Path(loader.phase_records[validation_phase]["inputs"]["assignment"])
    phase_core_count = {
        phase: len(np.load(args.contract_root / "phases" / phase / "training_core_id.npy", mmap_mode="r"))
        for phase in phases
    }
    pair_parity = run_real_view_parity_gate(
        loader=loader,
        phases=phases,
        validation_phase=validation_phase,
        validation_core=validation_core,
        final_normalization=final_normalization,
        dense_normalization=dense_normalization,
        device=args.device,
        mask_spec=mask_spec,
        seed=args.seed,
        data_contract=data_contract,
        output=output,
    )

    model = PairedDegradeJEPA(
        base=int(architecture["unet_base_channels"]),
        latent_channels=int(architecture["latent_channels"]),
        head_width=int(architecture["point_head_width"]),
    ).to(args.device)
    trainable = list(model.student.parameters())
    if args.arm == "jepa":
        trainable += list(model.predictors.parameters())
    elif args.arm == "masked_reconstruction":
        trainable += list(model.reconstruction.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    epoch_length = len(loader.training_epoch(seed=args.seed, epoch=1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(optimization["epochs"]) * epoch_length
    )
    epoch = 1
    cursor = 0
    global_step = 0
    history: list[dict] = []
    best_score = -np.inf
    best_epoch = -1
    early_best = -np.inf
    stale_epochs = 0
    target_accumulators = {phase: EpochLossAccumulator() for phase in phases}
    aux_sums = {name: 0.0 for name in ("alignment", "spread", "covariance", "reconstruction")}
    aux_steps = 0
    target_voxels = 0
    eligible_voxels = 0
    maximum_memory = 0

    if resume:
        state = torch_load(checkpoint_path, args.device)
        if state.get("schema_version") != "p11-paired-degrade-jepa-checkpoint-v1":
            raise RuntimeError("unsupported P11 JEPA checkpoint")
        if (
            state["source_contract"] != sources
            or state["frozen_execution"] != frozen
            or state.get("data_contract") != data_contract
        ):
            raise RuntimeError("P11 source/contract changed since checkpoint")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        resume_row = state["resume"]
        epoch = int(resume_row["epoch"])
        cursor = int(resume_row["cursor"])
        refs = loader.training_epoch(seed=args.seed, epoch=epoch)
        validate_resume_state(resume_row, refs)
        global_step = int(state["global_step"])
        history = list(state["history"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        early_best = float(state["early_best"])
        stale_epochs = int(state["stale_epochs"])
        target_accumulators = {
            phase: EpochLossAccumulator.from_dict(state["target_accumulators"].get(phase))
            for phase in phases
        }
        aux_sums = {name: float(value) for name, value in state["aux_sums"].items()}
        aux_steps = int(state["aux_steps"])
        target_voxels = int(state["target_voxels"])
        eligible_voxels = int(state["eligible_voxels"])
        maximum_memory = int(state["maximum_memory"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([row.cpu() for row in state["cuda_rng_state_all"]])
        reconcile_loss_trace(output / "loss_trace.jsonl", maximum_global_step=global_step)
        rewrite_jsonl(output / "epoch_history.jsonl", history)
    else:
        (output / "loss_trace.jsonl").write_text("")
        (output / "epoch_history.jsonl").write_text("")

    finite_pre_parameters = module_parameters_finite(model)
    if not finite_pre_parameters:
        raise RuntimeError("P11 model parameters are non-finite before the canary")

    run_manifest = {
        "schema_version": "p11-paired-degrade-jepa-run-v1",
        "created_utc": utc_now(),
        "contract_id": contract["contract_id"],
        "contract": str(args.contract),
        "contract_sha256": contract_hash,
        "arm": args.arm,
        "seed": args.seed,
        "training_phases": list(phases),
        "validation_and_selection_phase": validation_phase,
        "sealed_blind_phase": loader.blind_phase,
        "blind_truth_accessed": False,
        "student_view": contract["views"]["student"],
        "teacher_view": contract["views"]["teacher"] if args.arm == "jepa" else None,
        "teacher_has_target_head": False,
        "student_only_deployment": True,
        "epoch_length": epoch_length,
        "arguments": jsonable_args(args),
        "frozen_execution": frozen,
        "source_contract": sources,
        "data_contract": data_contract,
        "real_view_pair_parity": pair_parity,
        "git_revision_at_launch": git_revision(),
    }
    atomic_json(output / "run_manifest.json", run_manifest)

    def fixed_probe_cores() -> np.ndarray:
        """Choose a stable, fold-balanced ph006 latent probe without ph001."""
        maximum = int(diagnostics["fixed_validation_probe_cores"])
        folds = np.asarray(loader.field_adapter(validation_phase).base.core_fold)
        selected: list[int] = []
        per_fold = max(1, int(np.ceil(maximum / 5)))
        for fold in range(5):
            candidates = np.sort(
                validation_core[folds[validation_core.astype(np.int64)] == fold]
            )
            selected.extend(int(value) for value in candidates[:per_fold])
        result = np.asarray(selected[:maximum], dtype=np.int64)
        result_folds = folds[result]
        if len(result) < 2 or not np.any(result_folds <= 1) or not np.any(result_folds >= 2):
            raise RuntimeError("fixed ph006 latent probe does not span folds 0--1 and 2--4")
        return result

    probe_core_ids = fixed_probe_cores()

    def checkpoint(refs: tuple[PatchRef, ...], checkpoint_cursor: int) -> None:
        payload = {
            "schema_version": "p11-paired-degrade-jepa-checkpoint-v1",
            "created_utc": utc_now(),
            "source_contract": sources,
            "frozen_execution": frozen,
            "data_contract": data_contract,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "resume": resume_state(
                seed=args.seed,
                epoch=epoch,
                cursor=checkpoint_cursor,
                refs=refs,
                loss_accumulator={
                    phase: accumulator.as_dict()
                    for phase, accumulator in target_accumulators.items()
                },
            ),
            "global_step": global_step,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "early_best": early_best,
            "stale_epochs": stale_epochs,
            "target_accumulators": {
                phase: accumulator.as_dict()
                for phase, accumulator in target_accumulators.items()
            },
            "aux_sums": aux_sums,
            "aux_steps": aux_steps,
            "target_voxels": target_voxels,
            "eligible_voxels": eligible_voxels,
            "maximum_memory": maximum_memory,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        atomic_torch_save(payload, checkpoint_path)

    def validate_latent_export(
        export_path: Path, report_path: Path, expected_step: int
    ) -> dict:
        """Fail closed on incomplete, stale or arm-inconsistent latent exports."""
        if not export_path.is_file() or not report_path.is_file():
            raise RuntimeError("P11 latent snapshot/report pair is incomplete")
        snapshot = load_latent_snapshot(export_path)
        report = json.loads(report_path.read_text())
        expected_run_id = f"{args.run_name}/{args.arm}/seed_{args.seed}"
        if (
            snapshot.metadata.get("run_id") != expected_run_id
            or snapshot.metadata.get("arm") != args.arm
            or int(snapshot.metadata.get("global_step", -1)) != int(expected_step)
            or snapshot.metadata.get("phase") != validation_phase
            or snapshot.metadata.get("sealed_phase_opened", True)
        ):
            raise RuntimeError("P11 latent snapshot metadata violates the frozen run")
        if snapshot.response_only_latent is None:
            raise RuntimeError("P11 latent snapshot omitted the response-only representation")
        if args.arm == "jepa" and snapshot.predicted_dense_latent is None:
            raise RuntimeError("JEPA latent snapshot omitted the trained predictor output")
        if args.arm != "jepa" and snapshot.predicted_dense_latent is not None:
            raise RuntimeError("control latent snapshot contains an untrained JEPA predictor")
        if (
            int(report.get("global_step", -1)) != int(expected_step)
            or report.get("arm") != args.arm
            or report.get("file_sha256") != sha256(export_path)
            or report.get("ph001_opened", True)
        ):
            raise RuntimeError("P11 latent report does not validate its snapshot")
        return {
            "path": str(export_path),
            "sha256": sha256(export_path),
            "rows": int(len(snapshot.sample_id)),
            "global_step": int(expected_step),
            "run_id": expected_run_id,
        }

    def export_latents(step: int, export_epoch: int) -> None:
        export_dir = output / "latent_exports"
        export_path = export_dir / f"step_{step:09d}.npz"
        report_path = export_path.with_suffix(".json")
        if export_path.exists() and report_path.exists():
            validate_latent_export(export_path, report_path, step)
            return
        export_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        student_parts, teacher_parts, predicted_parts = [], [], []
        response_only_latent_parts = []
        parent_parts, target_parts, response_parts = [], [], []
        weight_parts, core_parts, fold_parts, split_parts = [], [], [], []
        maximum_rows = int(diagnostics["maximum_exported_galaxies"])
        per_core_rows = max(2, maximum_rows // len(probe_core_ids))
        with torch.no_grad():
            for core_id in probe_core_ids:
                paired = paired_inputs(
                    loader,
                    validation_phase,
                    int(core_id),
                    final_normalization=final_normalization,
                    dense_normalization=dense_normalization,
                    device=args.device,
                    mask_spec=mask_spec,
                    seed=args.seed,
                    epoch=max(export_epoch, 1),
                    phase_index=0,
                )
                response_only_values = masked_student_values(
                    paired.final_values,
                    paired.target_mask,
                    response_only=True,
                )
                student_export_values = (
                    response_only_values
                    if args.arm == "response_only"
                    else paired.final_values
                )
                student_feature = model.encode_student(student_export_values)["latent"]
                teacher_feature = (
                    model.encode_teacher(paired.dense_values)["latent"]
                    if args.arm == "jepa"
                    else model.encode_student(paired.dense_values)["latent"]
                )
                predicted_feature = (
                    model.predictors["latent"](student_feature)
                    if args.arm == "jepa"
                    else None
                )
                response_only_feature = model.encode_student(response_only_values)["latent"]
                parent = np.asarray(
                    paired.final_patch.authoritative_parent_id[:per_core_rows], dtype=np.int64
                )
                rows = len(parent)
                student_parts.append(
                    sample_feature(student_feature, paired.points).cpu().numpy()[:rows]
                )
                teacher_parts.append(
                    sample_feature(teacher_feature, paired.points).cpu().numpy()[:rows]
                )
                response_only_latent_parts.append(
                    sample_feature(response_only_feature, paired.points).cpu().numpy()[:rows]
                )
                if predicted_feature is not None:
                    predicted_parts.append(
                        sample_feature(predicted_feature, paired.points).cpu().numpy()[:rows]
                    )
                parent_parts.append(parent)
                target_parts.append(loader.targets_by_parent(validation_phase)[parent])
                # Model-order channel 2 is the deployable exposure/response channel.
                observed = sample_feature(paired.final_values, paired.points).cpu().numpy()[:rows]
                response_parts.append(observed[:, 2:3])
                weight_parts.append(validation_runtime.parent_weight[parent])
                fold = int(paired.final_patch.fold)
                core_parts.append(np.full(rows, int(core_id), dtype=np.int64))
                fold_parts.append(np.full(rows, fold, dtype=np.int8))
                split_parts.append(np.full(rows, 0 if fold <= 1 else 1, dtype=np.int8))
        student = np.concatenate(student_parts)[:maximum_rows].astype(np.float32)
        teacher = np.concatenate(teacher_parts)[:maximum_rows].astype(np.float32)
        predicted = (
            np.concatenate(predicted_parts)[:maximum_rows].astype(np.float32)
            if predicted_parts
            else None
        )
        response_only_latent = np.concatenate(response_only_latent_parts)[
            :maximum_rows
        ].astype(np.float32)
        parent = np.concatenate(parent_parts)[:maximum_rows]
        targets = np.concatenate(target_parts)[:maximum_rows].astype(np.float32)
        response_features = np.concatenate(response_parts)[:maximum_rows].astype(np.float32)
        weights = np.concatenate(weight_parts)[:maximum_rows].astype(np.float32)
        cores = np.concatenate(core_parts)[:maximum_rows]
        folds = np.concatenate(fold_parts)[:maximum_rows]
        probe_split = np.concatenate(split_parts)[:maximum_rows]
        sample_id = np.asarray(
            [f"{validation_phase}:{int(value)}" for value in parent], dtype="U40"
        )
        save_latent_snapshot(
            export_path,
            metadata={
                "run_id": f"{args.run_name}/{args.arm}/seed_{args.seed}",
                "arm": args.arm,
                "predictor_trained": args.arm == "jepa",
                "dense_encoder": (
                    "ema_stop_gradient_teacher" if args.arm == "jepa" else "student"
                ),
                "epoch": int(export_epoch),
                "global_step": int(step),
                "phase": validation_phase,
                "sealed_phase_opened": False,
                "student_view": "V_final",
                "teacher_view": "V_dense",
                "source_paths": [str(args.contract), str(args.adapter_contract)],
            },
            sample_id=sample_id,
            dense_latent=teacher,
            degraded_latent=student,
            predicted_dense_latent=predicted,
            response_only_latent=response_only_latent,
            response_strength=response_features[:, 0],
            response_features=response_features,
            probe_split=probe_split,
            target=targets,
            sample_weight=weights,
            group_id=cores,
            core_id=cores,
            fold_id=folds,
        )
        report = {
            "schema_version": "p11-jepa-latent-export-v1",
            "created_utc": utc_now(),
            "global_step": step,
            "epoch": export_epoch,
            "phase": validation_phase,
            "arm": args.arm,
            "predictor_trained": args.arm == "jepa",
            "dense_encoder": (
                "ema_stop_gradient_teacher" if args.arm == "jepa" else "student"
            ),
            "fixed_probe_core_ids": probe_core_ids.tolist(),
            "probe_folds": sorted(np.unique(folds).astype(int).tolist()),
            "sample_id_sha256": hashlib.sha256(
                "\n".join(sample_id.tolist()).encode()
            ).hexdigest(),
            "file": str(export_path),
            "file_sha256": sha256(export_path),
            "statistics": representation_statistics(
                student, teacher, predicted, response_only_latent
            ),
            "teacher_is_training_only": args.arm == "jepa",
            "ph001_opened": False,
        }
        atomic_json(report_path, report)
        append_jsonl(output / "latent_trace.jsonl", report)
        validate_latent_export(export_path, report_path, step)

    def pause(refs: tuple[PatchRef, ...], checkpoint_cursor: int, reason: str) -> None:
        checkpoint(refs, checkpoint_cursor)
        atomic_json(
            output / "ALLOCATION_PAUSED.json",
            {
                "schema_version": "p11-jepa-allocation-pause-v1",
                "created_utc": utc_now(),
                "reason": reason,
                "epoch": epoch,
                "cursor": checkpoint_cursor,
                "epoch_length": len(refs),
                "global_step": global_step,
                "resume_exit_code": CONTINUE_EXIT_CODE,
            },
        )
        raise SystemExit(CONTINUE_EXIT_CODE)

    started = time.monotonic()
    loss_window: list[dict[str, float]] = []
    try:
        technical_marker_path = output / "TECHNICAL_CANARY_COMPLETE.json"
        if args.stop_after_updates is not None and technical_marker_path.is_file():
            prior = json.loads(technical_marker_path.read_text())
            if (
                prior.get("pass")
                and not prior.get("ph001_opened")
                and prior.get("data_contract_aggregate_sha256")
                == data_contract["aggregate_sha256"]
                and int(prior.get("global_step", -1)) >= args.stop_after_updates
                and global_step >= args.stop_after_updates
            ):
                print(json.dumps(prior, indent=2), flush=True)
                return
            raise RuntimeError("stale or failing P11 technical-canary marker")
        if global_step == 0:
            # Step zero is durable before diagnostics touch the run directory.
            # A failed export is therefore an exact, whitelisted retry from the
            # frozen cursor-zero checkpoint rather than an ambiguous fresh run.
            initial_refs = loader.training_epoch(seed=args.seed, epoch=epoch)
            checkpoint(initial_refs, cursor)
            export_latents(0, 0)
        elif global_step in REGISTERED_TRAJECTORY_STEPS:
            # Repair a snapshot/report pair interrupted after its same-step
            # atomic checkpoint, before advancing the optimizer again.
            export_latents(global_step, epoch)
        while epoch <= int(optimization["epochs"]):
            refs = loader.training_epoch(seed=args.seed, epoch=epoch)
            if cursor > len(refs):
                raise RuntimeError("checkpoint cursor exceeds reconstructed epoch")
            for position in range(cursor, len(refs)):
                if (
                    args.max_runtime_seconds is not None
                    and time.monotonic() - started >= args.max_runtime_seconds
                ):
                    pause(refs, position, "interactive allocation runtime budget")
                ref = refs[position]
                phase_state = runtime[ref.phase]
                model.train()
                paired = paired_inputs(
                    loader,
                    ref.phase,
                    ref.core_id,
                    final_normalization=final_normalization,
                    dense_normalization=dense_normalization,
                    device=args.device,
                    mask_spec=mask_spec,
                    seed=args.seed,
                    epoch=epoch,
                    phase_index=ref.phase_index,
                )
                student_values = masked_student_values(
                    paired.final_values,
                    paired.target_mask,
                    response_only=args.arm == "response_only",
                )
                student_features = model.encode_student(student_values)
                prediction = model.predict_targets(student_features, paired.points)
                parent = np.asarray(paired.final_patch.authoritative_parent_id, dtype=np.int64)
                weight_np = np.asarray(phase_state.parent_weight[parent], dtype=np.float32)
                actual_weight = float(np.sum(weight_np, dtype=np.float64))
                expected_weight = phase_state.expected_core_weight[ref.core_id]
                if not np.isclose(actual_weight, expected_weight, rtol=2e-6, atol=1e-7):
                    raise RuntimeError("paired JEPA patch changed registered target weight")
                target = torch.from_numpy(phase_state.target_scaled[parent]).to(args.device)
                weight = torch.from_numpy(weight_np).to(args.device)
                loss_per_row = torch.mean((prediction - target) ** 2, dim=1)
                target_objective = phase_equal_patch_objective(
                    torch.sum(weight * loss_per_row),
                    phase_weight_denominator=phase_state.weight_denominator,
                    phase_objective_scale=ref.phase_objective_scale,
                )
                auxiliary = arm_auxiliary_losses(
                    arm=args.arm,
                    model=model,
                    student_features=student_features,
                    dense_values=paired.dense_values,
                    unmasked_final_values=paired.final_values,
                    target_mask=paired.target_mask,
                    layer_weights=architecture["aligned_layers"],
                )
                scale = lambda value: phase_scaled_patch_loss(
                    value, ref, phase_core_count[ref.phase]
                )
                loss = target_objective
                loss = loss + float(objective["alignment_weight"]) * scale(auxiliary["alignment"])
                loss = loss + float(objective["spread_weight"]) * scale(auxiliary["spread"])
                loss = loss + float(objective["covariance_weight"]) * scale(auxiliary["covariance"])
                loss = loss + float(objective["reconstruction_weight"]) * scale(
                    auxiliary["reconstruction"]
                )
                finite_loss = bool(torch.isfinite(loss).detach().cpu().item())
                if not finite_loss:
                    raise RuntimeError("P11 objective became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(objective["gradient_clip"])
                    ).detach().cpu()
                )
                if not np.isfinite(gradient_norm):
                    raise RuntimeError("P11 gradient norm became non-finite")
                if any(parameter.grad is not None for parameter in model.teacher.parameters()):
                    raise RuntimeError("stop-gradient teacher received gradients")
                optimizer.step()
                scheduler.step()
                if args.arm == "jepa":
                    model.ema_update(float(objective["teacher_ema_momentum"]))
                global_step += 1
                maximum_memory = max(maximum_memory, int(torch.cuda.max_memory_allocated()))

                loss_np = loss_per_row.detach().cpu().numpy()
                target_accumulators[ref.phase].add(loss_np, weight_np)
                aux_values = {
                    name: float(value.detach().cpu()) for name, value in auxiliary.items()
                }
                for name, value in aux_values.items():
                    aux_sums[name] += value
                aux_steps += 1
                target_voxels += paired.target_voxels
                eligible_voxels += paired.eligible_voxels
                loss_window.append(
                    {
                        "total": float(loss.detach().cpu()),
                        "target": float(target_objective.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        **aux_values,
                    }
                )

                if global_step % args.loss_log_every == 0 or position + 1 == len(refs):
                    row = {
                        "epoch": epoch,
                        "cursor": position + 1,
                        "epoch_length": len(refs),
                        "global_step": global_step,
                        "arm": args.arm,
                        "loss_window_mean": {
                            name: float(np.mean([item[name] for item in loss_window]))
                            for name in loss_window[0]
                        },
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "target_mask_fraction_cumulative": target_voxels / max(eligible_voxels, 1),
                    }
                    append_jsonl(output / "loss_trace.jsonl", row)
                    loss_window.clear()

                if global_step % args.checkpoint_every == 0 or position + 1 == len(refs):
                    checkpoint(refs, position + 1)
                if global_step % latent_export_every == 0:
                    export_latents(global_step, epoch)
                if args.stop_after_updates is not None and global_step >= args.stop_after_updates:
                    checkpoint(refs, position + 1)
                    reloaded = torch_load(checkpoint_path, args.device)
                    checkpoint_reload_valid = bool(
                        reloaded.get("schema_version")
                        == "p11-paired-degrade-jepa-checkpoint-v1"
                        and reloaded.get("source_contract") == sources
                        and reloaded.get("frozen_execution") == frozen
                        and reloaded.get("data_contract") == data_contract
                        and int(reloaded.get("global_step", -1)) == global_step
                        and int(reloaded["resume"].get("cursor", -1)) == position + 1
                    )
                    if checkpoint_reload_valid:
                        validate_resume_state(reloaded["resume"], refs)
                        model.load_state_dict(reloaded["model_state"])
                        optimizer.load_state_dict(reloaded["optimizer_state"])
                        scheduler.load_state_dict(reloaded["scheduler_state"])
                    export_latents(global_step, epoch)
                    registered_exports = [
                        validate_latent_export(
                            output / "latent_exports" / f"step_{step:09d}.npz",
                            output / "latent_exports" / f"step_{step:09d}.json",
                            step,
                        )
                        for step in REGISTERED_TRAJECTORY_STEPS
                    ]
                    final_export = registered_exports[-1]
                    mask_fraction = target_voxels / max(eligible_voxels, 1)
                    technical = technical_canary_gates(
                        finite_pre_parameters=finite_pre_parameters,
                        finite_post_parameters=module_parameters_finite(model),
                        finite_loss=finite_loss,
                        gradient_norm=gradient_norm,
                        mask_fraction=mask_fraction,
                        registered_mask_fraction_range=tuple(
                            float(value)
                            for value in mask_spec["registered_mask_fraction_range"]
                        ),
                        checkpoint_reload_valid=checkpoint_reload_valid,
                        latent_snapshot_valid=bool(
                            len(registered_exports) == len(REGISTERED_TRAJECTORY_STEPS)
                            and int(final_export["global_step"])
                            == max(REGISTERED_TRAJECTORY_STEPS)
                        ),
                    )
                    marker = {
                        "schema_version": "p11-jepa-technical-canary-v1",
                        "created_utc": utc_now(),
                        "arm": args.arm,
                        "global_step": global_step,
                        "epoch": epoch,
                        "cursor": position + 1,
                        "finite_loss": finite_loss,
                        "gradient_norm": gradient_norm,
                        "target_mask_fraction_cumulative": mask_fraction,
                        "registered_mask_fraction_range": mask_spec[
                            "registered_mask_fraction_range"
                        ],
                        "teacher_gradient_free": True,
                        "data_contract_aggregate_sha256": data_contract[
                            "aggregate_sha256"
                        ],
                        "checkpoint": str(checkpoint_path),
                        "registered_latent_exports": registered_exports,
                        "final_latent_snapshot": final_export,
                        "ph001_opened": False,
                        **technical,
                    }
                    atomic_json(output / "TECHNICAL_CANARY_COMPLETE.json", marker)
                    print(json.dumps(marker, indent=2), flush=True)
                    if not marker["pass"]:
                        raise RuntimeError("P11 technical canary gates failed")
                    return

            cursor = len(refs)
            if any(accumulator.patches == 0 for accumulator in target_accumulators.values()):
                raise RuntimeError("complete JEPA epoch omitted a training phase")
            if sum(accumulator.patches for accumulator in target_accumulators.values()) != len(refs):
                raise RuntimeError("complete JEPA epoch patch accounting mismatch")
            if (
                args.max_runtime_seconds is not None
                and time.monotonic() - started
                >= args.max_runtime_seconds - args.validation_reserve_seconds
            ):
                pause(refs, cursor, "validation deferred to next interactive allocation")

            # Production comparison always evaluates the unmasked final-view student.
            val_parent, val_scaled, failures = unet_impl.predict_fold(
                model.student,
                loader.field_adapter(validation_phase).base,
                validation_core,
                final_normalization,
                args.device,
            )
            val_eigen = increments_to_eigenvalues(
                unscale_increments(val_scaled, scaler)
            ).astype(np.float32)
            assignment = np.load(validation_assignment_path, mmap_mode="r")
            truth = loader.targets_by_parent(validation_phase)
            report = evaluate_complete_phase(
                parent_node_id=val_parent,
                predicted_eigenvalues=val_eigen,
                truth_by_parent=truth,
                assignment=assignment,
                phase=validation_phase,
                runtime={
                    "epoch": epoch,
                    "global_step": global_step,
                    "patch_failures": failures,
                    "arm": args.arm,
                    "student_view": "V_final",
                    "teacher_used_at_inference": False,
                    "maximum_cuda_memory_bytes": maximum_memory,
                },
            )
            assignment.close()
            val_row_loss = np.mean(
                (
                    np.asarray(val_scaled, dtype=np.float64)
                    - np.asarray(validation_runtime.target_scaled[val_parent], dtype=np.float64)
                )
                ** 2,
                axis=1,
            )
            score = float(report["primary_macro_r2_lambda1"])
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_sha256": epoch_hash(refs),
                "complete_epoch_coverage": True,
                "phase_patches": {
                    phase: accumulator.patches
                    for phase, accumulator in target_accumulators.items()
                },
                "phase_weighted_target_mse": {
                    phase: accumulator.mean
                    for phase, accumulator in target_accumulators.items()
                },
                "auxiliary_loss_mean": {
                    name: value / max(aux_steps, 1) for name, value in aux_sums.items()
                },
                "target_mask_fraction": target_voxels / max(eligible_voxels, 1),
                "validation_all_rows_scaled_mse": float(np.mean(val_row_loss)),
                "primary_macro_r2_lambda1": score,
                "diagnostic_first_three_shell_macro_r2_lambda1": report[
                    "diagnostic_first_three_shell_macro_r2_lambda1"
                ],
                "worst_shell_r2_lambda1": report["worst_shell_r2_lambda1"],
                "per_shell_lambda1_r2": {
                    name: report["per_shell"][name]["lambda1"]["r2"]
                    for name in SHELL_NAMES
                },
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "invocation_elapsed_seconds": time.monotonic() - started,
            }
            history.append(epoch_row)
            append_jsonl(output / "epoch_history.jsonl", epoch_row)
            print(json.dumps(epoch_row), flush=True)
            export_latents(global_step, epoch)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_payload = {
                    "schema_version": "p11-jepa-best-student-v1",
                    "student_state_dict": copy.deepcopy(model.student.state_dict()),
                    "arm": args.arm,
                    "seed": args.seed,
                    "epoch": epoch,
                    "global_step": global_step,
                    "score": score,
                    "scaler": scaler,
                    "final_normalization": final_normalization,
                    "source_contract": sources,
                    "teacher_is_training_only": args.arm == "jepa",
                }
                if args.arm == "jepa":
                    best_payload.update(
                        teacher_state_dict=copy.deepcopy(model.teacher.state_dict()),
                        predictor_state_dict=copy.deepcopy(model.predictors.state_dict()),
                    )
                atomic_torch_save(best_payload, output / "best_checkpoint.pt")
                np.save(output / "best_validation_parent_node_id.npy", val_parent)
                np.save(output / "best_validation_eigenvalues.npy", val_eigen)
                atomic_json(output / "best_validation_report.json", report)

            if improved(score, early_best, float(optimization["minimum_macro_r2_delta"])):
                early_best = score
                stale_epochs = 0
            else:
                stale_epochs += 1
            stopped = should_stop(
                epoch=epoch,
                stale_epochs=stale_epochs,
                min_epochs=int(optimization["minimum_epochs"]),
                patience=int(optimization["patience"]),
            )
            epoch += 1
            cursor = 0
            target_accumulators = {phase: EpochLossAccumulator() for phase in phases}
            aux_sums = {name: 0.0 for name in aux_sums}
            aux_steps = 0
            target_voxels = 0
            eligible_voxels = 0
            if epoch <= int(optimization["epochs"]):
                checkpoint(loader.training_epoch(seed=args.seed, epoch=epoch), 0)
            if stopped:
                break

        final_epoch = history[-1]["epoch"] if history else 0
        final = {
            **run_manifest,
            "completed_utc": utc_now(),
            "epochs_completed": int(final_epoch),
            "global_steps": int(global_step),
            "best_epoch": int(best_epoch),
            "best_primary_macro_r2_lambda1": float(best_score),
            "history": history,
            "student_only_deployment": True,
            "ph001_opened": False,
            "status": (
                "CONVERGED_EARLY_STOP"
                if final_epoch < int(optimization["epochs"])
                else "TRAINING_COMPLETE"
            ),
        }
        atomic_json(output / "p11_matched_arm_summary.json", final)
        atomic_json(complete_marker, final)
        paused = output / "ALLOCATION_PAUSED.json"
        if paused.exists():
            paused.unlink()
        print(json.dumps(final, indent=2), flush=True)
    finally:
        for adapter in loader._p11_field.values():
            adapter.close()
            adapter.base.close()
        lock.close()


if __name__ == "__main__":
    main()
