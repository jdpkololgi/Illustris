#!/usr/bin/env python3
"""Freeze and validate the fail-closed P12-F3-D2 experiment contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import h5py
import json
import numpy as np
from pathlib import Path
import subprocess
from typing import Any

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_train_conditional_gaussian import split_selected
from workflows.sbi.p12f3_train_conditional_generative import load_config as load_parent_config
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_d2_diffusion_v1.json"
DEFAULT_OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1"
)
CONFIG_SCHEMA = "p12f3-d2-diffusion-v1"
CONTRACT_SCHEMA = "p12f3-d2-contract-frozen-v1"
RUN_SCHEMA = "p12f3-d2-run-v1"
CHECKPOINT_SCHEMA = "p12f3-d2-checkpoint-v1"
CANARY_SCHEMA = "p12f3-d2-canary-complete-v1"
SELECTION_SCHEMA = "p12f3-d2-funnel-selection-v1"
TRAINED_SCHEMA = "p12f3-d2-trained-v1"
MODEL_SOURCE_FIELDS = (
    "counts",
    "exposure_apodized",
    "log_count_ratio",
    "distance_to_support_boundary",
)
EXACT_SUPPORT_FIELD = "support_random"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _assert_no_blind_source(sources: dict[str, Any]) -> None:
    for name, value in sources.items():
        if "ph001" in str(value).lower():
            raise PermissionError(f"D2 source {name} points at sealed ph001")


def load_d2_config(path: Path = DEFAULT_CONFIG) -> tuple[dict, dict, Path]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("unsupported D2 configuration schema")
    roles = config.get("roles", {})
    expected_training = ["ph000", "ph002", "ph003", "ph004", "ph005"]
    if (
        roles.get("training") != expected_training
        or roles.get("validation") != "ph006"
        or roles.get("sealed_blind_test") != "ph001"
        or config.get("scope", {}).get("ph001_opened")
    ):
        raise PermissionError("D2 phase roles differ from the registered contract")
    _assert_no_blind_source(config.get("sources", {}))
    matched = config.get("matched_contract", {})
    if (
        matched.get("condition_channels_count") != 7
        or matched.get("additional_smoothing") is not False
        or matched.get("voxel_mpc_h") != 5.0
        or matched.get("training_cores_per_phase_before_internal_split") != 512
        or matched.get("ph006_panel_cores") != 256
        or matched.get("ph006_draws") != 64
        or matched.get("mask_only_metadata")
        != "support_random exact M; never concatenated as an eighth learned condition channel"
    ):
        raise RuntimeError("D2 changed a matched F3-L2d target/condition/evaluation unit")
    diffusion = config.get("diffusion", {})
    if (
        diffusion.get("path") != "variance_preserving_cosine"
        or diffusion.get("prediction") != "v"
        or diffusion.get("periodic_padding") is not False
        or diffusion.get("classifier_free_guidance") is not False
        or "channel_layernorm" not in diffusion.get("residual_blocks", "")
        or "GroupNorm" not in diffusion.get("normalization_safety", "")
        or float(diffusion.get("ema_decay", -1)) != 0.999
        or not diffusion.get("ema_warmup")
    ):
        raise RuntimeError("D2 diffusion or patch-safety contract changed")
    reproducibility = config.get("reproducibility", {})
    expected_reproducibility = {
        "cuda_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "float32_matmul_precision": "highest",
        "numerical_replay_absolute_tolerance": 1.0e-6,
        "resume_claim_policy": (
            "state serialization is exact; post-update numerical replay is claimed only "
            "after the registered one-GPU smoke passes and is not generalized beyond "
            "the tested topology"
        ),
    }
    if reproducibility != expected_reproducibility:
        raise RuntimeError("D2 one-GPU deterministic replay policy changed")
    arms = config.get("arms", {})
    if set(arms) != {"modern_base4", "modern_base8", "modern_base8_attention"}:
        raise RuntimeError("D2 bounded architecture arms changed")
    if arms["modern_base4"] != {
        "base": 4,
        "time_channels": 64,
        "coarse_attention": False,
        "attention_heads": 4,
    }:
        raise RuntimeError("D2 base4 control changed")
    if arms["modern_base8"]["coarse_attention"]:
        raise RuntimeError("D2 capacity arm must not bundle attention")
    if arms["modern_base8"] != {
        "base": 8,
        "time_channels": 64,
        "coarse_attention": False,
        "attention_heads": 4,
    }:
        raise RuntimeError("D2 base8 capacity control changed")
    if not arms["modern_base8_attention"]["coarse_attention"]:
        raise RuntimeError("D2 attention arm is not identifiable")
    if arms["modern_base8_attention"] != {
        "base": 8,
        "time_channels": 64,
        "coarse_attention": True,
        "attention_heads": 4,
        "attention_support_metadata": "support_random",
        "attention_support_rule": (
            "exact random-derived binary support; apodized exposure is forbidden "
            "as an attention mask"
        ),
    }:
        raise RuntimeError("D2 exact-support attention contract changed")
    funnel = config.get("funnel", {})
    accumulation = int(funnel.get("gradient_accumulation_steps", 0))
    canary_presentations = int(funnel.get("canary_presentations", 0))
    science_presentations = int(funnel.get("science_total_presentations", 0))
    if (
        canary_presentations != 2_500
        or science_presentations != 12_500
        or accumulation != 2
        or canary_presentations % accumulation
        or science_presentations % accumulation
        or not funnel.get("no_update_extension")
        or int(funnel.get("maximum_programme_presentations_including_replication", 0))
        != 30_000
        or int(funnel.get("internal_selection_cores", 0)) != 128
        or int(funnel.get("internal_confirmation_cores", 0)) != 127
        or int(funnel.get("internal_sample_draws", 0)) != 32
        or int(funnel.get("internal_sample_draw_batch", 0)) != 4
        or int(funnel.get("internal_sample_network_evaluations", 0)) != 50
        or float(funnel.get("learning_rate", -1)) != 2.0e-4
        or float(funnel.get("weight_decay", -1)) != 1.0e-4
        or float(funnel.get("gradient_clip", -1)) != 5.0
        or int(funnel.get("seed", -1)) != 42
        or int(funnel.get("replication_seed", -1)) != 314159
        or list(funnel.get("internal_sample_milestone_presentations", ()))
        != [2500, 5000, 7500, 10000, 12500]
        or float(funnel.get("capacity_energy_relative_improvement_required", -1))
        != 0.01
        or float(funnel.get("attention_energy_relative_improvement_required", -1))
        != 0.01
        or not funnel.get("earliest_checkpoint_within_one_standard_error")
        or funnel.get("internal_confirmation_policy")
        != (
            "before winner continuation, open the 127 cores once and repeat each "
            "already-frozen sequential arm contrast with the arm-specific raw_or_ema "
            "choices and common sample seeds; apply the identical paired energy and "
            "feasibility rule; any contradicted decision closes D2 and never promotes "
            "a runner-up"
        )
    ):
        raise RuntimeError("D2 hard compute cap changed")
    maximum_funnel_presentations = (
        2 * canary_presentations
        + canary_presentations
        + (science_presentations - canary_presentations)
        + science_presentations
    )
    if maximum_funnel_presentations != int(
        funnel["maximum_programme_presentations_including_replication"]
    ):
        raise RuntimeError("D2 arm/continuation/replication budget does not close")
    ladder = config.get("sampler", {}).get(
        "deterministic_ladder_network_evaluations"
    )
    if ladder != [50, 100]:
        raise RuntimeError("D2 sampler ladder changed")
    if config["sampler"]["primary"] != {
        "type": "ddim",
        "eta": 0.0,
        "network_evaluations": 100,
    }:
        raise RuntimeError("D2 primary sampler changed")
    if int(config["sampler"].get("draw_batch", -1)) != 4:
        raise RuntimeError("D2 sampler draw batch changed")
    evaluation = config.get("evaluation", {})
    if evaluation != {
        "common_evaluator_seed": 42,
        "higher_order_seed": 20260904,
        "paired_bootstrap_repeats": 5000,
        "paired_bootstrap_seed": 20260904,
        "deployable_conditioning_gates": [
            "shell",
            "random_response",
            "boundary_distance",
            "tracer_density",
            "frozen_g1_mean_scaled",
            "frozen_g1_log_std",
            "frozen_g1_traceless_shear_amplitude",
        ],
        "conditional_coverage_gate_scope": (
            "maximum 68/90 percent component coverage error across supported voxel "
            "delta and galaxy-sampled lambda1/lambda2/lambda3/gap12/gap23 for every "
            "deployable conditioning variable; true environment is descriptive only"
        ),
        "matched_reference_methods": ["g1", "f3l2b", "f3l2d_nfe100"],
        "matched_reference_policy": (
            "recompute every frozen reference archive and D2 with the same "
            "p12f_common_evaluator seed and core_joint_scores subsets before "
            "any paired proper-score gate"
        ),
    }:
        raise RuntimeError("D2 matched evaluation seed/reference contract changed")
    gates = config.get("ph006_gate", {})
    expected_gate_values = {
        "low_band_power_ratio_absolute_tolerance": 0.10,
        "joint_tarp_maximum": 0.05,
        "five_shear_tarp_maximum": 0.05,
        "five_shear_marginal_coverage_error_maximum": 0.05,
        "global_coverage_error_maximum": 0.05,
        "deployable_proxy_conditional_coverage_error_maximum": 0.10,
        "proper_score_worsening_maximum": 0.01,
        "primary_paired_energy_relative_improvement_over_f3l2b_minimum": 0.02,
        "f3l2d_nfe100_proper_score_worsening_maximum": 0.01,
    }
    if any(float(gates.get(name, -1)) != value for name, value in expected_gate_values.items()):
        raise RuntimeError("D2 ph006 numerical gates changed")
    for name in (
        "positive_paired_core_energy_improvement_over_g1_required",
        "positive_paired_core_energy_improvement_over_f3l2b_required",
        "paired_core_interval_excludes_zero",
        "finite_non_degenerate_required",
        "physics_trace_and_order_closure_required",
        "selected_arm_second_seed_required",
    ):
        if gates.get(name) is not True:
            raise RuntimeError(f"D2 ph006 gate disabled: {name}")
    if (
        gates.get("phase_sensitive_bispectrum_proxy_required") is not False
        or gates.get("phase_sensitive_bispectrum_proxy_role")
        != "descriptive_only_no_promotion_gate"
    ):
        raise RuntimeError("D2 higher-order diagnostic was incorrectly made a gate")
    if not config.get("scope", {}).get("off_p12a_v1_critical_path"):
        raise RuntimeError("D2 was moved back onto the P12-A production spine")

    parent_path = _repo_path(config["sources"]["parent_config"])
    parent, grandparent, _ = load_parent_config(parent_path)
    if parent.get("schema_version") != "p12f3-conditional-calibration-v1":
        raise RuntimeError("D2 parent is not the frozen F3-L2c/L2d contract")
    if tuple(parent["target"]["band_edges_h_mpc"]) != tuple(
        matched["band_edges_h_mpc"]
    ):
        raise RuntimeError("D2 Fourier bands differ from F3-L2d")
    expected_tensor_channels = [
        "counts_normalized",
        "density_proxy_from_log_count_ratio",
        "exposure_apodized",
        "frozen_g1_mean_scaled",
        "frozen_g1_log_std",
        "frozen_g1_traceless_shear_amplitude",
        "random_support_boundary_distance_clipped_0_120_mpc_h",
    ]
    if matched["condition_channels"] != expected_tensor_channels:
        raise RuntimeError("D2 seven-channel conditioner differs from F3-L2d")
    if set(parent["proxy_contract"]["base_channels"]) != {
        "counts", "exposure_apodized", "log_count_ratio"
    } or parent["proxy_contract"]["deployable_added_channels"] != expected_tensor_channels[3:]:
        raise RuntimeError("D2 source fields differ from the F3-L2d conditioner")
    return config, parent, parent_path


def arm_config(config: dict, arm: str) -> dict:
    if arm not in config["arms"]:
        raise ValueError(f"unknown frozen D2 arm {arm}")
    return dict(config["arms"][arm])


def split_internal_refs(
    internal: dict[str, list[int]],
    phases: tuple[str, ...],
    *,
    seed: int,
    selection_count: int,
    confirmation_count: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Deterministically randomize within phase, then phase-interleave 128/127."""
    shuffled: dict[str, list[int]] = {}
    for phase_index, phase in enumerate(phases):
        values = np.asarray(internal[phase], dtype=np.int64).copy()
        np.random.default_rng(int(seed) + 104729 * (phase_index + 1)).shuffle(values)
        shuffled[phase] = values.astype(int).tolist()
    interleaved: list[tuple[str, int]] = []
    maximum = max(len(shuffled[phase]) for phase in phases)
    for index in range(maximum):
        for phase in phases:
            if index < len(shuffled[phase]):
                interleaved.append((phase, int(shuffled[phase][index])))
    if len(interleaved) != selection_count + confirmation_count:
        raise RuntimeError("D2 internal 128/127 core contract changed")
    return interleaved[:selection_count], interleaved[selection_count:]


def _safe_marker(path: Path, *, schema: str, required_method: str | None = None) -> dict:
    marker = json.loads(path.read_text())
    if (
        marker.get("schema_version") != schema
        or not marker.get("pass")
        or marker.get("ph001_opened")
    ):
        raise RuntimeError(f"unsafe D2 parent marker {path}")
    if required_method is not None and marker.get("method") != required_method:
        raise RuntimeError(f"unexpected method in D2 parent marker {path}")
    return marker


def _same_patch_geometry(left: Any, right: Any) -> bool:
    for name in ("core_id", "fold", "cap"):
        if int(getattr(left, name)) != int(getattr(right, name)):
            return False
    for name in (
        "context_start",
        "context_stop",
        "core_start",
        "core_stop",
        "authoritative_parent_id",
        "authoritative_frac_index_local",
    ):
        if not np.array_equal(
            np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        ):
            return False
    return left.core_slice == right.core_slice


def _freeze_exact_support_contract(
    loader: Any,
    visible_phases: tuple[str, ...],
    selected: dict[str, list[int]],
    *,
    halo: int,
    alignment: int,
) -> dict[str, Any]:
    """Audit and freeze exact P3b-R support without making it a model channel."""
    result: dict[str, Any] = {}
    for phase in visible_phases:
        if phase == "ph001":
            raise PermissionError("D2 exact-support audit refuses sealed ph001")
        adapter = loader.field_adapter(phase)
        manifest_path = adapter.root / "adapter_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != "p3br-r1-field-patch-adapter-v1"
            or not manifest.get("pass")
            or manifest.get("ph001_opened")
            or tuple(manifest.get("channel_order", ()))
            != ("counts", "exposure_apodized", "log_count_ratio")
        ):
            raise RuntimeError(f"unsafe P3b-R adapter for {phase}")
        p3_manifest_path = Path(manifest["p3_manifest"])
        if sha256(p3_manifest_path) != manifest.get("p3_manifest_sha256"):
            raise RuntimeError(f"P3b-R manifest hash changed for {phase}")
        phase_record: dict[str, Any] = {
            "adapter_manifest": str(manifest_path.resolve()),
            "adapter_manifest_sha256": sha256(manifest_path),
            "p3_manifest": str(p3_manifest_path.resolve()),
            "p3_manifest_sha256": manifest["p3_manifest_sha256"],
            "caps": {},
        }
        selected_ids = np.asarray(selected[phase], dtype=np.int64)
        for cap_name, cap_id in (("SGC", 0), ("NGC", 1)):
            cap = manifest["caps"][cap_name]
            field_path = Path(cap["field_path"])
            if not field_path.is_file() or len(str(cap.get("field_sha256", ""))) != 64:
                raise RuntimeError(f"missing registered response artifact for {phase}/{cap_name}")
            stat = field_path.stat()
            with h5py.File(field_path, "r") as handle:
                if not set(MODEL_SOURCE_FIELDS + (EXACT_SUPPORT_FIELD,)).issubset(handle):
                    raise RuntimeError(f"P3b-R response schema incomplete for {phase}/{cap_name}")
                support_dataset = handle[EXACT_SUPPORT_FIELD]
                reference_dataset = handle["counts"]
                if (
                    tuple(support_dataset.shape) != tuple(reference_dataset.shape)
                    or tuple(support_dataset.shape) != tuple(cap["shape"])
                    or support_dataset.dtype != np.dtype("uint8")
                    or support_dataset.attrs.get("units") != "indicator"
                ):
                    raise RuntimeError(f"P3b-R exact support schema changed for {phase}/{cap_name}")
            candidates = selected_ids[np.asarray(adapter.core_cap[selected_ids]) == cap_id]
            if not len(candidates):
                raise RuntimeError(f"D2 selected panel has no {phase}/{cap_name} support audit core")
            audit_core = int(candidates[0])
            model_patch = adapter.extract(
                audit_core,
                halo,
                MODEL_SOURCE_FIELDS,
                alignment_voxels=alignment,
            )
            support_patch = adapter.extract(
                audit_core,
                halo,
                (EXACT_SUPPORT_FIELD,),
                alignment_voxels=alignment,
            )
            support = np.asarray(support_patch.values[0])
            if (
                not _same_patch_geometry(model_patch, support_patch)
                or support.shape != model_patch.values.shape[1:]
                or not np.all(np.isfinite(support))
                or not np.all((support == 0) | (support == 1))
                or not np.any(support)
            ):
                raise RuntimeError(f"D2 support/model geometry parity failed for {phase}/{cap_name}")
            phase_record["caps"][cap_name] = {
                "field_path": str(field_path.resolve()),
                "registered_field_sha256": cap["field_sha256"],
                "field_size_bytes": int(stat.st_size),
                "field_mtime_ns": int(stat.st_mtime_ns),
                "shape": list(map(int, cap["shape"])),
                "support_dtype": "uint8",
                "support_units": "indicator",
                "geometry_parity_audit_core": audit_core,
                "geometry_parity_pass": True,
            }
        result[phase] = phase_record
    return result


def _validate_exact_support_artifacts(frozen: dict[str, Any]) -> None:
    support_contract = frozen.get("exact_support_contract", {})
    if set(support_contract) != set(frozen.get("training_phases", ())) | {"ph006"}:
        raise RuntimeError("D2 exact-support phase contract changed")
    for phase, phase_record in support_contract.items():
        if phase == "ph001":
            raise PermissionError("D2 frozen support contract references sealed ph001")
        for cap_name, cap in phase_record["caps"].items():
            field_path = Path(cap["field_path"])
            stat = field_path.stat()
            if (
                stat.st_size != int(cap["field_size_bytes"])
                or stat.st_mtime_ns != int(cap["field_mtime_ns"])
            ):
                raise RuntimeError(f"D2 response artifact changed: {phase}/{cap_name}")
            with h5py.File(field_path, "r") as handle:
                support = handle[EXACT_SUPPORT_FIELD]
                if (
                    tuple(support.shape) != tuple(cap["shape"])
                    or support.dtype != np.dtype(cap["support_dtype"])
                    or support.attrs.get("units") != cap["support_units"]
                ):
                    raise RuntimeError(f"D2 exact support dataset changed: {phase}/{cap_name}")


def _freeze_field_target_contract(
    phase_root: Path, visible_phases: tuple[str, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in visible_phases:
        if phase == "ph001":
            raise PermissionError("D2 target contract refuses sealed ph001")
        marker_path = phase_root / phase / "p12f_field_targets_v1/FIELD_TARGET_READY.json"
        marker = json.loads(marker_path.read_text())
        if (
            not marker.get("pass")
            or marker.get("ph001_opened")
            or marker.get("contract", {}).get("double_smoothing_applied") is not False
            or marker.get("contract", {}).get("target")
            != "delta_R7 = lambda1 + lambda2 + lambda3"
        ):
            raise RuntimeError(f"unsafe D2 field target marker for {phase}")
        components = {}
        for cap_name in ("SGC", "NGC"):
            component = marker["components"][cap_name]
            target_path = Path(component["file"])
            stat = target_path.stat()
            if (
                len(str(component.get("file_sha256", ""))) != 64
                or stat.st_size != int(component["file_bytes"])
                or not component.get("target", {}).get("all_finite")
            ):
                raise RuntimeError(f"unsafe D2 field target for {phase}/{cap_name}")
            components[cap_name] = {
                "path": str(target_path.resolve()),
                "registered_sha256": component["file_sha256"],
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "shape": list(map(int, component["grid"]["shape"])),
            }
        result[phase] = {
            "marker": str(marker_path.resolve()),
            "marker_sha256": sha256(marker_path),
            "components": components,
        }
    return result


def _validate_field_target_artifacts(frozen: dict[str, Any]) -> None:
    for phase, phase_record in frozen.get("field_target_contract", {}).items():
        if phase == "ph001":
            raise PermissionError("D2 frozen target contract references sealed ph001")
        for cap_name, component in phase_record["components"].items():
            stat = Path(component["path"]).stat()
            if (
                stat.st_size != int(component["size_bytes"])
                or stat.st_mtime_ns != int(component["mtime_ns"])
            ):
                raise RuntimeError(f"D2 target artifact changed: {phase}/{cap_name}")


def _freeze_reference_contract(
    config: dict, validation_core_ids: list[int]
) -> dict[str, Any]:
    result = {}
    expected_ids = [int(value) for value in validation_core_ids]
    for key in ("g1", "f3l2b", "f3l2d_nfe100"):
        archive_path = Path(config["sources"][f"{key}_reference_archive"])
        report_path = Path(config["sources"][f"{key}_reference_report"])
        archive = json.loads(archive_path.read_text())
        report = json.loads(report_path.read_text())
        found_ids = [int(row["core_id"]) for row in archive.get("entries", ())]
        if (
            archive.get("schema_version") != "p12f-sample-archive-v1"
            or archive.get("phase") != "ph006"
            or int(archive.get("draws", -1)) != 64
            or archive.get("ph001_opened")
            or found_ids != expected_ids
            or report.get("phase") != "ph006"
            or int(report.get("cores", -1)) != 256
            or report.get("ph001_opened")
        ):
            raise RuntimeError(f"unsafe or unmatched D2 {key} reference")
        result[key] = {
            "archive": str(archive_path.resolve()),
            "archive_sha256": sha256(archive_path),
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "method": archive["method"],
            "core_ids": expected_ids,
        }
    return result


def build_frozen_contract(config_path: Path, output_root: Path) -> dict:
    config, parent, parent_path = load_d2_config(config_path)
    _, f3_parent, f3_parent_path = load_parent_config(parent_path)
    conditional_root = Path(config["sources"]["conditional_output_root"])
    gaussian_root = (
        conditional_root
        / "gaussian"
        / config["sources"]["conditional_gaussian_arm"]
        / config["sources"]["conditional_gaussian_run"]
    )
    gaussian_marker_path = gaussian_root / "P12F3_CONDITIONAL_GAUSSIAN_TRAINED.json"
    gaussian_marker = _safe_marker(
        gaussian_marker_path, schema="p12f3-conditional-gaussian-trained-v1"
    )
    gaussian_checkpoint = Path(gaussian_marker["checkpoint"])
    if gaussian_marker.get("checkpoint_sha256") != sha256(gaussian_checkpoint):
        raise RuntimeError("conditional location/scale checkpoint hash changed")

    whitening_path = Path(config["sources"]["conditional_whitening"])
    whitening = _safe_marker(
        whitening_path, schema="p12f3-conditional-whitening-v1"
    )
    if whitening.get("validation_phase_used_for_fit"):
        raise RuntimeError("D2 whitening used the external validation phase")

    baseline_marker_path = Path(config["sources"]["f3l2d_trained_marker"])
    baseline = _safe_marker(
        baseline_marker_path,
        schema="p12f3-conditional-generative-trained-v1",
        required_method="diffusion",
    )
    baseline_checkpoint = Path(config["sources"]["f3l2d_checkpoint"])
    if (
        Path(baseline["checkpoint"]).resolve() != baseline_checkpoint.resolve()
        or baseline.get("checkpoint_sha256") != sha256(baseline_checkpoint)
    ):
        raise RuntimeError("frozen F3-L2d baseline checkpoint changed")

    sampler_path = _repo_path(config["sources"]["f3l2d_sampler_convergence"])
    sampler = json.loads(sampler_path.read_text())
    if (
        sampler.get("schema_version") != "p12f3-diffusion-sampler-convergence-v2"
        or sampler.get("ph001_opened")
        or not sampler.get("converged_at_nfe100")
    ):
        raise RuntimeError("unsafe F3-L2d sampler-convergence evidence")

    g1_sources = {
        name: Path(f3_parent["sources"][name])
        for name in (
            "g1_checkpoint",
            "g1_trained_marker",
            "g1_run_manifest",
            "g1_filter",
        )
    }
    g1_trained = json.loads(g1_sources["g1_trained_marker"].read_text())
    g1_run = json.loads(g1_sources["g1_run_manifest"].read_text())
    g1_filter = json.loads(g1_sources["g1_filter"].read_text())
    if (
        g1_trained.get("schema_version") != "p12f-matched-challenger-trained-v1"
        or not g1_trained.get("pass")
        or g1_trained.get("ph001_opened")
        or Path(g1_trained.get("checkpoint", "")).resolve()
        != g1_sources["g1_checkpoint"].resolve()
        or g1_trained.get("checkpoint_sha256")
        != sha256(g1_sources["g1_checkpoint"])
        or g1_run.get("schema_version") != "p12f-matched-challenger-run-v1"
        or g1_run.get("ph001_opened")
        or not isinstance(g1_run.get("frozen", {}).get("target_scaler"), dict)
        or g1_filter.get("schema_version") != "p12f-g1-radial-residual-filter-v2"
        or not g1_filter.get("pass")
        or g1_filter.get("ph001_opened")
    ):
        raise RuntimeError("unsafe or incomplete frozen G1 source contract")

    _, _, phases, validation, _, loader, store, selected = _open_common(f3_parent)
    try:
        if list(phases) != config["roles"]["training"]:
            raise RuntimeError("D2 inherited training phases changed")
        training, internal = split_selected(
            selected,
            phases,
            float(parent["training"]["internal_validation_fraction_per_phase"]),
            int(config["funnel"]["seed"]),
        )
        selection_refs, confirmation_refs = split_internal_refs(
            internal,
            phases,
            seed=int(config["funnel"]["seed"]),
            selection_count=int(config["funnel"]["internal_selection_cores"]),
            confirmation_count=int(config["funnel"]["internal_confirmation_cores"]),
        )
        exact_support_contract = _freeze_exact_support_contract(
            loader,
            phases + (validation,),
            selected,
            halo=int(f3_parent["patch"]["conditioning_halo_voxels"]),
            alignment=int(f3_parent["patch"]["alignment_voxels"]),
        )
        field_target_contract = _freeze_field_target_contract(
            Path(f3_parent["sources"]["phase_root"]), phases + (validation,)
        )
        panel_path = Path(parent["sources"]["source_panel"])
        panel = json.loads(panel_path.read_text())
        if (
            panel.get("phase") != "ph006"
            or panel.get("selection_uses_truth")
            or panel.get("ph001_opened")
            or len(panel.get("selected_core_id", ())) != 256
        ):
            raise RuntimeError("unsafe D2 ph006 evaluation panel")
        reference_contract = _freeze_reference_contract(
            config, panel["selected_core_id"]
        )
        internal_split_balance = {}
        for split_name, refs in (
            ("selection", selection_refs),
            ("confirmation", confirmation_refs),
        ):
            split_balance = {}
            for phase in phases:
                adapter = loader.field_adapter(phase)
                phase_refs = [core_id for ref_phase, core_id in refs if ref_phase == phase]
                caps = np.asarray(adapter.core_cap, dtype=np.int8)[phase_refs]
                split_balance[phase] = {
                    "cores": int(len(phase_refs)),
                    "SGC": int(np.count_nonzero(caps == 0)),
                    "NGC": int(np.count_nonzero(caps == 1)),
                }
            internal_split_balance[split_name] = split_balance
    finally:
        store.close()
        loader.close()
    before = {phase: len(selected[phase]) for phase in phases}
    if any(value != 512 for value in before.values()):
        raise RuntimeError("D2 inherited selected core count changed")
    examples_per_update = int(config["funnel"]["gradient_accumulation_steps"])
    training_count = sum(len(training[phase]) for phase in phases)
    science_presentations = int(config["funnel"]["science_total_presentations"])
    effective_passes = science_presentations / training_count
    if effective_passes > float(config["funnel"]["maximum_effective_training_passes"]):
        raise RuntimeError("D2 effective-pass budget exceeds its frozen cap")

    source_paths = {
        "config": config_path.resolve(),
        "parent_config": parent_path.resolve(),
        "f3_parent_config": f3_parent_path.resolve(),
        "conditional_gaussian_marker": gaussian_marker_path.resolve(),
        "conditional_gaussian_checkpoint": gaussian_checkpoint.resolve(),
        "conditional_whitening": whitening_path.resolve(),
        "f3l2d_marker": baseline_marker_path.resolve(),
        "f3l2d_checkpoint": baseline_checkpoint.resolve(),
        "f3l2d_sampler_convergence": sampler_path.resolve(),
        "d2_models": REPO_ROOT / "workflows/sbi/p12f3_d2_models.py",
        "d2_contract": Path(__file__).resolve(),
        "d2_trainer": REPO_ROOT / "workflows/sbi/p12f3_d2_train.py",
        "d2_selector": REPO_ROOT / "workflows/sbi/p12f3_d2_select.py",
        "d2_confirmation": REPO_ROOT / "workflows/sbi/p12f3_d2_confirm.py",
        "d2_exporter": REPO_ROOT / "workflows/sbi/p12f3_d2_export.py",
        "d2_evaluator": REPO_ROOT / "workflows/sbi/p12f3_d2_evaluate.py",
        "d2_reference_builder": REPO_ROOT
        / "workflows/sbi/p12f3_d2_build_references.py",
        "d2_transform_roundtrip": REPO_ROOT
        / "workflows/sbi/p12f3_d2_roundtrip.py",
        "d2_gpu_smoke": REPO_ROOT / "workflows/sbi/p12f3_d2_gpu_smoke.py",
        "d2_decision": REPO_ROOT / "workflows/sbi/p12f3_d2_decide.py",
        "d2_interactive_launcher": REPO_ROOT
        / "workflows/sbi/run_p12f3_d2_in_allocation.sh",
        "d2_slurm_launcher": REPO_ROOT
        / "workflows/sbi/submit_p12f3_d2_stage.slurm",
        "shared_conditional_models": REPO_ROOT
        / "workflows/sbi/p12f3_conditional_models.py",
        "shared_fourier_modes": REPO_ROOT / "workflows/sbi/p12f3_fourier_modes.py",
        "shared_lowmode_training": REPO_ROOT
        / "workflows/sbi/p12f3_train_lowmode_flow.py",
        "shared_fourier_training": REPO_ROOT
        / "workflows/sbi/p12f3_train_fourier_lowmode_flow.py",
        "shared_conditional_training": REPO_ROOT
        / "workflows/sbi/p12f3_train_conditional_generative.py",
        "shared_conditional_gaussian_training": REPO_ROOT
        / "workflows/sbi/p12f3_train_conditional_gaussian.py",
        "shared_response_loader": REPO_ROOT
        / "workflows/abacus_tweb/p3br_training_contract.py",
        "shared_patch_adapter": REPO_ROOT
        / "workflows/abacus_tweb/p6_field_patch_utils.py",
        "shared_hybrid_export": REPO_ROOT
        / "workflows/sbi/p12f3_export_hybrid_archive.py",
        "shared_gaussian_controls": REPO_ROOT
        / "workflows/sbi/p12f_gaussian_controls.py",
        "shared_common_evaluator": REPO_ROOT
        / "workflows/sbi/p12f_common_evaluator.py",
        "shared_shear_audit": REPO_ROOT / "workflows/sbi/p12f3l2_shear_audit.py",
        "shared_visual_analyzer": REPO_ROOT
        / "workflows/sbi/plot_p12f3_hierarchical_comparison.py",
        "shared_conditional_evaluator": REPO_ROOT
        / "workflows/sbi/p12f3_evaluate_conditional_archive.py",
        "shared_field_diagnostics": REPO_ROOT
        / "workflows/sbi/p12f_field_posterior_diagnostics.py",
        "shared_calibration_diagnostics": REPO_ROOT
        / "workflows/sbi/p12_calibration_diagnostics.py",
        "shared_dependency_evaluator": REPO_ROOT
        / "workflows/sbi/p12f_dependency_rescue_evaluator.py",
        "g1_checkpoint": g1_sources["g1_checkpoint"],
        "g1_trained_marker": g1_sources["g1_trained_marker"],
        "g1_run_manifest": g1_sources["g1_run_manifest"],
        "g1_filter": g1_sources["g1_filter"],
    }
    conditioning_root = Path(f3_parent["sources"]["conditioning_contract"])
    loader_ready_path = conditioning_root / "TRAINING_LOADER_READY.json"
    loader_ready = json.loads(loader_ready_path.read_text())
    field_transform_path = conditioning_root / "transforms/field/field_transform.json"
    if (
        not loader_ready.get("pass")
        or loader_ready.get("ph001_opened")
        or loader_ready.get("field_transform") != str(field_transform_path)
        or loader_ready.get("field_transform_sha256") != sha256(field_transform_path)
    ):
        raise RuntimeError("D2 response loader/normalization contract changed")
    source_paths["response_loader_ready"] = loader_ready_path
    source_paths["response_field_transform"] = field_transform_path
    source_paths["ph006_panel"] = panel_path
    source_paths.update(
        {
            f"response_adapter_{phase}": Path(record["adapter_manifest"])
            for phase, record in exact_support_contract.items()
        }
    )
    for key, record in reference_contract.items():
        source_paths[f"reference_archive_{key}"] = Path(record["archive"])
        source_paths[f"reference_report_{key}"] = Path(record["report"])
    source_paths.update(
        {
            f"field_target_marker_{phase}": Path(record["marker"])
            for phase, record in field_target_contract.items()
        }
    )
    source_paths.update(
        {
            f"response_manifest_{phase}": Path(record["p3_manifest"])
            for phase, record in exact_support_contract.items()
        }
    )
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    frozen = {
        "config": str(config_path.resolve()),
        "output_root": str(output_root.resolve()),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_hashes": source_hashes,
        "selected_core_ids": selected,
        "training_core_ids": training,
        "internal_validation_core_ids": internal,
        "internal_selection_refs": [
            {"phase": phase, "core_id": core_id} for phase, core_id in selection_refs
        ],
        "internal_confirmation_refs": [
            {"phase": phase, "core_id": core_id}
            for phase, core_id in confirmation_refs
        ],
        "internal_split_balance": internal_split_balance,
        "training_phases": list(phases),
        "selected_cores_per_phase": before,
        "training_examples": training_count,
        "examples_per_optimizer_update": examples_per_update,
        "canary_presentations": int(config["funnel"]["canary_presentations"]),
        "maximum_presentations": science_presentations,
        "maximum_optimizer_updates": science_presentations // examples_per_update,
        "maximum_effective_passes": float(effective_passes),
        "target_contract": config["matched_contract"],
        "exact_support_contract": exact_support_contract,
        "field_target_contract": field_target_contract,
        "reference_contract": reference_contract,
        "g1_target_scaler": g1_run["frozen"]["target_scaler"],
        "g1_target_scaler_digest": digest(g1_run["frozen"]["target_scaler"]),
        "sampler_contract": config["sampler"],
        "ph006_used_for_selection": False,
        "truth_arrays_read": [],
        "truth_metadata_read": [
            *[
                str(Path(record["marker"]).resolve())
                for record in field_target_contract.values()
            ],
            *[
                str(Path(record["report"]).resolve())
                for record in reference_contract.values()
            ],
        ],
        "ph001_opened": False,
    }
    return frozen


def validate_frozen_contract(marker_path: Path, config_path: Path) -> tuple[dict, dict]:
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("schema_version") != CONTRACT_SCHEMA
        or not marker.get("pass")
        or marker.get("ph001_opened")
        or marker.get("truth_arrays_read") != []
        or marker.get("frozen", {}).get("truth_arrays_read") != []
        or marker.get("truth_metadata_read")
        != marker.get("frozen", {}).get("truth_metadata_read")
    ):
        raise RuntimeError("unsafe D2 frozen contract marker")
    frozen = marker.get("frozen", {})
    if marker.get("frozen_digest") != digest(frozen):
        raise RuntimeError("D2 frozen contract digest changed")
    if marker.get("git_revision_at_freeze") != git_revision():
        raise RuntimeError("D2 execution Git revision differs from contract freeze")
    if frozen.get("config") != str(config_path.resolve()):
        raise RuntimeError("D2 config path differs from frozen contract")
    for name, path_value in frozen.get("source_paths", {}).items():
        path = Path(path_value)
        if frozen["source_hashes"].get(name) != sha256(path):
            raise RuntimeError(f"D2 frozen source changed: {name}")
    _validate_exact_support_artifacts(frozen)
    _validate_field_target_artifacts(frozen)
    config, _, _ = load_d2_config(config_path)
    return marker, config


def validate_output_root(
    contract: dict, output_root: Path, contract_path: Path
) -> Path:
    """Bind every D2 stage to the single root frozen by the contract.

    This is part of the hard 30k-presentation ledger: a valid contract may not
    be reused to create a second set of arm directories under another root.
    Requiring the canonical marker location also prevents an explicit
    ``--contract`` from silently decoupling evidence from its frozen tree.
    """
    actual = output_root.resolve()
    expected_value = contract.get("frozen", {}).get("output_root")
    if not expected_value or actual != Path(expected_value).resolve():
        raise RuntimeError("D2 runtime output root differs from frozen contract")
    expected_contract = actual / "D2_CONTRACT_FROZEN.json"
    if contract_path.resolve() != expected_contract.resolve():
        raise RuntimeError("D2 contract is not in its frozen output root")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    load_d2_config(args.config)
    if args.validate_only:
        print(json.dumps({"config": str(args.config), "pass": True}, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_root / "D2_CONTRACT_FROZEN.json"
    if marker_path.exists():
        marker, _ = validate_frozen_contract(marker_path, args.config)
        validate_output_root(marker, args.output_root, marker_path)
        print(json.dumps(marker, indent=2))
        return
    if any(args.output_root.iterdir()):
        raise RuntimeError("non-empty D2 root has no valid frozen contract")
    frozen = build_frozen_contract(args.config, args.output_root)
    marker = {
        "schema_version": CONTRACT_SCHEMA,
        "created_utc": utc_now(),
        "git_revision_at_freeze": git_revision(),
        "pass": True,
        "frozen": frozen,
        "frozen_digest": digest(frozen),
        "truth_arrays_read": [],
        "truth_metadata_read": frozen["truth_metadata_read"],
        "ph006_opened_for_d2": False,
        "ph001_opened": False,
    }
    atomic_json(marker_path, marker)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
