#!/usr/bin/env python3
"""Freeze the P12-A ph001 acceptance contract before blind truth is opened.

This module may read training-phase and ph006 evidence, but it refuses every
path containing ``ph001``.  In particular, the shell-conditioned web-class
climatology is fitted once from the frozen multi-phase training sample rather
than estimated from the eventual blind outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_phase_assets import expand_phase, load_registry
from workflows.sbi.p12a_immutable_io import write_json_exclusive
from workflows.sbi.p12_production_contract import (
    P12A_SCHEMA,
    assert_ph001_sealed_payload,
    assert_truth_free_payload,
)


SCHEMA = "p12a-blind-evaluation-contract-v2"
DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
DEFAULT_CANDIDATE = Path("docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json")
DEFAULT_GAUSSIAN = Path("docs/evidence/p12/production_aux_v1/P12A_GAUSSIAN_BASELINE.json")
DEFAULT_OUTPUT = Path("docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
BLIND_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/blind_predictions/ph001"
)
FROZEN_PYTHON = Path(
    "/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python"
).resolve()
CANONICAL_OUTPUTS = {
    "opened_marker": str(BLIND_ROOT / "P12_BLIND_OPENED.json"),
    "energy_score_array": str(BLIND_ROOT / "evaluation/P12A_PH001_ENERGY_SCORES.npz"),
    "energy_score_report": str(BLIND_ROOT / "evaluation/P12A_PH001_ENERGY_SCORE.json"),
    "evaluation_report": str(BLIND_ROOT / "evaluation/P12A_PH001_BLIND_EVALUATION.json"),
    "evaluation_figure": str(BLIND_ROOT / "evaluation/P12A_PH001_BLIND_EVALUATION.png"),
    "plot_manifest": str(BLIND_ROOT / "evaluation/P12A_PH001_BLIND_PLOTS.json"),
}
EVALUATION_PROTOCOL = {
    "posterior_draws": 512,
    "audit_rows": 50_000,
    "posterior_shard_seed_base": 42,
    "joint_energy_norm": "euclidean physical ordered-eigenvalue coordinates",
    "energy_pairing_offset": 257,
    "gaussian_ordering_transform": (
        "none; evaluate the frozen unconstrained shell/cap residual Gaussian exactly, "
        "including any unordered draws"
    ),
    "gaussian_draw_seed": 202609041,
    "bootstrap_seed": 202609042,
    "bootstrap_repetitions": 4000,
    "tarp_seed": 202609043,
    "tarp_repetitions": 20,
    "tarp_gate_aggregation": (
        "p90 of per-seed maximum deviation across 20 consecutive fixed seeds"
    ),
    "eigengap_tarp_seed_offset": 1,
    "rank_seed_offset": 2,
    "rank_repetitions": 1,
    "bootstrap_unit": "authoritative core",
}
IMPLEMENTATION_FILES = {
    "evaluation_contract": Path(__file__),
    "blind_evaluator": Path(__file__).with_name("p12a_evaluate_blind.py"),
    "proper_score_evaluator": Path(__file__).with_name("p12a_blind_proper_score.py"),
    "proper_score_primitives": Path(__file__).with_name(
        "p12a_blind_energy_score.py"
    ),
    "blind_plotter": Path(__file__).with_name("p12a_plot_blind_evaluation.py"),
    "one_open_guard": Path(__file__).with_name("p12a_open_blind.py"),
    "deep_preauthorization_audit": Path(__file__).with_name(
        "p12a_blind_preauthorization.py"
    ),
    "immutable_publication": Path(__file__).with_name("p12a_immutable_io.py"),
    "production_contract": Path(__file__).with_name("p12_production_contract.py"),
    "blind_shard_contract": Path(__file__).with_name("p12a_blind_shards.py"),
    "blind_array_worker": Path(__file__).with_name("p12a_blind_array_worker.py"),
    "blind_export_slurm": Path(__file__).with_name("submit_p12a_blind_export.slurm"),
    "blind_inference": Path(__file__).with_name("p12a_blind_inference.py"),
    "blind_classical_predictions": Path(__file__).with_name(
        "p12_blind_classical_predictions.py"
    ),
    "tarp_dependency": Path(__file__).with_name(
        "p12f_dependency_rescue_evaluator.py"
    ),
    "rank_dependency": Path(__file__).with_name(
        "p12f_field_posterior_diagnostics.py"
    ),
    "tarp_field_contract_dependency": Path(__file__).with_name(
        "p12f_challenger_common.py"
    ),
    "tarp_common_evaluator_dependency": Path(__file__).with_name(
        "p12f_common_evaluator.py"
    ),
    "posterior_sampling_dependency": Path(__file__).with_name(
        "p12_train_base_response_fmpe.py"
    ),
    "posterior_transform_dependency": Path(__file__).with_name(
        "p12_prepare_base_response_dataset.py"
    ),
    "checkpoint_loader_dependency": Path(__file__).parents[1]
    / "abacus_tweb/p8_train_patch_recovery.py",
    "blind_unet_dependency": Path(__file__).parents[1]
    / "abacus_tweb/p8_train_unet_patch.py",
    "blind_field_adapter_dependency": Path(__file__).parents[1]
    / "abacus_tweb/p6_field_patch_utils.py",
    "blind_unet_summary_dependency": Path(__file__).with_name(
        "p12_export_unet_summaries.py"
    ),
    "shared_deterministic_dependency": Path(__file__).parents[1]
    / "abacus_tweb/p8_deterministic_common.py",
    "postopen_chain": Path(__file__).with_name(
        "submit_p12a_ph001_postopen_chain.sh"
    ),
    "postopen_finalize_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_finalize.slurm"
    ),
    "postopen_score_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_energy_score.slurm"
    ),
    "postopen_evaluate_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_evaluate.slurm"
    ),
    "postopen_plot_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_plot.slurm"
    ),
}
SCIENTIFIC_CONTRACT_FILES = {
    "p12_production_blind_config": REPO_ROOT / "configs/p12_production_blind_v1.json",
}
FORBIDDEN_ARTIFACTS = {
    "p12a_strict_calibration_pass": Path(
        "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
        "p12a_base_response_v1/fmpe_seed42/P12A_CALIBRATION_PASS.json"
    ),
}
TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES = {
    "authorized_truth_wrapper": Path(__file__).with_name("p12a_authorized_truth.py"),
    "particle_b_restore_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_particle_b.slurm"
    ),
    "density_slurm": Path(__file__).with_name("submit_p12a_ph001_density.slurm"),
    "tweb_slurm": Path(__file__).with_name("submit_p12a_ph001_tweb.slurm"),
    "annotation_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_annotation.slurm"
    ),
    "compact_truth_slurm": Path(__file__).with_name(
        "submit_p12a_ph001_compact_truth.slurm"
    ),
    "dependency_chain": Path(__file__).with_name(
        "submit_p12a_ph001_truth_chain.sh"
    ),
    "particle_b_restore_physics": Path(__file__).parents[1]
    / "abacus_tweb/p10_stage_particle_b.py",
    "density_physics": Path(__file__).parents[1]
    / "abacus_tweb/p10_build_density_field.py",
    "tweb_physics": Path(__file__).parents[1] / "abacus_tweb/p10_run_tweb.py",
    "annotation_physics": Path(__file__).parents[1]
    / "abacus_tweb/annotate_cutsky_with_tweb_eigs.py",
    "phase_registry_loader": Path(__file__).parents[1]
    / "abacus_tweb/p10_phase_assets.py",
    "tweb_solver": Path(__file__).parents[1]
    / "abacus_tweb/abacus_process_particles2.py",
    "resource_guards": REPO_ROOT / "shared/resource_requirements.py",
    "path_configuration": REPO_ROOT / "config_paths.py",
    "immutable_publication": Path(__file__).with_name("p12a_immutable_io.py"),
}
TRUTH_FREE_IDENTITY_INPUTS = {
    "phase_registry": REPO_ROOT / "configs/p10_phase_registry_v1.json",
    "blind_parent_completion": Path(
        "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/catalogues/"
        "blind_parent/ph001_bgs_bright_parent_linkage.fits.complete.json"
    ),
    "p1_canonical_manifest": Path(
        "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/"
        "p1_canonical/manifest.json"
    ),
    "blind_parent_catalogue": Path(
        "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/catalogues/"
        "blind_parent/ph001_bgs_bright_parent_linkage.fits"
    ),
    "p1_canonical_index": Path(
        "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/"
        "p1_canonical/canonical_index.npz"
    ),
}
TRUTH_CONSTRUCTION_CONTRACT = {
    "phase": "ph001",
    "truth_root": "/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1",
    "ordinary_phase_tree_truth_writes_allowed": False,
    "phase_registry_mutation_allowed": False,
    "expected_supported_context_rows": 4_897_905,
    "density_grid_size": 2048,
    "density_box_size_mpc_h": 2000.0,
    "mass_assignment": "TSC",
    "particle_a_fraction": 0.03,
    "particle_b_fraction": 0.07,
    "tidal_smoothing_mpc_h": 7.0,
    "web_threshold": 0.2,
    "eigenvalue_order": "lambda1<=lambda2<=lambda3",
    "tweb_mpi_ranks": 16,
    "halo_position_field": "x_com",
    "compact_join": "frozen context parent_node_id -> P1 TARGETID -> annotated parent",
}
GATES = {
    "joint_eigenvalue_tarp_maximum": 0.05,
    "joint_eigengap_tarp_maximum": 0.05,
    "physical_rank_cdf_maximum": 0.05,
    "global_coverage_absolute_error_maximum": 0.03,
    "nonsparse_conditional_coverage_absolute_error_maximum": 0.06,
    "sparse_shell_green_absolute_error_maximum": 0.03,
    "sparse_shell_release_absolute_error_maximum": 0.06,
    "posterior_mean_lambda1_r2_delta_minimum": -0.02,
    "multiclass_brier_skill_minimum": 0.0,
    "gaussian_minus_fmpe_energy_score_ci95_lower_minimum": 0.0,
}
CONDITIONAL_STRATA = {
    "redshift_shell": "shell",
    "radial_selection_density_quartile": "ntilde_mpc3",
    "random_support_boundary_distance_quartile": "distance_to_support_boundary_mpc",
}
COVARIATE_DEFINITIONS = {
    "ntilde_mpc3": (
        "training-frozen BRIGHT radial selection density evaluated at galaxy redshift; "
        "not angular random density, fibre completeness, or redshift success"
    ),
    "distance_to_support_boundary_mpc": (
        "P3b-R random-support boundary distance stored in comoving Mpc; the frozen "
        "quality thresholds 10.3458469 and 20.6916938 Mpc correspond to 7 and 14 "
        "Mpc/h at h=0.6766"
    ),
    "quality_bit_legacy_response_outside_training_range": (
        "the existing production bit name is retained for binary compatibility, but its "
        "actual covariate is log(ntilde_mpc3); it must not be described as a random-"
        "completeness or full survey-response OOD flag"
    ),
    "quartile_binning": (
        "deterministic quartiles of the frozen truth-free ph001 covariate rows; no truth "
        "or post-open fit enters bin assignment"
    ),
}
RELEASE_POLICY = {
    "green": "all gates pass and sparse-shell coverage error <= 0.03",
    "amber": (
        "all non-sparse gates pass and 0.03 < sparse-shell coverage error <= 0.06; "
        "release requires the frozen high-redshift quality flag"
    ),
    "blocked": (
        "any global/non-sparse gate fails or sparse-shell coverage error > 0.06"
    ),
}
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "torch",
    "sbi",
    "tarp",
    "matplotlib",
    "fitsio",
    "asdf",
    "abacusutils",
    "mpi4py",
    "astropy",
    "numba",
)
RUNTIME_UNSET_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "LD_PRELOAD",
)


def validate_runtime_environment() -> None:
    contaminated = {
        name: os.environ[name]
        for name in RUNTIME_UNSET_VARIABLES
        if os.environ.get(name)
    }
    if contaminated:
        raise RuntimeError(
            "blind runtime contains mutable Python/library overrides: "
            + ", ".join(sorted(contaminated))
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
        raise RuntimeError("blind runtime must disable the Python user site")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _record(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def runtime_fingerprint() -> dict:
    """Return the exact interpreter and relevant installed distributions."""

    validate_runtime_environment()
    executable = Path(sys.executable).resolve()
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    all_distributions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip().lower()
        if name:
            all_distributions[name] = str(distribution.version)
    environment_root = executable.parent.parent
    conda_meta = environment_root / "conda-meta"
    conda_records = []
    if conda_meta.is_dir():
        for path in sorted(conda_meta.glob("*.json")):
            conda_records.append((path.name, sha256(path), path.stat().st_size))
    serialized_conda_records = json.dumps(conda_records, separators=(",", ":"))
    return {
        "python": _record(executable),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "distributions": versions,
        "all_installed_distributions": all_distributions,
        "conda_metadata": {
            "environment_root": str(environment_root),
            "record_count": len(conda_records),
            "records_sha256": hashlib.sha256(
                serialized_conda_records.encode()
            ).hexdigest(),
            "history": (
                _record(conda_meta / "history")
                if (conda_meta / "history").is_file()
                else None
            ),
        },
        "python_no_user_site": bool(sys.flags.no_user_site),
        "python_no_user_site_environment": os.environ.get("PYTHONNOUSERSITE"),
        "environment_variables_unset": list(RUNTIME_UNSET_VARIABLES),
    }


def validate_runtime_fingerprint(expected: dict) -> None:
    observed = runtime_fingerprint()
    if Path(expected.get("python", {}).get("path", "")).resolve() != FROZEN_PYTHON:
        raise RuntimeError("blind contract was not built with the frozen Python")
    if expected != observed:
        raise RuntimeError("P12-A Python/package runtime differs from the frozen contract")


def truth_source_contract(registry_path: Path) -> dict:
    """Freeze source locations without opening any truth-bearing payload."""

    registry = load_registry(registry_path)
    expanded = expand_phase(registry, "ph001")
    snapshot_root = Path(expanded["assets"]["snapshot_root"]).resolve()
    particle_b = dict(expanded["particle_b"])
    if particle_b.get("kind") != "hpss":
        raise RuntimeError("ph001 Particle-B source is no longer the registered HPSS archive")
    return {
        "phase_registry": _record(registry_path),
        "particle_a_root": str(snapshot_root),
        "particle_a_directories": ["field_rv_A", "halo_rv_A"],
        "particle_b": particle_b,
        "halo_info_root": str(snapshot_root / "halo_info"),
        "payload_content_may_be_read_only_after_authorization": True,
    }


def validate_nested_truth_free_identities() -> None:
    parent_marker = json.loads(
        TRUTH_FREE_IDENTITY_INPUTS["blind_parent_completion"].read_text()
    )
    parent_path = TRUTH_FREE_IDENTITY_INPUTS["blind_parent_catalogue"].resolve()
    parent_record = parent_marker.get("output", {})
    if (
        parent_marker.get("phase") != "ph001"
        or parent_marker.get("target_truth_present") is not False
        or Path(parent_record.get("path", "")).resolve() != parent_path
        or parent_record.get("sha256") != sha256(parent_path)
    ):
        raise PermissionError("blind parent nested identity is stale or truth-bearing")
    p1_manifest = json.loads(
        TRUTH_FREE_IDENTITY_INPUTS["p1_canonical_manifest"].read_text()
    )
    p1_index = TRUTH_FREE_IDENTITY_INPUTS["p1_canonical_index"].resolve()
    if (
        p1_manifest.get("phase") != "ph001"
        or p1_manifest.get("target_truth_present") is not False
        or p1_manifest.get("index_sha256") != sha256(p1_index)
    ):
        raise PermissionError("P1 nested identity is stale or truth-bearing")
    with np.load(p1_index, mmap_mode="r") as canonical:
        required = {
            "parent_node_id",
            "targetid",
            "cap",
            "shell",
            "active",
            "context",
            "valid_target",
        }
        if not required.issubset(canonical.files):
            raise RuntimeError("P1 canonical identity archive schema changed")
        parent = np.asarray(canonical["parent_node_id"], dtype=np.int64)
        targetid = np.asarray(canonical["targetid"], dtype=np.int64)
        if (
            not np.array_equal(parent, np.arange(len(parent), dtype=np.int64))
            or np.any(targetid <= 0)
            or len(np.unique(targetid)) != len(targetid)
            or int(p1_manifest.get("counts", {}).get("total", -1)) != len(parent)
        ):
            raise RuntimeError("P1 canonical parent/TARGETID identity is invalid")


def shell_class_climatology(
    truth: np.ndarray, shell: np.ndarray, weight: np.ndarray, threshold: float = 0.2
) -> dict[str, dict]:
    values = np.asarray(truth, dtype=np.float64)
    shell = np.asarray(shell, dtype=np.int8)
    weight = np.asarray(weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("training eigenvalues must have shape [rows,3]")
    if not (len(values) == len(shell) == len(weight)) or np.any(weight <= 0):
        raise ValueError("training climatology rows/weights are invalid")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values, axis=1) < 0):
        raise ValueError("training climatology truth is non-finite or unordered")
    web_class = np.sum(values > float(threshold), axis=1)
    result: dict[str, dict] = {}
    for value in range(4):
        chosen = shell == value
        if not np.any(chosen):
            raise RuntimeError(f"training climatology lacks shell {value}")
        mass = np.asarray(
            [np.sum(weight[chosen & (web_class == index)]) for index in range(4)],
            dtype=np.float64,
        )
        probability = mass / mass.sum()
        result[str(value)] = {
            "rows": int(np.count_nonzero(chosen)),
            "probability_void_sheet_filament_knot": probability.tolist(),
        }
    return result


def build_contract(
    *, candidate_path: Path, gaussian_path: Path, dataset_marker_path: Path
) -> dict:
    for path in (candidate_path, gaussian_path, dataset_marker_path):
        if "ph001" in str(path).lower():
            raise PermissionError("blind truth/path cannot enter evaluation-contract fitting")
    candidate = json.loads(candidate_path.read_text())
    assert_truth_free_payload(candidate)
    if candidate.get("schema_version") != P12A_SCHEMA or candidate.get("pass") is not True:
        raise RuntimeError("P12-A production candidate is not frozen")
    if candidate.get("strict_calibration_pass_marker_present") is not False:
        raise RuntimeError("P12-A candidate no longer records the strict calibration failure")
    for name, path in FORBIDDEN_ARTIFACTS.items():
        if path.exists():
            raise RuntimeError(f"forbidden post-fit artifact is present: {name}: {path}")
    dataset = json.loads(dataset_marker_path.read_text())
    assert_truth_free_payload(dataset)
    if dataset.get("schema_version") != "p12a-base-response-dataset-v2" or not dataset.get("pass"):
        raise RuntimeError("P12-A dataset marker is not frozen")
    if candidate.get("artifacts", {}).get("dataset", {}).get("sha256") != sha256(dataset_marker_path):
        raise RuntimeError("evaluation contract dataset differs from the frozen candidate")
    gaussian = json.loads(gaussian_path.read_text())
    assert_ph001_sealed_payload(gaussian)
    if gaussian.get("schema_version") != "p12a-shell-cap-residual-gaussian-v1" or not gaussian.get("pass"):
        raise RuntimeError("P12-A Gaussian control is not frozen")
    if candidate.get("artifacts", {}).get("gaussian_baseline", {}).get("sha256") != sha256(gaussian_path):
        raise RuntimeError("Gaussian control differs from the frozen candidate")
    candidate_artifacts = {}
    for name, item in candidate.get("artifacts", {}).items():
        artifact_path = Path(str(item.get("path", "")))
        if not artifact_path.is_file() or item.get("sha256") != sha256(artifact_path):
            raise RuntimeError(f"frozen P12-A candidate artifact is stale: {name}")
        candidate_artifacts[name] = _record(artifact_path)
    validate_nested_truth_free_identities()
    training_spec = dataset.get("training", {})
    training_path = Path(training_spec.get("path", ""))
    if "ph001" in str(training_path).lower() or sha256(training_path) != training_spec.get("sha256"):
        raise PermissionError("training-only climatology source is stale or blind-bearing")
    with np.load(training_path, mmap_mode="r") as training:
        required = {"truth_eigenvalues", "shell", "natural_weight"}
        if not required.issubset(training.files):
            raise RuntimeError("training sample lacks climatology arrays")
        climatology = shell_class_climatology(
            training["truth_eigenvalues"], training["shell"], training["natural_weight"]
        )
    runtime = runtime_fingerprint()
    if Path(runtime["python"]["path"]).resolve() != FROZEN_PYTHON:
        raise RuntimeError(
            f"build the blind contract with the frozen interpreter {FROZEN_PYTHON}"
        )
    marker = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "selection_phase": "ph006",
        "fit_scope": "training phases only; no ph001 outcomes",
        "candidate": _record(candidate_path),
        "candidate_artifacts": candidate_artifacts,
        "gaussian_baseline": _record(gaussian_path),
        "dataset_marker": _record(dataset_marker_path),
        "training_sample": _record(training_path),
        "evaluation_implementation": {
            name: _record(path) for name, path in IMPLEMENTATION_FILES.items()
        },
        "scientific_contract_sources": {
            name: _record(path) for name, path in SCIENTIFIC_CONTRACT_FILES.items()
        },
        "forbidden_artifacts_absent": {
            name: str(path.resolve()) for name, path in FORBIDDEN_ARTIFACTS.items()
        },
        "truth_construction_implementation": {
            name: _record(path)
            for name, path in TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES.items()
        },
        "truth_free_identity_inputs": {
            name: _record(path) for name, path in TRUTH_FREE_IDENTITY_INPUTS.items()
        },
        "truth_source_contract": truth_source_contract(
            TRUTH_FREE_IDENTITY_INPUTS["phase_registry"]
        ),
        "runtime": runtime,
        "truth_construction_contract": TRUTH_CONSTRUCTION_CONTRACT,
        "class_threshold": 0.2,
        "shell_class_climatology": climatology,
        "gates": GATES,
        "conditional_strata": CONDITIONAL_STRATA,
        "covariate_definitions": COVARIATE_DEFINITIONS,
        "release_policy": RELEASE_POLICY,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "canonical_outputs": CANONICAL_OUTPUTS,
        "primary_proper_score": (
            "physical joint energy score on frozen 50k x 512 production draws; "
            "positive Gaussian-minus-FMPE difference favours FMPE"
        ),
        "unnormalized_fmpe_log_score_is_decisive": False,
        "bootstrap_unit": EVALUATION_PROTOCOL["bootstrap_unit"],
        "post_open_refit_allowed": False,
        "truth_files_read": [str(training_path.resolve())],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    assert_ph001_sealed_payload(marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--gaussian-baseline", type=Path, default=DEFAULT_GAUSSIAN)
    parser.add_argument(
        "--dataset-marker",
        type=Path,
        default=DEFAULT_ROOT / "p12a_base_response_v1/P12A_DATASET_READY.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation contract: {args.output}")
    marker = build_contract(
        candidate_path=args.candidate,
        gaussian_path=args.gaussian_baseline,
        dataset_marker_path=args.dataset_marker,
    )
    write_json_exclusive(args.output, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
