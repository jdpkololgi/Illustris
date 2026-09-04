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
import json
from pathlib import Path
import subprocess

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_production_contract import (
    P12A_SCHEMA,
    assert_ph001_sealed_payload,
    assert_truth_free_payload,
)


SCHEMA = "p12a-blind-evaluation-contract-v1"
DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
DEFAULT_CANDIDATE = Path("docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json")
DEFAULT_GAUSSIAN = Path("docs/evidence/p12/production_aux_v1/P12A_GAUSSIAN_BASELINE.json")
DEFAULT_OUTPUT = Path("docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json")
IMPLEMENTATION_FILES = {
    "blind_evaluator": Path(__file__).with_name("p12a_evaluate_blind.py"),
    "proper_score_evaluator": Path(__file__).with_name("p12a_blind_proper_score.py"),
    "one_open_guard": Path(__file__).with_name("p12a_open_blind.py"),
}
GATES = {
    "joint_eigenvalue_tarp_maximum": 0.05,
    "joint_eigengap_tarp_maximum": 0.05,
    "physical_rank_cdf_maximum": 0.05,
    "global_coverage_absolute_error_maximum": 0.03,
    "conditional_coverage_absolute_error_maximum": 0.06,
    "posterior_mean_lambda1_r2_delta_minimum": -0.02,
    "multiclass_brier_skill_minimum": 0.0,
    "fmpe_minus_gaussian_log_score_ci95_lower_minimum": 0.0,
}
CONDITIONAL_STRATA = ["shell", "ntilde_quartile", "boundary_distance_quartile"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _record(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


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
    marker = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "selection_phase": "ph006",
        "fit_scope": "training phases only; no ph001 outcomes",
        "candidate": _record(candidate_path),
        "gaussian_baseline": _record(gaussian_path),
        "dataset_marker": _record(dataset_marker_path),
        "training_sample": _record(training_path),
        "evaluation_implementation": {
            name: _record(path) for name, path in IMPLEMENTATION_FILES.items()
        },
        "class_threshold": 0.2,
        "shell_class_climatology": climatology,
        "gates": GATES,
        "conditional_strata": CONDITIONAL_STRATA,
        "primary_proper_score": "physical joint log score on the frozen 50k audit rows",
        "bootstrap_unit": "authoritative core",
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
    atomic_json(args.output, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
