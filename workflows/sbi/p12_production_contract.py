#!/usr/bin/env python3
"""Fail-closed contracts for the P12-A production candidate and shared blind opening.

This module is deliberately independent of Torch/SBI so contract validation can run
on a CPU login preflight. It never opens ph001 truth. The controlled open transition
is represented and tested here, but production code must call it only after all
truth-free prediction manifests have been frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json


P12A_SCHEMA = "p12a-production-candidate-frozen-v1"
BLIND_SCHEMA = "p12-blind-predictions-frozen-v1"
OPEN_SCHEMA = "p12-blind-opened-v1"

QUALITY_SPARSE_SHELL = np.uint16(1 << 0)
QUALITY_BOUNDARY_LT_R7 = np.uint16(1 << 1)
QUALITY_BOUNDARY_LT_2R7 = np.uint16(1 << 2)
QUALITY_RESPONSE_OOD = np.uint16(1 << 3)
QUALITY_PRIOR_DOMINATED_WIDTH = np.uint16(1 << 4)

QUALITY_BITS = {
    "sparse_shell_z_ge_0p45": int(QUALITY_SPARSE_SHELL),
    "boundary_distance_lt_7_mpc_h": int(QUALITY_BOUNDARY_LT_R7),
    "boundary_distance_lt_14_mpc_h": int(QUALITY_BOUNDARY_LT_2R7),
    "response_outside_training_range": int(QUALITY_RESPONSE_OOD),
    "prior_dominated_width": int(QUALITY_PRIOR_DOMINATED_WIDTH),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def require_file(path: Path | str, *, expected_sha256: str | None = None) -> dict:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = sha256(candidate)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"hash mismatch for {candidate}")
    return {"path": str(candidate), "sha256": digest, "bytes": candidate.stat().st_size}


def assert_truth_free_payload(payload: Mapping[str, Any]) -> None:
    """Reject evidence that admits any blind truth access."""
    if payload.get("open_count", 0) != 0:
        raise PermissionError("blind payload has a nonzero open count")
    if payload.get("truth_files_read", []) not in ([], ()):
        raise PermissionError("blind payload records truth-file access")
    for key in ("sealed_phase_opened", "ph001_opened", "blind_truth_opened"):
        if bool(payload.get(key, False)):
            raise PermissionError(f"blind payload has {key}=true")


def validate_ordered_draws(draws: np.ndarray) -> np.ndarray:
    values = np.asarray(draws, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("draws must have shape [rows,draws,3]")
    if values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("posterior draws must be finite and nontrivial")
    tolerance = 32.0 * np.finfo(np.float32).eps
    if np.any(np.diff(values, axis=-1) < -tolerance):
        raise ValueError("posterior eigenvalue draws are not ordered")
    return values


def posterior_summaries(draws: np.ndarray, *, threshold: float = 0.2) -> dict[str, np.ndarray]:
    """Return the complete production summary for ordered eigenvalue draws."""
    values = validate_ordered_draws(draws)
    quantile = np.quantile(values, [0.05, 0.16, 0.50, 0.84, 0.95], axis=1)
    above = np.mean(values > float(threshold), axis=1)
    web_index = np.sum(values > float(threshold), axis=-1)
    web_probability = np.stack(
        [np.mean(web_index == index, axis=1) for index in range(4)], axis=-1
    )
    if not np.allclose(web_probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("web-class posterior probabilities do not normalize")
    safe = np.clip(web_probability, 1e-12, 1.0)
    trace = values.sum(axis=-1)
    trace_quantile = np.quantile(trace, [0.05, 0.16, 0.50, 0.84, 0.95], axis=1)
    return {
        "eigenvalue_mean": values.mean(axis=1, dtype=np.float64).astype(np.float32),
        "eigenvalue_std": values.std(axis=1, ddof=1, dtype=np.float64).astype(np.float32),
        "eigenvalue_q05": quantile[0].astype(np.float32),
        "eigenvalue_q16": quantile[1].astype(np.float32),
        "eigenvalue_q50": quantile[2].astype(np.float32),
        "eigenvalue_q84": quantile[3].astype(np.float32),
        "eigenvalue_q95": quantile[4].astype(np.float32),
        "probability_eigenvalue_gt_0p2": above.astype(np.float32),
        "web_class_probability": web_probability.astype(np.float32),
        "web_class_entropy_nats": (-np.sum(web_probability * np.log(safe), axis=1)).astype(
            np.float32
        ),
        "trace_mean": trace.mean(axis=1, dtype=np.float64).astype(np.float32),
        "trace_std": trace.std(axis=1, ddof=1, dtype=np.float64).astype(np.float32),
        "trace_q05": trace_quantile[0].astype(np.float32),
        "trace_q16": trace_quantile[1].astype(np.float32),
        "trace_q50": trace_quantile[2].astype(np.float32),
        "trace_q84": trace_quantile[3].astype(np.float32),
        "trace_q95": trace_quantile[4].astype(np.float32),
    }


def quality_bitmask(
    *,
    redshift: np.ndarray,
    boundary_distance_mpc_h: np.ndarray,
    response_covariate: np.ndarray,
    posterior_width: np.ndarray,
    response_training_range: tuple[float, float],
    prior_width_threshold: np.ndarray | float,
) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=np.float64)
    boundary = np.asarray(boundary_distance_mpc_h, dtype=np.float64)
    response = np.asarray(response_covariate, dtype=np.float64)
    width = np.asarray(posterior_width, dtype=np.float64)
    if width.ndim == 2:
        width_metric = np.max(width, axis=1)
    elif width.ndim == 1:
        width_metric = width
    else:
        raise ValueError("posterior width must be [rows] or [rows,coordinates]")
    if not (len(redshift) == len(boundary) == len(response) == len(width_metric)):
        raise ValueError("quality covariates are not row aligned")
    if not all(np.all(np.isfinite(item)) for item in (redshift, boundary, response, width)):
        raise ValueError("quality covariates must be finite")
    low, high = map(float, response_training_range)
    if not low < high:
        raise ValueError("invalid response training range")
    threshold = np.asarray(prior_width_threshold, dtype=np.float64)
    if threshold.ndim > 1:
        raise ValueError("prior-width threshold must be scalar or one-dimensional")
    if threshold.ndim == 1 and threshold.shape not in ((len(width_metric),), (3,)):
        raise ValueError("unsupported prior-width threshold shape")
    if threshold.shape == (3,) and width.ndim == 2:
        prior_dominated = np.any(width > threshold[None], axis=1)
    else:
        prior_dominated = width_metric > threshold
    mask = np.zeros(len(redshift), dtype=np.uint16)
    mask[redshift >= 0.45] |= QUALITY_SPARSE_SHELL
    mask[boundary < 7.0] |= QUALITY_BOUNDARY_LT_R7
    mask[boundary < 14.0] |= QUALITY_BOUNDARY_LT_2R7
    mask[(response < low) | (response > high)] |= QUALITY_RESPONSE_OOD
    mask[prior_dominated] |= QUALITY_PRIOR_DOMINATED_WIDTH
    return mask


def deterministic_audit_subset(
    parent_node_id: np.ndarray,
    shell: np.ndarray,
    cap: np.ndarray,
    boundary_distance: np.ndarray,
    *,
    maximum: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Select a stable shell/cap/boundary-stratified audit population."""
    parent = np.asarray(parent_node_id, dtype=np.int64)
    shell = np.asarray(shell, dtype=np.int8)
    cap = np.asarray(cap, dtype=np.int8)
    boundary = np.asarray(boundary_distance, dtype=np.float64)
    if not (len(parent) == len(shell) == len(cap) == len(boundary)):
        raise ValueError("audit covariates are not row aligned")
    if len(np.unique(parent)) != len(parent):
        raise ValueError("parent identifiers must be unique")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(parent) <= maximum:
        return np.arange(len(parent), dtype=np.int64)
    edges = np.quantile(boundary, [0.0, 0.25, 0.5, 0.75, 1.0])
    boundary_bin = np.searchsorted(edges[1:-1], boundary, side="right")
    group = shell.astype(np.int64) * 8 + cap.astype(np.int64) * 4 + boundary_bin
    unique, counts = np.unique(group, return_counts=True)
    quota = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))
    while quota.sum() > maximum:
        index = int(np.argmax(quota))
        quota[index] -= 1
    while quota.sum() < maximum:
        available = counts - quota
        index = int(np.argmax(available))
        quota[index] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for value, count in zip(unique, quota, strict=True):
        rows = np.flatnonzero(group == value)
        chosen.append(rng.choice(rows, size=int(count), replace=False))
    return np.sort(np.concatenate(chosen).astype(np.int64))


def fit_shell_cap_gaussian(
    residual: np.ndarray, shell: np.ndarray, cap: np.ndarray, weight: np.ndarray | None = None
) -> dict:
    """Fit the registered train-only Gaussian residual baseline."""
    residual = np.asarray(residual, dtype=np.float64)
    shell = np.asarray(shell, dtype=np.int8)
    cap = np.asarray(cap, dtype=np.int8)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError("residual must have shape [rows,3]")
    if not (len(residual) == len(shell) == len(cap)) or not np.all(np.isfinite(residual)):
        raise ValueError("Gaussian baseline rows are invalid")
    if weight is None:
        weight = np.ones(len(residual), dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    output: dict[str, Any] = {}
    for shell_value in np.unique(shell):
        for cap_value in np.unique(cap):
            selected = (shell == shell_value) & (cap == cap_value)
            if int(selected.sum()) < 4:
                continue
            w = weight[selected]
            w = w / w.sum()
            values = residual[selected]
            mean = np.sum(values * w[:, None], axis=0)
            centered = values - mean
            covariance = (centered * w[:, None]).T @ centered
            covariance += np.eye(3) * max(1e-8, 1e-6 * np.trace(covariance) / 3.0)
            output[f"shell{int(shell_value)}_cap{int(cap_value)}"] = {
                "rows": int(selected.sum()),
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
            }
    return {
        "schema_version": "p12a-shell-cap-residual-gaussian-v1",
        "fit_scope": "training phases only",
        "groups": output,
    }


def build_p12a_candidate_marker(config: Mapping[str, Any]) -> dict:
    artifacts = {
        name: require_file(spec["path"], expected_sha256=spec.get("sha256"))
        for name, spec in config["artifacts"].items()
        if name != "calibration_pass_absent"
    }
    absent = Path(config["artifacts"]["calibration_pass_absent"]["path"])
    if absent.exists():
        raise RuntimeError("strict P12A_CALIBRATION_PASS marker must remain absent")
    completion = json.loads(Path(artifacts["completion"]["path"]).read_text())
    dataset = json.loads(Path(artifacts["dataset"]["path"]).read_text())
    audit = json.loads(Path(artifacts["calibration_audit"]["path"]).read_text())
    for payload in (completion, dataset, audit):
        assert_truth_free_payload(payload)
    if not completion.get("technical_complete") or completion.get("calibration_pass"):
        raise RuntimeError("P12-A must be technically complete and uncorrected")
    if dataset.get("pass") is not True:
        raise RuntimeError("P12-A dataset marker does not pass")
    checkpoint = artifacts["checkpoint"]
    if completion.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError("completion/checkpoint hash mismatch")
    marker = {
        "schema_version": P12A_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "estimand": "q(lambda_ordered | U_PATCH_R1_prediction, response, H_fid)",
        "posterior": "FMPE conditional flow matching in ordered-softplus coordinates",
        "tempering": None,
        "recalibration": None,
        "strict_calibration_pass_marker_present": False,
        "ph006_caveat": "sparse-shell lambda2/lambda3 conditional residual",
        "quality_bits": QUALITY_BITS,
        "posterior_draws": int(config["posterior_draws"]),
        "audit_draw_rows": int(config["audit_draw_rows"]),
        "artifacts": artifacts,
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    assert_truth_free_payload(marker)
    return marker


def freeze_blind_predictions(
    *,
    candidate_marker: Path,
    method_selection_marker: Path,
    prediction_manifests: Iterable[Path],
    deterministic_contract: Path,
) -> dict:
    candidate = json.loads(candidate_marker.read_text())
    selection = json.loads(method_selection_marker.read_text())
    assert_truth_free_payload(candidate)
    assert_truth_free_payload(selection)
    if candidate.get("schema_version") != P12A_SCHEMA or not candidate.get("pass"):
        raise RuntimeError("P12-A production candidate is not frozen")
    if selection.get("schema_version") not in (
        "p12f-method-selection-frozen-v1",
        "p12f-no-field-finalist-v1",
    ):
        raise RuntimeError("P12-F method selection is not frozen")
    manifests = []
    for path in prediction_manifests:
        payload = json.loads(path.read_text())
        assert_truth_free_payload(payload)
        if not payload.get("pass"):
            raise RuntimeError(f"prediction manifest does not pass: {path}")
        manifests.append(require_file(path))
    deterministic = json.loads(deterministic_contract.read_text())
    assert_truth_free_payload(deterministic)
    marker = {
        "schema_version": BLIND_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "p12a_candidate": require_file(candidate_marker),
        "p12f_selection": require_file(method_selection_marker),
        "p10_deterministic_contract": require_file(deterministic_contract),
        "prediction_manifests": manifests,
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "post_open_tuning_allowed": False,
        "pass": True,
    }
    assert_truth_free_payload(marker)
    return marker


def build_opened_marker(
    frozen_marker: Mapping[str, Any],
    *,
    truth_artifacts: Iterable[Path],
    explicit_authorization: str,
) -> dict:
    """Represent the sole state transition; never invoke it during preparation."""
    assert_truth_free_payload(frozen_marker)
    if frozen_marker.get("schema_version") != BLIND_SCHEMA or not frozen_marker.get("pass"):
        raise RuntimeError("blind predictions are not frozen")
    if explicit_authorization != "OPEN_PH001_ONCE":
        raise PermissionError("explicit one-open authorization token is absent")
    truth = [require_file(path) for path in truth_artifacts]
    if not truth:
        raise RuntimeError("controlled opening requires frozen truth artifacts")
    return {
        "schema_version": OPEN_SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "frozen_predictions": frozen_marker,
        "truth_files_read": truth,
        "open_count": 1,
        "sealed_phase_opened": True,
        "post_open_tuning_allowed": False,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    marker = build_p12a_candidate_marker(config.get("p12a", config))
    atomic_json(args.output, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
