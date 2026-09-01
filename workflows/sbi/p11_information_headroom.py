#!/usr/bin/env python3
"""Estimate information headroom retained by the frozen P12-A summary.

This diagnostic deliberately does *not* claim to estimate the Bayes ceiling of
the raw final-view catalogue.  P12-A conditions on a seven-dimensional summary
containing the OOF U-PATCH prediction and response covariates.  If its posterior
is a calibrated approximation to p(lambda | S_U), then

    R2_max(S_U) = 1 - E[Var(lambda | S_U)] / Var(lambda).

The same posterior must also satisfy the Bayes-risk identity

    E[(lambda - E[lambda | S_U])**2] = E[Var(lambda | S_U)].

We therefore report the conditional-variance estimate together with the
identity residual.  Spatial cap+superblock resampling supplies uncertainty.
ph001 is never opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def theta_to_eigenvalues(theta: np.ndarray) -> np.ndarray:
    """Invert the frozen P12 ordered-softplus target coordinates."""
    theta = np.asarray(theta, dtype=np.float64)
    result = np.empty_like(theta)
    result[..., 0] = theta[..., 0]
    result[..., 1] = result[..., 0] + np.logaddexp(0.0, theta[..., 1])
    result[..., 2] = result[..., 1] + np.logaddexp(0.0, theta[..., 2])
    return result


def weighted_components(
    truth: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    base_prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the sufficient statistics used by every report and bootstrap."""
    truth = np.asarray(truth, dtype=np.float64)
    posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
    posterior_variance = np.asarray(posterior_variance, dtype=np.float64)
    base_prediction = np.asarray(base_prediction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if truth.shape != posterior_mean.shape or truth.shape != posterior_variance.shape:
        raise ValueError("truth/posterior shape mismatch")
    if truth.shape != base_prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("expected N x 3 physical eigenvalue arrays")
    if weight.shape != (len(truth),) or np.any(weight <= 0):
        raise ValueError("weights must be finite positive row weights")
    if not all(
        np.all(np.isfinite(value))
        for value in (truth, posterior_mean, posterior_variance, base_prediction, weight)
    ):
        raise ValueError("non-finite information-headroom input")
    if np.any(posterior_variance < 0):
        raise ValueError("posterior variance must be non-negative")
    return {
        "weight": weight,
        "weighted_truth": weight[:, None] * truth,
        "weighted_truth2": weight[:, None] * np.square(truth),
        "weighted_posterior_variance": weight[:, None] * posterior_variance,
        "weighted_posterior_squared_error": weight[:, None]
        * np.square(truth - posterior_mean),
        "weighted_base_squared_error": weight[:, None]
        * np.square(truth - base_prediction),
    }


def metrics_from_sums(
    sum_weight: float,
    sum_truth: np.ndarray,
    sum_truth2: np.ndarray,
    sum_posterior_variance: np.ndarray,
    sum_posterior_squared_error: np.ndarray,
    sum_base_squared_error: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate R2 quantities from weighted sufficient statistics."""
    if sum_weight <= 0:
        raise ValueError("empty weighted sample")
    truth_variance = sum_truth2 / sum_weight - np.square(sum_truth / sum_weight)
    if np.any(truth_variance <= 0):
        raise ValueError("truth variance must be positive")
    posterior_variance = sum_posterior_variance / sum_weight
    posterior_mse = sum_posterior_squared_error / sum_weight
    base_mse = sum_base_squared_error / sum_weight
    variance_ceiling = 1.0 - posterior_variance / truth_variance
    posterior_mean_r2 = 1.0 - posterior_mse / truth_variance
    base_r2 = 1.0 - base_mse / truth_variance
    identity_gap = variance_ceiling - posterior_mean_r2
    return {
        "truth_variance": truth_variance,
        "expected_posterior_variance": posterior_variance,
        "posterior_mean_mse": posterior_mse,
        "base_mse": base_mse,
        "posterior_variance_r2_estimate": variance_ceiling,
        "posterior_mean_r2": posterior_mean_r2,
        "base_r2": base_r2,
        "bayes_identity_gap_r2": identity_gap,
        "nonnegative_same_summary_headroom_r2": np.maximum(identity_gap, 0.0),
    }


def summarize_metrics(value: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {name: float(array[index]) for index, name in enumerate(EIGEN_NAMES)}
        for key, array in value.items()
    }


def aggregate_by_block_shell(
    components: dict[str, np.ndarray],
    group: np.ndarray,
    shell: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate row statistics to cap+superblock x shell cells."""
    group = np.asarray(group, dtype=np.int64)
    shell = np.asarray(shell, dtype=np.int64)
    if group.shape != shell.shape or group.shape != components["weight"].shape:
        raise ValueError("group/shell shape mismatch")
    unique, inverse = np.unique(group, return_inverse=True)
    if len(unique) < 10:
        raise ValueError("too few spatial blocks")
    cells = len(unique) * 4
    flat = inverse * 4 + shell
    result: dict[str, np.ndarray] = {}
    for key, values in components.items():
        if values.ndim == 1:
            result[key] = np.bincount(flat, weights=values, minlength=cells).reshape(
                len(unique), 4
            )
        else:
            result[key] = np.column_stack(
                [
                    np.bincount(flat, weights=values[:, i], minlength=cells)
                    for i in range(values.shape[1])
                ]
            ).reshape(len(unique), 4, values.shape[1])
    return result, unique


def metric_set_from_aggregate(
    aggregate: dict[str, np.ndarray], multiplicity: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Evaluate global, per-shell and macro-shell metrics for one block draw."""
    multiplicity = np.asarray(multiplicity, dtype=np.float64)

    def combine(shell_index: int | None) -> dict[str, np.ndarray]:
        chosen = slice(None) if shell_index is None else shell_index
        selected_weight = aggregate["weight"][:, chosen]
        if selected_weight.ndim == 1:
            weight = float(np.sum(multiplicity * selected_weight))
        else:
            weight = float(np.sum(multiplicity[:, None] * selected_weight))

        def total(key: str) -> np.ndarray:
            values = aggregate[key][:, chosen]
            if values.ndim == 2:
                values = values[:, None, :]
            return np.sum(multiplicity[:, None, None] * values, axis=(0, 1))

        return metrics_from_sums(
            weight,
            total("weighted_truth"),
            total("weighted_truth2"),
            total("weighted_posterior_variance"),
            total("weighted_posterior_squared_error"),
            total("weighted_base_squared_error"),
        )

    global_metrics = combine(None)
    shells = [combine(index) for index in range(4)]
    macro = {
        key: np.mean(np.stack([value[key] for value in shells], axis=0), axis=0)
        for key in global_metrics
    }
    return global_metrics, shells, macro


def bootstrap_report(
    aggregate: dict[str, np.ndarray], repeats: int, seed: int
) -> dict[str, Any]:
    """Cap+superblock cluster bootstrap, retaining shell correlations."""
    blocks = aggregate["weight"].shape[0]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[dict[str, np.ndarray]]] = {
        "global": [],
        "macro_shell": [],
        **{f"shell_{index}": [] for index in range(4)},
    }
    for _ in range(repeats):
        selected = rng.integers(0, blocks, size=blocks)
        multiplicity = np.bincount(selected, minlength=blocks)
        global_metrics, shells, macro = metric_set_from_aggregate(
            aggregate, multiplicity
        )
        draws["global"].append(global_metrics)
        draws["macro_shell"].append(macro)
        for index, value in enumerate(shells):
            draws[f"shell_{index}"].append(value)

    result: dict[str, Any] = {}
    for region, region_draws in draws.items():
        result[region] = {}
        for metric in region_draws[0]:
            values = np.stack([draw[metric] for draw in region_draws], axis=0)
            quantile = np.quantile(values, [0.025, 0.5, 0.975], axis=0)
            result[region][metric] = {
                name: {
                    "q025": float(quantile[0, index]),
                    "q500": float(quantile[1, index]),
                    "q975": float(quantile[2, index]),
                }
                for index, name in enumerate(EIGEN_NAMES)
            }
    return result


def load_physical_posterior(
    samples_path: Path,
    training_path: Path,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cached untempered standardized draws to physical eigenvalues."""
    # Recompute the exact frozen scaler from the registered training array.  This
    # matches p12_train_base_response_fmpe.py and avoids importing the GPU/PyTorch
    # stack for a CPU-only evidence reduction.
    training = np.load(training_path)
    training_theta = np.asarray(training["theta_softplus"], dtype=np.float32)
    theta_mean = training_theta.mean(axis=0, dtype=np.float64)
    theta_std = training_theta.std(axis=0, dtype=np.float64)
    samples = np.load(samples_path, mmap_mode="r")
    if samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError(f"unexpected posterior sample shape {samples.shape}")
    mean_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []
    for start in range(0, len(samples), chunk):
        scaled = np.asarray(samples[start : start + chunk], dtype=np.float64)
        eigen = theta_to_eigenvalues(scaled * theta_std + theta_mean)
        mean_parts.append(eigen.mean(axis=1))
        variance_parts.append(eigen.var(axis=1))
    return np.concatenate(mean_parts), np.concatenate(variance_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "fmpe_seed42" / "calibration_audit_v1",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "fmpe_seed42" / "fmpe_estimator.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT.parent
            / "p11_factorial_views_v1"
            / "information_headroom_v1"
            / "P11_INFORMATION_HEADROOM.json"
        ),
    )
    parser.add_argument("--repo-evidence", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--identity-tolerance-r2", type=float, default=0.03)
    parser.add_argument("--large-headroom-r2", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ready_path = args.root / "P12A_DATASET_READY.json"
    complete_path = args.root / "fmpe_seed42" / "P12A_COMPLETE.json"
    audit_report_path = args.audit_root / "P12A_CALIBRATION_AUDIT.json"
    index_path = args.audit_root / "evaluation_index.npy"
    samples_path = args.audit_root / "evaluation_samples_scaled.npy"
    ready = json.loads(ready_path.read_text())
    complete = json.loads(complete_path.read_text())
    audit = json.loads(audit_report_path.read_text())
    if any(value.get("sealed_phase_opened") for value in (ready, complete, audit)):
        raise RuntimeError("sealed ph001 guard failed")
    if ready["validation_phase"] != "ph006" or complete["selection_phase"] != "ph006":
        raise RuntimeError("information headroom must use ph006 selection data")
    if sha256(ready_path) != complete["dataset_marker_sha256"]:
        raise RuntimeError("P12 dataset/checkpoint contract mismatch")
    if sha256(args.checkpoint) != complete["checkpoint_sha256"]:
        raise RuntimeError("P12 checkpoint hash mismatch")

    validation = np.load(ready["validation"]["path"])
    evaluation_index = np.load(index_path)
    posterior_mean, posterior_variance = load_physical_posterior(
        samples_path, Path(ready["training"]["path"]), args.chunk
    )
    if len(evaluation_index) != len(posterior_mean):
        raise RuntimeError("posterior/evaluation-index length mismatch")
    truth = np.asarray(validation["truth_eigenvalues"])[evaluation_index]
    base = np.asarray(validation["base_prediction_eigenvalues"])[evaluation_index]
    weight = np.asarray(validation["natural_weight"])[evaluation_index]
    shell = np.asarray(validation["shell"])[evaluation_index]
    cap = np.asarray(validation["cap"])[evaluation_index].astype(np.int64)
    superblock = np.asarray(validation["superblock_id"])[evaluation_index].astype(np.int64)
    group = (cap << 32) + superblock

    components = weighted_components(
        truth, posterior_mean, posterior_variance, base, weight
    )
    aggregate, unique_groups = aggregate_by_block_shell(components, group, shell)
    unit = np.ones(len(unique_groups), dtype=np.int64)
    global_metrics, shell_metrics, macro_metrics = metric_set_from_aggregate(
        aggregate, unit
    )
    bootstrap = bootstrap_report(aggregate, args.bootstrap_repeats, args.seed)

    lambda1_gap = float(global_metrics["bayes_identity_gap_r2"][0])
    lambda1_gap_upper = float(
        bootstrap["global"]["bayes_identity_gap_r2"]["lambda1"]["q975"]
    )
    maximum_identity_gap = float(
        max(
            np.max(np.abs(global_metrics["bayes_identity_gap_r2"])),
            np.max(np.abs(macro_metrics["bayes_identity_gap_r2"])),
        )
    )
    identity_pass = maximum_identity_gap <= args.identity_tolerance_r2
    large_same_summary_headroom = (
        lambda1_gap >= args.large_headroom_r2
        and lambda1_gap_upper >= args.large_headroom_r2
    )
    summary_saturated = identity_pass and lambda1_gap_upper < args.large_headroom_r2

    report = {
        "schema_version": "p11-information-headroom-v1",
        "created_utc": utc_now(),
        "estimand": "R2_max(S_U), where S_U is the frozen P12-A seven-feature summary",
        "not_estimands": [
            "R2_max(X_final_raw_catalogue)",
            "R2_max(X_dense)",
            "mutual information in nats or bits",
        ],
        "interpretation": (
            "posterior conditional variance estimates a Bayes R2 ceiling only if the "
            "Bayes-risk identity agrees with held-out posterior-mean squared error"
        ),
        "selection_phase": "ph006",
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "evaluation_folds": ready["validation"]["evaluation_folds"],
        "rows": int(len(evaluation_index)),
        "posterior_draws_per_row": int(np.load(samples_path, mmap_mode="r").shape[1]),
        "spatial_blocks": int(len(unique_groups)),
        "weighting": "frozen P12-A natural-volume row weights",
        "coordinates": "physical ordered lambda1 <= lambda2 <= lambda3",
        "posterior_version": "frozen untempered P12-A; rejected affine correction absent",
        "point_estimates": {
            "global": summarize_metrics(global_metrics),
            "by_shell": {
                str(index): summarize_metrics(value)
                for index, value in enumerate(shell_metrics)
            },
            "macro_shell": summarize_metrics(macro_metrics),
        },
        "spatial_block_bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "intervals": bootstrap,
        },
        "gates": {
            "identity_tolerance_r2": args.identity_tolerance_r2,
            "large_headroom_r2": args.large_headroom_r2,
            "max_abs_global_or_macro_identity_gap_r2": maximum_identity_gap,
            "posterior_bayes_identity_pass": bool(identity_pass),
            "lambda1_same_summary_headroom_r2": lambda1_gap,
            "lambda1_same_summary_headroom_q975_r2": lambda1_gap_upper,
            "large_same_summary_estimator_headroom": bool(large_same_summary_headroom),
            "summary_downstream_saturated_at_0p03_scale": bool(summary_saturated),
            "raw_final_view_ceiling_estimated": False,
            "dense_view_ceiling_estimated": False,
        },
        "decision": {
            "dense_teacher_role": "advisory, not a JEPA veto",
            "same_summary_posterior_head": (
                "no material lambda1 headroom at the registered 0.03 scale"
                if summary_saturated
                else "material headroom unresolved"
            ),
            "jepa_readiness": (
                "READY_FOR_BOUNDED_MATCHED_CONTROL_CANARY"
                if summary_saturated
                else "WAIT_FOR_HEADROOM_DIAGNOSIS"
            ),
            "claim_limit": (
                "A JEPA can test whether a new final-view representation preserves "
                "more target-relevant information. This report cannot show that the "
                "raw final view contains more information than S_U."
            ),
        },
        "remaining_information_ladder": [
            "cross-fitted higher-capacity q(lambda|X_final) proper-score/variance control",
            "matched q(lambda|X_dense) control",
            "paired final+dense proper-score increment as an operational conditional-information estimate",
            "the same audit for Z_JEPA if the bounded canary trains",
        ],
        "provenance": {
            "dataset_ready": str(ready_path),
            "dataset_ready_sha256": sha256(ready_path),
            "validation_dataset": ready["validation"]["path"],
            "validation_dataset_sha256": ready["validation"]["sha256"],
            "p12_complete": str(complete_path),
            "p12_complete_sha256": sha256(complete_path),
            "calibration_audit": str(audit_report_path),
            "calibration_audit_sha256": sha256(audit_report_path),
            "posterior_samples": str(samples_path),
            "posterior_samples_sha256": sha256(samples_path),
            "evaluation_index": str(index_path),
            "evaluation_index_sha256": sha256(index_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
    }
    atomic_json(args.output, report)
    if args.repo_evidence is not None:
        atomic_json(args.repo_evidence, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
