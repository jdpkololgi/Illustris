#!/usr/bin/env python3
"""Diagnose the frozen P12-A calibration failure without opening ph001.

The audit uses randomized finite-sample posterior ranks, reports both FMPE
training coordinates and physical ordered eigenvalues, matches SBC and TARP
row/draw budgets, and stratifies failures before any correction is fitted.
ph006 folds 0--1 remain calibration-only; folds 2--4 remain selection-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import cramervonmises, kstest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json
from workflows.sbi.p12_train_base_response_fmpe import (
    sample_posterior,
    theta_to_eigenvalues,
    weighted_coverage,
    weighted_r2,
)

ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")
THETA_NAMES = ("lambda1", "gap12_softplus", "gap23_softplus")
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_indices(
    fold: np.ndarray,
    calibration_folds: list[int],
    evaluation_folds: list[int],
    calibration_rows: int,
    evaluation_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly reproduce the frozen P12-A row selection."""
    rng = np.random.default_rng(seed + 12000)

    def choose(mask: np.ndarray, maximum: int) -> np.ndarray:
        index = np.flatnonzero(mask)
        if len(index) > maximum:
            index = rng.choice(index, size=maximum, replace=False)
        return np.asarray(index, dtype=np.int64)

    calibration = choose(np.isin(fold, calibration_folds), calibration_rows)
    evaluation = choose(np.isin(fold, evaluation_folds), evaluation_rows)
    if np.intersect1d(calibration, evaluation).size:
        raise RuntimeError("calibration and evaluation rows overlap")
    return calibration, evaluation


def randomized_pit(samples: np.ndarray, truth: np.ndarray, seed: int) -> np.ndarray:
    """Randomized finite-ensemble ranks suitable for a continuous-uniform test."""
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    expected = (samples.shape[0], samples.shape[2])
    if samples.ndim != 3 or truth.shape != expected:
        raise ValueError("samples/truth shape mismatch")
    rng = np.random.default_rng(seed)
    below = np.sum(samples < truth[:, None, :], axis=1)
    equal = np.sum(samples == truth[:, None, :], axis=1)
    return (below + rng.random(below.shape) * (equal + 1.0)) / (
        samples.shape[1] + 1.0
    )


def weighted_ks_uniform(values: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(values)
    x = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weight, dtype=np.float64)[order]
    w /= w.sum()
    upper = np.cumsum(w)
    lower = upper - w
    return float(max(np.max(np.abs(upper - x)), np.max(np.abs(lower - x))))


def effective_rows(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.float64)
    return float(weight.sum() ** 2 / np.square(weight).sum())


def rank_summary(
    ranks: np.ndarray, weight: np.ndarray, names: tuple[str, str, str]
) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    result: dict[str, Any] = {
        "rows": int(len(ranks)),
        "effective_weighted_rows": effective_rows(weight),
        "components": {},
    }
    for component, name in enumerate(names):
        value = np.asarray(ranks[:, component], dtype=np.float64)
        histogram, _ = np.histogram(value, bins=bins, weights=weight)
        histogram = histogram / histogram.sum()
        ks = kstest(value, "uniform")
        mean = float(np.average(value, weights=weight))
        edge_excess = float(histogram[[0, 9]].sum() - 0.2)
        centre_excess = float(histogram[[4, 5]].sum() - 0.2)
        flags: list[str] = []
        if mean > 0.515:
            flags.append("posterior_location_low_relative_to_truth")
        elif mean < 0.485:
            flags.append("posterior_location_high_relative_to_truth")
        if edge_excess > 0.03:
            flags.append("posterior_too_narrow_or_heavy_truth_tails")
        if centre_excess > 0.03:
            flags.append("posterior_too_wide_or_overconcentrated_truth_centre")
        if abs(histogram[0] - histogram[-1]) > 0.03:
            flags.append("asymmetric_tail_or_location_error")
        if not flags:
            flags.append("no_large_shape_signature_at_heuristic_scale")
        result["components"][name] = {
            "weighted_mean_rank": mean,
            "weighted_ks_distance": weighted_ks_uniform(value, weight),
            "unweighted_ks_distance": float(ks.statistic),
            "unweighted_ks_p": float(ks.pvalue),
            "cramer_von_mises": float(cramervonmises(value, "uniform").statistic),
            "decile_mass": histogram.tolist(),
            "edge_mass_excess_over_uniform": edge_excess,
            "centre_mass_excess_over_uniform": centre_excess,
            "interpretation_flags": flags,
        }
    return result


def interval_report(
    samples: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    mean = samples.mean(axis=1)
    return {
        "coverage68": weighted_coverage(samples, truth, weight, 0.68).tolist(),
        "coverage90": weighted_coverage(samples, truth, weight, 0.90).tolist(),
        "posterior_mean_r2": [
            weighted_r2(truth[:, i], mean[:, i], weight) for i in range(3)
        ],
        "base_r2": [weighted_r2(truth[:, i], base[:, i], weight) for i in range(3)],
    }


def quantile_groups(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 5)))
    result = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        final = index == len(edges) - 2
        mask = (values >= lo) & (values <= hi if final else values < hi)
        result.append((f"q{index}:{lo:.6g}:{hi:.6g}", mask))
    return result


def conditional_report(
    theta_ranks: np.ndarray,
    eigen_ranks: np.ndarray,
    eigen_samples: np.ndarray,
    validation: Any,
    index: np.ndarray,
) -> dict[str, Any]:
    context = np.asarray(validation["context"])[index]
    truth = np.asarray(validation["truth_eigenvalues"])[index]
    base = np.asarray(validation["base_prediction_eigenvalues"])[index]
    weight = np.asarray(validation["natural_weight"])[index]
    shell = np.asarray(validation["shell"])[index]
    cap = np.asarray(validation["cap"])[index]
    variables = {
        "shell": [(str(value), shell == value) for value in range(4)],
        "cap": [(str(value), cap == value) for value in (0, 1)],
        "redshift_quartile": quantile_groups(context[:, 3]),
        "ntilde_quartile": quantile_groups(context[:, 4]),
        "support_boundary_distance_quartile": quantile_groups(context[:, 6]),
        "truth_lambda1_quartile": quantile_groups(truth[:, 0]),
        "base_lambda1_quartile": quantile_groups(base[:, 0]),
        "posterior_width_lambda1_quartile": quantile_groups(
            np.quantile(eigen_samples[:, :, 0], 0.84, axis=1)
            - np.quantile(eigen_samples[:, :, 0], 0.16, axis=1)
        ),
    }
    result: dict[str, Any] = {}
    for variable, groups in variables.items():
        result[variable] = {}
        for label, mask in groups:
            if np.count_nonzero(mask) < 100:
                continue
            result[variable][label] = {
                "rows": int(np.count_nonzero(mask)),
                "theta_rank": rank_summary(theta_ranks[mask], weight[mask], THETA_NAMES),
                "eigen_rank": rank_summary(eigen_ranks[mask], weight[mask], EIGEN_NAMES),
                "intervals": interval_report(
                    eigen_samples[mask], truth[mask], base[mask], weight[mask]
                ),
            }
    return result


def tarp_diagnostic(
    samples: np.ndarray,
    truth: np.ndarray,
    seed: int,
    bootstrap_repeats: int,
    bootstrap_rows: int,
) -> dict[str, Any]:
    try:
        import tarp

        posterior = np.transpose(samples, (1, 0, 2))
        ecp, alpha = tarp.get_tarp_coverage(
            posterior, truth, norm=True, bootstrap=False, seed=seed
        )
        full = float(np.max(np.abs(ecp - alpha)))
        rng = np.random.default_rng(seed + 91)
        maxima = []
        size = min(bootstrap_rows, len(truth))
        for repeat in range(bootstrap_repeats):
            chosen = rng.choice(len(truth), size=size, replace=True)
            b_ecp, b_alpha = tarp.get_tarp_coverage(
                posterior[:, chosen],
                truth[chosen],
                norm=True,
                bootstrap=False,
                seed=seed + repeat + 1,
            )
            maxima.append(float(np.max(np.abs(b_ecp - b_alpha))))
        return {
            "available": True,
            "rows": int(len(truth)),
            "full_max_abs_ecp_minus_alpha": full,
            "bootstrap_rows": int(size),
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_max_abs_quantiles": np.quantile(
                maxima, [0.05, 0.5, 0.95]
            ).tolist(),
            "pass_0p05": bool(full <= 0.05),
        }
    except Exception as error:
        return {"available": False, "error": repr(error), "pass_0p05": False}


def build_posterior(checkpoint: dict[str, Any], device: str) -> Any:
    from sbi.inference import FMPE
    from sbi.neural_nets import posterior_flow_nn
    from sbi.utils import BoxUniform

    prior = BoxUniform(
        low=torch.as_tensor(checkpoint["prior_low"], device=device),
        high=torch.as_tensor(checkpoint["prior_high"], device=device),
        device=device,
    )
    builder = posterior_flow_nn(
        model="mlp",
        hidden_features=int(checkpoint["hidden_features"]),
        num_layers=int(checkpoint["num_layers"]),
        z_score_theta="none",
        z_score_x="none",
    )
    dummy_theta = torch.zeros((2, 3), device=device)
    dummy_context = torch.zeros((2, len(checkpoint["context_mean"])), device=device)
    estimator = builder(dummy_theta, dummy_context)
    estimator.load_state_dict(checkpoint["state_dict"])
    estimator.to(device)
    estimator.eval()
    inference = FMPE(prior=prior, density_estimator=builder, device=device)
    return inference.build_posterior(estimator)


def sample_posterior_resumable(
    posterior: Any,
    context: np.ndarray,
    n_samples: int,
    chunk: int,
    device: str,
    sample_path: Path,
    progress_path: Path,
) -> np.ndarray:
    """Sample into an NPY memmap and atomically checkpoint completed rows."""
    expected = (len(context), n_samples, 3)
    if sample_path.exists() or progress_path.exists():
        if not sample_path.exists() or not progress_path.exists():
            raise RuntimeError("incomplete sampling cache/progress pair")
        progress = json.loads(progress_path.read_text())
        samples = np.lib.format.open_memmap(sample_path, mode="r+")
        if tuple(samples.shape) != expected:
            raise RuntimeError("cached posterior samples have the wrong shape")
        if progress.get("expected_shape") != list(expected):
            raise RuntimeError("sampling progress shape mismatch")
        completed = int(progress.get("completed_rows", 0))
    else:
        samples = np.lib.format.open_memmap(
            sample_path, mode="w+", dtype=np.float32, shape=expected
        )
        completed = 0
        atomic_json(
            progress_path,
            {"expected_shape": list(expected), "completed_rows": completed},
        )
    if not 0 <= completed <= len(context):
        raise RuntimeError("invalid completed-row counter")
    for start in range(completed, len(context), chunk):
        stop = min(start + chunk, len(context))
        samples[start:stop] = sample_posterior(
            posterior, context[start:stop], n_samples, chunk, device
        )
        samples.flush()
        atomic_json(
            progress_path,
            {"expected_shape": list(expected), "completed_rows": stop},
        )
    return np.load(sample_path, mmap_mode="r")


def plot_ranks(theta_ranks: np.ndarray, eigen_ranks: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    panels = (
        (theta_ranks, THETA_NAMES, "FMPE training coordinates"),
        (eigen_ranks, EIGEN_NAMES, "Physical ordered eigenvalues"),
    )
    for row, (ranks, names, title) in enumerate(panels):
        for column, name in enumerate(names):
            axes[row, column].hist(
                ranks[:, column], bins=bins, density=True, color="#3A86FF"
            )
            axes[row, column].axhline(1.0, color="black", linestyle="--")
            axes[row, column].set_title(f"{title}\n{name}")
            axes[row, column].set_xlabel("randomized posterior rank")
            axes[row, column].set_ylabel("density")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "fmpe_seed42/fmpe_estimator.pt"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "fmpe_seed42/calibration_audit_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-posterior-samples", type=int, default=512)
    parser.add_argument("--calibration-rows", type=int, default=20_000)
    parser.add_argument("--evaluation-rows", type=int, default=50_000)
    parser.add_argument("--sample-chunk", type=int, default=2048)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-rows", type=int, default=10_000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("posterior sampling requires a GPU interactive allocation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    ready_path = args.dataset_root / "P12A_DATASET_READY.json"
    complete_path = args.checkpoint.parent / "P12A_COMPLETE.json"
    ready = json.loads(ready_path.read_text())
    complete = json.loads(complete_path.read_text())
    if ready.get("sealed_phase_opened") or complete.get("sealed_phase_opened"):
        raise RuntimeError("sealed-phase guard failed")
    if sha256(ready_path) != complete["dataset_marker_sha256"]:
        raise RuntimeError("dataset marker differs from trained posterior contract")

    validation = np.load(ready["validation"]["path"])
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if checkpoint["dataset_marker_sha256"] != sha256(ready_path):
        raise RuntimeError("checkpoint dataset provenance mismatch")
    calibration_index, evaluation_index = choose_indices(
        np.asarray(validation["fold"], dtype=np.uint8),
        ready["validation"]["calibration_folds"],
        ready["validation"]["evaluation_folds"],
        args.calibration_rows,
        args.evaluation_rows,
        args.seed,
    )
    np.save(args.output_root / "calibration_index.npy", calibration_index)
    np.save(args.output_root / "evaluation_index.npy", evaluation_index)

    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float64)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float64)
    theta_mean = np.asarray(checkpoint["theta_mean"], dtype=np.float64)
    theta_std = np.asarray(checkpoint["theta_std"], dtype=np.float64)
    context_scaled = (
        (np.asarray(validation["context"], dtype=np.float32) - context_mean)
        / context_std
    ).astype(np.float32)
    theta_truth_scaled = (
        (np.asarray(validation["theta_softplus"], dtype=np.float32) - theta_mean)
        / theta_std
    ).astype(np.float32)

    sample_path = args.output_root / "evaluation_samples_scaled.npy"
    progress_path = args.output_root / "sampling_progress.json"
    posterior = build_posterior(checkpoint, "cuda")
    samples_scaled = sample_posterior_resumable(
        posterior,
        context_scaled[evaluation_index],
        args.n_posterior_samples,
        args.sample_chunk,
        "cuda",
        sample_path,
        progress_path,
    )

    theta_samples = np.asarray(samples_scaled, dtype=np.float64) * theta_std + theta_mean
    eigen_samples = theta_to_eigenvalues(theta_samples)
    truth_theta = np.asarray(validation["theta_softplus"])[evaluation_index]
    truth_eigen = np.asarray(validation["truth_eigenvalues"])[evaluation_index]
    base = np.asarray(validation["base_prediction_eigenvalues"])[evaluation_index]
    weight = np.asarray(validation["natural_weight"])[evaluation_index]
    theta_ranks = randomized_pit(theta_samples, truth_theta, args.seed + 1)
    eigen_ranks = randomized_pit(eigen_samples, truth_eigen, args.seed + 2)

    report = {
        "schema_version": "p12a-calibration-audit-v1",
        "created_utc": utc_now(),
        "purpose": "diagnose before correction; no recalibration fitted",
        "selection_phase": ready["validation_phase"],
        "sealed_phase": ready["sealed_phase"],
        "sealed_phase_opened": False,
        "evaluation_folds": ready["validation"]["evaluation_folds"],
        "calibration_folds_reserved": ready["validation"]["calibration_folds"],
        "evaluation_rows": int(len(evaluation_index)),
        "posterior_samples_per_row": int(args.n_posterior_samples),
        "rank_definition": "(# draws below truth + U[0,1])/(S+1)",
        "coordinate_warning": (
            "original SBC ranked lambda1 plus two softplus eigengap coordinates, "
            "not physical lambda1/lambda2/lambda3"
        ),
        "global": {
            "theta_rank": rank_summary(theta_ranks, weight, THETA_NAMES),
            "eigen_rank": rank_summary(eigen_ranks, weight, EIGEN_NAMES),
            "intervals": interval_report(eigen_samples, truth_eigen, base, weight),
        },
        "conditional": conditional_report(
            theta_ranks, eigen_ranks, eigen_samples, validation, evaluation_index
        ),
        "tarp_matched_rows_and_draws": tarp_diagnostic(
            np.asarray(samples_scaled),
            theta_truth_scaled[evaluation_index],
            args.seed,
            args.bootstrap_repeats,
            args.bootstrap_rows,
        ),
        "provenance": {
            "dataset_marker": str(ready_path),
            "dataset_marker_sha256": sha256(ready_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "source_result": str(complete_path),
            "source_result_sha256": sha256(complete_path),
            "samples": str(sample_path),
            "evaluation_index": str(args.output_root / "evaluation_index.npy"),
            "calibration_index": str(args.output_root / "calibration_index.npy"),
        },
    }
    atomic_json(args.output_root / "P12A_CALIBRATION_AUDIT.json", report)
    plot_ranks(theta_ranks, eigen_ranks, args.output_root / "p12a_rank_histograms")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

