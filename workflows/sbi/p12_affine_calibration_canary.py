#!/usr/bin/env python3
"""Cross-fit a bounded shell-aware affine correction for frozen P12-A.

The map is fitted only on ph006 calibration folds 0--1 in scaled ordered-
softplus coordinates.  It corrects a per-shell location offset and posterior
width.  Cross-fold stability is required before a map fitted on both calibration
folds is evaluated once on selection folds 2--4.  ph001 is never opened.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json
from workflows.sbi.p12_calibration_diagnostics import (
    EIGEN_NAMES,
    ROOT,
    build_posterior,
    choose_indices,
    interval_report,
    randomized_pit,
    rank_summary,
    sample_posterior_resumable,
    spatial_block_bootstrap,
    tarp_diagnostic,
)
from workflows.sbi.p12_train_base_response_fmpe import (
    paired_posterior_log_prob,
    theta_to_eigenvalues,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fit_moment_correction(
    samples: np.ndarray,
    truth: np.ndarray,
    shell: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    """Fit offset and scale using conditional posterior moment identities."""
    centre = np.asarray(samples, dtype=np.float64).mean(axis=1)
    variance = np.asarray(samples, dtype=np.float64).var(axis=1, ddof=1)
    offset = np.zeros((4, 3), dtype=np.float64)
    scale = np.ones((4, 3), dtype=np.float64)
    rows = {}
    for shell_value in range(4):
        chosen = shell == shell_value
        if np.count_nonzero(chosen) < 100:
            raise RuntimeError(f"too few calibration rows in shell {shell_value}")
        w = np.asarray(weight[chosen], dtype=np.float64)
        residual = truth[chosen] - centre[chosen]
        offset[shell_value] = np.average(residual, axis=0, weights=w)
        centred = residual - offset[shell_value]
        residual_second = np.average(np.square(centred), axis=0, weights=w)
        posterior_second = np.average(variance[chosen], axis=0, weights=w)
        if np.any(posterior_second <= 0):
            raise RuntimeError("degenerate posterior variance")
        scale[shell_value] = np.sqrt(residual_second / posterior_second)
        rows[str(shell_value)] = {
            "rows": int(np.count_nonzero(chosen)),
            "offset_scaled_theta": offset[shell_value].tolist(),
            "scale_scaled_theta": scale[shell_value].tolist(),
            "residual_second_moment": residual_second.tolist(),
            "posterior_variance_mean": posterior_second.tolist(),
        }
    return {"offset": offset, "scale": scale, "by_shell": rows}


def apply_correction(
    samples: np.ndarray,
    shell: np.ndarray,
    fit: dict[str, Any],
) -> np.ndarray:
    centre = np.asarray(samples, dtype=np.float64).mean(axis=1, keepdims=True)
    offset = np.asarray(fit["offset"], dtype=np.float64)[shell, None, :]
    scale = np.asarray(fit["scale"], dtype=np.float64)[shell, None, :]
    return centre + offset + scale * (np.asarray(samples, dtype=np.float64) - centre)


def calibration_summary(
    samples_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    truth_eigen: np.ndarray,
    base: np.ndarray,
    shell: np.ndarray,
    weight: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    theta = np.asarray(samples_scaled, dtype=np.float64) * theta_std + theta_mean
    eigen = theta_to_eigenvalues(theta)
    ranks = randomized_pit(eigen, truth_eigen, seed)
    global_rank = rank_summary(ranks, weight, EIGEN_NAMES)
    intervals = interval_report(eigen, truth_eigen, base, weight)
    by_shell = {}
    for value in range(4):
        chosen = shell == value
        by_shell[str(value)] = {
            "rows": int(np.count_nonzero(chosen)),
            "rank": rank_summary(ranks[chosen], weight[chosen], EIGEN_NAMES),
            "intervals": interval_report(
                eigen[chosen], truth_eigen[chosen], base[chosen], weight[chosen]
            ),
        }
    rank_distance = sum(
        row["weighted_ks_distance"]
        for row in global_rank["components"].values()
    )
    coverage_error = float(
        np.abs(np.asarray(intervals["coverage68"]) - 0.68).sum()
        + np.abs(np.asarray(intervals["coverage90"]) - 0.90).sum()
    )
    return {
        "global_rank": global_rank,
        "intervals": intervals,
        "by_shell": by_shell,
        "calibration_score_rank_plus_coverage": float(
            rank_distance + coverage_error
        ),
        "_ranks": ranks,
        "_eigen_samples": eigen,
    }


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def paired_log_prob_array(
    posterior: Any,
    theta: np.ndarray,
    context: np.ndarray,
    chunk: int,
    device: str,
) -> np.ndarray:
    values = []
    for start in range(0, len(theta), chunk):
        stop = min(start + chunk, len(theta))
        y = torch.as_tensor(theta[start:stop], dtype=torch.float32, device=device)
        x = torch.as_tensor(context[start:stop], dtype=torch.float32, device=device)
        with torch.no_grad():
            log_prob = paired_posterior_log_prob(posterior, y, x)
        values.append(np.asarray(log_prob.detach().cpu(), dtype=np.float64))
    return np.concatenate(values)


def block_bootstrap_mean_difference(
    difference: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    unique, inverse = np.unique(groups, return_inverse=True)
    blocks = len(unique)
    block_weight = np.bincount(inverse, weights=weight, minlength=blocks)
    block_value = np.bincount(
        inverse, weights=weight * difference, minlength=blocks
    )
    rng = np.random.default_rng(seed)
    probability = np.full(blocks, 1.0 / blocks)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        multiplicity = rng.multinomial(blocks, probability)
        draws[repeat] = (
            multiplicity @ block_value / (multiplicity @ block_weight)
        )
    return {
        "spatial_blocks": int(blocks),
        "bootstrap_repeats": int(repeats),
        "mean": float(np.average(difference, weights=weight)),
        "mean_95ci": np.quantile(draws, [0.025, 0.5, 0.975]).tolist(),
    }


def corrected_log_score(
    posterior: Any,
    samples_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    truth_eigen: np.ndarray,
    context_scaled: np.ndarray,
    shell: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
    fit: dict[str, Any],
    theta_std: np.ndarray,
    chunk: int,
    device: str,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    centre = np.asarray(samples_scaled, dtype=np.float64).mean(axis=1)
    offset = np.asarray(fit["offset"], dtype=np.float64)[shell]
    scale = np.asarray(fit["scale"], dtype=np.float64)[shell]
    preimage = centre + (truth_scaled - centre - offset) / scale
    base_log = paired_log_prob_array(
        posterior, truth_scaled, context_scaled, chunk, device
    )
    corrected_log = paired_log_prob_array(
        posterior, preimage, context_scaled, chunk, device
    ) - np.log(scale).sum(axis=1)
    gaps = np.maximum(np.diff(truth_eigen, axis=1), 1.0e-12)
    jacobian = (
        -float(np.log(theta_std).sum())
        - np.log1p(-np.exp(-gaps[:, 0]))
        - np.log1p(-np.exp(-gaps[:, 1]))
    )
    difference = corrected_log - base_log
    result = block_bootstrap_mean_difference(
        difference, weight, groups, bootstrap_repeats, seed
    )
    result.update(
        {
            "base_mean_physical_log_score": float(
                np.average(base_log + jacobian, weights=weight)
            ),
            "corrected_mean_physical_log_score": float(
                np.average(corrected_log + jacobian, weights=weight)
            ),
            "finite": bool(
                np.all(np.isfinite(base_log))
                and np.all(np.isfinite(corrected_log))
            ),
        }
    )
    return result


def parameter_stability(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    offset_difference = np.abs(first["offset"] - second["offset"])
    scale_difference = np.abs(first["scale"] - second["scale"])
    return {
        "max_abs_offset_difference_scaled_theta": float(offset_difference.max()),
        "max_abs_scale_difference": float(scale_difference.max()),
        "offset_difference_by_shell": offset_difference.tolist(),
        "scale_difference_by_shell": scale_difference.tolist(),
        "pass_offset_0p05": bool(offset_difference.max() <= 0.05),
        "pass_scale_0p10": bool(scale_difference.max() <= 0.10),
    }


def selection_gates(
    uncorrected: dict[str, Any],
    corrected: dict[str, Any],
    crossfits: list[dict[str, Any]],
    stability: dict[str, Any],
    log_score: dict[str, Any],
    tarp: dict[str, Any],
) -> dict[str, Any]:
    unc_r2 = np.asarray(uncorrected["intervals"]["posterior_mean_r2"])
    cor_r2 = np.asarray(corrected["intervals"]["posterior_mean_r2"])
    unc_d = np.asarray(
        [
            row["weighted_ks_distance"]
            for row in uncorrected["global_rank"]["components"].values()
        ]
    )
    cor_d = np.asarray(
        [
            row["weighted_ks_distance"]
            for row in corrected["global_rank"]["components"].values()
        ]
    )
    coverage_degradation = []
    for shell_value in range(4):
        unc = np.asarray(
            uncorrected["by_shell"][str(shell_value)]["intervals"]["coverage68"]
        )
        cor = np.asarray(
            corrected["by_shell"][str(shell_value)]["intervals"]["coverage68"]
        )
        coverage_degradation.extend(
            (np.abs(cor - 0.68) - np.abs(unc - 0.68)).tolist()
        )
    unc_sparse = np.asarray(
        uncorrected["by_shell"]["3"]["intervals"]["coverage68"]
    )[1:]
    cor_sparse = np.asarray(
        corrected["by_shell"]["3"]["intervals"]["coverage68"]
    )[1:]
    gates = {
        "crossfit_calibration_score_improves_both_directions": bool(
            all(
                row["corrected"]["calibration_score_rank_plus_coverage"]
                < row["uncorrected"]["calibration_score_rank_plus_coverage"]
                for row in crossfits
            )
        ),
        "parameter_offset_stable_0p05": stability["pass_offset_0p05"],
        "parameter_scale_stable_0p10": stability["pass_scale_0p10"],
        "selection_log_score_improves_spatial_95ci": bool(
            log_score["mean_95ci"][0] > 0.0
        ),
        "posterior_mean_r2_degradation_le_0p002": bool(
            np.all(cor_r2 >= unc_r2 - 0.002)
        ),
        "global_rank_distance_no_degradation_gt_0p002": bool(
            np.all(cor_d <= unc_d + 0.002)
        ),
        "no_shell_coverage_error_degradation_gt_0p01": bool(
            max(coverage_degradation) <= 0.01
        ),
        "sparse_lambda2_lambda3_coverage_error_improves": bool(
            np.abs(cor_sparse - 0.68).sum()
            < np.abs(unc_sparse - 0.68).sum()
        ),
        "corrected_tarp_pass": bool(tarp.get("pass_0p05")),
    }
    gates["promote_affine_correction"] = bool(all(gates.values()))
    return gates


def plot_comparison(
    uncorrected_ranks: np.ndarray,
    corrected_ranks: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for row, (ranks, title) in enumerate(
        ((uncorrected_ranks, "uncorrected"), (corrected_ranks, "affine corrected"))
    ):
        for component, name in enumerate(EIGEN_NAMES):
            axes[row, component].hist(
                ranks[:, component], bins=bins, density=True, color="#3A86FF"
            )
            axes[row, component].axhline(1.0, color="black", linestyle="--")
            axes[row, component].set_title(f"{title}: {name}")
            axes[row, component].set_xlabel("randomized posterior rank")
            axes[row, component].set_ylabel("density")
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
        "--audit-root", type=Path, default=ROOT / "fmpe_seed42/calibration_audit_v1"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "fmpe_seed42/affine_calibration_canary_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-posterior-samples", type=int, default=512)
    parser.add_argument("--calibration-rows", type=int, default=20_000)
    parser.add_argument("--evaluation-rows", type=int, default=50_000)
    parser.add_argument("--sample-chunk", type=int, default=2048)
    parser.add_argument("--log-prob-chunk", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=1_000)
    parser.add_argument("--tarp-bootstrap-repeats", type=int, default=100)
    parser.add_argument("--tarp-bootstrap-rows", type=int, default=10_000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P12 affine canary requires a GPU interactive allocation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    ready_path = args.dataset_root / "P12A_DATASET_READY.json"
    ready = json.loads(ready_path.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if ready.get("sealed_phase_opened"):
        raise RuntimeError("sealed phase was opened")
    if checkpoint["dataset_marker_sha256"] != sha256(ready_path):
        raise RuntimeError("checkpoint/dataset provenance mismatch")
    validation = np.load(ready["validation"]["path"])
    calibration_index, evaluation_index = choose_indices(
        np.asarray(validation["fold"], dtype=np.uint8),
        ready["validation"]["calibration_folds"],
        ready["validation"]["evaluation_folds"],
        args.calibration_rows,
        args.evaluation_rows,
        args.seed,
    )
    frozen_evaluation_index = np.load(args.audit_root / "evaluation_index.npy")
    if not np.array_equal(evaluation_index, frozen_evaluation_index):
        raise RuntimeError("selection rows differ from frozen calibration audit")
    evaluation_samples = np.load(
        args.audit_root / "evaluation_samples_scaled.npy", mmap_mode="r"
    )
    expected = (len(evaluation_index), args.n_posterior_samples, 3)
    if evaluation_samples.shape != expected:
        raise RuntimeError("frozen selection posterior sample shape mismatch")

    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float64)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float64)
    theta_mean = np.asarray(checkpoint["theta_mean"], dtype=np.float64)
    theta_std = np.asarray(checkpoint["theta_std"], dtype=np.float64)
    context_scaled = (
        (np.asarray(validation["context"], dtype=np.float32) - context_mean)
        / context_std
    ).astype(np.float32)
    truth_scaled = (
        (np.asarray(validation["theta_softplus"], dtype=np.float32) - theta_mean)
        / theta_std
    ).astype(np.float32)

    posterior = build_posterior(checkpoint, "cuda")
    calibration_samples = sample_posterior_resumable(
        posterior,
        context_scaled[calibration_index],
        args.n_posterior_samples,
        args.sample_chunk,
        "cuda",
        args.output_root / "calibration_samples_scaled.npy",
        args.output_root / "calibration_sampling_progress.json",
    )
    calibration_fold = np.asarray(validation["fold"])[calibration_index]
    calibration_shell = np.asarray(validation["shell"])[calibration_index]
    calibration_weight = np.asarray(validation["natural_weight"])[calibration_index]
    calibration_truth_scaled = truth_scaled[calibration_index]
    calibration_truth_eigen = np.asarray(
        validation["truth_eigenvalues"]
    )[calibration_index]
    calibration_base = np.asarray(
        validation["base_prediction_eigenvalues"]
    )[calibration_index]

    crossfits = []
    fits = {}
    for fit_fold, held_fold in ((0, 1), (1, 0)):
        fit_rows = calibration_fold == fit_fold
        held_rows = calibration_fold == held_fold
        fit = fit_moment_correction(
            calibration_samples[fit_rows],
            calibration_truth_scaled[fit_rows],
            calibration_shell[fit_rows],
            calibration_weight[fit_rows],
        )
        fits[str(fit_fold)] = fit
        uncorrected = calibration_summary(
            calibration_samples[held_rows],
            calibration_truth_scaled[held_rows],
            calibration_truth_eigen[held_rows],
            calibration_base[held_rows],
            calibration_shell[held_rows],
            calibration_weight[held_rows],
            theta_mean,
            theta_std,
            args.seed + 100 + held_fold,
        )
        corrected_samples = apply_correction(
            calibration_samples[held_rows], calibration_shell[held_rows], fit
        )
        corrected = calibration_summary(
            corrected_samples,
            calibration_truth_scaled[held_rows],
            calibration_truth_eigen[held_rows],
            calibration_base[held_rows],
            calibration_shell[held_rows],
            calibration_weight[held_rows],
            theta_mean,
            theta_std,
            args.seed + 100 + held_fold,
        )
        crossfits.append(
            {
                "fit_fold": fit_fold,
                "held_fold": held_fold,
                "fit": fit["by_shell"],
                "uncorrected": public_summary(uncorrected),
                "corrected": public_summary(corrected),
            }
        )
    stability = parameter_stability(fits["0"], fits["1"])
    full_fit = fit_moment_correction(
        calibration_samples,
        calibration_truth_scaled,
        calibration_shell,
        calibration_weight,
    )

    evaluation_shell = np.asarray(validation["shell"])[evaluation_index]
    evaluation_weight = np.asarray(validation["natural_weight"])[evaluation_index]
    evaluation_truth_eigen = np.asarray(
        validation["truth_eigenvalues"]
    )[evaluation_index]
    evaluation_base = np.asarray(
        validation["base_prediction_eigenvalues"]
    )[evaluation_index]
    evaluation_cap = np.asarray(validation["cap"])[evaluation_index].astype(np.int64)
    evaluation_superblock = np.asarray(
        validation["superblock_id"]
    )[evaluation_index].astype(np.int64)
    spatial_group = (evaluation_cap << 32) + evaluation_superblock
    uncorrected = calibration_summary(
        evaluation_samples,
        truth_scaled[evaluation_index],
        evaluation_truth_eigen,
        evaluation_base,
        evaluation_shell,
        evaluation_weight,
        theta_mean,
        theta_std,
        args.seed + 500,
    )
    corrected_samples = apply_correction(
        evaluation_samples, evaluation_shell, full_fit
    )
    corrected = calibration_summary(
        corrected_samples,
        truth_scaled[evaluation_index],
        evaluation_truth_eigen,
        evaluation_base,
        evaluation_shell,
        evaluation_weight,
        theta_mean,
        theta_std,
        args.seed + 500,
    )
    spatial = spatial_block_bootstrap(
        corrected["_ranks"],
        corrected["_eigen_samples"],
        evaluation_truth_eigen,
        evaluation_weight,
        spatial_group,
        args.bootstrap_repeats,
        args.seed + 600,
    )
    tarp = tarp_diagnostic(
        corrected_samples,
        truth_scaled[evaluation_index],
        spatial_group,
        args.seed + 700,
        args.tarp_bootstrap_repeats,
        args.tarp_bootstrap_rows,
    )
    log_score = corrected_log_score(
        posterior,
        evaluation_samples,
        truth_scaled[evaluation_index],
        evaluation_truth_eigen,
        context_scaled[evaluation_index],
        evaluation_shell,
        evaluation_weight,
        spatial_group,
        full_fit,
        theta_std,
        args.log_prob_chunk,
        "cuda",
        args.bootstrap_repeats,
        args.seed + 800,
    )
    gates = selection_gates(
        uncorrected, corrected, crossfits, stability, log_score, tarp
    )
    report = {
        "schema_version": "p12a-affine-calibration-canary-v1",
        "created_utc": utc_now(),
        "purpose": "bounded correction challenger; not production unless every gate passes",
        "sealed_phase": ready["sealed_phase"],
        "sealed_phase_opened": False,
        "calibration_phase": ready["validation_phase"],
        "calibration_folds": ready["validation"]["calibration_folds"],
        "selection_folds": ready["validation"]["evaluation_folds"],
        "target_coordinates": "scaled ordered-softplus lambda1/gap12/gap23",
        "correction": (
            "per-shell posterior-centre offset plus scale times centred draws"
        ),
        "crossfits": crossfits,
        "parameter_stability": stability,
        "full_calibration_fit": full_fit["by_shell"],
        "selection_uncorrected": public_summary(uncorrected),
        "selection_corrected": public_summary(corrected),
        "selection_corrected_spatial_bootstrap": spatial,
        "selection_corrected_tarp": tarp,
        "selection_log_score": log_score,
        "gates": gates,
        "promote_affine_correction": gates["promote_affine_correction"],
        "provenance": {
            "dataset_marker": str(ready_path),
            "dataset_marker_sha256": sha256(ready_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "audit_report": str(args.audit_root / "P12A_CALIBRATION_AUDIT.json"),
            "audit_report_sha256": sha256(
                args.audit_root / "P12A_CALIBRATION_AUDIT.json"
            ),
            "evaluation_samples": str(
                args.audit_root / "evaluation_samples_scaled.npy"
            ),
            "calibration_samples": str(
                args.output_root / "calibration_samples_scaled.npy"
            ),
        },
    }
    marker = args.output_root / "P12A_AFFINE_CALIBRATION_CANARY.json"
    atomic_json(marker, report)
    if gates["promote_affine_correction"]:
        atomic_json(
            args.output_root / "P12A_AFFINE_CORRECTION_CANDIDATE.json", report
        )
    else:
        atomic_json(
            args.output_root / "P12A_AFFINE_CORRECTION_REJECTED.json", report
        )
    plot_comparison(
        uncorrected["_ranks"],
        corrected["_ranks"],
        args.output_root / "p12a_affine_rank_comparison",
    )
    print(
        json.dumps(
            {
                "marker": str(marker),
                "promote_affine_correction": gates[
                    "promote_affine_correction"
                ],
                "gates": gates,
                "log_score": log_score,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
