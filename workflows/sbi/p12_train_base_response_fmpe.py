#!/usr/bin/env python3
"""Fit and evaluate the P12-A FMPE posterior on OOF U-PATCH predictions.

ph006 folds 0--1 are used for width calibration and folds 2--4 for the
selection report.  ph001 is never opened.  The posterior target uses ordered
softplus coordinates, so every physical posterior draw is ordered.
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
from scipy.stats import kstest, spearmanr
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def softplus(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.logaddexp(0.0, value)


def theta_to_eigenvalues(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    result = np.empty_like(theta)
    result[..., 0] = theta[..., 0]
    result[..., 1] = result[..., 0] + softplus(theta[..., 1])
    result[..., 2] = result[..., 1] + softplus(theta[..., 2])
    return result


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weight))


def weighted_r2(truth: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mean = weighted_mean(truth, weight)
    residual = np.sum(weight * np.square(truth - prediction))
    total = np.sum(weight * np.square(truth - mean))
    return float(1.0 - residual / total)


def weighted_coverage(
    samples: np.ndarray, truth: np.ndarray, weight: np.ndarray, q: float
) -> np.ndarray:
    lo = np.quantile(samples, (1.0 - q) / 2.0, axis=1)
    hi = np.quantile(samples, 1.0 - (1.0 - q) / 2.0, axis=1)
    inside = (truth >= lo) & (truth <= hi)
    return np.asarray(
        [weighted_mean(inside[:, index], weight) for index in range(inside.shape[1])]
    )


def calibrate_lambda1_tau(
    scaled_samples: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    truth: np.ndarray,
    weight: np.ndarray,
    target: float = 0.68,
) -> float:
    centre = scaled_samples.mean(axis=1, keepdims=True)

    def coverage(tau: float) -> float:
        tempered = centre + tau * (scaled_samples - centre)
        theta = tempered * theta_std + theta_mean
        eigen = theta_to_eigenvalues(theta)
        return float(weighted_coverage(eigen, truth, weight, target)[0])

    low, high = 0.25, 8.0
    if coverage(low) >= target:
        return low
    if coverage(high) <= target:
        return high
    for _ in range(40):
        middle = 0.5 * (low + high)
        if coverage(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sample_posterior(
    posterior: Any,
    context: np.ndarray,
    n_samples: int,
    chunk: int,
    device: str,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(context), chunk):
        values = torch.as_tensor(context[start : start + chunk], dtype=torch.float32, device=device)
        try:
            samples = posterior.sample_batched(
                (n_samples,), x=values, show_progress_bars=False
            )
            array = np.asarray(samples.detach().cpu(), dtype=np.float32)
            if array.shape[:2] == (n_samples, len(values)):
                array = np.transpose(array, (1, 0, 2))
        except (AttributeError, NotImplementedError):
            array = np.stack(
                [
                    np.asarray(
                        posterior.sample(
                            (n_samples,), x=values[index : index + 1], show_progress_bars=False
                        ).detach().cpu(),
                        dtype=np.float32,
                    )
                    for index in range(len(values))
                ],
                axis=0,
            )
        if array.shape != (len(values), n_samples, 3):
            raise RuntimeError(f"unexpected posterior sample shape {array.shape}")
        parts.append(array)
    return np.concatenate(parts, axis=0)


def paired_posterior_log_prob(
    posterior: Any, theta: torch.Tensor, context: torch.Tensor
) -> torch.Tensor:
    """Evaluate paired q(theta_i | x_i) across SBI posterior API versions."""
    if hasattr(posterior, "log_prob_batched"):
        return posterior.log_prob_batched(
            theta, x=context, norm_posterior=False, track_gradients=False
        )
    potential = getattr(posterior, "potential_fn", None)
    if potential is None or not hasattr(potential, "set_x"):
        raise AttributeError("posterior has no paired batched log-probability path")
    potential.set_x(context, x_is_iid=False)
    flow = getattr(potential, "flow", None)
    if flow is None:
        raise AttributeError("vector-field potential did not build a conditional flow")
    value = flow.log_prob(theta.unsqueeze(0)).reshape(-1)
    prior = getattr(potential, "prior", None)
    if prior is not None:
        in_support = torch.isfinite(prior.log_prob(theta))
        value = torch.where(in_support, value, torch.full_like(value, float("-inf")))
    return value


def physical_log_score(
    posterior: Any,
    theta_scaled: np.ndarray,
    context_scaled: np.ndarray,
    truth_eigenvalues: np.ndarray,
    theta_std: np.ndarray,
    weight: np.ndarray,
    chunk: int,
    device: str,
) -> dict:
    """Evaluate transformed FMPE log density in physical eigenvalue space."""
    values: list[np.ndarray] = []
    for start in range(0, len(theta_scaled), chunk):
        stop = min(start + chunk, len(theta_scaled))
        y = torch.as_tensor(theta_scaled[start:stop], dtype=torch.float32, device=device)
        x = torch.as_tensor(context_scaled[start:stop], dtype=torch.float32, device=device)
        log_prob = paired_posterior_log_prob(posterior, y, x)
        array = np.asarray(log_prob.detach().cpu(), dtype=np.float64).reshape(-1)
        if len(array) != stop - start:
            raise RuntimeError(
                f"paired posterior log-probability shape mismatch: {tuple(log_prob.shape)}"
            )
        values.append(array)
    log_q_y = np.concatenate(values)
    gaps = np.maximum(np.diff(truth_eigenvalues, axis=1), 1.0e-12)
    log_jacobian = (
        -float(np.log(theta_std).sum())
        - np.log1p(-np.exp(-gaps[:, 0]))
        - np.log1p(-np.exp(-gaps[:, 1]))
    )
    log_q_lambda = log_q_y + log_jacobian
    return {
        "rows": int(len(log_q_lambda)),
        "mean_log_prob_softplus_scaled": weighted_mean(log_q_y, weight),
        "mean_log_prob_physical_eigenvalues": weighted_mean(log_q_lambda, weight),
        "finite": bool(np.all(np.isfinite(log_q_lambda))),
    }


def tarp_diagnostic(samples_scaled: np.ndarray, theta_scaled: np.ndarray, seed: int) -> dict:
    try:
        import tarp

        ecp, alpha = tarp.get_tarp_coverage(
            np.transpose(samples_scaled, (1, 0, 2)),
            theta_scaled,
            norm=True,
            bootstrap=False,
            seed=seed,
        )
        maximum = float(np.max(np.abs(ecp - alpha)))
        return {
            "available": True,
            "max_abs_ecp_minus_alpha": maximum,
            "pass_0p05": bool(maximum <= 0.05),
        }
    except Exception as error:  # pragma: no cover - dependency/runtime diagnostic
        return {"available": False, "error": repr(error), "pass_0p05": False}


def reliability(probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray) -> dict:
    bins = np.linspace(0.0, 1.0, 11)
    index = np.minimum(np.digitize(probability, bins[1:-1]), 9)
    rows = []
    ece = 0.0
    total = float(weight.sum())
    for ibin in range(10):
        chosen = index == ibin
        if not np.any(chosen):
            continue
        w = weight[chosen]
        fraction = float(w.sum() / total)
        predicted = weighted_mean(probability[chosen], w)
        observed = weighted_mean(outcome[chosen], w)
        ece += fraction * abs(predicted - observed)
        rows.append(
            {
                "lo": float(bins[ibin]),
                "hi": float(bins[ibin + 1]),
                "weight_fraction": fraction,
                "predicted": predicted,
                "observed": observed,
            }
        )
    return {"ece_10bin": float(ece), "bins": rows}


def report_samples(
    scaled_samples: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
    theta_truth_scaled: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    shell: np.ndarray,
    cap: np.ndarray,
    support_log_distance: np.ndarray,
    weight: np.ndarray,
    tau: float,
) -> dict:
    centre = scaled_samples.mean(axis=1, keepdims=True)
    tempered = centre + tau * (scaled_samples - centre)
    theta_samples = tempered * theta_std + theta_mean
    eigen_samples = theta_to_eigenvalues(theta_samples)
    posterior_mean = eigen_samples.mean(axis=1)
    ranks = (tempered < theta_truth_scaled[:, None, :]).mean(axis=1)
    coverage68 = weighted_coverage(eigen_samples, truth, weight, 0.68)
    coverage90 = weighted_coverage(eigen_samples, truth, weight, 0.90)
    widths68 = np.quantile(eigen_samples, 0.84, axis=1) - np.quantile(
        eigen_samples, 0.16, axis=1
    )
    abs_error = np.abs(posterior_mean - truth)
    knot_probability = np.mean(eigen_samples[:, :, 0] > 0.2, axis=1)
    knot_truth = truth[:, 0] > 0.2
    result: dict[str, Any] = {
        "tau": float(tau),
        "rows": int(len(truth)),
        "coverage68": coverage68.tolist(),
        "coverage90": coverage90.tolist(),
        "sbc_ks_p": [float(kstest(ranks[:, i], "uniform").pvalue) for i in range(3)],
        "posterior_mean_r2": [
            weighted_r2(truth[:, i], posterior_mean[:, i], weight) for i in range(3)
        ],
        "base_r2": [weighted_r2(truth[:, i], base[:, i], weight) for i in range(3)],
        "posterior_mean_spearman": [
            float(spearmanr(truth[:, i], posterior_mean[:, i]).statistic)
            for i in range(3)
        ],
        "width_error_spearman": [
            float(spearmanr(widths68[:, i], abs_error[:, i]).statistic)
            for i in range(3)
        ],
        "posterior_variance_over_truth_variance": [
            float(
                np.average(np.var(eigen_samples[:, :, i], axis=1), weights=weight)
                / np.cov(truth[:, i], aweights=weight)
            )
            for i in range(3)
        ],
        "knot_brier": weighted_mean(np.square(knot_probability - knot_truth), weight),
        "knot_reliability": reliability(knot_probability, knot_truth, weight),
    }
    by_shell = {}
    for value in range(4):
        chosen = shell == value
        if np.any(chosen):
            by_shell[str(value)] = {
                "rows": int(chosen.sum()),
                "coverage68": weighted_coverage(
                    eigen_samples[chosen], truth[chosen], weight[chosen], 0.68
                ).tolist(),
                "coverage90": weighted_coverage(
                    eigen_samples[chosen], truth[chosen], weight[chosen], 0.90
                ).tolist(),
                "r2": [
                    weighted_r2(
                        truth[chosen, i], posterior_mean[chosen, i], weight[chosen]
                    )
                    for i in range(3)
                ],
            }
    result["by_shell"] = by_shell
    result["by_cap"] = {
        str(value): {
            "rows": int(np.count_nonzero(cap == value)),
            "coverage68": weighted_coverage(
                eigen_samples[cap == value], truth[cap == value], weight[cap == value], 0.68
            ).tolist(),
        }
        for value in (0, 1)
    }
    quantiles = np.quantile(support_log_distance, [0.0, 0.25, 0.5, 0.75, 1.0])
    support_rows = []
    for lo, hi in zip(quantiles[:-1], quantiles[1:], strict=True):
        chosen = (support_log_distance >= lo) & (
            support_log_distance <= hi if hi == quantiles[-1] else support_log_distance < hi
        )
        support_rows.append(
            {
                "lo_log1p_mpc": float(lo),
                "hi_log1p_mpc": float(hi),
                "rows": int(chosen.sum()),
                "coverage68": weighted_coverage(
                    eigen_samples[chosen], truth[chosen], weight[chosen], 0.68
                ).tolist(),
            }
        )
    result["by_random_support_boundary_distance_quantile"] = support_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "fmpe_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--n-posterior-samples", type=int, default=256)
    parser.add_argument("--calibration-rows", type=int, default=20_000)
    parser.add_argument("--evaluation-rows", type=int, default=50_000)
    parser.add_argument("--score-rows", type=int, default=10_000)
    parser.add_argument("--sample-chunk", type=int, default=2048)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    args = parser.parse_args()
    if args.dataloader_workers != 0:
        raise ValueError(
            "FMPE stores the training TensorDataset on CUDA; --dataloader-workers "
            "must remain zero to prevent forked CUDA initialization"
        )
    terminal = args.output_root / "P12A_COMPLETE.json"
    if terminal.exists():
        existing = json.loads(terminal.read_text())
        if existing.get("technical_complete") and not existing.get("sealed_phase_opened"):
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "path": str(terminal),
                        "calibration_pass": bool(existing.get("calibration_pass")),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
    if not torch.cuda.is_available():
        raise RuntimeError("P12-A FMPE requires a GPU interactive allocation")
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ready = json.loads((args.dataset_root / "P12A_DATASET_READY.json").read_text())
    if not ready.get("pass") or ready.get("sealed_phase_opened"):
        raise RuntimeError("P12-A dataset contract is not ready")
    train = np.load(ready["training"]["path"])
    validation = np.load(ready["validation"]["path"])
    context = np.asarray(train["context"], dtype=np.float32)
    theta = np.asarray(train["theta_softplus"], dtype=np.float32)
    context_mean = context.mean(axis=0, dtype=np.float64)
    context_std = context.std(axis=0, dtype=np.float64)
    theta_mean = theta.mean(axis=0, dtype=np.float64)
    theta_std = theta.std(axis=0, dtype=np.float64)
    if np.any(context_std <= 0) or np.any(theta_std <= 0):
        raise RuntimeError("degenerate P12-A transformation")
    x = ((context - context_mean) / context_std).astype(np.float32)
    y = ((theta - theta_mean) / theta_std).astype(np.float32)

    from sbi.inference import FMPE
    from sbi.neural_nets import posterior_flow_nn
    from sbi.utils import BoxUniform

    low = np.min(y, axis=0) - 0.2 * np.ptp(y, axis=0)
    high = np.max(y, axis=0) + 0.2 * np.ptp(y, axis=0)
    prior = BoxUniform(
        low=torch.as_tensor(low, dtype=torch.float32, device=device),
        high=torch.as_tensor(high, dtype=torch.float32, device=device),
    )
    builder = posterior_flow_nn(
        model="mlp",
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        z_score_theta="none",
        z_score_x="none",
    )
    inference = FMPE(prior=prior, density_estimator=builder, device=device)
    inference.append_simulations(torch.from_numpy(y), torch.from_numpy(x))
    estimator = inference.train(
        training_batch_size=args.batch_size,
        validation_fraction=0.1,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_epochs,
        clip_max_norm=5.0,
        show_train_summary=True,
        dataloader_kwargs={
            "num_workers": args.dataloader_workers,
            "pin_memory": False,
        },
    )
    posterior = inference.build_posterior(estimator)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "fmpe_estimator.pt"
    torch.save(
        {
            "schema_version": "p12a-fmpe-estimator-v1",
            "state_dict": estimator.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "prior_low": low,
            "prior_high": high,
            "hidden_features": args.hidden_features,
            "num_layers": args.num_layers,
            "seed": args.seed,
            "dataset_marker_sha256": sha256(args.dataset_root / "P12A_DATASET_READY.json"),
        },
        checkpoint,
    )

    vx = ((np.asarray(validation["context"], dtype=np.float32) - context_mean) / context_std).astype(np.float32)
    vtheta = ((np.asarray(validation["theta_softplus"], dtype=np.float32) - theta_mean) / theta_std).astype(np.float32)
    fold = np.asarray(validation["fold"], dtype=np.uint8)
    rng = np.random.default_rng(args.seed + 12000)

    def choose(mask: np.ndarray, maximum: int) -> np.ndarray:
        index = np.flatnonzero(mask)
        if len(index) > maximum:
            index = rng.choice(index, size=maximum, replace=False)
        return index

    calibration_index = choose(np.isin(fold, ready["validation"]["calibration_folds"]), args.calibration_rows)
    evaluation_index = choose(np.isin(fold, ready["validation"]["evaluation_folds"]), args.evaluation_rows)
    calibration_superblock = (
        validation["cap"][calibration_index].astype(np.int64) << 32
    ) + validation["superblock_id"][calibration_index].astype(np.int64)
    evaluation_superblock = (
        validation["cap"][evaluation_index].astype(np.int64) << 32
    ) + validation["superblock_id"][evaluation_index].astype(np.int64)
    if np.intersect1d(calibration_superblock, evaluation_superblock).size:
        raise RuntimeError("ph006 calibration/evaluation superblocks overlap")
    calibration_samples = sample_posterior(
        posterior, vx[calibration_index], args.n_posterior_samples, args.sample_chunk, device
    )
    tau = calibrate_lambda1_tau(
        calibration_samples,
        theta_mean,
        theta_std,
        validation["truth_eigenvalues"][calibration_index],
        validation["natural_weight"][calibration_index],
    )
    evaluation_samples = sample_posterior(
        posterior, vx[evaluation_index], args.n_posterior_samples, args.sample_chunk, device
    )
    untempered = report_samples(
        evaluation_samples,
        validation["truth_eigenvalues"][evaluation_index],
        validation["base_prediction_eigenvalues"][evaluation_index],
        vtheta[evaluation_index],
        theta_mean,
        theta_std,
        validation["shell"][evaluation_index],
        validation["cap"][evaluation_index],
        validation["context"][evaluation_index, -1],
        validation["natural_weight"][evaluation_index],
        1.0,
    )
    tempered = report_samples(
        evaluation_samples,
        validation["truth_eigenvalues"][evaluation_index],
        validation["base_prediction_eigenvalues"][evaluation_index],
        vtheta[evaluation_index],
        theta_mean,
        theta_std,
        validation["shell"][evaluation_index],
        validation["cap"][evaluation_index],
        validation["context"][evaluation_index, -1],
        validation["natural_weight"][evaluation_index],
        tau,
    )
    score_index = evaluation_index[: min(args.score_rows, len(evaluation_index))]
    log_score = physical_log_score(
        posterior,
        vtheta[score_index],
        vx[score_index],
        validation["truth_eigenvalues"][score_index],
        theta_std,
        validation["natural_weight"][score_index],
        min(args.sample_chunk, 512),
        device,
    )
    tempered_scaled = evaluation_samples * 1.0
    centre = tempered_scaled.mean(axis=1, keepdims=True)
    tempered_scaled = centre + tau * (tempered_scaled - centre)
    tarp_rows = min(args.score_rows, len(evaluation_index))
    tarp_result = tarp_diagnostic(
        tempered_scaled[:tarp_rows], vtheta[evaluation_index[:tarp_rows]], args.seed
    )
    marginal_coverage_pass = bool(
        np.all(np.abs(np.asarray(tempered["coverage68"]) - 0.68) <= 0.03)
        and np.all(np.abs(np.asarray(tempered["coverage90"]) - 0.90) <= 0.03)
    )
    shell_coverage_pass = bool(
        all(
            abs(value - 0.68) <= 0.06
            for row in tempered["by_shell"].values()
            for value in row["coverage68"]
        )
    )
    mean_accuracy_pass = bool(
        tempered["posterior_mean_r2"][0] >= tempered["base_r2"][0] - 0.02
    )
    report = {
        "schema_version": "p12a-base-response-fmpe-result-v1",
        "created_utc": utc_now(),
        "estimand": "per-galaxy ordered tidal-eigenvalue posterior conditional on H_fid",
        "not_joint_field_posterior": True,
        "training_phases": ready["training_phases"],
        "selection_phase": ready["validation_phase"],
        "sealed_phase": ready["sealed_phase"],
        "sealed_phase_opened": False,
        "conditioning_features": ready["feature_names"],
        "target_parameterization": "ordered softplus increments",
        "train_rows": int(len(train["shell"])),
        "calibration_rows": int(len(calibration_index)),
        "evaluation_rows": int(len(evaluation_index)),
        "calibration_folds": ready["validation"]["calibration_folds"],
        "evaluation_folds": ready["validation"]["evaluation_folds"],
        "calibration_evaluation_superblocks_disjoint": True,
        "lambda1_tau": float(tau),
        "untempered": untempered,
        "tempered": tempered,
        "gates": {
            "marginal_coverage": marginal_coverage_pass,
            "shell_conditional_coverage": shell_coverage_pass,
            "posterior_mean_accuracy": mean_accuracy_pass,
            "sbc_shape": bool(all(value > 0.01 for value in tempered["sbc_ks_p"])),
            "tarp": tarp_result["pass_0p05"],
            "held_out_degradation": None,
            "ph001_blind": None,
        },
        "technical_complete": True,
        "calibration_pass": bool(
            marginal_coverage_pass
            and shell_coverage_pass
            and mean_accuracy_pass
            and all(value > 0.01 for value in tempered["sbc_ks_p"])
            and tarp_result["pass_0p05"]
        ),
        "log_score": log_score,
        "tarp": tarp_result,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_marker": str(args.dataset_root / "P12A_DATASET_READY.json"),
        "dataset_marker_sha256": sha256(args.dataset_root / "P12A_DATASET_READY.json"),
    }
    atomic_json(output / "P12A_COMPLETE.json", report)
    if report["calibration_pass"]:
        atomic_json(output / "P12A_CALIBRATION_PASS.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
