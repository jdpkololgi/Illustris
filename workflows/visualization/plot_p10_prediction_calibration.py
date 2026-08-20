#!/usr/bin/env python3
"""Prediction-conditioned calibration diagnostics for frozen P10 U-PATCH.

The familiar residual-versus-truth plot diagnoses shrinkage but is not a
calibration curve.  A deterministic conditional-mean estimator is calibrated
when ``E[truth | prediction] = prediction``.  This script therefore bins the
independent ph006 truth by the *predicted* value, reports reliability residuals,
and contrasts them with the existing truth-conditioned residuals.

Only the visible ph006 validation contract is supported.  ph001 is never
discovered or opened, and no fitted correction is written or applied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import TEXT_COLOR, apply_style
from workflows.visualization.plot_p10_predicted_vs_true import (
    DEFAULT_CONTRACT,
    DEFAULT_RUN_ROOT,
    LAMBDA_LABELS,
    SHELL_LABELS,
    align_model,
    load_validation_contract,
    sha256,
)


DEFAULT_OUTPUT = REPO_ROOT / "docs/figures/p10_multiphase_review_20260818"
DEFAULT_EVIDENCE = REPO_ROOT / "docs/evidence/p10/prediction_conditioned_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile-bins", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def quantile_reliability(
    prediction: np.ndarray,
    truth: np.ndarray,
    bins: int,
) -> dict:
    """Return deterministic equal-count reliability bins and summary metrics."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("prediction and truth must be aligned one-dimensional arrays")
    if len(prediction) < bins or bins < 2:
        raise ValueError("not enough rows for requested reliability bins")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("reliability inputs must be finite")

    order = np.argsort(prediction, kind="mergesort")
    groups = np.array_split(order, bins)
    rows = []
    for index, selected in enumerate(groups):
        p = prediction[selected]
        y = truth[selected]
        residual = y - p
        rows.append(
            {
                "bin": index,
                "n": int(len(selected)),
                "prediction_min": float(p[0]),
                "prediction_max": float(p[-1]),
                "mean_prediction": float(np.mean(p)),
                "mean_truth": float(np.mean(y)),
                "mean_truth_minus_prediction": float(np.mean(residual)),
                "standard_error_truth_minus_prediction": float(
                    np.std(residual, ddof=1) / np.sqrt(len(residual))
                ),
            }
        )

    counts = np.asarray([row["n"] for row in rows], dtype=np.float64)
    residuals = np.asarray(
        [row["mean_truth_minus_prediction"] for row in rows], dtype=np.float64
    )
    truth_sigma = float(np.std(truth))
    truth_on_prediction_slope, truth_on_prediction_intercept = np.polyfit(
        prediction, truth, 1
    )
    prediction_on_truth_slope, prediction_on_truth_intercept = np.polyfit(
        truth, prediction, 1
    )
    return {
        "n": int(len(prediction)),
        "bins": rows,
        "mean_bias_truth_minus_prediction": float(np.mean(truth - prediction)),
        "weighted_mean_absolute_calibration_error": float(
            np.average(np.abs(residuals), weights=counts)
        ),
        "normalized_weighted_mean_absolute_calibration_error": float(
            np.average(np.abs(residuals), weights=counts) / truth_sigma
        ),
        "maximum_absolute_bin_residual": float(np.max(np.abs(residuals))),
        "truth_on_prediction_slope": float(truth_on_prediction_slope),
        "truth_on_prediction_intercept": float(truth_on_prediction_intercept),
        "prediction_on_truth_slope": float(prediction_on_truth_slope),
        "prediction_on_truth_intercept": float(prediction_on_truth_intercept),
        "truth_standard_deviation": truth_sigma,
    }


def plot_reliability(report: dict, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 17), constrained_layout=True)
    for shell_id in range(4):
        for eigen_id in range(3):
            row = report["per_shell"][str(shell_id)][f"lambda{eigen_id + 1}"]
            bins = row["bins"]
            x = np.asarray([entry["mean_prediction"] for entry in bins])
            y = np.asarray([entry["mean_truth"] for entry in bins])
            error = np.asarray(
                [entry["standard_error_truth_minus_prediction"] for entry in bins]
            )
            low = min(float(x.min()), float(y.min()))
            high = max(float(x.max()), float(y.max()))
            pad = 0.05 * max(high - low, 1e-6)
            ax = axes[shell_id, eigen_id]
            ax.plot((low - pad, high + pad), (low - pad, high + pad), "--",
                    color=TEXT_COLOR, lw=1.0, label="conditional-mean calibration")
            ax.errorbar(x, y, yerr=error, fmt="o-", ms=3.5, lw=1.2,
                        color="#3A86FF", ecolor="#80AFFF", capsize=2)
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
            ax.text(
                0.03,
                0.97,
                "WMACE/σ="
                f"{row['normalized_weighted_mean_absolute_calibration_error']:.3f}\n"
                f"truth|pred slope={row['truth_on_prediction_slope']:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.72},
            )
            if shell_id == 0:
                ax.set_title(LAMBDA_LABELS[eigen_id])
            if shell_id == 3:
                ax.set_xlabel(f"Mean predicted {LAMBDA_LABELS[eigen_id]}")
            if eigen_id == 0:
                ax.set_ylabel(f"{SHELL_LABELS[shell_id]}\nMean true")
    fig.suptitle(
        "Frozen epoch-20 U-PATCH conditional-mean calibration on ph006\n"
        "Rows are equal-count bins conditioned on prediction; no correction is fitted",
        fontsize=18,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quantile_bins < 2:
        raise ValueError("quantile-bins must be at least two")
    apply_style()
    contract, parent, shell, truth = load_validation_contract(args.contract)
    model = align_model(args.run_root, "unet", args.seed, parent, truth)
    prediction = model["prediction"]
    runtime_epoch = int(model["report"]["runtime"]["epoch"])
    if runtime_epoch != 20:
        raise RuntimeError(f"expected frozen epoch-20 U-PATCH, got epoch {runtime_epoch}")

    per_shell = {}
    for shell_id in range(4):
        selected = shell == shell_id
        per_shell[str(shell_id)] = {
            f"lambda{eigen_id + 1}": quantile_reliability(
                prediction[selected, eigen_id], truth[selected, eigen_id], args.quantile_bins
            )
            for eigen_id in range(3)
        }
    pooled = {
        f"lambda{eigen_id + 1}": quantile_reliability(
            prediction[:, eigen_id], truth[:, eigen_id], args.quantile_bins
        )
        for eigen_id in range(3)
    }
    report = {
        "schema_version": "p10-ph006-prediction-calibration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_phase": "ph006",
        "model": "U-PATCH",
        "checkpoint_epoch": runtime_epoch,
        "conditioning_direction": "truth conditioned on deterministic prediction",
        "interpretation": (
            "E[truth|prediction]=prediction is the deterministic conditional-mean "
            "calibration target; this diagnostic does not fit or apply a correction"
        ),
        "quantile_bins": args.quantile_bins,
        "n_authoritative": int(len(parent)),
        "pooled": pooled,
        "per_shell": per_shell,
        "contract": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "prediction": str(model["prediction_path"]),
        "prediction_sha256": sha256(model["prediction_path"]),
        "parent_ids": str(model["parent_path"]),
        "parent_ids_sha256": sha256(model["parent_path"]),
        "sealed_phase_opened": False,
        "correction_fitted": False,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(
        report,
        args.output_dir / "prediction_conditioned_calibration_unet",
        args.dpi,
    )
    print(json.dumps({"evidence": str(args.evidence), "epoch": runtime_epoch}, indent=2))


if __name__ == "__main__":
    main()
