#!/usr/bin/env python3
"""Render the frozen P12-A TARP coverage curve from cached ph006 draws.

This is plotting-only post-processing. It uses the exact 50k folds-2--4 rows,
512 posterior draws, standardized ordered-softplus target coordinates and seed
registered by ``p12_calibration_diagnostics.py``. It never samples the posterior,
fits a correction, or opens ph001.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import ACCENT_COLORS, TEXT_COLOR, apply_style
from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")
AUDIT = ROOT / "fmpe_seed42" / "calibration_audit_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ordered_curve(ecp: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, one-dimensional TARP coordinates ordered by alpha."""
    ecp = np.asarray(ecp, dtype=np.float64).squeeze()
    alpha = np.asarray(alpha, dtype=np.float64).squeeze()
    if ecp.ndim != 1 or alpha.ndim != 1 or ecp.shape != alpha.shape:
        raise ValueError(f"invalid TARP curve shapes: ecp={ecp.shape}, alpha={alpha.shape}")
    if not np.all(np.isfinite(ecp)) or not np.all(np.isfinite(alpha)):
        raise ValueError("TARP curve contains non-finite values")
    order = np.argsort(alpha)
    alpha = alpha[order]
    ecp = ecp[order]
    if np.any(np.diff(alpha) <= 0.0):
        raise ValueError("TARP alpha coordinates are not unique")
    return ecp, alpha


def curve_subsample_indices(
    alpha: np.ndarray, ecp: np.ndarray, max_points: int = 501
) -> np.ndarray:
    """Keep a compact representative curve plus the exact worst-deviation point."""
    if max_points < 3:
        raise ValueError("max_points must be at least three")
    if len(alpha) <= max_points:
        return np.arange(len(alpha), dtype=np.int64)
    evenly_spaced = np.linspace(0, len(alpha) - 1, max_points, dtype=np.int64)
    worst = int(np.argmax(np.abs(ecp - alpha)))
    return np.unique(np.concatenate((evenly_spaced, np.asarray([worst]))))


def render_tarp(
    output: Path,
    alpha: np.ndarray,
    ecp: np.ndarray,
    max_deviation: float,
    bootstrap_q95_max_deviation: float,
    rows: int,
    draws: int,
) -> None:
    """Plot coverage and residual panels using the canonical GraphWeb style."""
    apply_style()
    figure, (coverage_ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        constrained_layout=True,
    )
    grid = np.linspace(0.0, 1.0, 501)
    lower_gate = np.maximum(0.0, grid - 0.05)
    upper_gate = np.minimum(1.0, grid + 0.05)
    lower_spatial = np.maximum(0.0, grid - bootstrap_q95_max_deviation)
    upper_spatial = np.minimum(1.0, grid + bootstrap_q95_max_deviation)

    coverage_ax.fill_between(
        grid,
        lower_gate,
        upper_gate,
        color=TEXT_COLOR,
        alpha=0.06,
        label=r"Registered $|\mathrm{ECP}-\alpha|\leq0.05$ gate",
    )
    coverage_ax.fill_between(
        grid,
        lower_spatial,
        upper_spatial,
        color=ACCENT_COLORS["magenta"],
        alpha=0.12,
        label=(
            "Spatial resample 95th-percentile max-deviation "
            f"envelope (±{bootstrap_q95_max_deviation:.3f})"
        ),
    )
    coverage_ax.plot(
        grid,
        grid,
        color=TEXT_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Ideal calibration",
    )
    coverage_ax.plot(
        alpha,
        ecp,
        color=ACCENT_COLORS["blue"],
        linewidth=2.8,
        label=f"P12-A (max deviation {max_deviation:.4f})",
    )
    coverage_ax.set(
        title="P12-A TARP coverage on held-out ph006 spatial folds",
        ylabel="Expected coverage probability",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    coverage_ax.grid(True, alpha=0.15)
    coverage_ax.legend(loc="lower right", fontsize=10)
    coverage_ax.text(
        0.03,
        0.97,
        f"{rows:,} galaxies × {draws} draws\n"
        "joint 3-D standardized ordered-softplus coordinates",
        transform=coverage_ax.transAxes,
        va="top",
        fontsize=10,
    )

    residual = ecp - alpha
    residual_ax.axhspan(-0.05, 0.05, color=TEXT_COLOR, alpha=0.06)
    residual_ax.axhspan(
        -bootstrap_q95_max_deviation,
        bootstrap_q95_max_deviation,
        color=ACCENT_COLORS["magenta"],
        alpha=0.12,
    )
    residual_ax.axhline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.5)
    residual_ax.plot(alpha, residual, color=ACCENT_COLORS["blue"], linewidth=2.5)
    residual_ax.set(
        xlabel=r"Credibility level $\alpha$",
        ylabel=r"ECP $-\alpha$",
        ylim=(-0.055, 0.055),
    )
    residual_ax.grid(True, alpha=0.15)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "fmpe_seed42" / "fmpe_estimator.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    marker_path = args.dataset_root / "P12A_DATASET_READY.json"
    audit_path = args.audit_root / "P12A_CALIBRATION_AUDIT.json"
    evaluation_index_path = args.audit_root / "evaluation_index.npy"
    sample_path = args.audit_root / "evaluation_samples_scaled.npy"
    marker = json.loads(marker_path.read_text())
    audit = json.loads(audit_path.read_text())
    if marker.get("sealed_phase_opened") or audit.get("sealed_phase_opened"):
        raise RuntimeError("sealed ph001 guard failed")
    if marker["validation_phase"] != "ph006" or audit["selection_phase"] != "ph006":
        raise RuntimeError("TARP plot is frozen to ph006")
    if sha256(marker_path) != audit["provenance"]["dataset_marker_sha256"]:
        raise RuntimeError("dataset marker differs from the calibration audit")
    if sha256(args.checkpoint) != audit["provenance"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint differs from the calibration audit")

    evaluation_index = np.load(evaluation_index_path)
    samples_scaled = np.load(sample_path, mmap_mode="r")
    validation = np.load(marker["validation"]["path"])
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    theta_mean = np.asarray(checkpoint["theta_mean"], dtype=np.float64)
    theta_std = np.asarray(checkpoint["theta_std"], dtype=np.float64)
    truth_scaled = (
        np.asarray(validation["theta_softplus"])[evaluation_index] - theta_mean
    ) / theta_std
    expected_shape = (
        audit["evaluation_rows"],
        audit["posterior_samples_per_row"],
        3,
    )
    if samples_scaled.shape != expected_shape or truth_scaled.shape != expected_shape[::2]:
        raise RuntimeError(
            f"cached TARP shape mismatch: samples={samples_scaled.shape}, "
            f"truth={truth_scaled.shape}, expected={expected_shape}"
        )

    import tarp

    ecp, alpha = tarp.get_tarp_coverage(
        np.transpose(samples_scaled, (1, 0, 2)),
        truth_scaled,
        norm=True,
        bootstrap=False,
        seed=args.seed,
    )
    ecp, alpha = ordered_curve(ecp, alpha)
    max_deviation = float(np.max(np.abs(ecp - alpha)))
    registered = audit["tarp_matched_rows_and_draws"]
    if not np.isclose(
        max_deviation,
        registered["full_max_abs_ecp_minus_alpha"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "recomputed TARP curve does not reproduce the frozen audit: "
            f"{max_deviation} vs {registered['full_max_abs_ecp_minus_alpha']}"
        )
    bootstrap_q95 = float(registered["bootstrap_max_abs_quantiles"][2])
    output = args.audit_root / "p12a_tarp_curve"
    render_tarp(
        output,
        alpha,
        ecp,
        max_deviation,
        bootstrap_q95,
        len(evaluation_index),
        samples_scaled.shape[1],
    )
    stored = curve_subsample_indices(alpha, ecp)
    report = {
        "schema_version": "p12a-tarp-curve-v1",
        "created_utc": utc_now(),
        "purpose": "plot frozen TARP audit; no fitting, correction, or posterior resampling",
        "selection_phase": "ph006",
        "evaluation_folds": audit["evaluation_folds"],
        "sealed_phase": marker["sealed_phase"],
        "sealed_phase_opened": False,
        "rows": int(len(evaluation_index)),
        "posterior_draws_per_row": int(samples_scaled.shape[1]),
        "coordinate_system": "standardized ordered-softplus lambda1 plus two eigengap coordinates",
        "joint_dimension": 3,
        "tarp_norm": True,
        "seed": args.seed,
        "full_curve_points": int(len(alpha)),
        "stored_curve_points": int(len(stored)),
        "alpha": alpha[stored].tolist(),
        "expected_coverage_probability": ecp[stored].tolist(),
        "ecp_minus_alpha": (ecp[stored] - alpha[stored]).tolist(),
        "max_abs_ecp_minus_alpha": max_deviation,
        "registered_gate": 0.05,
        "pass_0p05": bool(max_deviation <= 0.05),
        "spatial_bootstrap_note": (
            "shaded narrow envelope is the frozen audit's 95th percentile of the "
            "maximum absolute deviation across 100 cap+superblock resamples of 10k rows; "
            "it is not a pointwise confidence interval"
        ),
        "spatial_bootstrap_q95_max_abs_deviation": bootstrap_q95,
        "provenance": {
            "dataset_marker": str(marker_path),
            "dataset_marker_sha256": sha256(marker_path),
            "audit_report": str(audit_path),
            "audit_report_sha256": sha256(audit_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "evaluation_index": str(evaluation_index_path),
            "evaluation_index_sha256": sha256(evaluation_index_path),
            "posterior_samples": str(sample_path),
            "posterior_samples_sha256": sha256(sample_path),
            "validation_sample": marker["validation"]["path"],
            "validation_sample_sha256": marker["validation"]["sha256"],
        },
    }
    atomic_json(args.audit_root / "P12A_TARP_CURVE.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
