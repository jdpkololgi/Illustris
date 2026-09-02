#!/usr/bin/env python3
"""Visual-first comparison of the P12-F G2 covariance control against G1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plot_style import ACCENT_COLORS, TEXT_COLOR, apply_style
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-report", type=Path, required=True)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--proper-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser.parse_args()


def _curve(report: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    payload = report["nested_draw_reports"]["256"]["tarp"][key]
    return (
        np.asarray(payload["alpha"], dtype=np.float64),
        np.asarray(payload["expected_coverage_probability"], dtype=np.float64),
    )


def _tarp_panel(ax: plt.Axes, g1: dict, g2: dict, key: str, title: str) -> None:
    alpha = np.linspace(0.0, 1.0, 501)
    ax.fill_between(
        alpha,
        np.maximum(0.0, alpha - 0.05),
        np.minimum(1.0, alpha + 0.05),
        color=TEXT_COLOR,
        alpha=0.06,
    )
    ax.plot(alpha, alpha, "--", color=TEXT_COLOR, linewidth=1.1)
    for report, label, color in (
        (g1, "G1 global covariance", ACCENT_COLORS["magenta"]),
        (g2, "G2 shell/scale covariance", ACCENT_COLORS["blue"]),
    ):
        x, y = _curve(report, key)
        maximum = report["nested_draw_reports"]["256"]["tarp"][key][
            "maximum_deviation"
        ]
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{label}: {maximum:.3f}")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"Credibility level $\alpha$",
        ylabel="Expected coverage",
        title=title,
    )
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="lower right")


def main() -> None:
    args = parse_args()
    g1 = json.loads(args.g1_report.read_text())
    g2 = json.loads(args.g2_report.read_text())
    proper = json.loads(args.proper_scores.read_text())
    if (
        g1.get("method") != "gaussian_correlated_g1"
        or g2.get("method") != "gaussian_shell_correlated_g2"
        or g1.get("ph001_opened")
        or g2.get("ph001_opened")
        or proper.get("ph001_opened")
    ):
        raise RuntimeError("G1/G2 visual comparison received unsafe evidence")

    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    _tarp_panel(axes[0, 0], g1, g2, "ordered_eigenvalues", "Ordered eigenvalue TARP")
    _tarp_panel(axes[0, 1], g1, g2, "eigengaps", "Eigengap TARP")

    for report, label, color in (
        (g1, "G1", ACCENT_COLORS["magenta"]),
        (g2, "G2", ACCENT_COLORS["blue"]),
    ):
        spectral = report["spectral_dependence_256_draws"]
        edges = np.asarray(spectral["k_edges_cycles_per_voxel"])
        k = 0.5 * (edges[:-1] + edges[1:]) / 5.0 * (2.0 * np.pi)
        ratio = np.asarray(spectral["posterior_residual_power"]) / np.maximum(
            np.asarray(spectral["truth_innovation_power"]), 1e-12
        )
        axes[1, 0].plot(k, ratio, marker="o", color=color, label=label)
    axes[1, 0].axhline(1.0, color=TEXT_COLOR, linestyle="--")
    axes[1, 0].set(
        xlabel=r"$k$ [$h$ Mpc$^{-1}$]",
        ylabel="posterior / held-out residual power",
        title="Did shell conditioning repair the scale dependence?",
    )
    axes[1, 0].grid(True, alpha=0.15)
    axes[1, 0].legend()

    names = ("energy", "variogram_p0p5", "coarse_energy", "marginal_crps")
    labels = ("Energy", "Variogram", "Coarse energy", "CRPS")
    improvement = [
        proper["paired_core_bootstrap"][name]["fractional_improvement"]
        for name in names
    ]
    interval = np.asarray(
        [proper["paired_core_bootstrap"][name]["interval95"] for name in names]
    )
    error = np.vstack((np.asarray(improvement) - interval[:, 0], interval[:, 1] - improvement))
    color = [ACCENT_COLORS["blue"] if value >= 0 else ACCENT_COLORS["red"] for value in improvement]
    axes[1, 1].bar(np.arange(4), 100.0 * np.asarray(improvement), color=color, alpha=0.8)
    axes[1, 1].errorbar(
        np.arange(4),
        100.0 * np.asarray(improvement),
        yerr=100.0 * error,
        fmt="none",
        ecolor=TEXT_COLOR,
        capsize=3,
    )
    axes[1, 1].axhline(0.0, color=TEXT_COLOR, linestyle="--")
    axes[1, 1].axhline(2.0, color=TEXT_COLOR, linestyle=":", label="primary +2% gate")
    axes[1, 1].set(
        xticks=np.arange(4),
        xticklabels=labels,
        ylabel="G2 improvement over G1 [%]",
        title="Paired 1,024-core proper scores (64 draws)",
    )
    axes[1, 1].grid(True, axis="y", alpha=0.15)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "P12-F v2 bounded covariance rescue: G2 versus frozen G1",
        fontsize=18,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    atomic_json(
        args.evidence_output,
        {
            "schema_version": "p12f-g2-vs-g1-plot-v1",
            "g1_report_sha256": sha256(args.g1_report),
            "g2_report_sha256": sha256(args.g2_report),
            "proper_scores_sha256": sha256(args.proper_scores),
            "figure_png": str(args.output.with_suffix(".png").resolve()),
            "figure_pdf": str(args.output.with_suffix(".pdf").resolve()),
            "ph001_opened": False,
            "pass": True,
        },
    )


if __name__ == "__main__":
    main()
