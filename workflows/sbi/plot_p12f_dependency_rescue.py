#!/usr/bin/env python3
"""Render the P12-F v2 expanded-panel calibration and dependency audit."""
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


DRAW_COLORS = {
    "64": ACCENT_COLORS["blue"],
    "128": ACCENT_COLORS["magenta"],
    "256": ACCENT_COLORS["red"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser.parse_args()


def validate_report(report: dict) -> None:
    if report.get("schema_version") != "p12f-dependency-rescue-evaluation-v2":
        raise RuntimeError("unsupported dependency-rescue report")
    if report.get("phase") != "ph006" or report.get("ph001_opened"):
        raise PermissionError("dependency plots accept sealed ph006 evidence only")
    if set(report.get("nested_draw_reports", {})) != {"64", "128", "256"}:
        raise RuntimeError("nested draw reports are incomplete")
    if set(report.get("subpanel_reports_256_draws", {})) != {"0", "1", "2", "3"}:
        raise RuntimeError("four disjoint subpanel reports are required")
    if not report.get("physics_closure", {}).get("all_finite") or not report.get(
        "physics_closure", {}
    ).get("all_ordered"):
        raise RuntimeError("field-physics closure failed")


def _curve(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(payload["alpha"], dtype=np.float64),
        np.asarray(payload["expected_coverage_probability"], dtype=np.float64),
    )


def _gate(ax: plt.Axes) -> None:
    grid = np.linspace(0.0, 1.0, 501)
    ax.fill_between(
        grid,
        np.maximum(0.0, grid - 0.05),
        np.minimum(1.0, grid + 0.05),
        color=TEXT_COLOR,
        alpha=0.06,
    )
    ax.plot(grid, grid, "--", color=TEXT_COLOR, linewidth=1.2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"Credibility level $\alpha$", ylabel="Expected coverage")
    ax.grid(True, alpha=0.15)


def render_nested_tarp(report: dict, output: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        constrained_layout=True,
    )
    for column, key in enumerate(("ordered_eigenvalues", "eigengaps")):
        _gate(axes[0, column])
        axes[0, column].set_title(
            "Physical ordered eigenvalues" if column == 0 else "Physical eigengaps"
        )
        axes[1, column].axhspan(-0.05, 0.05, color=TEXT_COLOR, alpha=0.06)
        axes[1, column].axhline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.2)
        axes[1, column].set(xlim=(0, 1), ylim=(-0.14, 0.14), xlabel=r"Credibility level $\alpha$", ylabel=r"ECP $-\alpha$")
        axes[1, column].grid(True, alpha=0.15)
        for draws in ("64", "128", "256"):
            payload = report["nested_draw_reports"][draws]["tarp"][key]
            alpha, ecp = _curve(payload)
            label = f"{draws} draws (max {payload['maximum_deviation']:.3f})"
            axes[0, column].plot(alpha, ecp, color=DRAW_COLORS[draws], linewidth=2.0, label=label)
            axes[1, column].plot(alpha, ecp - alpha, color=DRAW_COLORS[draws], linewidth=1.8)
        axes[0, column].legend(loc="lower right")
    figure.suptitle(
        f"P12-F G1 calibration stability on {report['cores']:,} held-out ph006 cores",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_subpanels(report: dict, output: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for index in range(4):
        payload = report["subpanel_reports_256_draws"][str(index)]
        for row, key in enumerate(("ordered_eigenvalue_tarp", "eigengap_tarp")):
            ax = axes[row, index]
            _gate(ax)
            alpha, ecp = _curve(payload[key])
            ax.plot(alpha, ecp, color=DRAW_COLORS["256"], linewidth=2.1)
            ax.set_title(
                f"Subpanel {index + 1}: {'eigenvalues' if row == 0 else 'eigengaps'}\n"
                f"max |ECP-alpha|={payload[key]['maximum_deviation']:.3f}"
            )
        axes[0, index].text(
            0.03,
            0.96,
            f"{payload['cores']} cores\n{payload['galaxies']:,} galaxies\n"
            f"shells {payload['shell_counts']}",
            transform=axes[0, index].transAxes,
            va="top",
            fontsize=9,
        )
    figure.suptitle(
        "P12-F G1 panel-composition stability — four disjoint balanced panels",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_sbc(report: dict, output: Path) -> None:
    apply_style()
    names = ("lambda1", "lambda2", "lambda3", "gap12", "gap23")
    titles = (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_2-\lambda_1$", r"$\lambda_3-\lambda_2$")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    centres = np.arange(10) + 0.5
    for ax, name, title in zip(axes.ravel(), names, titles, strict=False):
        payload = report["sbc_256_draws"][name]
        mass = np.asarray(payload["decile_mass"])
        interval = np.asarray(payload["decile_mass_95ci"])
        error = np.maximum(np.vstack((mass - interval[:, 0], interval[:, 2] - mass)), 0.0)
        ax.bar(centres, mass, width=0.88, color=ACCENT_COLORS["blue"], alpha=0.75, edgecolor=TEXT_COLOR, linewidth=0.35)
        ax.errorbar(centres, mass, yerr=error, fmt="none", ecolor=TEXT_COLOR, capsize=2)
        ax.axhline(0.1, color=TEXT_COLOR, linestyle="--")
        ax.set(xlim=(0, 10), xlabel="Randomized rank decile", ylabel="Mass")
        ax.set_title(f"{title}: rank-CDF max {payload['rank_cdf_maximum_deviation']:.3f}")
        ax.grid(True, axis="y", alpha=0.15)
    axes.ravel()[-1].axis("off")
    figure.suptitle(
        "P12-F G1 joint-physics SBC — 256 draws, core-bootstrap intervals",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_dependency(report: dict, output: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    spatial = report["spatial_dependence_256_draws"]
    edges = np.asarray(spatial["distance_edges_voxels"])
    centre = 0.5 * (edges[:-1] + edges[1:]) * 5.0
    valid = np.asarray([row["pairs"] > 0 for row in spatial["bins"]])
    for key, label, color in (
        ("truth_residual_covariance", "truth innovation", ACCENT_COLORS["red"]),
        ("posterior_residual_covariance", "posterior draws", ACCENT_COLORS["blue"]),
    ):
        values = np.asarray([row.get(key, np.nan) for row in spatial["bins"]])
        axes[0].plot(centre[valid], values[valid], marker="o", label=label, color=color)
    axes[0].axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=1)
    axes[0].set(xlabel="Separation [Mpc/h]", ylabel="Residual covariance", title="Spatial covariance")
    axes[0].legend()
    for key, label, color in (
        ("truth_residual_variogram", "truth innovation", ACCENT_COLORS["red"]),
        ("posterior_residual_variogram", "posterior draws", ACCENT_COLORS["blue"]),
    ):
        values = np.asarray([row.get(key, np.nan) for row in spatial["bins"]])
        axes[1].plot(centre[valid], values[valid], marker="o", label=label, color=color)
    axes[1].set(xlabel="Separation [Mpc/h]", ylabel="Mean squared difference", title="Residual variogram")
    axes[1].legend()
    spectral = report["spectral_dependence_256_draws"]
    k_edges = np.asarray(spectral["k_edges_cycles_per_voxel"])
    k = 0.5 * (k_edges[:-1] + k_edges[1:]) / 5.0 * (2.0 * np.pi)
    truth = np.asarray(spectral["truth_innovation_power"])
    posterior = np.asarray(spectral["posterior_residual_power"])
    axes[2].plot(k, truth, marker="o", label="truth innovation", color=ACCENT_COLORS["red"])
    axes[2].plot(k, posterior, marker="o", label="posterior draws", color=ACCENT_COLORS["blue"])
    axes[2].set(xlabel=r"$k$ [$h$ Mpc$^{-1}$]", ylabel="Masked residual power", title="Scale dependence", yscale="log")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.15)
    figure.suptitle(
        "Where G1's dependency structure differs from held-out truth",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_conditional(report: dict, output: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    shell = report["conditional_reports_256_draws"]["shell"]
    for value in range(4):
        payload = shell[str(value)]
        for row, key in enumerate(("ordered_eigenvalue_tarp", "eigengap_tarp")):
            _gate(axes[row, value])
            alpha, ecp = _curve(payload[key])
            axes[row, value].plot(alpha, ecp, color=ACCENT_COLORS["blue"], linewidth=2)
            axes[row, value].set_title(
                f"Shell {value}: {'eigenvalues' if row == 0 else 'eigengaps'}\n"
                f"max |ECP-alpha|={payload[key]['maximum_deviation']:.3f}"
            )
    figure.suptitle(
        "P12-F G1 calibration versus redshift shell — 256 draws",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_seed_and_response_stability(report: dict, output: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    positions = np.arange(3)
    for offset, key, label, color in (
        (-0.12, "ordered_eigenvalues", "ordered eigenvalues", ACCENT_COLORS["blue"]),
        (0.12, "eigengaps", "eigengaps", ACCENT_COLORS["red"]),
    ):
        values = [
            report["nested_draw_reports"][str(draws)]["tarp"]["reference_seed_maxima"][key]
            for draws in (64, 128, 256)
        ]
        axes[0].boxplot(
            values,
            positions=positions + offset,
            widths=0.20,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.6},
            medianprops={"color": TEXT_COLOR},
        )
        axes[0].plot([], [], color=color, linewidth=6, alpha=0.6, label=label)
    axes[0].axhline(0.05, color=TEXT_COLOR, linestyle="--", label="registered gate")
    axes[0].set(xticks=positions, xticklabels=("64", "128", "256"), xlabel="Posterior draws", ylabel="Maximum |ECP-alpha|", title="TARP reference-seed sensitivity")
    axes[0].legend(fontsize=9)
    conditional = report["conditional_reports_256_draws"]
    for variable, label, color in (
        ("random_response", "random response", ACCENT_COLORS["blue"]),
        ("boundary_distance", "boundary distance", ACCENT_COLORS["magenta"]),
        ("tracer_density", "tracer density", ACCENT_COLORS["red"]),
    ):
        error68 = [conditional[variable][str(value)]["coverage"]["0.68"]["absolute_error"] for value in range(4)]
        error90 = [conditional[variable][str(value)]["coverage"]["0.90"]["absolute_error"] for value in range(4)]
        axes[1].plot(np.arange(4), error68, marker="o", color=color, label=f"{label}: 68%")
        axes[1].plot(np.arange(4), error90, marker="s", linestyle="--", color=color, alpha=0.75, label=f"{label}: 90%")
    axes[1].axhline(0.10, color=TEXT_COLOR, linestyle="--", label="conditional gate")
    axes[1].set(xticks=np.arange(4), xticklabels=("lowest", "q2", "q3", "highest"), xlabel="Observed-information quartile", ylabel="Absolute coverage error", title="Voxel calibration versus response/information")
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.15)
    figure.suptitle("P12-F G1 stability controls", fontsize=18, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text())
    validate_report(report)
    stems = {
        "nested_tarp": args.output_dir / "p12f_v2_nested_tarp",
        "subpanel_tarp": args.output_dir / "p12f_v2_subpanel_tarp",
        "sbc": args.output_dir / "p12f_v2_sbc",
        "dependency": args.output_dir / "p12f_v2_dependency",
        "conditional": args.output_dir / "p12f_v2_conditional_tarp",
        "stability": args.output_dir / "p12f_v2_seed_response_stability",
    }
    render_nested_tarp(report, stems["nested_tarp"])
    render_subpanels(report, stems["subpanel_tarp"])
    render_sbc(report, stems["sbc"])
    render_dependency(report, stems["dependency"])
    render_conditional(report, stems["conditional"])
    render_seed_and_response_stability(report, stems["stability"])
    atomic_json(
        args.evidence_output,
        {
            "schema_version": "p12f-dependency-rescue-plots-v2",
            "report": str(args.report.resolve()),
            "report_sha256": sha256(args.report),
            "figures": {name: str(path.with_suffix(".png").resolve()) for name, path in stems.items()},
            "selection_phase": "ph006",
            "ph001_opened": False,
            "pass": True,
        },
    )


if __name__ == "__main__":
    main()
