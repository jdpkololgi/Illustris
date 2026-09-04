#!/usr/bin/env python3
"""Render the frozen visual comparison for the P12-F3 conditional rescue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plot_style import apply_style
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.plot_p12f3_hierarchical_comparison import moving_average


KEYS = ("base3", "shuffled", "proxy7", "flow", "diffusion")
ORDER = ("reference",) + KEYS
LABELS = {
    "reference": "F3-L2b (30k)",
    "base3": "conditional Gaussian: base",
    "shuffled": "conditional Gaussian: shuffled proxies",
    "proxy7": "conditional Gaussian: aligned proxies",
    "flow": "F3-L2c conditional flow",
    "diffusion": "F3-L2d conditional diffusion",
}
COLORS = {
    "reference": "#555555",
    "base3": "#8c8c8c",
    "shuffled": "#e69f00",
    "proxy7": "#0072b2",
    "flow": "#cc3311",
    "diffusion": "#7a3e9d",
}
VARIABLES = (
    ("shell", "redshift shell"),
    ("random_response", "random response quartile"),
    ("boundary_distance", "boundary-distance quartile"),
    ("tracer_density", "tracer-density quartile"),
    ("true_environment", "true-density quartile (diagnostic)"),
)
DEPLOYABLE_CONDITIONS = ("shell", "random_response", "boundary_distance", "tracer_density")


def parse_keyed(values: list[str], *, required: tuple[str, ...]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in output:
            raise ValueError("keyed inputs must be unique KEY=PATH values")
        output[key] = Path(path)
    if set(output) != set(required):
        raise RuntimeError(f"expected keys {required}; received {tuple(output)}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--visual", action="append", required=True)
    parser.add_argument("--shear", action="append", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-visual", type=Path, required=True)
    parser.add_argument("--reference-shear", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--flow-loss", type=Path, required=True)
    parser.add_argument("--diffusion-loss", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("ph001_opened"):
        raise RuntimeError(f"blind phase was opened in {path}")
    return value


def deployable_conditional_error(report: dict) -> float:
    if "maximum_deployable_conditional_coverage_error" in report:
        return float(report["maximum_deployable_conditional_coverage_error"])
    rows = report["conditional_voxel_coverage"]
    return max(
        float(row["coverage"][level]["absolute_error"])
        for name in DEPLOYABLE_CONDITIONS
        for row in rows[name].values()
        for level in ("0.68", "0.90")
        if level in row.get("coverage", {})
    )


def curve(visual: dict, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    row = visual[name]
    return (
        np.asarray(row["alpha"], dtype=np.float64),
        np.asarray(row["expected_coverage_probability"], dtype=np.float64),
        float(row["maximum_deviation"]),
    )


def plot_overview(
    reports: dict[str, dict],
    visuals: dict[str, dict],
    selection: dict,
    flow_loss_trace: Path,
    diffusion_loss_trace: Path,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    comparisons = selection["comparisons"]
    names = ("proxy7_minus_base3", "proxy7_minus_proxy7_shuffled")
    labels = ("aligned - base", "aligned - shuffled")
    means = np.asarray([comparisons[name]["mean"] for name in names])
    intervals = np.asarray([comparisons[name]["q025_q50_q975"] for name in names])
    errors = np.vstack((means - intervals[:, 0], intervals[:, 2] - means))
    axes[0, 0].errorbar(
        np.arange(2), means, yerr=errors, fmt="o", capsize=5, color=COLORS["proxy7"]
    )
    axes[0, 0].axhline(0.0, color="black", linestyle="--")
    axes[0, 0].set(
        xticks=np.arange(2), xticklabels=labels, ylabel="paired held-out NLL difference",
        title="A. Observable proxies carry train-phase information",
    )

    for key, path in (("flow", flow_loss_trace), ("diffusion", diffusion_loss_trace)):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        update = np.asarray([row["update"] for row in rows])
        loss = np.asarray([row["loss"] for row in rows])
        smooth = moving_average(loss, 21)
        normalizer = float(np.median(smooth[: min(20, len(smooth))]))
        axes[0, 1].plot(
            update, smooth / normalizer, color=COLORS[key], linewidth=2,
            label=LABELS[key],
        )
    axes[0, 1].set(
        xlabel="optimizer update", ylabel="moving loss / initial window",
        title="B. Training trajectories (objective units normalized)",
    )
    axes[0, 1].legend(fontsize=7)

    for axis, name, title in (
        (axes[0, 2], "eigen_tarp", "C. Joint ordered-eigenvalue TARP"),
        (axes[1, 0], "gap_tarp", "D. Joint eigengap TARP"),
    ):
        axis.plot((0, 1), (0, 1), "--", color="black", linewidth=1, label="ideal")
        axis.fill_between(
            (0, 1), (0, 0.95), (0.05, 1), color="#7fbf7b", alpha=0.10,
            label=r"$\pm0.05$ gate",
        )
        for key in ORDER:
            alpha, ecp, maximum = curve(visuals[key], name)
            axis.plot(alpha, ecp, color=COLORS[key], linewidth=1.8,
                      label=f"{LABELS[key]} ({maximum:.3f})")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="nominal coverage",
                 ylabel="empirical coverage", title=title)
        axis.legend(fontsize=7, loc="lower right")

    gate_names = ("eigen TARP", "gap TARP", "global 68/90", "deployable conditional")
    limits = np.asarray((0.05, 0.05, 0.05, 0.10))
    for key in ORDER:
        report = reports[key]
        values = np.asarray(
            (
                report["tarp"]["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"],
                report["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"],
                max(report["global_coverage_error"].values()),
                deployable_conditional_error(report),
            ),
            dtype=np.float64,
        )
        axes[1, 1].plot(np.arange(4), values / limits, marker="o", color=COLORS[key],
                        label=LABELS[key])
    axes[1, 1].axhline(1.0, color="black", linestyle="--", label="registered limit")
    axes[1, 1].set(xticks=np.arange(4), xticklabels=gate_names,
                   ylabel="error / registered limit", title="E. Simultaneous calibration gates")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].legend(fontsize=7)

    score_names = ("energy", "coarse_energy", "marginal_crps", "variogram_p0p5")
    score_labels = ("energy", "coarse", "CRPS", "variogram")
    x = np.arange(4)
    for key in KEYS:
        ratio = np.asarray(
            [reports[key]["proper_scores"][name] / reports["reference"]["proper_scores"][name]
             for name in score_names]
        )
        axes[1, 2].plot(x, 100.0 * (ratio - 1.0), marker="o", color=COLORS[key],
                        label=LABELS[key])
    axes[1, 2].axhline(0.0, color="black", linestyle="--", label="F3-L2b")
    axes[1, 2].set(xticks=x, xticklabels=score_labels, ylabel="change from F3-L2b [%]",
                   title="F. Proper scores (lower is better)")
    axes[1, 2].legend(fontsize=7)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("P12-F3 conditional-width rescue — held-out ph006", fontsize=17)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_conditional(reports: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, len(VARIABLES), figsize=(22, 8), constrained_layout=True)
    for row_index, level in enumerate(("0.68", "0.90")):
        nominal = float(level)
        for column, (variable, label) in enumerate(VARIABLES):
            axis = axes[row_index, column]
            axis.axhline(nominal, color="black", linestyle="--", linewidth=1)
            axis.axhspan(nominal - (0.10 if row_index else 0.10),
                        nominal + (0.10 if row_index else 0.10), color="#7fbf7b", alpha=0.08)
            for key in ORDER:
                strata = reports[key]["conditional_voxel_coverage"][variable]
                labels_sorted = sorted(strata, key=lambda value: int(value))
                empirical = [strata[value]["coverage"][level]["empirical"] for value in labels_sorted]
                axis.plot(np.arange(len(empirical)), empirical, marker="o", color=COLORS[key],
                          linewidth=1.5, label=LABELS[key])
            strata_count = len(reports["reference"]["conditional_voxel_coverage"][variable])
            tick_labels = (
                [f"S{index + 1}" for index in range(strata_count)]
                if variable == "shell"
                else [f"Q{index + 1}" for index in range(strata_count)]
            )
            axis.set(xticks=np.arange(strata_count), xticklabels=tick_labels,
                     ylim=(0.45 if level == "0.68" else 0.68, 1.01),
                     ylabel=f"empirical {int(nominal * 100)}% coverage" if column == 0 else None,
                     title=label)
            axis.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=7, loc="lower left")
    figure.suptitle(
        "Where field-posterior widths succeed or fail\n"
        "green band is the registered conditional tolerance; true density is evaluation-only",
        fontsize=16,
    )
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_shear(reports: dict[str, dict], shear: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].plot((0, 1), (0, 1), "--", color="black", linewidth=1, label="ideal")
    axes[0, 0].fill_between((0, 1), (0, 0.95), (0.05, 1), color="#7fbf7b", alpha=0.10)
    for key in ORDER:
        row = shear[key]["joint_tarp"]
        axes[0, 0].plot(row["alpha"], row["expected_coverage_probability"],
                        color=COLORS[key], linewidth=1.8,
                        label=f"{LABELS[key]} ({row['maximum_deviation']:.3f})")
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="nominal coverage",
                   ylabel="empirical coverage", title="A. Five-component low-k shear TARP")
    axes[0, 0].legend(fontsize=7, loc="lower right")

    components = shear["reference"]["components"]
    x = np.arange(len(components))
    width = 0.12
    for panel, level in enumerate(("0.68", "0.90")):
        axis = axes[0, 1] if panel == 0 else axes[1, 0]
        for offset, key in enumerate(ORDER):
            values = [shear[key]["marginal"][name]["coverage"][level]["empirical"]
                      for name in components]
            axis.bar(x + (offset - 2.5) * width, values, width=width,
                     color=COLORS[key], alpha=0.82, label=LABELS[key])
        axis.axhline(float(level), color="black", linestyle="--")
        axis.set(xticks=x, xticklabels=components, ylim=(0.55 if panel == 0 else 0.72, 1.0),
                 ylabel="empirical coverage",
                 title=f"{'B' if panel == 0 else 'C'}. Marginal shear {int(float(level)*100)}% intervals")
        if panel == 0:
            axis.legend(fontsize=6)

    names = ("ordered", "eigengap", "shear TARP", "shear marginal")
    for key in ORDER:
        values = np.asarray(
            (
                reports[key]["tarp"]["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"],
                reports[key]["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"],
                shear[key]["joint_tarp_blocked"]["full_max_abs_ecp_minus_alpha"],
                shear[key]["maximum_marginal_coverage_error"],
            )
        )
        axes[1, 1].plot(np.arange(4), values / 0.05, marker="o", color=COLORS[key],
                        label=LABELS[key])
    axes[1, 1].axhline(1.0, color="black", linestyle="--", label="registered limit")
    axes[1, 1].set(xticks=np.arange(4), xticklabels=names, ylabel="error / 0.05 limit",
                   title="D. Joint and marginal dependence gates")
    axes[1, 1].legend(fontsize=7)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("Does conditional width modelling preserve tidal dependence?", fontsize=16)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    report_paths = parse_keyed(args.report, required=KEYS)
    visual_paths = parse_keyed(args.visual, required=KEYS)
    shear_paths = parse_keyed(args.shear, required=KEYS)
    reports = {key: read_json(path) for key, path in report_paths.items()}
    visuals = {key: read_json(path) for key, path in visual_paths.items()}
    shear = {key: read_json(path) for key, path in shear_paths.items()}
    reports["reference"] = read_json(args.reference_report)
    old_visual = read_json(args.reference_visual)
    visuals["reference"] = old_visual["methods"]["fourier_flow_h24"]
    shear["reference"] = read_json(args.reference_shear)
    selection = read_json(args.selection)
    if selection.get("selected_arm") != "proxy7" or not selection.get("pass"):
        raise RuntimeError("aligned observable-proxy gate did not pass")
    for key in ORDER:
        if reports[key].get("phase") != "ph006" or shear[key].get("phase") != "ph006":
            raise RuntimeError(f"{key} is not a ph006-only result")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(reports, visuals, selection, args.flow_loss, args.diffusion_loss,
                  args.output_dir / "p12f3_conditional_rescue_overview")
    plot_conditional(reports, args.output_dir / "p12f3_conditional_coverage")
    plot_shear(reports, shear, args.output_dir / "p12f3_conditional_shear")
    inputs = [*report_paths.values(), *visual_paths.values(), *shear_paths.values(),
              args.reference_report, args.reference_visual, args.reference_shear,
              args.selection, args.flow_loss, args.diffusion_loss]
    atomic_json(
        args.output_dir / "P12F3_CONDITIONAL_VISUAL_AUDIT.json",
        {
            "schema_version": "p12f3-conditional-visual-audit-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {str(path.resolve()): sha256(path) for path in inputs},
            "figures": [
                "p12f3_conditional_rescue_overview.png",
                "p12f3_conditional_coverage.png",
                "p12f3_conditional_shear.png",
            ],
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        },
    )


if __name__ == "__main__":
    main()
