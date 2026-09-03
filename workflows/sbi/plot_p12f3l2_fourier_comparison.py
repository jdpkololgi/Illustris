#!/usr/bin/env python3
"""Render the visual, anti-gaming P12-F3-L2 comparison on ph006."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.plot_p12f3_hierarchical_comparison import analyze_archive, moving_average


METHODS = ("g1_wide_h24", "fourier_gaussian_h24", "fourier_flow_h24")
LABELS = {
    "g1_wide_h24": "G1 stationary residual",
    "fourier_gaussian_h24": "direct Fourier Gaussian",
    "fourier_flow_h24": "direct conditional Fourier flow",
}
COLORS = {
    "g1_wide_h24": "#555555",
    "fourier_gaussian_h24": "#2878b5",
    "fourier_flow_h24": "#d1495b",
}


def parse_keyed(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in output:
            raise ValueError("inputs must be unique METHOD=PATH values")
        output[key] = Path(path)
    if set(output) != set(METHODS):
        raise RuntimeError("plot requires the exact registered F3-L2 method triplet")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--shear-report", action="append", required=True)
    parser.add_argument("--loss-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return (
        np.asarray([row["update"] for row in rows], dtype=np.int64),
        np.asarray([row["loss"] for row in rows], dtype=np.float64),
    )


def plot_main(summary: dict, reports: dict, shear: dict, trace: tuple[np.ndarray, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    update, loss = trace
    axes[0, 0].plot(update, loss, color=COLORS["fourier_flow_h24"], alpha=.20, linewidth=.7)
    axes[0, 0].plot(update, moving_average(loss, 9), color=COLORS["fourier_flow_h24"], linewidth=2)
    axes[0, 0].set(title="A. Conditional Fourier-flow training", xlabel="optimizer update", ylabel="equal-band flow loss")

    x = np.arange(2)
    width = .24
    for offset, method in enumerate(METHODS):
        axes[0, 1].bar(x + (offset - 1) * width, summary[method]["posterior_to_truth_power"][:2], width, color=COLORS[method], label=LABELS[method])
    axes[0, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhspan(.9, 1.1, color="#8bc34a", alpha=.12)
    axes[0, 1].set(xticks=x, xticklabels=("longest band K1", "second band K2"), ylabel="posterior / truth residual power", title="B. Registered low-k amplitudes")
    axes[0, 1].legend(fontsize=8)

    for axis, key, title in (
        (axes[0, 2], "eigen_tarp", "C. Joint ordered-eigenvalue TARP"),
        (axes[1, 0], "gap_tarp", "D. Joint eigengap TARP"),
    ):
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
        for method in METHODS:
            row = summary[method][key]
            blocked = reports[method]["tarp"][
                "ordered_eigenvalues" if key == "eigen_tarp" else "eigengaps"
            ]["full_max_abs_ecp_minus_alpha"]
            axis.plot(
                row["alpha"], row["expected_coverage_probability"],
                color=COLORS[method],
                label=(f"{LABELS[method]} "
                       f"(pooled {row['maximum_deviation']:.3f}; "
                       f"blocked {blocked:.3f})"),
            )
        axis.fill_between([0, 1], [0, .95], [.05, 1], color="#8bc34a", alpha=.06)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="nominal coverage", ylabel="empirical coverage", title=title)
        axis.legend(fontsize=7)

    axis = axes[1, 1]
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
    for method in METHODS:
        row = shear[method]["joint_tarp"]
        blocked = shear[method]["joint_tarp_blocked"][
            "full_max_abs_ecp_minus_alpha"
        ]
        axis.plot(
            row["alpha"], row["expected_coverage_probability"],
            color=COLORS[method],
            label=(f"{LABELS[method]} "
                   f"(pooled {row['maximum_deviation']:.3f}; "
                   f"blocked {blocked:.3f})"),
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="nominal coverage", ylabel="empirical coverage", title="E. Joint five-component low-k shear TARP")
    axis.legend(fontsize=7)

    axis = axes[1, 2]
    names = ("eigen TARP", "gap TARP", "global cov.", "conditional cov.", "shear cov.", "shear TARP")
    limits = np.asarray((.05, .05, .05, .10, .05, .05))
    for method in METHODS:
        values = np.asarray((
            reports[method]["tarp"]["ordered_eigenvalues"]
            ["full_max_abs_ecp_minus_alpha"],
            reports[method]["tarp"]["eigengaps"]
            ["full_max_abs_ecp_minus_alpha"],
            max(reports[method]["global_coverage_error"].values()),
            reports[method]["maximum_conditional_coverage_error"],
            shear[method]["maximum_marginal_coverage_error"],
            shear[method]["joint_tarp_blocked"]
            ["full_max_abs_ecp_minus_alpha"],
        ))
        axis.plot(np.arange(len(names)), values / limits, marker="o", color=COLORS[method], label=LABELS[method])
    axis.axhline(1, color="black", linestyle="--", linewidth=1, label="gate")
    axis.set(xticks=np.arange(len(names)), xticklabels=names, ylabel="error / registered limit", title="F. Simultaneous anti-gaming gates")
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=7)

    for axis in axes.ravel():
        axis.grid(alpha=.2)
    figure.suptitle("P12-F3-L2 direct low-mode posterior — held-out ph006", fontsize=16)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_shear_coverage(shear: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(5); width = .24
    for panel, level in enumerate(("0.68", "0.90")):
        nominal = float(level)
        for offset, method in enumerate(METHODS):
            values = [shear[method]["marginal"][name]["coverage"][level]["empirical"] for name in shear[method]["components"]]
            axes[panel].bar(x + (offset - 1) * width, values, width, color=COLORS[method], label=LABELS[method])
        axes[panel].axhline(nominal, color="black", linestyle="--", linewidth=1)
        axes[panel].axhspan(nominal - .05, nominal + .05, color="#8bc34a", alpha=.12)
        axes[panel].set(xticks=x, xticklabels=shear[METHODS[0]]["components"], ylim=(0, 1), ylabel="empirical coverage", title=f"Nominal {int(100*nominal)}% low-k shear intervals")
        axes[panel].grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Directional tidal-shear calibration, not merely scalar P(k)")
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_examples(examples: dict, output: Path) -> None:
    figure, axes = plt.subplots(3, len(METHODS), figsize=(15, 10), constrained_layout=True)
    truth_limit = np.quantile(np.abs(np.concatenate([examples[m]["truth"].ravel() for m in METHODS])), .99)
    for column, method in enumerate(METHODS):
        row = examples[method]; z = row["truth"].shape[2] // 2
        for axis, field, label in zip(axes[:, column], (row["truth"][:, :, z], row["mean"][:, :, z], row["std"][:, :, z]), ("truth", "posterior mean", "posterior std")):
            if label == "posterior std":
                artist = axis.imshow(field.T, origin="lower", cmap="magma", vmin=0)
            else:
                artist = axis.imshow(field.T, origin="lower", cmap="coolwarm", vmin=-truth_limit, vmax=truth_limit)
            axis.set(xticks=[], yticks=[])
            if column == 0: axis.set_ylabel(label)
            figure.colorbar(artist, ax=axis, fraction=.046)
        axes[0, column].set_title(f"{LABELS[method]}\ncore {row['core_id']}")
    figure.suptitle("Matched ph006 field-posterior examples")
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    archives = parse_keyed(args.archive); report_paths = parse_keyed(args.report); shear_paths = parse_keyed(args.shear_report)
    reports = {method: json.loads(path.read_text()) for method, path in report_paths.items()}
    shear = {method: json.loads(path.read_text()) for method, path in shear_paths.items()}
    summary, examples = {}, {}
    for method in METHODS:
        row, example = analyze_archive(archives[method], device=args.device)
        if row["method"] != method or reports[method].get("method") != method or shear[method].get("method") != method:
            raise RuntimeError("F3-L2 plot inputs are mismatched")
        summary[method] = row; examples[method] = example
    evidence = {
        "schema_version": "p12f3l2-visual-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "ph006", "methods": summary,
        "artifacts": {
            "archives": {m: {"path": str(p.resolve()), "sha256": sha256(p)} for m, p in archives.items()},
            "reports": {m: {"path": str(p.resolve()), "sha256": sha256(p)} for m, p in report_paths.items()},
            "shear": {m: {"path": str(p.resolve()), "sha256": sha256(p)} for m, p in shear_paths.items()},
        },
        "truth_files_read": ["ph006 density/T-web"], "ph001_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "P12F3L2_VISUAL_AUDIT.json", evidence)
    trace = read_trace(args.loss_trace)
    plot_main(summary, reports, shear, trace, args.output_dir / "p12f3l2_fourier_summary")
    plot_shear_coverage(shear, args.output_dir / "p12f3l2_shear_coverage")
    plot_examples(examples, args.output_dir / "p12f3l2_field_examples")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "methods": list(METHODS)}, indent=2))


if __name__ == "__main__":
    main()
