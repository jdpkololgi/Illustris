#!/usr/bin/env python3
"""Render the P12-F causal-autopsy results as interpretation-first figures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


COLORS = {
    "g1": "#6b7280",
    "lowk_mean_oracle": "#2563eb",
    "lowk_power_oracle": "#f59e0b",
    "lowk_mean_power_oracle": "#059669",
    "empirical_residual_patch": "#7c3aed",
    "trace": "#2563eb",
    "shear_q": "#dc2626",
    "lode_eta": "#7c3aed",
    "gap12": "#f59e0b",
    "gap23": "#059669",
}
LABELS = {
    "g1": "G1",
    "lowk_mean_oracle": "low-k mean oracle",
    "lowk_power_oracle": "low-k power oracle",
    "lowk_mean_power_oracle": "mean + power oracle",
    "empirical_residual_patch": "whole residual patches",
    "trace": r"trace $I_1$",
    "shear_q": r"shear amplitude $q$",
    "lode_eta": r"shear shape $\eta$",
    "gap12": r"gap $g_{12}$",
    "gap23": r"gap $g_{23}$",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_trace_shear(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    probabilities = (0.5, 0.68, 0.9)
    x = np.arange(len(probabilities))
    for name, row in report["scalar"].items():
        error = [row["coverage_core_bootstrap"][str(p)]["error"] for p in probabilities]
        axes[0, 0].plot(x, error, marker="o", label=LABELS[name], color=COLORS[name])
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_xticks(x, [f"{int(100*p)}%" for p in probabilities])
    axes[0, 0].set_ylabel("empirical minus nominal coverage")
    axes[0, 0].set_title("A. Marginal interval calibration")
    axes[0, 0].legend(fontsize=8, ncol=2)

    center = np.arange(10) + 0.5
    for name, row in report["scalar"].items():
        axes[0, 1].plot(center, row["sbc"]["decile_mass"], marker="o", ms=3, label=LABELS[name], color=COLORS[name])
    axes[0, 1].axhline(0.1, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("SBC rank decile")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].set_title("B. Location and shape of posterior ranks")

    tarp_colors = ("#2563eb", "#dc2626", "#f59e0b")
    for (name, row), color in zip(report["joint_tarp"].items(), tarp_colors, strict=True):
        axes[1, 0].plot(row["alpha"], row["expected_coverage_probability"], color=color, label=f"{name.replace('_', ' ')}  D={row['maximum_deviation']:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("nominal TARP coverage")
    axes[1, 0].set_ylabel("empirical coverage")
    axes[1, 0].set_title("C. Joint calibration after the fixed tidal map")
    axes[1, 0].legend(fontsize=8)

    maxima = {
        "trace rank": report["scalar"]["trace"]["sbc"]["rank_cdf_maximum_deviation"],
        "shear q rank": report["scalar"]["shear_q"]["sbc"]["rank_cdf_maximum_deviation"],
        "shape eta rank": report["scalar"]["lode_eta"]["sbc"]["rank_cdf_maximum_deviation"],
        "gap joint TARP": report["joint_tarp"]["eigengaps"]["maximum_deviation"],
    }
    bars = axes[1, 1].barh(list(maxima), list(maxima.values()), color=["#2563eb", "#dc2626", "#7c3aed", "#f59e0b"])
    axes[1, 1].axvline(0.05, color="black", linestyle="--", linewidth=1, label="0.05 reference")
    axes[1, 1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1, 1].set_xlim(0, max(0.11, 1.2 * max(maxima.values())))
    axes[1, 1].set_xlabel("maximum calibration deviation")
    axes[1, 1].set_title("D. Where the calibration error enters")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("P12-F G1 causal bridge: density-like trace versus tidal shear", fontsize=15)
    figure.tight_layout()
    _save(figure, output)


def render_method_comparison(report: dict, output: Path, title: str) -> None:
    methods = list(report["methods"])
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for method in methods:
        row = report["methods"][method]
        color = COLORS[method]
        eigen = row["ordered_eigenvalue_tarp"]
        gap = row["eigengap_tarp"]
        axes[0, 0].plot(eigen["alpha"], eigen["expected_coverage_probability"], color=color, label=f"{LABELS[method]}  D={eigen['maximum_deviation']:.3f}")
        axes[0, 1].plot(gap["alpha"], gap["expected_coverage_probability"], color=color, label=f"{LABELS[method]}  D={gap['maximum_deviation']:.3f}")
    for axis, subtitle in zip(axes[0], ("A. Joint ordered eigenvalues", "B. Joint eigengaps"), strict=True):
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("nominal TARP coverage")
        axis.set_ylabel("empirical coverage")
        axis.set_title(subtitle)
        axis.legend(fontsize=8)

    scores = ("energy", "coarse_energy", "variogram_p0p5")
    width = 0.78 / len(methods)
    base = np.asarray([report["methods"]["g1"]["proper_scores_64_draws"][name] for name in scores])
    for index, method in enumerate(methods):
        values = np.asarray([report["methods"][method]["proper_scores_64_draws"][name] for name in scores]) / base
        axes[1, 0].bar(np.arange(len(scores)) - 0.39 + width / 2 + index * width, values, width=width, color=COLORS[method], label=LABELS[method])
    axes[1, 0].axhline(1.0, color="black", linewidth=1)
    axes[1, 0].set_xticks(np.arange(len(scores)), ["energy", "coarse energy", "variogram"])
    axes[1, 0].set_ylabel("score / G1 score  (lower is better)")
    axes[1, 0].set_title("C. Joint field proper scores")
    axes[1, 0].legend(fontsize=8)

    gap_names = ("gap12", "gap23")
    x = np.arange(len(gap_names))
    for method in methods:
        errors = [report["methods"][method]["scalar"][name]["coverage_core_bootstrap"]["0.68"]["error"] for name in gap_names]
        axes[1, 1].plot(x, errors, marker="o", color=COLORS[method], label=LABELS[method])
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_xticks(x, [r"$g_{12}$", r"$g_{23}$"])
    axes[1, 1].set_ylabel("68% empirical minus nominal coverage")
    axes[1, 1].set_title("D. Does the intervention repair gap intervals?")
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    _save(figure, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    artifacts = []
    trace = args.input_root / "P12F_TRACE_SHEAR_AUTOPSY.json"
    low_k = args.input_root / "P12F_LOW_K_CAUSAL_AUTOPSY.json"
    empirical = args.input_root / "P12F_EMPIRICAL_RESIDUAL_CAUSAL_AUTOPSY.json"
    if trace.exists():
        path = args.output_root / "p12f_trace_vs_shear_autopsy.png"
        render_trace_shear(json.loads(trace.read_text()), path)
        artifacts.extend((path, path.with_suffix(".pdf")))
    if low_k.exists():
        path = args.output_root / "p12f_low_k_oracle_autopsy.png"
        render_method_comparison(json.loads(low_k.read_text()), path, "P12-F low-k causal interventions (truth-assisted diagnostics)")
        artifacts.extend((path, path.with_suffix(".pdf")))
    if empirical.exists():
        path = args.output_root / "p12f_empirical_residual_autopsy.png"
        render_method_comparison(json.loads(empirical.read_text()), path, "P12-F whole training-residual-patch control")
        artifacts.extend((path, path.with_suffix(".pdf")))
    if not artifacts:
        raise RuntimeError("no causal-autopsy reports were available to plot")
    manifest = {
        "schema_version": "p12f-causal-autopsy-plots-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in artifacts],
        "source_reports": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (trace, low_k, empirical)
            if path.exists()
        ],
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(args.output_root / "P12F_CAUSAL_AUTOPSY_PLOTS.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
