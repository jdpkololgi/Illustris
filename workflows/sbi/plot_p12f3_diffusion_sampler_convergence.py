#!/usr/bin/env python3
"""Compare frozen 24-, 50-, and 100-evaluation DDIM archives without retraining."""
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


LABELS = {
    "nfe24": "DDIM 24 evaluations",
    "nfe50": "DDIM 50 evaluations",
    "nfe100": "DDIM 100 evaluations",
}
COLORS = {"nfe24": "#7a3e9d", "nfe50": "#009e73", "nfe100": "#0072b2"}
SCORES = ("energy", "coarse_energy", "marginal_crps", "variogram_p0p5")
DEPLOYABLE_CONDITIONS = ("shell", "random_response", "boundary_distance", "tracer_density")


def parse_keyed(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in result:
            raise ValueError("inputs must be unique KEY=PATH values")
        result[key] = Path(path)
    if set(result) != set(LABELS):
        raise RuntimeError("sampler audit requires nfe24, nfe50 and nfe100")
    return result


def read_safe(path: Path) -> dict:
    value = json.loads(path.read_text())
    truth = value.get("truth_files_read")
    if (
        value.get("ph001_opened")
        or not isinstance(truth, list)
        or not truth
        or any(not str(item).startswith("ph006") for item in truth)
    ):
        raise RuntimeError(f"unsafe sampler-audit input {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--visual", action="append", required=True)
    parser.add_argument("--shear", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tarp_curve(value: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    row = value[name]
    return np.asarray(row["alpha"]), np.asarray(row["expected_coverage_probability"])


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


def comparison_metrics(
    lower: str,
    upper: str,
    report: dict[str, dict],
    visual: dict[str, dict],
) -> dict:
    tarp_delta = max(
        abs(
            report[upper]["tarp"][name]["full_max_abs_ecp_minus_alpha"]
            - report[lower]["tarp"][name]["full_max_abs_ecp_minus_alpha"]
        )
        for name in ("ordered_eigenvalues", "eigengaps")
    )
    coverage_delta = max(
        abs(
            report[upper]["global_coverage_error"][level]
            - report[lower]["global_coverage_error"][level]
        )
        for level in ("0.68", "0.90")
    )
    low_power_delta = float(
        np.max(
            np.abs(
                np.asarray(visual[upper]["posterior_to_truth_power"][:2])
                - np.asarray(visual[lower]["posterior_to_truth_power"][:2])
            )
        )
    )
    score_delta = max(
        abs(
            report[upper]["proper_scores"][name]
            / report[lower]["proper_scores"][name]
            - 1.0
        )
        for name in SCORES
    )
    changes = {
        "joint_tarp": tarp_delta,
        "global_coverage_error": coverage_delta,
        "low_band_power_ratio": low_power_delta,
        "proper_score_relative": score_delta,
    }
    gates = {
        "joint_tarp_change_le_0p01": tarp_delta <= 0.01,
        "global_coverage_error_change_le_0p01": coverage_delta <= 0.01,
        "low_band_power_change_le_0p05": low_power_delta <= 0.05,
        "proper_score_relative_change_le_0p01": score_delta <= 0.01,
    }
    return {"lower": lower, "upper": upper, "maximum_changes": changes,
            "gates": gates, "converged": all(gates.values())}


def absolute_metrics(key: str, report: dict, visual: dict, shear: dict) -> dict:
    return {
        "network_evaluations": int(key.removeprefix("nfe")),
        "ordered_eigenvalue_tarp": float(
            report["tarp"]["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"]
        ),
        "eigengap_tarp": float(
            report["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"]
        ),
        "global_coverage_error": {
            level: float(report["global_coverage_error"][level])
            for level in ("0.68", "0.90")
        },
        "deployable_conditional_coverage_error": deployable_conditional_error(report),
        "low_band_power_ratio": [float(value) for value in visual["posterior_to_truth_power"][:2]],
        "joint_shear_tarp": float(shear["joint_tarp"]["maximum_deviation"]),
        "maximum_marginal_shear_coverage_error": float(
            shear["maximum_marginal_coverage_error"]
        ),
        "proper_scores": {name: float(report["proper_scores"][name]) for name in SCORES},
    }


def main() -> None:
    args = parse_args()
    paths = {
        "report": parse_keyed(args.report),
        "visual": parse_keyed(args.visual),
        "shear": parse_keyed(args.shear),
    }
    values = {
        kind: {key: read_safe(path) for key, path in keyed.items()}
        for kind, keyed in paths.items()
    }
    report = values["report"]
    visual = values["visual"]
    shear = values["shear"]

    comparisons = {
        "nfe24_to_nfe50": comparison_metrics("nfe24", "nfe50", report, visual),
        "nfe50_to_nfe100": comparison_metrics("nfe50", "nfe100", report, visual),
    }

    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, name, title in (
        (axes[0, 0], "eigen_tarp", "A. Ordered-eigenvalue TARP"),
        (axes[0, 1], "gap_tarp", "B. Eigengap TARP"),
    ):
        axis.plot((0, 1), (0, 1), "--", color="black", label="ideal")
        for key in LABELS:
            alpha, ecp = tarp_curve(visual[key], name)
            maximum = float(visual[key][name]["maximum_deviation"])
            axis.plot(alpha, ecp, color=COLORS[key], label=f"{LABELS[key]} ({maximum:.3f})")
        axis.set(xlabel="nominal coverage", ylabel="empirical coverage", title=title)
        axis.legend(fontsize=8)

    x = np.arange(2)
    width = 0.25
    for index, key in enumerate(LABELS):
        axes[0, 2].bar(
            x + (index - 1.0) * width,
            visual[key]["posterior_to_truth_power"][:2],
            width,
            color=COLORS[key],
            label=LABELS[key],
        )
    axes[0, 2].axhline(1.0, color="black", linestyle="--")
    axes[0, 2].axhspan(0.9, 1.1, color="#7fbf7b", alpha=0.12)
    axes[0, 2].set(xticks=x, xticklabels=("longest band", "second band"),
                   ylabel="posterior / truth residual power", title="C. Low-mode power")
    axes[0, 2].legend(fontsize=8)

    labels = ("global 68%", "global 90%", "deployable conditional")
    for key in LABELS:
        y = (
            report[key]["global_coverage_error"]["0.68"],
            report[key]["global_coverage_error"]["0.90"],
            deployable_conditional_error(report[key]),
        )
        axes[1, 0].plot(np.arange(3), y, marker="o", color=COLORS[key], label=LABELS[key])
    axes[1, 0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.10, color="black", linestyle=":", linewidth=1)
    axes[1, 0].set(xticks=np.arange(3), xticklabels=labels, ylabel="absolute coverage error",
                   title="D. Coverage errors")
    axes[1, 0].tick_params(axis="x", rotation=20)

    for key in LABELS:
        y = (
            shear[key]["joint_tarp"]["maximum_deviation"],
            shear[key]["maximum_marginal_coverage_error"],
        )
        axes[1, 1].plot(np.arange(2), y, marker="o", color=COLORS[key], label=LABELS[key])
    axes[1, 1].axhline(0.05, color="black", linestyle="--")
    axes[1, 1].set(xticks=np.arange(2), xticklabels=("joint shear TARP", "max marginal shear"),
                   ylabel="maximum deviation", title="E. Tidal-shear calibration")

    x = np.arange(4)
    width = 0.34
    for index, key in enumerate(("nfe50", "nfe100")):
        ratios = [report[key]["proper_scores"][name] / report["nfe24"]["proper_scores"][name]
                  for name in SCORES]
        axes[1, 2].bar(
            x + (index - 0.5) * width, 100.0 * (np.asarray(ratios) - 1.0), width,
            color=COLORS[key], label=f"{LABELS[key]} vs NFE24",
        )
    axes[1, 2].axhline(0.0, color="black", linestyle="--")
    axes[1, 2].axhspan(-1.0, 1.0, color="#7fbf7b", alpha=0.12)
    axes[1, 2].set(xticks=x, xticklabels=("energy", "coarse", "CRPS", "variogram"),
                   ylabel="change from NFE24 [%]", title="F. Proper-score convergence")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("P12-F3-L2d DDIM sampler convergence — frozen ph006 panel", fontsize=16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "p12f3_diffusion_sampler_convergence"
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    payload = {
        "schema_version": "p12f3-diffusion-sampler-convergence-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "converged_at_nfe50": comparisons["nfe24_to_nfe50"]["converged"],
        "converged_at_nfe100": comparisons["nfe50_to_nfe100"]["converged"],
        "comparisons": comparisons,
        "absolute_metrics": {
            key: absolute_metrics(key, report[key], visual[key], shear[key])
            for key in LABELS
        },
        "inputs": {
            str(path.resolve()): sha256(path)
            for keyed in paths.values()
            for path in keyed.values()
        },
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
    }
    atomic_json(args.output_dir / "P12F3_DIFFUSION_SAMPLER_CONVERGENCE.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
