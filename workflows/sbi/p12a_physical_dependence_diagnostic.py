#!/usr/bin/env python3
"""Render physical eigenvalue/eigengap calibration for frozen P12-A draws.

This is a checkpoint-free posterior diagnostic: it reuses the frozen 50,000-row
ph006 sample cache and opens the checkpoint only to recover the already-fitted
target standardisation.  It neither resamples nor fits/recalibrates anything and
never opens ph001.  TARP and SBC deliberately reuse the P12-F v2 implementations
so the coordinate- and algorithm-matched comparison is meaningful.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.plot_style import ACCENT_COLORS, TEXT_COLOR, apply_style
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_dependency_rescue_evaluator import TARP_SEEDS, _sbc, tarp_curve
from workflows.sbi.p12f_field_posterior_diagnostics import scalar_posterior_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_EVIDENCE = REPO_ROOT / "docs/evidence/p12/P12A_TARP_CURVE.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
    "p12a_base_response_v1/fmpe_seed42/physical_dependence_v1"
)
DEFAULT_FIGURE_DIR = REPO_ROOT / "docs/figures/p12a_physical_dependence_20260903"
DEFAULT_EVIDENCE_OUTPUT = (
    REPO_ROOT / "docs/evidence/p12/P12A_PHYSICAL_DEPENDENCE_DIAGNOSTIC.json"
)
DEFAULT_G1_REPORT = (
    REPO_ROOT
    / "docs/evidence/p12/p12f_dependency_rescue_v2/"
    "P12F_DEPENDENCY_RESCUE_V2_REPORT.json"
)
DEFAULT_G2_REPORT = (
    REPO_ROOT
    / "docs/evidence/p12/p12f_g2_conditional_covariance_v2/P12F_G2_REPORT.json"
)
DRAW_COUNTS = (64, 128, 256, 512)
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")
GAP_NAMES = ("gap12", "gap23")
COLORS = {
    64: "#7FDBFF",
    128: ACCENT_COLORS["blue"],
    256: ACCENT_COLORS["magenta"],
    512: ACCENT_COLORS["red"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE_OUTPUT)
    parser.add_argument("--g1-report", type=Path, default=DEFAULT_G1_REPORT)
    parser.add_argument("--g2-report", type=Path, default=DEFAULT_G2_REPORT)
    parser.add_argument("--sbc-bootstrap-repeats", type=int, default=4000)
    return parser.parse_args()


def validate_source_evidence(payload: dict[str, Any]) -> dict[str, Path]:
    if payload.get("schema_version") != "p12a-tarp-curve-v1":
        raise RuntimeError("unsupported P12-A source evidence")
    if (
        payload.get("selection_phase") != "ph006"
        or payload.get("sealed_phase") != "ph001"
        or payload.get("sealed_phase_opened")
    ):
        raise PermissionError("P12-A physical diagnostic requires sealed ph006 evidence")
    if payload.get("rows") != 50_000 or payload.get("posterior_draws_per_row") != 512:
        raise RuntimeError("frozen P12-A sample dimensions changed")
    provenance = payload.get("provenance", {})
    required = {
        "audit_report",
        "checkpoint",
        "dataset_marker",
        "evaluation_index",
        "posterior_samples",
        "validation_sample",
    }
    if not required.issubset(provenance):
        raise RuntimeError("P12-A provenance inventory is incomplete")
    paths: dict[str, Path] = {}
    for name in sorted(required):
        path = Path(provenance[name]).resolve()
        if "ph001" in path.parts:
            raise PermissionError(f"blind path forbidden for {name}")
        expected = provenance.get(f"{name}_sha256")
        if not path.is_file() or not expected or sha256(path) != expected:
            raise RuntimeError(f"frozen P12-A artifact failed hash validation: {name}")
        paths[name] = path
    return paths


def validate_validation_contract(
    source: dict[str, Any], paths: dict[str, Path], evaluation_index: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    dataset = json.loads(paths["dataset_marker"].read_text())
    audit = json.loads(paths["audit_report"].read_text())
    if (
        dataset.get("validation_phase") != "ph006"
        or dataset.get("sealed_phase") != "ph001"
        or dataset.get("sealed_phase_opened")
        or audit.get("selection_phase") != "ph006"
        or audit.get("sealed_phase_opened")
    ):
        raise PermissionError("P12-A dataset/audit phase contract changed")
    if list(audit.get("evaluation_folds", [])) != [2, 3, 4]:
        raise RuntimeError("P12-A evaluation folds changed")
    if evaluation_index.shape != (source["rows"],):
        raise RuntimeError("P12-A evaluation index length changed")
    if len(np.unique(evaluation_index)) != len(evaluation_index):
        raise RuntimeError("P12-A evaluation index contains duplicate rows")
    validation = np.load(paths["validation_sample"], mmap_mode="r")
    required_arrays = {
        "truth_eigenvalues",
        "shell",
        "cap",
        "superblock_id",
        "fold",
    }
    if not required_arrays.issubset(validation.files):
        raise RuntimeError("P12-A validation sample lacks physical diagnostic arrays")
    folds = np.asarray(validation["fold"])[evaluation_index]
    if not np.all(np.isin(folds, [2, 3, 4])):
        raise RuntimeError("P12-A selected rows escaped evaluation folds 2--4")
    return validation, audit


def scaled_to_physical_eigenvalues(
    scaled: np.ndarray, theta_mean: np.ndarray, theta_std: np.ndarray
) -> np.ndarray:
    values = np.asarray(scaled, dtype=np.float64)
    mean = np.asarray(theta_mean, dtype=np.float64)
    std = np.asarray(theta_std, dtype=np.float64)
    if values.shape[-1] != 3 or mean.shape != (3,) or std.shape != (3,):
        raise ValueError("ordered-softplus arrays must end in three components")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError("invalid frozen P12-A target standardisation")
    theta = values * std + mean
    result = np.empty_like(theta)
    result[..., 0] = theta[..., 0]
    result[..., 1] = result[..., 0] + np.logaddexp(0.0, theta[..., 1])
    result[..., 2] = result[..., 1] + np.logaddexp(0.0, theta[..., 2])
    if not np.all(np.isfinite(result)) or np.any(np.diff(result, axis=-1) < 0):
        raise RuntimeError("physical P12-A draws are non-finite or unordered")
    return result


def spatial_groups(cap: np.ndarray, superblock: np.ndarray) -> np.ndarray:
    cap = np.asarray(cap, dtype=np.int64)
    superblock = np.asarray(superblock, dtype=np.int64)
    if cap.shape != superblock.shape:
        raise ValueError("cap/superblock shape mismatch")
    return (cap << 32) + superblock


def reference_seed_report(
    eigen: np.ndarray, truth: np.ndarray, *, rows: int = 50_000
) -> dict[str, Any]:
    index = np.linspace(0, len(truth) - 1, min(rows, len(truth)), dtype=np.int64)
    gaps = np.diff(eigen, axis=-1)
    truth_gaps = np.diff(truth, axis=-1)
    ordered = [
        tarp_curve(eigen[:, index], truth[index], seed=seed)["maximum_deviation"]
        for seed in TARP_SEEDS
    ]
    gap = [
        tarp_curve(gaps[:, index], truth_gaps[index], seed=seed + 100)[
            "maximum_deviation"
        ]
        for seed in TARP_SEEDS
    ]
    return {
        "rows": int(len(index)),
        "ordered_seeds": list(TARP_SEEDS),
        "eigengap_seeds": [seed + 100 for seed in TARP_SEEDS],
        "ordered_eigenvalues": ordered,
        "eigengaps": gap,
        "ordered_p90": float(np.quantile(ordered, 0.9)),
        "eigengap_p90": float(np.quantile(gap, 0.9)),
    }


def evaluate_draws(
    eigen_row_first: np.ndarray,
    truth: np.ndarray,
    draw_count: int,
    *,
    with_reference_seeds: bool,
) -> dict[str, Any]:
    eigen = np.transpose(eigen_row_first[:, :draw_count], (1, 0, 2))
    gaps = np.diff(eigen, axis=-1)
    truth_gaps = np.diff(truth, axis=-1)
    report: dict[str, Any] = {
        "draws": draw_count,
        "tarp": {
            "ordered_eigenvalues": tarp_curve(eigen, truth, seed=42),
            "eigengaps": tarp_curve(gaps, truth_gaps, seed=43),
        },
        "ordered_eigenvalues": {
            name: scalar_posterior_report(eigen[..., index], truth[..., index], seed=62 + index)
            for index, name in enumerate(EIGEN_NAMES)
        },
        "eigengaps": {
            name: scalar_posterior_report(gaps[..., index], truth_gaps[..., index], seed=72 + index)
            for index, name in enumerate(GAP_NAMES)
        },
    }
    if with_reference_seeds:
        report["tarp"]["reference_seed_maxima"] = reference_seed_report(eigen, truth)
    return report


def shell_tarp(eigen: np.ndarray, truth: np.ndarray, shell: np.ndarray) -> dict[str, Any]:
    gaps = np.diff(eigen, axis=-1)
    truth_gaps = np.diff(truth, axis=-1)
    report: dict[str, Any] = {}
    for value in range(4):
        selected = np.asarray(shell) == value
        if np.count_nonzero(selected) < 100:
            raise RuntimeError(f"too few P12-A rows in shell {value}")
        report[str(value)] = {
            "rows": int(np.count_nonzero(selected)),
            "ordered_eigenvalues": tarp_curve(eigen[:, selected], truth[selected], seed=82 + value),
            "eigengaps": tarp_curve(gaps[:, selected], truth_gaps[selected], seed=92 + value),
        }
    return report


def sbc_report(
    eigen: np.ndarray, truth: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    gaps = np.diff(eigen, axis=-1)
    truth_gaps = np.diff(truth, axis=-1)
    report = {
        name: _sbc(eigen[..., index], truth[..., index], groups, seed=162 + index)
        for index, name in enumerate(EIGEN_NAMES)
    }
    report.update(
        {
            name: _sbc(gaps[..., index], truth_gaps[..., index], groups, seed=172 + index)
            for index, name in enumerate(GAP_NAMES)
        }
    )
    return report


def _curve(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(payload["alpha"], dtype=np.float64),
        np.asarray(payload["expected_coverage_probability"], dtype=np.float64),
    )


def _calibration_axis(ax: plt.Axes, *, residual: bool = False) -> None:
    grid = np.linspace(0.0, 1.0, 501)
    if residual:
        ax.axhspan(-0.05, 0.05, color=TEXT_COLOR, alpha=0.06)
        ax.axhline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.2)
        ax.set(xlim=(0, 1), ylim=(-0.14, 0.14), ylabel=r"ECP $-\alpha$")
    else:
        ax.fill_between(
            grid,
            np.maximum(0.0, grid - 0.05),
            np.minimum(1.0, grid + 0.05),
            color=TEXT_COLOR,
            alpha=0.06,
        )
        ax.plot(grid, grid, "--", color=TEXT_COLOR, linewidth=1.2)
        ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Empirical coverage")
    ax.set_xlabel(r"Nominal credibility $\alpha$")
    ax.grid(True, alpha=0.15)


def render_nested(report: dict[str, Any], stem: Path) -> None:
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
        _calibration_axis(axes[0, column])
        _calibration_axis(axes[1, column], residual=True)
        axes[0, column].set_title(
            "Physical ordered eigenvalues" if column == 0 else "Physical eigengaps"
        )
        for draws in DRAW_COUNTS:
            payload = report["nested_draw_reports"][str(draws)]["tarp"][key]
            alpha, ecp = _curve(payload)
            label = f"{draws} draws (max {payload['maximum_deviation']:.3f})"
            axes[0, column].plot(alpha, ecp, color=COLORS[draws], linewidth=2, label=label)
            axes[1, column].plot(alpha, ecp - alpha, color=COLORS[draws], linewidth=1.8)
        axes[0, column].legend(loc="lower right")
    figure.suptitle(
        "P12-A physical joint calibration on 50,000 held-out ph006 galaxies",
        fontsize=18,
        fontweight="bold",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_sbc(report: dict[str, Any], stem: Path) -> None:
    apply_style()
    names = (*EIGEN_NAMES, *GAP_NAMES)
    titles = (
        r"$\lambda_1$",
        r"$\lambda_2$",
        r"$\lambda_3$",
        r"$\lambda_2-\lambda_1$",
        r"$\lambda_3-\lambda_2$",
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    centres = np.arange(10) + 0.5
    for ax, name, title in zip(axes.ravel(), names, titles, strict=False):
        payload = report["sbc_512_draws"][name]
        mass = np.asarray(payload["decile_mass"])
        interval = np.asarray(payload["decile_mass_95ci"])
        error = np.maximum(
            np.vstack((mass - interval[:, 0], interval[:, 2] - mass)), 0.0
        )
        ax.bar(
            centres,
            mass,
            width=0.88,
            color=ACCENT_COLORS["blue"],
            alpha=0.75,
            edgecolor=TEXT_COLOR,
            linewidth=0.35,
        )
        ax.errorbar(centres, mass, yerr=error, fmt="none", ecolor=TEXT_COLOR, capsize=2)
        ax.axhline(0.1, color=TEXT_COLOR, linestyle="--")
        ax.set(xlim=(0, 10), xlabel="Randomized rank decile", ylabel="Mass")
        ax.set_title(f"{title}: rank-CDF max {payload['rank_cdf_maximum_deviation']:.3f}")
        ax.grid(True, axis="y", alpha=0.15)
    axes.ravel()[-1].axis("off")
    figure.suptitle(
        "P12-A physical eigenvalue/eigengap SBC — cap+superblock intervals",
        fontsize=18,
        fontweight="bold",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_shells(report: dict[str, Any], stem: Path) -> None:
    apply_style()
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for shell in range(4):
        payload = report["shell_tarp_512_draws"][str(shell)]
        for row, key in enumerate(("ordered_eigenvalues", "eigengaps")):
            ax = axes[row, shell]
            _calibration_axis(ax)
            alpha, ecp = _curve(payload[key])
            ax.plot(alpha, ecp, color=ACCENT_COLORS["blue"], linewidth=2)
            ax.set_title(
                f"Shell {shell}: {'eigenvalues' if row == 0 else 'eigengaps'}\n"
                f"{payload['rows']:,} rows; max {payload[key]['maximum_deviation']:.3f}"
            )
    figure.suptitle(
        "P12-A physical joint calibration versus redshift shell — 512 draws",
        fontsize=18,
        fontweight="bold",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def validate_field_report(payload: dict[str, Any], name: str) -> None:
    if (
        payload.get("schema_version") != "p12f-dependency-rescue-evaluation-v2"
        or payload.get("phase") != "ph006"
        or payload.get("ph001_opened")
        or "256" not in payload.get("nested_draw_reports", {})
    ):
        raise RuntimeError(f"invalid frozen {name} comparison report")


def render_field_comparison(
    report: dict[str, Any], g1: dict[str, Any], g2: dict[str, Any], stem: Path
) -> None:
    apply_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        constrained_layout=True,
    )
    methods = (
        ("P12-A galaxy posterior", report["nested_draw_reports"]["256"], ACCENT_COLORS["blue"]),
        ("P12-F G1 global covariance", g1["nested_draw_reports"]["256"], ACCENT_COLORS["magenta"]),
        ("P12-F G2 shell covariance", g2["nested_draw_reports"]["256"], ACCENT_COLORS["red"]),
    )
    for column, key in enumerate(("ordered_eigenvalues", "eigengaps")):
        _calibration_axis(axes[0, column])
        _calibration_axis(axes[1, column], residual=True)
        axes[0, column].set_title(
            "Physical ordered eigenvalues" if column == 0 else "Physical eigengaps"
        )
        for label, payload, color in methods:
            curve = payload["tarp"][key]
            alpha, ecp = _curve(curve)
            axes[0, column].plot(
                alpha,
                ecp,
                color=color,
                linewidth=2,
                label=f"{label} (max {curve['maximum_deviation']:.3f})",
            )
            axes[1, column].plot(alpha, ecp - alpha, color=color, linewidth=1.8)
        axes[0, column].legend(loc="lower right", fontsize=9)
    figure.suptitle(
        "Matched 256-draw physical TARP implementation\n"
        "P12-A galaxies versus P12-F local field-derived galaxy samples",
        fontsize=17,
        fontweight="bold",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def decision_summary(report: dict[str, Any], gate: float = 0.05) -> dict[str, Any]:
    full = report["nested_draw_reports"]["512"]["tarp"]
    reference = full["reference_seed_maxima"]
    shell = report["shell_tarp_512_draws"]
    maxima = {
        "ordered_eigenvalues": float(full["ordered_eigenvalues"]["maximum_deviation"]),
        "eigengaps": float(full["eigengaps"]["maximum_deviation"]),
        "ordered_reference_seed_p90": float(reference["ordered_p90"]),
        "eigengap_reference_seed_p90": float(reference["eigengap_p90"]),
        "shell_ordered_maximum": max(
            float(payload["ordered_eigenvalues"]["maximum_deviation"])
            for payload in shell.values()
        ),
        "shell_eigengap_maximum": max(
            float(payload["eigengaps"]["maximum_deviation"])
            for payload in shell.values()
        ),
    }
    return {
        "registered_global_gate": gate,
        "maxima": maxima,
        "global_physical_joint_pass": bool(
            maxima["ordered_eigenvalues"] <= gate
            and maxima["eigengaps"] <= gate
            and maxima["ordered_reference_seed_p90"] <= gate
            and maxima["eigengap_reference_seed_p90"] <= gate
        ),
        "scope": (
            "within-galaxy physical joint calibration only; this does not test "
            "spatial coherence between distinct galaxies or field voxels"
        ),
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_evidence.read_text())
    paths = validate_source_evidence(source)
    evaluation_index = np.load(paths["evaluation_index"], mmap_mode="r")
    validation, audit = validate_validation_contract(source, paths, evaluation_index)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("dataset_marker_sha256") != source["provenance"]["dataset_marker_sha256"]:
        raise RuntimeError("P12-A checkpoint/dataset provenance mismatch")
    samples_scaled = np.load(paths["posterior_samples"], mmap_mode="r")
    if samples_scaled.shape != (50_000, 512, 3):
        raise RuntimeError(f"unexpected cached P12-A sample shape {samples_scaled.shape}")
    eigen_row_first = scaled_to_physical_eigenvalues(
        samples_scaled,
        np.asarray(checkpoint["theta_mean"]),
        np.asarray(checkpoint["theta_std"]),
    )
    truth = np.asarray(validation["truth_eigenvalues"])[evaluation_index].astype(np.float64)
    if truth.shape != (50_000, 3) or np.any(np.diff(truth, axis=-1) < 0):
        raise RuntimeError("invalid ordered physical truth array")
    shell = np.asarray(validation["shell"])[evaluation_index].astype(np.int8)
    groups = spatial_groups(
        np.asarray(validation["cap"])[evaluation_index],
        np.asarray(validation["superblock_id"])[evaluation_index],
    )
    nested = {
        str(draws): evaluate_draws(
            eigen_row_first,
            truth,
            draws,
            with_reference_seeds=draws in (256, 512),
        )
        for draws in DRAW_COUNTS
    }
    eigen_draw_first = np.transpose(eigen_row_first, (1, 0, 2))
    report: dict[str, Any] = {
        "schema_version": "p12a-physical-dependence-diagnostic-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "selection_phase": "ph006",
        "sealed_phase": "ph001",
        "ph001_opened": False,
        "truth_files_read": [str(paths["validation_sample"])],
        "rows": 50_000,
        "posterior_draws": 512,
        "evaluation_folds": [2, 3, 4],
        "nested_draw_reports": nested,
        "sbc_512_draws": sbc_report(eigen_draw_first, truth, groups),
        "shell_tarp_512_draws": shell_tarp(eigen_draw_first, truth, shell),
        "methodology": {
            "posterior_resampled": False,
            "checkpoint_use": "read frozen theta_mean/theta_std only",
            "fit_or_recalibration": False,
            "physical_coordinates": ["lambda1", "lambda2", "lambda3", "gap12", "gap23"],
            "tarp_implementation": "P12-F v2 official tarp.get_tarp_coverage wrapper",
            "sbc_bootstrap": "cap+superblock cluster bootstrap, 4000 repeats",
            "comparison_caveat": (
                "P12-A and P12-F share the physical diagnostic and 256-draw budget, "
                "but P12-A evaluates a frozen 50k galaxy subset whereas P12-F derives "
                "galaxy samples from local patch-field draws on 1024 cores"
            ),
        },
        "source_transformed_coordinate_tarp_maximum": float(source["max_abs_ecp_minus_alpha"]),
        "source_audit_physical_rank_available": bool(
            audit.get("global", {}).get("eigen_rank")
        ),
        "provenance": {
            "source_evidence": str(args.source_evidence.resolve()),
            "source_evidence_sha256": sha256(args.source_evidence),
            **{
                name: str(path)
                for name, path in paths.items()
            },
            **{
                f"{name}_sha256": source["provenance"][f"{name}_sha256"]
                for name in paths
            },
            "diagnostic_source": str(Path(__file__).resolve()),
            "diagnostic_source_sha256": sha256(Path(__file__)),
        },
    }
    report["decision"] = decision_summary(report)

    g1 = json.loads(args.g1_report.read_text())
    g2 = json.loads(args.g2_report.read_text())
    validate_field_report(g1, "G1")
    validate_field_report(g2, "G2")
    stems = {
        "nested_tarp": args.figure_dir / "p12a_physical_nested_tarp",
        "sbc": args.figure_dir / "p12a_physical_sbc",
        "shell_tarp": args.figure_dir / "p12a_physical_shell_tarp",
        "p12a_p12f_comparison": args.figure_dir / "p12a_vs_p12f_physical_tarp",
    }
    render_nested(report, stems["nested_tarp"])
    render_sbc(report, stems["sbc"])
    render_shells(report, stems["shell_tarp"])
    render_field_comparison(report, g1, g2, stems["p12a_p12f_comparison"])
    report["comparison_provenance"] = {
        "g1_report": str(args.g1_report.resolve()),
        "g1_report_sha256": sha256(args.g1_report),
        "g2_report": str(args.g2_report.resolve()),
        "g2_report_sha256": sha256(args.g2_report),
    }
    report["figures"] = {
        name: {
            "png": str(stem.with_suffix(".png").resolve()),
            "png_sha256": sha256(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf").resolve()),
            "pdf_sha256": sha256(stem.with_suffix(".pdf")),
        }
        for name, stem in stems.items()
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "P12A_PHYSICAL_DEPENDENCE_DIAGNOSTIC.json", report)
    atomic_json(args.evidence_output, report)
    print(json.dumps(report["decision"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
