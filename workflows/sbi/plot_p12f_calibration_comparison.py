#!/usr/bin/env python3
"""Reproduce P12-F TARP/SBC diagnostics from frozen ph006 field samples.

The workflow never fits, resamples, or corrects a posterior.  It applies the
registered fixed tidal operator to the archived 64-draw local field samples,
samples the resulting ordered eigenvalues at the authoritative galaxies, and
checks every reproduced scalar diagnostic against the frozen common-evaluator
report.  P12-A is loaded only from its durable ph006 calibration reports and is
shown in a separate panel because it is a direct per-galaxy posterior with a
different validation panel and target coordinate system.  ph001 is never read.
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
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_common_evaluator import (
    load_core_record,
    sample_eigenvalues_at_galaxies,
    validate_archive_manifest,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    fixed_tidal_eigenvalues,
    randomized_ranks,
    rank_cdf_maximum_deviation,
)
from workflows.sbi.plot_p12_tarp_curve import (
    curve_subsample_indices,
    ordered_curve,
)


P12F_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
    "p12f_matched_challengers_v1/matched_v1_seed42"
)
P12A_AUDIT = REPO_ROOT / "docs/evidence/p12/P12A_CALIBRATION_AUDIT.json"
P12A_TARP = REPO_ROOT / "docs/evidence/p12/P12A_TARP_CURVE.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
DEFAULT_PANEL = (
    REPO_ROOT / "docs/evidence/p12/p12f_matched_v1/P12F_PH006_PANEL_128.json"
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "docs/evidence/p12/p12f_matched_v1"
DEFAULT_OUTPUT = REPO_ROOT / "docs/figures/p12f_calibration_20260902"
DEFAULT_EVIDENCE = DEFAULT_REPORT_ROOT / "P12F_CALIBRATION_PLOTS.json"

METHODS = (
    ("gaussian_correlated_g1", "Correlated Gaussian G1", ACCENT_COLORS["blue"]),
    ("rectified_flow_f1b", "Rectified flow", ACCENT_COLORS["magenta"]),
    ("score_diffusion_v1", "Score diffusion", ACCENT_COLORS["red"]),
)
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decile_mass(ranks: np.ndarray) -> np.ndarray:
    """Return ten normalized posterior-rank bin masses."""
    values = np.asarray(ranks, dtype=np.float64).ravel()
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("finite non-empty ranks are required")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("posterior ranks must lie in [0,1]")
    counts = np.histogram(values, bins=np.linspace(0.0, 1.0, 11))[0]
    return counts.astype(np.float64) / counts.sum()


def clustered_decile_interval(
    ranks: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-bootstrap rank-bin masses by authoritative patch core."""
    values = np.asarray(ranks, dtype=np.float64).ravel()
    labels = np.asarray(groups).ravel()
    if values.shape != labels.shape or repeats <= 0:
        raise ValueError("rank/group alignment and positive repeats are required")
    unique, inverse = np.unique(labels, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two spatial groups are required")
    block_counts = np.zeros((len(unique), 10), dtype=np.int64)
    bins = np.minimum((values * 10.0).astype(np.int64), 9)
    np.add.at(block_counts, (inverse, bins), 1)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, 10), dtype=np.float64)
    for repeat in range(repeats):
        selected = rng.integers(0, len(unique), size=len(unique))
        counts = block_counts[selected].sum(axis=0)
        draws[repeat] = counts / counts.sum()
    return np.quantile(draws, [0.025, 0.5, 0.975], axis=0).T


def compact_curve(alpha: np.ndarray, ecp: np.ndarray) -> dict[str, Any]:
    selected = curve_subsample_indices(alpha, ecp)
    return {
        "full_curve_points": int(len(alpha)),
        "stored_curve_points": int(len(selected)),
        "alpha": alpha[selected].tolist(),
        "expected_coverage_probability": ecp[selected].tolist(),
        "ecp_minus_alpha": (ecp[selected] - alpha[selected]).tolist(),
        "maximum_deviation": float(np.max(np.abs(ecp - alpha))),
    }


def load_frozen_contracts(
    config_path: Path,
    panel_path: Path,
    report_root: Path,
) -> tuple[dict, dict, dict[str, dict], dict[str, dict]]:
    config = json.loads(config_path.read_text())
    panel = json.loads(panel_path.read_text())
    if (
        panel.get("schema_version") != "p12f-truth-free-selection-panel-v1"
        or not panel.get("pass")
        or panel.get("selection_uses_truth")
        or panel.get("truth_files_read")
        or panel.get("ph001_opened")
    ):
        raise RuntimeError("P12-F selection panel is not frozen, truth-free, and passing")
    reports: dict[str, dict] = {}
    archives: dict[str, dict] = {}
    for method, _, _ in METHODS:
        report_path = report_root / f"{method}.json"
        report = json.loads(report_path.read_text())
        archive_path = Path(report["archive_manifest"])
        archive = json.loads(archive_path.read_text())
        if report.get("phase") != "ph006" or report.get("ph001_opened"):
            raise RuntimeError(f"unsafe frozen report for {method}")
        if sha256(archive_path) != report["archive_manifest_sha256"]:
            raise RuntimeError(f"archive manifest changed for {method}")
        validate_archive_manifest(
            archive,
            archive_path=archive_path,
            panel=panel,
            panel_path=panel_path,
            config=config,
        )
        reports[method] = report
        archives[method] = archive
    return config, panel, reports, archives


def reconstruct_calibration(
    panel: dict,
    reports: dict[str, dict],
    archives: dict[str, dict],
    *,
    device: str,
    seed: int,
    bootstrap_repeats: int,
) -> dict[str, dict]:
    """Rebuild derived galaxy eigenvalue draws once per frozen core/method."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F calibration reconstruction requires a compute GPU")
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    entry_maps = {
        method: {int(row["core_id"]): row for row in archive["entries"]}
        for method, archive in archives.items()
    }
    selected = [int(value) for value in panel["selected_core_id"]]
    sample_parts: dict[str, list[np.ndarray]] = {method: [] for method, _, _ in METHODS}
    truth_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []

    for ordinal, core_id in enumerate(selected):
        base_method = METHODS[0][0]
        base_record = load_core_record(
            entry_maps[base_method][core_id], int(archives[base_method]["draws"])
        )
        truth_tensor = torch.from_numpy(
            np.asarray(base_record["delta_truth"], dtype=np.float32)
        ).to(device)
        truth_eigen = fixed_tidal_eigenvalues(truth_tensor).detach().cpu().numpy()
        sampled_truth: np.ndarray | None = None
        for method, _, _ in METHODS:
            record = (
                base_record
                if method == base_method
                else load_core_record(
                    entry_maps[method][core_id], int(archives[method]["draws"])
                )
            )
            for key in ("delta_truth", "support", "core_bounds", "galaxy_frac_index_local"):
                if not np.array_equal(record[key], base_record[key]):
                    raise RuntimeError(f"{method} core {core_id} changed common {key}")
            sample_tensor = torch.from_numpy(
                np.asarray(record["delta_samples"], dtype=np.float32)
            ).to(device)
            sample_eigen = fixed_tidal_eigenvalues(sample_tensor).detach().cpu().numpy()
            sampled, current_truth = sample_eigenvalues_at_galaxies(
                sample_eigen,
                truth_eigen,
                record["galaxy_frac_index_local"],
            )
            if sampled_truth is None:
                sampled_truth = current_truth
            elif not np.array_equal(sampled_truth, current_truth):
                raise RuntimeError(f"truth interpolation changed for core {core_id}")
            sample_parts[method].append(sampled)
            del sample_tensor
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        if sampled_truth is not None and sampled_truth.size:
            truth_parts.append(sampled_truth)
            group_parts.append(np.full(len(sampled_truth), core_id, dtype=np.int64))
        del truth_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"calibration core {ordinal + 1}/{len(selected)} id={core_id} "
            f"shell={metadata[core_id]['shell']}",
            flush=True,
        )

    truth = np.concatenate(truth_parts, axis=0)
    groups = np.concatenate(group_parts)
    import tarp

    products: dict[str, dict] = {}
    for method_index, (method, label, color) in enumerate(METHODS):
        samples = np.concatenate(sample_parts[method], axis=1)
        if samples.shape[1:] != truth.shape or samples.shape[0] != 64:
            raise RuntimeError(f"unexpected reconstructed sample shape for {method}")
        report = reports[method]
        if len(truth) != report["tarp"]["ordered_eigenvalues"]["rows"]:
            raise RuntimeError(f"row count does not reproduce {method}")

        ecp, alpha = tarp.get_tarp_coverage(
            samples,
            truth,
            norm=True,
            bootstrap=False,
            seed=seed + 30,
        )
        ecp, alpha = ordered_curve(ecp, alpha)
        gap_samples = samples[..., 1:] - samples[..., :-1]
        gap_truth = truth[..., 1:] - truth[..., :-1]
        gap_ecp, gap_alpha = tarp.get_tarp_coverage(
            gap_samples,
            gap_truth,
            norm=True,
            bootstrap=False,
            seed=seed + 31,
        )
        gap_ecp, gap_alpha = ordered_curve(gap_ecp, gap_alpha)
        expected_tarp = report["tarp"]
        for value, expected, name in (
            (
                np.max(np.abs(ecp - alpha)),
                expected_tarp["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"],
                "ordered eigenvalue TARP",
            ),
            (
                np.max(np.abs(gap_ecp - gap_alpha)),
                expected_tarp["eigengaps"]["full_max_abs_ecp_minus_alpha"],
                "eigengap TARP",
            ),
        ):
            if not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{method} did not reproduce {name}: {value} vs {expected}")

        sbc: dict[str, dict] = {}
        for component, name in enumerate(EIGEN_NAMES):
            ranks = randomized_ranks(
                samples[..., component], truth[..., component], seed=seed + 20 + component
            )
            expected_rank = report["derived_ordered_eigenvalues"][name]
            reproduced = rank_cdf_maximum_deviation(ranks)
            if not np.isclose(
                reproduced,
                expected_rank["rank_cdf_maximum_deviation"],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"{method} did not reproduce {name} ranks")
            sbc[name] = {
                "rows": int(len(ranks)),
                "decile_mass": decile_mass(ranks).tolist(),
                "decile_mass_95ci": clustered_decile_interval(
                    ranks,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=seed + 1000 * (method_index + 1) + component,
                ).tolist(),
                "rank_cdf_maximum_deviation": reproduced,
                "rank_mean": float(np.mean(ranks)),
                "rank_variance": float(np.var(ranks)),
            }
        products[method] = {
            "label": label,
            "color": color,
            "rows": int(len(truth)),
            "draws": int(samples.shape[0]),
            "ordered_eigenvalue_tarp": compact_curve(alpha, ecp),
            "eigengap_tarp": compact_curve(gap_alpha, gap_ecp),
            "sbc": sbc,
        }
    return products


def plot_gate(ax: plt.Axes) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 501)
    ax.fill_between(
        grid,
        np.maximum(0.0, grid - 0.05),
        np.minimum(1.0, grid + 0.05),
        color=TEXT_COLOR,
        alpha=0.06,
        label=r"Registered $|\mathrm{ECP}-\alpha|\leq0.05$ gate",
    )
    ax.plot(grid, grid, color=TEXT_COLOR, linestyle="--", linewidth=1.5)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), ylabel="Expected coverage")
    ax.grid(True, alpha=0.15)
    return grid


def render_tarp_comparison(
    output: Path,
    p12a: dict,
    products: dict[str, dict],
) -> None:
    apply_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        constrained_layout=True,
    )
    titles = (
        "U-PATCH + FMPE: direct posterior",
        "P12-F: physical ordered eigenvalues",
        "P12-F: physical eigengaps",
    )
    for column, title in enumerate(titles):
        plot_gate(axes[0, column])
        axes[0, column].set_title(title)
        axes[1, column].axhspan(-0.05, 0.05, color=TEXT_COLOR, alpha=0.06)
        axes[1, column].axhline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.3)
        axes[1, column].set(
            xlim=(0.0, 1.0),
            ylim=(-0.18, 0.18),
            xlabel=r"Credibility level $\alpha$",
            ylabel=r"ECP $-\alpha$",
        )
        axes[1, column].grid(True, alpha=0.15)

    p_alpha = np.asarray(p12a["alpha"])
    p_ecp = np.asarray(p12a["expected_coverage_probability"])
    axes[0, 0].plot(
        p_alpha,
        p_ecp,
        color=ACCENT_COLORS["blue"],
        linewidth=2.6,
        label=f"P12-A (max {p12a['max_abs_ecp_minus_alpha']:.4f})",
    )
    axes[1, 0].plot(
        p_alpha,
        p_ecp - p_alpha,
        color=ACCENT_COLORS["blue"],
        linewidth=2.3,
    )
    axes[0, 0].text(
        0.03,
        0.96,
        "50,000 galaxies × 512 draws\nstandardized ordered-softplus coordinates",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=10,
    )

    for method, _, _ in METHODS:
        product = products[method]
        for column, key in ((1, "ordered_eigenvalue_tarp"), (2, "eigengap_tarp")):
            curve = product[key]
            alpha = np.asarray(curve["alpha"])
            ecp = np.asarray(curve["expected_coverage_probability"])
            axes[0, column].plot(
                alpha,
                ecp,
                color=product["color"],
                linewidth=2.2,
                label=f"{product['label']} (max {curve['maximum_deviation']:.3f})",
            )
            axes[1, column].plot(
                alpha,
                ecp - alpha,
                color=product["color"],
                linewidth=2.0,
            )
    for column in range(3):
        axes[0, column].legend(loc="lower right", fontsize=9)
    axes[0, 1].text(
        0.03,
        0.96,
        "74,191 galaxies in 127 supported cores × 64 field draws",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=10,
    )
    figure.suptitle(
        "Held-out ph006 TARP calibration — separated posterior estimands",
        fontsize=20,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def p12a_sbc_rows(audit: dict) -> dict[str, dict]:
    global_rank = audit["global"]["eigen_rank"]["components"]
    bootstrap = audit["spatial_block_bootstrap"]["global"]["components"]
    return {
        name: {
            "decile_mass": global_rank[name]["decile_mass"],
            "decile_mass_95ci": bootstrap[name]["decile_mass_95ci"],
            "rank_cdf_maximum_deviation": global_rank[name]["weighted_ks_distance"],
        }
        for name in EIGEN_NAMES
    }


def render_sbc_comparison(
    output: Path,
    p12a: dict,
    products: dict[str, dict],
) -> None:
    apply_style()
    rows = [
        ("U-PATCH + FMPE", p12a, TEXT_COLOR),
        *[(products[m]["label"], products[m]["sbc"], products[m]["color"]) for m, _, _ in METHODS],
    ]
    maximum_interval = max(
        float(np.max(np.asarray(components[name]["decile_mass_95ci"])[:, 2]))
        for _, components, _ in rows
        for name in EIGEN_NAMES
    )
    y_maximum = max(0.16, min(1.0, 1.12 * maximum_interval))
    figure, axes = plt.subplots(4, 3, figsize=(17, 16), constrained_layout=True)
    centres = np.arange(10) + 0.5
    for row_index, (row_label, components, color) in enumerate(rows):
        for component, name in enumerate(EIGEN_NAMES):
            ax = axes[row_index, component]
            values = np.asarray(components[name]["decile_mass"], dtype=np.float64)
            interval = np.asarray(components[name]["decile_mass_95ci"], dtype=np.float64)
            error = np.vstack((values - interval[:, 0], interval[:, 2] - values))
            error = np.maximum(error, 0.0)
            ax.bar(
                centres,
                values,
                width=0.88,
                color=color,
                alpha=0.72,
                edgecolor=TEXT_COLOR,
                linewidth=0.35,
            )
            ax.errorbar(
                centres,
                values,
                yerr=error,
                fmt="none",
                ecolor=TEXT_COLOR,
                elinewidth=0.8,
                capsize=1.8,
            )
            ax.axhline(0.1, color=TEXT_COLOR, linestyle="--", linewidth=1.3)
            ax.set(
                xlim=(0.0, 10.0),
                ylim=(0.0, y_maximum),
                xticks=np.arange(0, 11, 2),
                xlabel="Randomized posterior-rank decile",
                ylabel="Mass",
            )
            ax.grid(True, axis="y", alpha=0.15)
            distance = float(components[name]["rank_cdf_maximum_deviation"])
            ax.set_title(
                f"{row_label}: $\\lambda_{component + 1}$\nrank-CDF max deviation {distance:.3f}",
                fontsize=13,
            )
    figure.suptitle(
        "Held-out ph006 SBC rank histograms\n"
        "bars are spatially correlated; intervals bootstrap authoritative cores",
        fontsize=20,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=P12F_ROOT / "archives")
    parser.add_argument("--p12a-audit", type=Path, default=P12A_AUDIT)
    parser.add_argument("--p12a-tarp", type=Path, default=P12A_TARP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    config, panel, reports, archives = load_frozen_contracts(
        args.config, args.panel, args.report_root
    )
    for method, _, _ in METHODS:
        expected = args.archive_root / method / "P12F_SAMPLE_ARCHIVE.json"
        if Path(reports[method]["archive_manifest"]).resolve() != expected.resolve():
            raise RuntimeError(f"archive-root contract differs for {method}")
    products = reconstruct_calibration(
        panel,
        reports,
        archives,
        device=args.device,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
    )

    p12a_audit = json.loads(args.p12a_audit.read_text())
    p12a_tarp = json.loads(args.p12a_tarp.read_text())
    if p12a_audit.get("sealed_phase_opened") or p12a_tarp.get("sealed_phase_opened"):
        raise RuntimeError("P12-A sealed-phase guard failed")
    if p12a_audit.get("selection_phase") != "ph006":
        raise RuntimeError("P12-A calibration reference is not ph006")
    render_tarp_comparison(args.output_dir / "p12f_tarp_comparison", p12a_tarp, products)
    render_sbc_comparison(
        args.output_dir / "p12f_sbc_comparison",
        p12a_sbc_rows(p12a_audit),
        products,
    )

    comparison = {
        "schema_version": "p12f-calibration-plot-comparison-v1",
        "created_utc": utc_now(),
        "purpose": "plot frozen ph006 calibration only; no fitting, correction, posterior sampling, or ph001 access",
        "selection_phase": "ph006",
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "comparison_warning": (
            "P12-A is a direct per-galaxy ordered-eigenvalue posterior on 50k folds-2--4 rows; "
            "P12-F is a local coherent-field posterior evaluated at 74,191 galaxies in a separate "
            "truth-free-stratified 128-core panel. Curves are adjacent diagnostics, not a matched-row leaderboard."
        ),
        "p12a": {
            "rows": p12a_audit["evaluation_rows"],
            "draws": p12a_audit["posterior_samples_per_row"],
            "base_r2": p12a_audit["global"]["intervals"]["base_r2"],
            "posterior_mean_r2": p12a_audit["global"]["intervals"]["posterior_mean_r2"],
            "coverage68": p12a_audit["global"]["intervals"]["coverage68"],
            "coverage90": p12a_audit["global"]["intervals"]["coverage90"],
            "tarp_maximum_deviation": p12a_tarp["max_abs_ecp_minus_alpha"],
            "audit_path": str(args.p12a_audit),
            "audit_sha256": sha256(args.p12a_audit),
            "tarp_path": str(args.p12a_tarp),
            "tarp_sha256": sha256(args.p12a_tarp),
        },
        "p12f": products,
        "frozen_reports": {
            method: {
                "path": str((args.report_root / f"{method}.json").resolve()),
                "sha256": sha256(args.report_root / f"{method}.json"),
                "voxel_posterior_mean_r2": reports[method]["voxel"]["posterior_mean_r2_diagnostic"],
                "derived_eigenvalue_posterior_mean_r2": [
                    reports[method]["derived_ordered_eigenvalues"][name]["posterior_mean_r2_diagnostic"]
                    for name in EIGEN_NAMES
                ],
                "primary_energy_score": reports[method]["proper_scores"]["primary_joint"],
                "marginal_crps": reports[method]["proper_scores"]["marginal_crps"],
                "maximum_conditional_coverage_error": reports[method]["maximum_conditional_coverage_error"],
            }
            for method, _, _ in METHODS
        },
        "provenance": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "panel": str(args.panel.resolve()),
            "panel_sha256": sha256(args.panel),
            "bootstrap_repeats": args.bootstrap_repeats,
            "random_seed": args.seed,
            "tarp_norm": True,
            "fixed_physics": "local-periodic no-second-smoothing tidal projector followed by eigvalsh",
        },
        "figures": {
            "tarp": str((args.output_dir / "p12f_tarp_comparison.png").resolve()),
            "sbc": str((args.output_dir / "p12f_sbc_comparison.png").resolve()),
        },
    }
    atomic_json(args.evidence_output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
