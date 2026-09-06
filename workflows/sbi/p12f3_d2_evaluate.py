#!/usr/bin/env python3
"""Run the frozen, visual D2 ph006 evaluation for one NFE archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from workflows.abacus_tweb.p6_field_patch_utils import trilinear_sample
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_models import configure_d2_determinism
from workflows.sbi.p12f3_evaluate_conditional_archive import (
    DEPLOYABLE_CONDITIONS,
    maximum_coverage_error,
)
from workflows.sbi.p12f3l2_shear_audit import audit_archive
from workflows.sbi.p12f_common_evaluator import (
    evaluate_records,
    load_core_record,
    sample_eigenvalues_at_galaxies,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    conditional_reports,
    fixed_tidal_eigenvalues,
    quantile_labels,
    scalar_posterior_report,
)
from workflows.sbi.plot_p12f3_hierarchical_comparison import analyze_archive


EVALUATION_SCHEMA = "p12f3-d2-ph006-evaluation-v1"
PROXY_CONDITIONS = (
    "frozen_g1_mean_scaled",
    "frozen_g1_log_std",
    "frozen_g1_traceless_shear_amplitude",
)
DERIVED_VARIABLES = ("lambda1", "lambda2", "lambda3", "gap12", "gap23")


def _spectral_edges() -> np.ndarray:
    cutoff = 0.1813799364234218
    maximum = np.sqrt(3.0) * np.pi / 5.0
    return np.concatenate(([0.0, cutoff / 2.0], np.linspace(cutoff, maximum, 17)))


def posterior_mean_spectral_diagnostics(
    records: list[tuple[dict, dict]], *, voxel_mpc_h: float = 5.0
) -> dict:
    """Power, transfer and cross-correlation of the posterior mean.

    These are reconstruction diagnostics only, never posterior promotion
    scores.  Each core is support-masked and mean-centred before its Fourier
    transform; sums are accumulated over identical bins and authoritative
    cores before ratios are formed.
    """
    edges = _spectral_edges()
    bins = len(edges) - 1
    truth_power = np.zeros(bins, dtype=np.float64)
    mean_power = np.zeros(bins, dtype=np.float64)
    cross_power = np.zeros(bins, dtype=np.float64)
    mode_count = np.zeros(bins, dtype=np.int64)
    for _, record in records:
        samples = np.asarray(record["delta_samples"], dtype=np.float64)
        truth = np.asarray(record["delta_truth"], dtype=np.float64)
        support = np.asarray(record["support"], dtype=bool)
        bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        core = tuple(
            slice(int(left), int(right))
            for left, right in zip(bounds[0], bounds[1], strict=True)
        )
        mask = support[core].astype(np.float64)
        if not np.any(mask):
            raise RuntimeError("D2 spectral diagnostic found an unsupported core")
        target = truth[core]
        posterior_mean = samples[(slice(None),) + core].mean(axis=0)

        def support_center(values: np.ndarray) -> np.ndarray:
            mean = float(np.sum(values * mask) / np.sum(mask))
            return (values - mean) * mask

        target_k = np.fft.rfftn(support_center(target), norm="ortho")
        mean_k = np.fft.rfftn(support_center(posterior_mean), norm="ortho")
        shape = target.shape
        kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=voxel_mpc_h)[:, None, None]
        ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=voxel_mpc_h)[None, :, None]
        kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=voxel_mpc_h)[None, None, :]
        radius = np.sqrt(kx * kx + ky * ky + kz * kz).ravel()
        labels = np.searchsorted(edges[1:-1], radius, side="right")
        keep = radius > 0
        truth_raw = np.square(np.abs(target_k)).ravel()
        mean_raw = np.square(np.abs(mean_k)).ravel()
        cross_raw = np.real(mean_k * np.conjugate(target_k)).ravel()
        truth_power += np.bincount(
            labels[keep], weights=truth_raw[keep], minlength=bins
        )
        mean_power += np.bincount(
            labels[keep], weights=mean_raw[keep], minlength=bins
        )
        cross_power += np.bincount(
            labels[keep], weights=cross_raw[keep], minlength=bins
        )
        mode_count += np.bincount(labels[keep], minlength=bins)
    per_mode_truth = np.divide(
        truth_power,
        mode_count,
        out=np.full(bins, np.nan),
        where=mode_count > 0,
    )
    per_mode_mean = np.divide(
        mean_power,
        mode_count,
        out=np.full(bins, np.nan),
        where=mode_count > 0,
    )
    per_mode_cross = np.divide(
        cross_power,
        mode_count,
        out=np.full(bins, np.nan),
        where=mode_count > 0,
    )
    cross_transfer = np.divide(
        cross_power,
        truth_power,
        out=np.full(bins, np.nan),
        where=truth_power > 0,
    )
    amplitude_transfer = np.sqrt(
        np.divide(
            mean_power,
            truth_power,
            out=np.full(bins, np.nan),
            where=truth_power > 0,
        )
    )
    cross_correlation = np.divide(
        cross_power,
        np.sqrt(mean_power * truth_power),
        out=np.full(bins, np.nan),
        where=(mean_power > 0) & (truth_power > 0),
    )
    finite = mode_count > 0
    if not all(
        np.all(np.isfinite(values[finite]))
        for values in (
            per_mode_truth,
            per_mode_mean,
            per_mode_cross,
            cross_transfer,
            amplitude_transfer,
            cross_correlation,
        )
    ):
        raise RuntimeError("D2 posterior-mean spectrum is non-finite")
    return {
        "role": "diagnostic_only_not_a_posterior_promotion_score",
        "field_preparation": "exact-support masked and support-weighted-mean centred per authoritative core",
        "spectral_k_edges_h_mpc": edges.tolist(),
        "spectral_k_centres_h_mpc": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "mode_count": mode_count.tolist(),
        "posterior_mean_power": per_mode_mean.tolist(),
        "truth_power": per_mode_truth.tolist(),
        "cross_power": per_mode_cross.tolist(),
        "p_cross_over_p_truth": cross_transfer.tolist(),
        "sqrt_p_mean_over_p_truth": amplitude_transfer.tolist(),
        "r_k": cross_correlation.tolist(),
    }


def derived_physics_conditionals(
    records: list[tuple[dict, dict]], *, device: str, seed: int
) -> dict:
    """Component coverage for galaxy-sampled eigenvalues and eigengaps."""
    draw_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    core_parts: list[np.ndarray] = []
    strata_parts = {
        name: []
        for name in (
            "shell",
            "random_response",
            "boundary_distance",
            "tracer_density",
            *PROXY_CONDITIONS,
        )
    }
    for metadata, record in records:
        coordinates = np.asarray(record["galaxy_frac_index_local"], dtype=np.float64)
        if not len(coordinates):
            continue
        support = np.asarray(record["support"], dtype=bool)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or not np.all(np.isfinite(coordinates))
            or np.any(coordinates < 0.0)
            or np.any(coordinates > (np.asarray(support.shape) - 1.0))
        ):
            raise RuntimeError(
                "D2 authoritative galaxy coordinates fall outside the frozen patch"
            )
        nearest = np.rint(coordinates).astype(np.int64)
        if not np.all(support[tuple(nearest.T)]):
            raise RuntimeError("D2 derived calibration includes an M=0 galaxy")
        samples = np.asarray(record["delta_samples"], dtype=np.float32)
        truth = np.asarray(record["delta_truth"], dtype=np.float32)
        with torch.inference_mode():
            sample_eigen = (
                fixed_tidal_eigenvalues(torch.from_numpy(samples).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            truth_eigen = (
                fixed_tidal_eigenvalues(torch.from_numpy(truth).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        sampled_draw, sampled_truth = sample_eigenvalues_at_galaxies(
            sample_eigen, truth_eigen, coordinates
        )
        if not len(sampled_truth):
            continue
        draw_parts.append(sampled_draw)
        truth_parts.append(sampled_truth)
        count = len(sampled_truth)
        core_parts.append(
            np.full(count, int(metadata["core_id"]), dtype=np.int64)
        )
        strata_parts["shell"].append(
            np.full(count, int(metadata["shell"]), dtype=np.int8)
        )
        for name, record_name in (
            ("random_response", "angular_response"),
            ("boundary_distance", "boundary_distance_mpc"),
            ("tracer_density", "tracer_density"),
            *[(name, name) for name in PROXY_CONDITIONS],
        ):
            values = trilinear_sample(np.asarray(record[record_name]), coordinates)
            strata_parts[name].append(np.asarray(values, dtype=np.float32))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not draw_parts:
        raise RuntimeError("D2 derived conditional audit found no galaxies")
    eigen_draw = np.concatenate(draw_parts, axis=1)
    eigen_truth = np.concatenate(truth_parts, axis=0)
    core_group = np.concatenate(core_parts)
    gap_draw = eigen_draw[..., 1:] - eigen_draw[..., :-1]
    gap_truth = eigen_truth[..., 1:] - eigen_truth[..., :-1]
    variables = {
        "lambda1": (eigen_draw[..., 0], eigen_truth[..., 0]),
        "lambda2": (eigen_draw[..., 1], eigen_truth[..., 1]),
        "lambda3": (eigen_draw[..., 2], eigen_truth[..., 2]),
        "gap12": (gap_draw[..., 0], gap_truth[..., 0]),
        "gap23": (gap_draw[..., 1], gap_truth[..., 1]),
    }
    labels = {
        "shell": np.concatenate(strata_parts["shell"]),
        **{
            name: quantile_labels(np.concatenate(strata_parts[name]))
            for name in strata_parts
            if name != "shell"
        },
        "true_environment": np.sum(eigen_truth > 0.2, axis=1).astype(np.int8),
    }
    conditional = {
        variable: {
            condition: conditional_reports(
                draws,
                target,
                condition_labels,
                seed=seed + 100 * variable_index + condition_index,
            )
            for condition_index, (condition, condition_labels) in enumerate(
                labels.items()
            )
        }
        for variable_index, (variable, (draws, target)) in enumerate(
            variables.items()
        )
    }
    deployable = (
        "shell",
        "random_response",
        "boundary_distance",
        "tracer_density",
        *PROXY_CONDITIONS,
    )
    deployable_errors = {
        variable: maximum_coverage_error(rows, deployable)
        for variable, rows in conditional.items()
    }
    true_environment_errors = {
        variable: maximum_coverage_error(rows, ("true_environment",))
        for variable, rows in conditional.items()
    }
    return {
        "role": "componentwise conditional coverage; pooled joint dependence is tested separately by TARP",
        "galaxies": int(eigen_truth.shape[0]),
        "authoritative_cores": int(len(np.unique(core_group))),
        "dependence_unit": "authoritative ph006 core; component coverage is a point estimate and no galaxy-IID uncertainty is claimed",
        "variables": conditional,
        "maximum_deployable_error_by_variable": deployable_errors,
        "maximum_deployable_error": max(deployable_errors.values()),
        "maximum_true_environment_error_by_variable": true_environment_errors,
        "maximum_true_environment_error_diagnostic": max(
            true_environment_errors.values()
        ),
        "true_environment_definition": "number of true ordered eigenvalues greater than lambda_th=0.2; evaluation only",
    }


def combined_deployable_conditional_error(
    voxel_error: float, derived_report: dict
) -> float:
    """Frozen D2 gate takes the worse voxel or derived-physics coverage error."""
    values = (float(voxel_error), float(derived_report["maximum_deployable_error"]))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("D2 deployable conditional coverage error is non-finite")
    return max(values)


def label_authoritative_core_bootstrap(*rows: dict) -> None:
    """Correct the inherited generic TARP label to the actual D2 blocks."""
    for row in rows:
        if row.get("available"):
            row["bootstrap_scheme"] = "authoritative ph006 core cluster resampling"
            row["bootstrap_unit"] = "authoritative ph006 patch core"


def proxy_conditionals(records: list[tuple[dict, dict]], *, seed: int) -> dict:
    """Voxel coverage stratified by the D2 deployable conditioner itself."""
    sample_parts = []
    truth_parts = []
    proxy_parts = {name: [] for name in PROXY_CONDITIONS}
    for _, record in records:
        samples = np.asarray(record["delta_samples"], dtype=np.float32)
        truth = np.asarray(record["delta_truth"], dtype=np.float32)
        support = np.asarray(record["support"], dtype=bool)
        bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        core = tuple(
            slice(int(left), int(right))
            for left, right in zip(bounds[0], bounds[1], strict=True)
        )
        valid = np.flatnonzero(support[core].ravel())
        if len(valid) > 2048:
            valid = valid[np.linspace(0, len(valid) - 1, 2048, dtype=np.int64)]
        if not len(valid):
            raise RuntimeError("D2 proxy-conditional audit found no exact support")
        sample_parts.append(samples[(slice(None),) + core].reshape(samples.shape[0], -1)[:, valid])
        truth_parts.append(truth[core].ravel()[valid])
        for name in PROXY_CONDITIONS:
            proxy_parts[name].append(np.asarray(record[name])[core].ravel()[valid])
    sample = np.concatenate(sample_parts, axis=1)
    truth = np.concatenate(truth_parts)
    return {
        name: conditional_reports(
            sample,
            truth,
            quantile_labels(np.concatenate(proxy_parts[name])),
            seed=seed + index,
        )
        for index, name in enumerate(PROXY_CONDITIONS)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--matched-reference-marker", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fixed_lag_three_point(archive: dict, *, seed: int) -> dict:
    """A fixed, phase-sensitive three-point (bispectrum-dual) diagnostic.

    The statistic is <r(x) r(x+ex) r(x+ey)> on the authoritative core and
    exact common support.  It is intentionally frozen and descriptive rather
    than a tunable family of higher-order summaries.
    """
    draw_values: list[np.ndarray] = []
    truth_values: list[float] = []
    core_ids: list[int] = []
    for entry in archive["entries"]:
        record = load_core_record(entry, int(archive["draws"]))
        samples = np.asarray(record["delta_samples"], dtype=np.float64)
        truth = np.asarray(record["delta_truth"], dtype=np.float64)
        bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        core = tuple(
            slice(int(left), int(right))
            for left, right in zip(bounds[0], bounds[1], strict=True)
        )
        sample = samples[(slice(None),) + core]
        target = truth[core]
        support = np.asarray(record["support"], dtype=bool)[core]
        posterior_mean = sample.mean(axis=0)
        residual = sample - posterior_mean[None]
        innovation = target - posterior_mean
        valid = support[:-1, :-1, :] & support[1:, :-1, :] & support[:-1, 1:, :]
        if not np.any(valid):
            continue
        draw_stat = np.mean(
            (
                residual[:, :-1, :-1, :]
                * residual[:, 1:, :-1, :]
                * residual[:, :-1, 1:, :]
            )[:, valid],
            axis=1,
        )
        truth_stat = float(
            np.mean(
                (
                    innovation[:-1, :-1, :]
                    * innovation[1:, :-1, :]
                    * innovation[:-1, 1:, :]
                )[valid]
            )
        )
        draw_values.append(draw_stat)
        truth_values.append(truth_stat)
        core_ids.append(int(entry["core_id"]))
    if len(draw_values) != len(archive["entries"]):
        raise RuntimeError("D2 phase-sensitive diagnostic lost an evaluation core")
    draws = np.stack(draw_values, axis=1)
    truth = np.asarray(truth_values, dtype=np.float64)
    report = scalar_posterior_report(draws, truth, seed=seed)
    ranks = np.mean(draws < truth[None, :], axis=0)
    rank_sorted = np.sort(ranks)
    rank_ecdf = np.arange(1, len(rank_sorted) + 1, dtype=np.float64) / len(
        rank_sorted
    )
    posterior_variance = float(np.var(draws))
    truth_variance = float(np.var(truth))
    finite = bool(np.all(np.isfinite(draws)) and np.all(np.isfinite(truth)))
    non_degenerate = bool(posterior_variance > 1.0e-14 and truth_variance > 1.0e-14)
    return {
        "name": "fixed_lag_xy_three_point_bispectrum_proxy",
        "definition": "mean r(x) r(x+1voxel_x) r(x+1voxel_y) on exact-supported authoritative cores",
        "cores": len(core_ids),
        "core_ids": core_ids,
        "draws": int(draws.shape[0]),
        "posterior_variance": posterior_variance,
        "truth_innovation_variance": truth_variance,
        "finite": finite,
        "non_degenerate": non_degenerate,
        "posterior": report,
        "rank_cdf": {
            "rank": rank_sorted.tolist(),
            "empirical_cdf": rank_ecdf.tolist(),
        },
        "available": bool(finite and non_degenerate),
    }


def _maximum_abs_tarp(report: dict, name: str) -> float:
    row = report["tarp"][name]
    if not row.get("available"):
        raise RuntimeError(f"D2 {name} TARP is unavailable")
    return float(row["full_max_abs_ecp_minus_alpha"])


def _plot_bundle(
    common: dict,
    shear: dict,
    visual: dict,
    higher: dict,
    references: dict[str, dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    colors = {
        "D2": "#2878b5",
        "G1": "#777777",
        "F3-L2b": "#f28e2b",
        "F3-L2d": "#59a14f",
    }
    report_rows = {
        "D2": common,
        "G1": references["g1"],
        "F3-L2b": references["f3l2b"],
        "F3-L2d": references["f3l2d_nfe100"],
    }
    axis = axes[0, 0]
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    for label, report in report_rows.items():
        for name, linestyle in (("ordered_eigenvalues", "-"), ("eigengaps", "--")):
            row = report["tarp"][name]
            alpha = np.asarray(row["alpha"])
            observed = np.asarray(row["expected_coverage_probability"])
            stride = max(1, len(alpha) // 500)
            axis.plot(
                alpha[::stride],
                observed[::stride],
                color=colors[label],
                linestyle=linestyle,
                linewidth=1.6 if label == "D2" else 1.0,
                alpha=1.0 if label == "D2" else .75,
                label=(
                    f"{label} {'eigen' if name == 'ordered_eigenvalues' else 'gap'} "
                    f"({row['full_max_abs_ecp_minus_alpha']:.3f})"
                ),
            )
    envelope = float(
        max(
            common["tarp"][name]["bootstrap_max_abs_quantiles"][2]
            for name in ("ordered_eigenvalues", "eigengaps")
        )
    )
    diagonal = np.linspace(0, 1, 200)
    axis.fill_between(
        diagonal,
        np.clip(diagonal - envelope, 0, 1),
        np.clip(diagonal + envelope, 0, 1),
        color=colors["D2"],
        alpha=.08,
        label="D2 block-bootstrap 95% max-deviation envelope",
    )
    shear_row = shear["joint_tarp_blocked"]
    alpha = np.asarray(shear_row["alpha"])
    observed = np.asarray(shear_row["expected_coverage_probability"])
    stride = max(1, len(alpha) // 500)
    axis.plot(alpha[::stride], observed[::stride], color="#6a4c93", label=f"5-shear ({shear_row['full_max_abs_ecp_minus_alpha']:.3f})")
    for label, report in report_rows.items():
        if label == "D2":
            continue
        row = report["_d2_shear"]["joint_tarp_blocked"]
        alpha = np.asarray(row["alpha"])
        observed = np.asarray(row["expected_coverage_probability"])
        stride = max(1, len(alpha) // 500)
        axis.plot(
            alpha[::stride],
            observed[::stride],
            color=colors[label],
            linestyle=":",
            linewidth=1.0,
            alpha=.75,
            label=f"{label} 5-shear ({row['full_max_abs_ecp_minus_alpha']:.3f})",
        )
    axis.set(xlabel="nominal credibility", ylabel="empirical coverage", title="A. Joint physical TARP", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=.2)
    axis.legend(fontsize=6, ncol=2)

    axis = axes[0, 1]
    visual_rows = {"D2": visual}
    visual_rows.update(
        {
            label: report["_d2_visual"]
            for label, report in report_rows.items()
            if label != "D2"
        }
    )
    plotted_ratios = []
    for label, row in visual_rows.items():
        ratios = np.asarray(row["posterior_to_truth_power"], dtype=np.float64)
        edges = np.asarray(row["spectral_k_edges_h_mpc"], dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plotted_ratios.append(ratios)
        axis.plot(
            centers,
            ratios,
            marker="o" if label == "D2" else None,
            markersize=3,
            color=colors[label],
            linewidth=1.6 if label == "D2" else 1.0,
            label=label,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.axhspan(.9, 1.1, color="#7cb518", alpha=.12, label="registered low-band gate")
    finite_power = np.concatenate([row.ravel() for row in plotted_ratios])
    finite_power = finite_power[np.isfinite(finite_power)]
    power_maximum = float(np.max(finite_power)) if finite_power.size else 2.0
    axis.set(
        xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
        ylabel="posterior / truth residual power",
        title="B. Residual power",
        ylim=(0, max(2.0, power_maximum * 1.1)),
    )
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    variables = list(DEPLOYABLE_CONDITIONS + PROXY_CONDITIONS) + ["true_environment"]
    x = np.arange(len(variables))
    for label, report in report_rows.items():
        errors = []
        for variable in variables:
            values = [
                float(row["coverage"][level]["absolute_error"])
                for row in report["conditional_voxel_coverage"][variable].values()
                for level in ("0.68", "0.90")
            ]
            errors.append(max(values))
        axis.plot(x, errors, marker="o", color=colors[label], label=label)
    axis.axhline(.10, color="black", linestyle="--", label="deployable gate")
    axis.set_xticks(x, variables, rotation=25, ha="right")
    axis.set(ylabel="maximum absolute coverage error", title="C. Conditional coverage")
    axis.grid(axis="y", alpha=.2)
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    methods = list(report_rows)
    values = [float(row["proper_scores"]["energy"]) for row in report_rows.values()]
    axis.bar(np.arange(4), values, color=[colors[name] for name in methods])
    axis.set_xticks(np.arange(4), methods, rotation=20, ha="right")
    axis.set(ylabel="energy score (lower is better)", title="D. Matched joint score")
    axis.grid(axis="y", alpha=.2)
    axis = axes[1, 1]
    rank = np.asarray(higher["rank_cdf"]["rank"], dtype=np.float64)
    empirical = np.asarray(higher["rank_cdf"]["empirical_cdf"])
    if rank.shape != empirical.shape:
        raise RuntimeError("D2 three-point rank-CDF arrays differ")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="uniform ranks")
    axis.step(rank, empirical, where="post", color=colors["D2"], label="D2")
    axis.set(
        xlabel="posterior rank",
        ylabel="empirical CDF",
        title=(
            "E. Fixed three-point rank diagnostic\n"
            f"KS-like deviation={higher['posterior']['rank_cdf_maximum_deviation']:.3f} (descriptive)"
        ),
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    metrics = {
        "eigen TARP": float(common["tarp"]["ordered_eigenvalues"]["full_max_abs_ecp_minus_alpha"]) / .05,
        "gap TARP": float(common["tarp"]["eigengaps"]["full_max_abs_ecp_minus_alpha"]) / .05,
        "5-shear TARP": float(shear_row["full_max_abs_ecp_minus_alpha"]) / .05,
        "global cov.": max(map(float, common["global_coverage_error"].values())) / .05,
        "conditional cov.\nvoxel+derived": float(
            common["maximum_deployable_conditional_coverage_error"]
        )
        / .10,
    }
    normalized = np.asarray(list(metrics.values()))
    axis.bar(
        np.arange(len(metrics)),
        normalized,
        color=["#7cb518" if value <= 1 else "#d1495b" for value in normalized],
    )
    axis.axhline(1, color="black", linestyle="--", label="registered limit")
    axis.set_xticks(np.arange(len(metrics)), list(metrics), rotation=25, ha="right")
    axis.set(ylabel="value / gate", title="F. Calibration gate intuition")
    axis.grid(axis="y", alpha=.2)
    axis.legend(fontsize=8)
    figure.suptitle(
        f"P12-F3-D2 held-out ph006 | three-point diagnostic {'available' if higher['available'] else 'unavailable'}",
        fontsize=14,
    )
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _plot_transfer_and_derived_coverage(
    spectral_rows: dict[str, dict], derived: dict, output: Path
) -> None:
    """Show reconstruction spectra separately from posterior coverage gates."""
    colors = {
        "D2": "#2878b5",
        "G1": "#777777",
        "F3-L2b": "#f28e2b",
        "F3-L2d": "#59a14f",
    }
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for label, row in spectral_rows.items():
        k = np.asarray(row["spectral_k_centres_h_mpc"], dtype=np.float64)
        axes[0].plot(
            k,
            row["p_cross_over_p_truth"],
            color=colors[label],
            linewidth=1.8 if label == "D2" else 1.0,
            label=f"{label} $P_{{cross}}/P_{{true}}$",
        )
        axes[0].plot(
            k,
            row["sqrt_p_mean_over_p_truth"],
            color=colors[label],
            linestyle="--",
            linewidth=1.2 if label == "D2" else .8,
            alpha=.8,
            label=rf"{label} $\sqrt{{P_{{mean}}/P_{{true}}}}$",
        )
        axes[1].plot(
            k,
            row["r_k"],
            color=colors[label],
            linewidth=1.8 if label == "D2" else 1.0,
            label=label,
        )
    for axis in axes[:2]:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axvspan(0.0, 0.1813799364234218, color="#f2c14e", alpha=.10)
        axis.grid(alpha=.2)
        axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axes[0].set_ylabel("amplitude diagnostic")
    axes[0].set_title("A. Posterior-mean transfer\n(diagnostic, not a calibration gate)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_ylabel(r"$r(k)$")
    axes[1].set_ylim(-.05, 1.05)
    axes[1].set_title("B. Posterior-mean cross-correlation\n(diagnostic, not a calibration gate)")
    axes[1].legend(fontsize=7)

    conditions = (
        "shell",
        "random_response",
        "boundary_distance",
        "tracer_density",
        *PROXY_CONDITIONS,
        "true_environment",
    )
    matrix = np.asarray(
        [
            [
                max(
                    float(report["coverage"][level]["absolute_error"])
                    for report in derived["variables"][variable][condition].values()
                    for level in ("0.68", "0.90")
                )
                for variable in DERIVED_VARIABLES
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    image = axes[2].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(.10, float(np.max(matrix))),
    )
    axes[2].set_xticks(np.arange(len(DERIVED_VARIABLES)), DERIVED_VARIABLES)
    axes[2].set_yticks(np.arange(len(conditions)), conditions)
    axes[2].set_title("C. Derived conditional coverage\nmax |empirical - nominal| at 68/90%")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[2].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > .05 else "black",
                fontsize=7,
            )
    figure.colorbar(image, ax=axes[2], fraction=.046, label="absolute coverage error")
    figure.suptitle(
        "P12-F3-D2 held-out ph006 — reconstruction diagnostics and derived posterior calibration",
        fontsize=14,
    )
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _plot_example_and_training(archive: dict, example: dict, output: Path) -> None:
    trained = json.loads(Path(archive["trained_marker"]).read_text())
    diagnostics = [
        json.loads(line)
        for line in Path(trained["milestone_diagnostics"]).read_text().splitlines()
        if line.strip()
    ]
    loss_path = Path(archive["trained_marker"]).parent / "loss_trace.jsonl"
    losses = [json.loads(line) for line in loss_path.read_text().splitlines() if line.strip()]
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    truth = np.asarray(example["truth"])
    mean = np.asarray(example["mean"])
    std = np.asarray(example["std"])
    z = truth.shape[2] // 2
    limit = float(np.quantile(np.abs(truth), .99))
    for axis, field, title, cmap, limits in (
        (axes[0, 0], truth[:, :, z], "truth", "coolwarm", (-limit, limit)),
        (axes[0, 1], mean[:, :, z], "posterior mean", "coolwarm", (-limit, limit)),
        (axes[0, 2], std[:, :, z], "posterior std", "magma", (0, None)),
    ):
        image = axis.imshow(field.T, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set(title=title, xticks=[], yticks=[])
        figure.colorbar(image, ax=axis, fraction=.046)
    samples = np.asarray(example["samples"])
    if samples.shape[0] < 3 or samples.shape[1:] != truth.shape:
        raise RuntimeError("D2 visual example lacks three fixed-index samples")
    for index in range(3):
        image = axes[1, index].imshow(
            samples[index, :, :, z].T,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[1, index].set(title=f"posterior draw {index}", xticks=[], yticks=[])
        figure.colorbar(image, ax=axes[1, index], fraction=.046)
    axis = axes[2, 0]
    axis.hist(
        ((truth - mean) / np.maximum(std, 1e-6)).ravel(),
        bins=80,
        density=True,
        histtype="stepfilled",
        alpha=.35,
        color="#2878b5",
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set(xlabel="standardized truth innovation", ylabel="density", title="fixed-core calibration texture")
    axis = axes[2, 1]
    if losses:
        axis.plot(
            [row["examples_seen"] for row in losses],
            [row["loss"] for row in losses],
            color="#777777",
            alpha=.5,
            label="update loss",
        )
        axis.plot(
            [row["examples_seen"] for row in losses],
            [row["mean_loss"] for row in losses],
            color="#111111",
            label="running mean",
        )
    axis.set(xlabel="patch presentations", ylabel="v-prediction loss", title="training trajectory")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    axis = axes[2, 2]
    for weight, color in (("raw", "#2878b5"), ("ema", "#d1495b")):
        axis.plot(
            [row["presentations"] for row in diagnostics],
            [row["selection"][weight]["energy_score"] for row in diagnostics],
            marker="o",
            color=color,
            label=weight,
        )
    axis.axvline(int(trained["selected_presentations"]), color="black", linestyle="--", label="frozen age")
    axis.set(xlabel="patch presentations", ylabel="128-core energy score", title="sample-based checkpoint selection")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    figure.suptitle(f"D2 field and optimization audit — core {example['core_id']}", fontsize=14)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    common_seed = int(config["evaluation"]["common_evaluator_seed"])
    higher_seed = int(config["evaluation"]["higher_order_seed"])
    reference_marker_path = (
        args.matched_reference_marker
        or args.output_root / "D2_MATCHED_REFERENCE_REPORTS.json"
    )
    reference_marker = json.loads(reference_marker_path.read_text())
    if (
        reference_marker.get("schema_version")
        != "p12f3-d2-matched-reference-reports-v1"
        or not reference_marker.get("pass")
        or reference_marker.get("frozen", {}).get("contract_digest")
        != contract["frozen_digest"]
        or int(reference_marker.get("frozen", {}).get("common_evaluator_seed", -1))
        != common_seed
        or reference_marker.get("ph001_opened")
    ):
        raise RuntimeError("unsafe D2 matched-reference reports")
    archive = json.loads(args.archive.read_text())
    if (
        archive.get("schema_version") != "p12f-sample-archive-v1"
        or not archive.get("pass")
        or archive.get("phase") != "ph006"
        or archive.get("ph001_opened")
        or archive.get("truth_files_read") != ["ph006"]
        or int(archive.get("draws", -1)) != 64
        or int(archive.get("network_evaluations", -1)) not in (50, 100)
        or int(archive.get("draw_batch", -1))
        != int(config["sampler"]["draw_batch"])
        or archive.get("d2_contract_sha256") != sha256(contract_path)
    ):
        raise RuntimeError("unsafe D2 ph006 archive")
    panel = json.loads(Path(archive["panel_marker"]).read_text())
    metadata = {
        int(row["core_id"]): row for row in panel["selected_core_metadata"]
    }
    expected = [int(value) for value in panel["selected_core_id"]]
    if (
        len(expected) != 256
        or [int(row["core_id"]) for row in archive["entries"]] != expected
    ):
        raise RuntimeError("D2 archive/panel identity changed")
    output_dir = args.output_dir or args.archive.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "D2_PH006_EVALUATION.json"
    evaluation_frozen = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "contract_digest": contract["frozen_digest"],
        "deterministic_runtime": deterministic_runtime,
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256(args.archive),
        "method": archive["method"],
        "network_evaluations": int(archive["network_evaluations"]),
        "draw_batch": int(archive["draw_batch"]),
        "sampler": archive.get("sampler", "deterministic"),
        "sampler_eta": float(archive.get("sampler_eta", 0.0)),
        "seed": int(archive["seed"]),
        "seed_role": archive["seed_role"],
        "selected_arm": archive["selected_arm"],
        "selected_presentations": int(archive["selected_presentations"]),
        "selected_weights": archive["selected_weights"],
        "checkpoint_sha256": archive["checkpoint_sha256"],
        "trained_marker_sha256": archive["trained_marker_sha256"],
        "export_frozen_digest": archive["export_frozen_digest"],
        "export_run_manifest_sha256": archive["export_run_manifest_sha256"],
        "second_seed_license_sha256": archive.get("second_seed_license_sha256"),
        "stochastic_control_license_sha256": archive.get(
            "stochastic_control_license_sha256"
        ),
        "panel_sha256": archive["panel_sha256"],
        "common_evaluator_seed": common_seed,
        "higher_order_seed": higher_seed,
        "matched_reference_marker": str(reference_marker_path.resolve()),
        "matched_reference_marker_sha256": sha256(reference_marker_path),
        "ph001_opened": False,
    }
    evaluation_digest = __import__("hashlib").sha256(
        json.dumps(evaluation_frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text())
        if (
            existing.get("schema_version") != EVALUATION_SCHEMA
            or not existing.get("pass")
            or existing.get("frozen_digest") != evaluation_digest
            or existing.get("ph001_opened")
        ):
            raise RuntimeError("existing D2 evaluation freeze changed")
        for path_key, hash_key in (
            ("common_report", "common_report_sha256"),
            ("shear_report", "shear_report_sha256"),
            ("visual_plot_data", "visual_plot_data_sha256"),
            ("spectral_transfer_plot_data", "spectral_transfer_plot_data_sha256"),
            ("higher_order_report", "higher_order_report_sha256"),
        ):
            if sha256(Path(existing[path_key])) != existing[hash_key]:
                raise RuntimeError(f"existing D2 evaluation component changed: {path_key}")
        for path_key, hash_key in (
            ("png", "png_sha256"),
            ("pdf", "pdf_sha256"),
            ("field_training_png", "field_training_png_sha256"),
            ("field_training_pdf", "field_training_pdf_sha256"),
            ("spectral_derived_png", "spectral_derived_png_sha256"),
            ("spectral_derived_pdf", "spectral_derived_pdf_sha256"),
        ):
            if sha256(Path(existing["figures"][path_key])) != existing["figures"][hash_key]:
                raise RuntimeError(f"existing D2 figure changed: {path_key}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    reserved = (
        "common_evaluation.json",
        "lowmode_shear_audit.json",
        "spectral_tarp_plot_data.json",
        "spectral_transfer_plot_data.json",
        "phase_sensitive_three_point.json",
        "p12f3_d2_ph006_diagnostics.png",
        "p12f3_d2_ph006_diagnostics.pdf",
        "p12f3_d2_field_and_training.png",
        "p12f3_d2_field_and_training.pdf",
        "p12f3_d2_transfer_and_derived_coverage.png",
        "p12f3_d2_transfer_and_derived_coverage.pdf",
    )
    if any((output_dir / name).exists() for name in reserved):
        raise RuntimeError("partial D2 evaluation exists without a terminal freeze")
    records = []
    for entry in archive["entries"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"] or "ph001" in str(path).lower():
            raise RuntimeError("D2 ph006 core artifact changed")
        record = load_core_record(entry, 64)
        with np.load(path, allow_pickle=False) as values:
            if not set(PROXY_CONDITIONS).issubset(values.files):
                raise RuntimeError("D2 archive lacks its deployable conditioner fields")
            record.update(
                {name: np.asarray(values[name]) for name in PROXY_CONDITIONS}
            )
        records.append((metadata[int(entry["core_id"])], record))
    common = evaluate_records(
        records, method=archive["method"], seed=common_seed, device=args.device
    )
    label_authoritative_core_bootstrap(
        common["tarp"]["ordered_eigenvalues"], common["tarp"]["eigengaps"]
    )
    missing = [
        name
        for name in ("ordered_eigenvalues", "eigengaps")
        if not common.get("tarp", {}).get(name, {}).get("available")
    ]
    if missing:
        raise RuntimeError(f"D2 TARP unavailable: {missing}")
    common["conditional_voxel_coverage"].update(
        proxy_conditionals(records, seed=common_seed + 100)
    )
    voxel_deployable_error = maximum_coverage_error(
        common["conditional_voxel_coverage"],
        tuple(config["evaluation"]["deployable_conditioning_gates"]),
    )
    voxel_true_environment_error = maximum_coverage_error(
        common["conditional_voxel_coverage"], ("true_environment",)
    )
    derived = derived_physics_conditionals(
        records, device=args.device, seed=common_seed + 300
    )
    candidate_spectral = posterior_mean_spectral_diagnostics(records)
    common["derived_conditional_physics"] = derived
    common["posterior_mean_spectral_diagnostic"] = candidate_spectral
    common["maximum_voxel_deployable_conditional_coverage_error"] = (
        voxel_deployable_error
    )
    common["maximum_derived_deployable_conditional_coverage_error"] = float(
        derived["maximum_deployable_error"]
    )
    common["maximum_deployable_conditional_coverage_error"] = (
        combined_deployable_conditional_error(voxel_deployable_error, derived)
    )
    common["maximum_true_environment_coverage_error_diagnostic"] = max(
        voxel_true_environment_error,
        float(derived["maximum_true_environment_error_diagnostic"]),
    )
    common.update(
        {
            "schema_version": "p12f3-d2-common-evaluation-v1",
            "created_utc": utc_now(),
            "archive": str(args.archive.resolve()),
            "archive_sha256": sha256(args.archive),
            "common_evaluator_seed": common_seed,
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        }
    )
    common_path = output_dir / "common_evaluation.json"
    atomic_json(common_path, common)
    shear = audit_archive(
        args.archive,
        device=args.device,
        draw_batch=8,
        maximum_k=0.1813799364234218,
    )
    label_authoritative_core_bootstrap(shear["joint_tarp_blocked"])
    shear["resampling_note"] = (
        "joint_tarp is pooled visualization; joint_tarp_blocked resamples "
        "authoritative ph006 patch cores"
    )
    shear_path = output_dir / "lowmode_shear_audit.json"
    atomic_json(shear_path, shear)
    visual, example = analyze_archive(args.archive, device=args.device)
    example_entry = next(
        row for row in archive["entries"] if int(row["core_id"]) == int(example["core_id"])
    )
    example_record = load_core_record(example_entry, 64)
    example_bounds = np.asarray(example_record["core_bounds"], dtype=np.int64)
    example_core = tuple(
        slice(int(left), int(right))
        for left, right in zip(example_bounds[0], example_bounds[1], strict=True)
    )
    example["samples"] = np.asarray(example_record["delta_samples"])[
        (slice(0, 3),) + example_core
    ]
    visual_path = output_dir / "spectral_tarp_plot_data.json"
    atomic_json(visual_path, visual)
    higher = fixed_lag_three_point(archive, seed=higher_seed)
    higher_path = output_dir / "phase_sensitive_three_point.json"
    atomic_json(higher_path, higher)
    candidate_records_by_core = {
        int(meta["core_id"]): record for meta, record in records
    }
    references = {}
    for key in config["evaluation"]["matched_reference_methods"]:
        row = reference_marker.get("reports", {}).get(key, {})
        report_path = Path(row.get("path", ""))
        if (
            row.get("sha256") != sha256(report_path)
            or row.get("archive_sha256")
            != contract["frozen"]["reference_contract"][key]["archive_sha256"]
        ):
            raise RuntimeError(f"D2 matched reference changed: {key}")
        report = json.loads(report_path.read_text())
        visual_reference_path = Path(row.get("visual_path", ""))
        shear_reference_path = Path(row.get("shear_path", ""))
        if (
            int(report.get("common_evaluator_seed", -1)) != common_seed
            or report.get("core_ids")
            != contract["frozen"]["reference_contract"][key]["core_ids"]
            or report.get("ph001_opened")
            or row.get("visual_sha256") != sha256(visual_reference_path)
            or row.get("shear_sha256") != sha256(shear_reference_path)
        ):
            raise RuntimeError(f"D2 matched reference identity changed: {key}")
        reference_visual = json.loads(visual_reference_path.read_text())
        reference_shear = json.loads(shear_reference_path.read_text())
        if (
            reference_visual.get("reference_key") != key
            or reference_visual.get("archive_sha256") != row.get("archive_sha256")
            or reference_visual.get("ph001_opened")
            or reference_shear.get("reference_key") != key
            or reference_shear.get("archive_sha256") != row.get("archive_sha256")
            or reference_shear.get("ph001_opened")
        ):
            raise RuntimeError(f"D2 matched reference visual changed: {key}")
        report["_d2_visual"] = reference_visual
        report["_d2_shear"] = reference_shear
        reference_archive_path = Path(row.get("archive", ""))
        if row.get("archive_sha256") != sha256(reference_archive_path):
            raise RuntimeError(f"D2 matched reference archive changed: {key}")
        reference_archive = json.loads(reference_archive_path.read_text())
        reference_records = []
        if [int(item["core_id"]) for item in reference_archive.get("entries", ())] != expected:
            raise RuntimeError(f"D2 matched reference panel changed: {key}")
        for entry in reference_archive["entries"]:
            core_id = int(entry["core_id"])
            reference_artifact = Path(entry["path"])
            if (
                "ph001" in str(reference_artifact).lower()
                or entry.get("sha256") != sha256(reference_artifact)
            ):
                raise RuntimeError(
                    f"D2 matched reference core artifact changed: {key}/{core_id}"
                )
            reference_record = load_core_record(entry, 64)
            proxy_record = candidate_records_by_core[core_id]
            if (
                not np.array_equal(
                    reference_record["core_bounds"], proxy_record["core_bounds"]
                )
                or not np.array_equal(
                    reference_record["support"], proxy_record["support"]
                )
            ):
                raise RuntimeError(f"D2/reference proxy geometry differs: {key}")
            reference_record.update(
                {name: proxy_record[name] for name in PROXY_CONDITIONS}
            )
            reference_records.append((metadata[core_id], reference_record))
        report["conditional_voxel_coverage"].update(
            proxy_conditionals(reference_records, seed=common_seed + 100)
        )
        report["_d2_posterior_mean_spectral"] = (
            posterior_mean_spectral_diagnostics(reference_records)
        )
        references[key] = report
    figure_base = output_dir / "p12f3_d2_ph006_diagnostics"
    _plot_bundle(common, shear, visual, higher, references, figure_base)
    example_base = output_dir / "p12f3_d2_field_and_training"
    _plot_example_and_training(archive, example, example_base)
    spectral_rows = {
        "D2": candidate_spectral,
        "G1": references["g1"]["_d2_posterior_mean_spectral"],
        "F3-L2b": references["f3l2b"]["_d2_posterior_mean_spectral"],
        "F3-L2d": references["f3l2d_nfe100"]["_d2_posterior_mean_spectral"],
    }
    spectral_transfer_path = output_dir / "spectral_transfer_plot_data.json"
    atomic_json(
        spectral_transfer_path,
        {
            "schema_version": "p12f3-d2-posterior-mean-spectrum-v1",
            "role": "diagnostic_only_not_a_posterior_promotion_score",
            "methods": spectral_rows,
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        },
    )
    spectral_derived_base = output_dir / "p12f3_d2_transfer_and_derived_coverage"
    _plot_transfer_and_derived_coverage(
        spectral_rows, derived, spectral_derived_base
    )
    evidence = {
        "schema_version": EVALUATION_SCHEMA,
        "created_utc": utc_now(),
        "method": archive["method"],
        "network_evaluations": int(archive["network_evaluations"]),
        "draw_batch": int(archive["draw_batch"]),
        "sampler": archive.get("sampler", "deterministic"),
        "sampler_eta": float(archive.get("sampler_eta", 0.0)),
        "seed": int(archive["seed"]),
        "seed_role": archive["seed_role"],
        "selected_arm": archive["selected_arm"],
        "selected_presentations": int(archive["selected_presentations"]),
        "selected_weights": archive["selected_weights"],
        "checkpoint_sha256": archive["checkpoint_sha256"],
        "trained_marker_sha256": archive["trained_marker_sha256"],
        "export_frozen_digest": archive["export_frozen_digest"],
        "export_run_manifest_sha256": archive["export_run_manifest_sha256"],
        "second_seed_license_sha256": archive.get("second_seed_license_sha256"),
        "stochastic_control_license_sha256": archive.get(
            "stochastic_control_license_sha256"
        ),
        "panel_sha256": archive["panel_sha256"],
        "common_evaluator_seed": common_seed,
        "higher_order_seed": higher_seed,
        "matched_reference_marker": str(reference_marker_path.resolve()),
        "matched_reference_marker_sha256": sha256(reference_marker_path),
        "frozen": evaluation_frozen,
        "frozen_digest": evaluation_digest,
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256(args.archive),
        "common_report": str(common_path.resolve()),
        "common_report_sha256": sha256(common_path),
        "shear_report": str(shear_path.resolve()),
        "shear_report_sha256": sha256(shear_path),
        "visual_plot_data": str(visual_path.resolve()),
        "visual_plot_data_sha256": sha256(visual_path),
        "spectral_transfer_plot_data": str(spectral_transfer_path.resolve()),
        "spectral_transfer_plot_data_sha256": sha256(spectral_transfer_path),
        "higher_order_report": str(higher_path.resolve()),
        "higher_order_report_sha256": sha256(higher_path),
        "figures": {
            "png": str(figure_base.with_suffix(".png").resolve()),
            "png_sha256": sha256(figure_base.with_suffix(".png")),
            "pdf": str(figure_base.with_suffix(".pdf").resolve()),
            "pdf_sha256": sha256(figure_base.with_suffix(".pdf")),
            "field_training_png": str(example_base.with_suffix(".png").resolve()),
            "field_training_png_sha256": sha256(example_base.with_suffix(".png")),
            "field_training_pdf": str(example_base.with_suffix(".pdf").resolve()),
            "field_training_pdf_sha256": sha256(example_base.with_suffix(".pdf")),
            "spectral_derived_png": str(
                spectral_derived_base.with_suffix(".png").resolve()
            ),
            "spectral_derived_png_sha256": sha256(
                spectral_derived_base.with_suffix(".png")
            ),
            "spectral_derived_pdf": str(
                spectral_derived_base.with_suffix(".pdf").resolve()
            ),
            "spectral_derived_pdf_sha256": sha256(
                spectral_derived_base.with_suffix(".pdf")
            ),
        },
        "compact_metrics": {
            "ordered_eigenvalue_tarp": _maximum_abs_tarp(common, "ordered_eigenvalues"),
            "eigengap_tarp": _maximum_abs_tarp(common, "eigengaps"),
            "five_shear_tarp": float(
                shear["joint_tarp_blocked"]["full_max_abs_ecp_minus_alpha"]
            ),
            "five_shear_marginal_coverage_error": float(
                shear["maximum_marginal_coverage_error"]
            ),
            "global_coverage_error": float(max(common["global_coverage_error"].values())),
            "deployable_conditional_coverage_error": float(
                common["maximum_deployable_conditional_coverage_error"]
            ),
            "voxel_deployable_conditional_coverage_error": float(
                common["maximum_voxel_deployable_conditional_coverage_error"]
            ),
            "derived_deployable_conditional_coverage_error": float(
                common["maximum_derived_deployable_conditional_coverage_error"]
            ),
            "posterior_mean_spectral_diagnostic": candidate_spectral,
            "low_band_power_ratios": list(
                map(float, visual["posterior_to_truth_power"][:2])
            ),
            "proper_scores": common["proper_scores"],
            "physics_closure": common["physics_closure"],
            "finite_non_degenerate": bool(common["finite_non_degenerate"]),
            "higher_order_available": bool(higher["available"]),
        },
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(evidence_path, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
