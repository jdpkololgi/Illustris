#!/usr/bin/env python3
"""Freeze D2 sampler/science gates, seed licence, and combined-seed decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_evaluate import EVALUATION_SCHEMA


DECISION_SCHEMA = "p12f3-d2-ph006-seed-decision-v1"
LICENSE_SCHEMA = "p12f3-d2-second-seed-license-v1"
COMBINED_SCHEMA = "p12f3-d2-combined-seed-decision-v1"
STOCHASTIC_LICENSE_SCHEMA = "p12f3-d2-stochastic-control-license-v1"
PROPER_NAMES = ("energy", "coarse_energy", "marginal_crps", "variogram_p0p5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--stage", choices=("seed", "combined"), required=True)
    parser.add_argument("--seed-role", choices=("primary", "replication"), default="primary")
    parser.add_argument("--nfe50-evaluation", type=Path)
    parser.add_argument("--nfe100-evaluation", type=Path)
    parser.add_argument("--matched-reference-marker", type=Path)
    parser.add_argument("--primary-decision", type=Path)
    parser.add_argument("--replication-decision", type=Path)
    return parser.parse_args()


def _read_evaluation(path: Path, nfe: int) -> tuple[dict, dict, dict, dict, dict]:
    marker = json.loads(path.read_text())
    if (
        marker.get("schema_version") != EVALUATION_SCHEMA
        or not marker.get("pass")
        or marker.get("ph001_opened")
        or int(marker.get("network_evaluations", -1)) != nfe
        or marker.get("sampler") != "deterministic"
        or float(marker.get("sampler_eta", -1)) != 0.0
    ):
        raise RuntimeError(f"unsafe D2 NFE{nfe} evaluation marker")
    reports = []
    for path_key, hash_key in (
        ("common_report", "common_report_sha256"),
        ("shear_report", "shear_report_sha256"),
        ("visual_plot_data", "visual_plot_data_sha256"),
        ("higher_order_report", "higher_order_report_sha256"),
    ):
        source = Path(marker[path_key])
        if sha256(source) != marker[hash_key]:
            raise RuntimeError(f"D2 NFE{nfe} evaluation component changed: {path_key}")
        reports.append(json.loads(source.read_text()))
    return (marker, *reports)


def paired_energy(candidate: dict, reference: dict, *, repeats: int, seed: int) -> dict:
    left = {int(row["core_id"]): float(row["energy"]) for row in candidate["per_core_proper_scores"]}
    right = {int(row["core_id"]): float(row["energy"]) for row in reference["per_core_proper_scores"]}
    if set(left) != set(right) or len(left) != 256:
        raise RuntimeError("D2 paired-energy panels differ")
    ids = np.asarray(sorted(left), dtype=np.int64)
    difference = np.asarray([left[value] - right[value] for value in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = difference[
        rng.integers(0, len(difference), size=(int(repeats), len(difference)))
    ].mean(axis=1)
    interval = np.quantile(samples, (0.025, 0.5, 0.975))
    reference_mean = float(np.mean(list(right.values())))
    relative_improvement = float(-difference.mean() / max(abs(reference_mean), 1.0e-12))
    return {
        "estimand": "candidate_minus_reference_energy",
        "mean": float(difference.mean()),
        "reference_mean": reference_mean,
        "relative_improvement": relative_improvement,
        "q025_q50_q975": interval.tolist(),
        "cores": int(len(ids)),
        "bootstrap_repeats": int(repeats),
        "bootstrap_unit": "ph006_authoritative_core",
        "pass": bool(interval[-1] < 0.0),
    }


def paired_energy_materiality(row: dict, minimum_relative_improvement: float) -> dict:
    """Apply the preregistered P12-F primary-score materiality rule."""
    ci_excludes_zero = bool(float(row["q025_q50_q975"][2]) < 0.0)
    relative = float(row["relative_improvement"])
    return {
        "relative_improvement": relative,
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "core_bootstrap_95pct_interval_excludes_zero": ci_excludes_zero,
        "pass": bool(relative >= minimum_relative_improvement and ci_excludes_zero),
    }


def score_changes(candidate: dict, reference: dict) -> dict:
    return {
        name: float(candidate["proper_scores"][name] / reference["proper_scores"][name] - 1.0)
        for name in PROPER_NAMES
    }


def validate_nfe_candidate_identity(
    nfe50: tuple,
    nfe100: tuple,
    *,
    expected_seed: int,
    seed_role: str,
    common_evaluator_seed: int,
    matched_reference_sha256: str,
) -> None:
    """Reject mixed arm/seed/checkpoint/panel/evaluator sampler ladders."""
    identity_keys = (
        "seed",
        "seed_role",
        "selected_arm",
        "selected_presentations",
        "selected_weights",
        "checkpoint_sha256",
        "trained_marker_sha256",
        "second_seed_license_sha256",
        "panel_sha256",
        "draw_batch",
        "sampler",
        "sampler_eta",
    )
    if (
        any(nfe50[0].get(key) != nfe100[0].get(key) for key in identity_keys)
        or int(nfe100[0].get("seed", -1)) != expected_seed
        or nfe100[0].get("seed_role") != seed_role
        or nfe100[0].get("sampler") != "deterministic"
        or float(nfe100[0].get("sampler_eta", -1)) != 0.0
        or any(
            int(bundle[0].get("common_evaluator_seed", -1))
            != common_evaluator_seed
            or bundle[0].get("matched_reference_marker_sha256")
            != matched_reference_sha256
            for bundle in (nfe50, nfe100)
        )
    ):
        raise RuntimeError("D2 NFE50/NFE100 candidate identity mismatch")


def _validate_reference(
    key: str, report: dict, frozen: dict, *, common_evaluator_seed: int
) -> None:
    core_ids = [int(row["core_id"]) for row in report.get("per_core_proper_scores", ())]
    if (
        report.get("phase") != "ph006"
        or report.get("ph001_opened")
        or int(report.get("cores", -1)) != 256
        or core_ids != frozen["core_ids"]
        or report.get("method") != frozen["method"]
        or report.get("archive_sha256") != frozen["archive_sha256"]
        or int(report.get("common_evaluator_seed", -1)) != common_evaluator_seed
    ):
        raise RuntimeError(f"D2 {key} reference report/archive binding changed")


def sampler_convergence(nfe50: tuple, nfe100: tuple, config: dict) -> dict:
    _, common50, shear50, visual50, _ = nfe50
    _, common100, shear100, visual100, _ = nfe100

    def tarp_curve_change(left: dict, right: dict) -> float:
        """Supremum distance between two matched TARP ECP curves."""
        alpha_left = np.asarray(left["alpha"], dtype=np.float64)
        alpha_right = np.asarray(right["alpha"], dtype=np.float64)
        ecp_left = np.asarray(left["expected_coverage_probability"], dtype=np.float64)
        ecp_right = np.asarray(right["expected_coverage_probability"], dtype=np.float64)
        if (
            not np.array_equal(alpha_left, alpha_right)
            or ecp_left.shape != ecp_right.shape
            or ecp_left.shape != alpha_left.shape
        ):
            raise RuntimeError("D2 NFE TARP grids or curve shapes differ")
        return float(np.max(np.abs(ecp_right - ecp_left)))

    tarp_changes = {
        name: tarp_curve_change(common50["tarp"][name], common100["tarp"][name])
        for name in ("ordered_eigenvalues", "eigengaps")
    }
    tarp_changes["five_shear"] = tarp_curve_change(
        shear50["joint_tarp_blocked"], shear100["joint_tarp_blocked"]
    )

    # Compare every physical coverage cell, not merely two scalar maxima.  A
    # change in which component dominates must not disappear from this gate.
    coverage_cells = {}
    coverage_groups = (
        ("voxel", ("voxel",)),
        ("lambda1", ("derived_ordered_eigenvalues", "lambda1")),
        ("lambda2", ("derived_ordered_eigenvalues", "lambda2")),
        ("lambda3", ("derived_ordered_eigenvalues", "lambda3")),
        ("gap12", ("derived_eigengaps", "gap12")),
        ("gap23", ("derived_eigengaps", "gap23")),
    )
    for label, path in coverage_groups:
        row50 = common50
        row100 = common100
        for key in path:
            row50 = row50[key]
            row100 = row100[key]
        for level in ("0.68", "0.90"):
            value50 = float(row50["coverage"][level]["empirical"])
            value100 = float(row100["coverage"][level]["empirical"])
            coverage_cells[f"{label}:{level}"] = abs(value100 - value50)
    global_change = max(coverage_cells.values())
    power_change = np.abs(
        np.asarray(visual100["posterior_to_truth_power"][:2], dtype=np.float64)
        - np.asarray(visual50["posterior_to_truth_power"][:2], dtype=np.float64)
    )
    proper_change = {
        name: abs(
            float(common100["proper_scores"][name] / common50["proper_scores"][name] - 1.0)
        )
        for name in PROPER_NAMES
    }
    sampler = config["sampler"]
    checks = {
        "joint_tarp": max(tarp_changes.values())
        <= float(sampler["convergence_tarp_change_maximum"]),
        "global_coverage": global_change
        <= float(sampler["convergence_global_coverage_change_maximum"]),
        "low_band_power": float(np.max(power_change))
        <= float(sampler["convergence_low_band_power_change_maximum"]),
        "proper_scores": max(proper_change.values())
        <= float(sampler["convergence_proper_score_relative_change_maximum"]),
    }
    return {
        "nfe_pair": [50, 100],
        "joint_tarp_pointwise_supremum_changes": tarp_changes,
        "global_coverage_cell_absolute_changes": coverage_cells,
        "global_coverage_pointwise_maximum_change": global_change,
        "low_band_power_absolute_changes": power_change.tolist(),
        "proper_score_relative_changes": proper_change,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def science_gates(
    evaluation: tuple,
    references: dict[str, dict],
    config: dict,
    *,
    repeats: int,
    seed: int,
) -> dict:
    _, common, shear, visual, higher = evaluation
    limits = config["ph006_gate"]
    ratios = np.asarray(visual["posterior_to_truth_power"][:2], dtype=np.float64)
    tarp = {
        name: float(common["tarp"][name]["full_max_abs_ecp_minus_alpha"])
        for name in ("ordered_eigenvalues", "eigengaps")
    }
    shear_tarp = float(
        shear["joint_tarp_blocked"]["full_max_abs_ecp_minus_alpha"]
    )
    shear_coverage = float(shear["maximum_marginal_coverage_error"])
    global_coverage = float(max(map(float, common["global_coverage_error"].values())))
    conditional_coverage = float(common["maximum_deployable_conditional_coverage_error"])
    voxel_conditional_coverage = float(
        common["maximum_voxel_deployable_conditional_coverage_error"]
    )
    derived_conditional_coverage = float(
        common["maximum_derived_deployable_conditional_coverage_error"]
    )
    proper_changes = {
        key: score_changes(common, reference)
        for key, reference in references.items()
    }
    paired = {
        key: paired_energy(
            common, references[key], repeats=repeats, seed=seed + index
        )
        for index, key in enumerate(("g1", "f3l2b"))
    }
    f3l2b_materiality = paired_energy_materiality(
        paired["f3l2b"],
        float(
            limits[
                "primary_paired_energy_relative_improvement_over_f3l2b_minimum"
            ]
        ),
    )
    physics = common["physics_closure"]
    checks = {
        "finite_non_degenerate": bool(common["finite_non_degenerate"]),
        "physics_trace_and_order": bool(
            physics.get("all_finite")
            and physics.get("all_ordered")
            and not physics.get("additional_gaussian_smoothing")
            and float(physics.get("maximum_trace_max_abs", np.inf)) <= 1.0e-5
        ),
        "low_band_power": bool(
            np.all(
                np.abs(ratios - 1.0)
                <= float(limits["low_band_power_ratio_absolute_tolerance"])
            )
        ),
        "ordered_eigenvalue_tarp": tarp["ordered_eigenvalues"]
        <= float(limits["joint_tarp_maximum"]),
        "eigengap_tarp": tarp["eigengaps"]
        <= float(limits["joint_tarp_maximum"]),
        "five_shear_tarp": shear_tarp <= float(limits["five_shear_tarp_maximum"]),
        "five_shear_marginal_coverage": shear_coverage
        <= float(limits["five_shear_marginal_coverage_error_maximum"]),
        "global_coverage": global_coverage
        <= float(limits["global_coverage_error_maximum"]),
        "deployable_conditional_coverage": conditional_coverage
        <= float(limits["deployable_proxy_conditional_coverage_error_maximum"]),
        "proper_nonworsening_g1": max(proper_changes["g1"].values())
        <= float(limits["proper_score_worsening_maximum"]),
        "paired_energy_improves_g1": bool(paired["g1"]["pass"]),
        "paired_energy_improves_f3l2b": bool(paired["f3l2b"]["pass"]),
        "f3l2b_primary_energy_materiality": bool(f3l2b_materiality["pass"]),
        "f3l2d_nfe100_noninferior": max(
            proper_changes["f3l2d_nfe100"].values()
        )
        <= float(limits["f3l2d_nfe100_proper_score_worsening_maximum"]),
    }
    return {
        "values": {
            "low_band_power_ratios": ratios.tolist(),
            "joint_tarp": tarp,
            "five_shear_tarp": shear_tarp,
            "five_shear_marginal_coverage_error": shear_coverage,
            "global_coverage_error": global_coverage,
            "deployable_conditional_coverage_error": conditional_coverage,
            "voxel_deployable_conditional_coverage_error": voxel_conditional_coverage,
            "derived_deployable_conditional_coverage_error": derived_conditional_coverage,
            "proper_score_relative_changes": proper_changes,
            "paired_energy": paired,
            "f3l2b_primary_energy_materiality": f3l2b_materiality,
            "physics_closure": physics,
            "higher_order": higher,
        },
        "limits": limits,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _plot_decision(
    gates: dict, convergence: dict, output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    paired = gates["values"]["paired_energy"]
    names = ("G1", "F3-L2b")
    rows = (paired["g1"], paired["f3l2b"])
    means = np.asarray([row["mean"] for row in rows])
    low = means - np.asarray([row["q025_q50_q975"][0] for row in rows])
    high = np.asarray([row["q025_q50_q975"][2] for row in rows]) - means
    axes[0].errorbar(
        means,
        np.arange(2),
        xerr=np.vstack((low, high)),
        fmt="o",
        color="#2878b5",
        capsize=4,
    )
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_yticks(np.arange(2), names)
    axes[0].set(xlabel="D2 minus reference energy (95% core bootstrap)", title="A. Paired proper-score evidence")
    axes[0].grid(alpha=.2)
    convergence_values = {
        "TARP": max(
            convergence["joint_tarp_pointwise_supremum_changes"].values()
        ),
        "coverage": convergence["global_coverage_pointwise_maximum_change"],
        "power": max(convergence["low_band_power_absolute_changes"]),
        "proper score": max(convergence["proper_score_relative_changes"].values()),
    }
    limits = (0.01, 0.01, 0.05, 0.01)
    normalized = [convergence_values[name] / limit for name, limit in zip(convergence_values, limits)]
    axes[1].bar(np.arange(4), normalized, color=["#7cb518" if value <= 1 else "#d1495b" for value in normalized])
    axes[1].axhline(1, color="black", linestyle="--", label="frozen tolerance")
    axes[1].set_xticks(np.arange(4), list(convergence_values), rotation=20, ha="right")
    axes[1].set(ylabel="NFE50-to-100 change / tolerance", title="B. Sampler convergence")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _plot_combined(primary: dict, replication: dict, output: Path) -> None:
    """Visualize the two independently trained seeds against frozen gates."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels = ("eigen TARP", "gap TARP", "5-shear", "global cov.", "conditional cov.")
    for marker, color, seed_label in (
        (primary, "#2878b5", f"seed {primary['seed']}"),
        (replication, "#d1495b", f"seed {replication['seed']}"),
    ):
        values = marker["science_gates_nfe100"]["values"]
        normalized = (
            float(values["joint_tarp"]["ordered_eigenvalues"]) / .05,
            float(values["joint_tarp"]["eigengaps"]) / .05,
            float(values["five_shear_tarp"]) / .05,
            float(values["global_coverage_error"]) / .05,
            float(values["deployable_conditional_coverage_error"]) / .10,
        )
        axes[0].plot(np.arange(len(labels)), normalized, marker="o", color=color, label=seed_label)
        convergence = marker["sampler_convergence_nfe50_nfe100"]
        convergence_values = (
            max(convergence["joint_tarp_pointwise_supremum_changes"].values()) / .01,
            float(convergence["global_coverage_pointwise_maximum_change"]) / .01,
            max(convergence["low_band_power_absolute_changes"]) / .05,
            max(convergence["proper_score_relative_changes"].values()) / .01,
        )
        axes[1].plot(
            np.arange(4),
            convergence_values,
            marker="o",
            color=color,
            label=seed_label,
        )
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", label="registered limit")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axes[0].set(ylabel="value / gate", title="A. Independent-seed science calibration")
    axes[1].set_xticks(
        np.arange(4), ("TARP curve", "coverage cells", "low-band power", "proper score"),
        rotation=25, ha="right"
    )
    axes[1].set(ylabel="NFE50-to-100 change / tolerance", title="B. Independent-seed sampler convergence")
    figure.suptitle("P12-F3-D2 two-seed frozen decision", fontsize=14)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _seed_stage(args, contract, config) -> None:
    if args.nfe50_evaluation is None or args.nfe100_evaluation is None:
        raise ValueError("seed decision requires both NFE50 and NFE100 evaluations")
    nfe50 = _read_evaluation(args.nfe50_evaluation, 50)
    nfe100 = _read_evaluation(args.nfe100_evaluation, 100)
    reference_marker_path = (
        args.matched_reference_marker
        or args.output_root / "D2_MATCHED_REFERENCE_REPORTS.json"
    )
    reference_marker = json.loads(reference_marker_path.read_text())
    common_seed = int(config["evaluation"]["common_evaluator_seed"])
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
        raise RuntimeError("unsafe D2 matched-reference freeze")
    expected_seed = int(
        config["funnel"][
            "seed" if args.seed_role == "primary" else "replication_seed"
        ]
    )
    validate_nfe_candidate_identity(
        nfe50,
        nfe100,
        expected_seed=expected_seed,
        seed_role=args.seed_role,
        common_evaluator_seed=common_seed,
        matched_reference_sha256=sha256(reference_marker_path),
    )
    for bundle in (nfe50, nfe100):
        archive_path = Path(bundle[0]["archive"])
        archive = json.loads(archive_path.read_text())
        run_path = Path(archive.get("export_run_manifest", ""))
        run = json.loads(run_path.read_text())
        if (
            sha256(archive_path) != bundle[0]["archive_sha256"]
            or archive.get("trained_marker_sha256")
            != bundle[0]["trained_marker_sha256"]
            or int(archive.get("seed", -1)) != expected_seed
            or archive.get("seed_role") != args.seed_role
            or archive.get("selected_arm") != bundle[0]["selected_arm"]
            or int(archive.get("selected_presentations", -1))
            != int(bundle[0]["selected_presentations"])
            or archive.get("selected_weights") != bundle[0]["selected_weights"]
            or int(archive.get("draw_batch", -1))
            != int(config["sampler"]["draw_batch"])
            or archive.get("export_run_manifest_sha256") != sha256(run_path)
            or run.get("frozen_digest") != archive.get("export_frozen_digest")
            or run.get("ph001_opened")
        ):
            raise RuntimeError("D2 evaluation-to-archive identity binding changed")
        if args.seed_role == "replication":
            license_path = args.output_root / "D2_SECOND_SEED_LICENSE.json"
            if archive.get("second_seed_license_sha256") != sha256(license_path):
                raise RuntimeError("D2 replication archive lost its seed licence")
    references = {}
    for key in config["evaluation"]["matched_reference_methods"]:
        row = reference_marker.get("reports", {}).get(key, {})
        report_path = Path(row.get("path", ""))
        frozen_reference = contract["frozen"]["reference_contract"][key]
        if (
            row.get("sha256") != sha256(report_path)
            or row.get("archive_sha256") != frozen_reference["archive_sha256"]
        ):
            raise RuntimeError(f"D2 matched reference changed: {key}")
        report = json.loads(report_path.read_text())
        _validate_reference(
            key,
            report,
            frozen_reference,
            common_evaluator_seed=common_seed,
        )
        references[key] = report
    bootstrap_repeats = int(config["evaluation"]["paired_bootstrap_repeats"])
    bootstrap_seed = int(config["evaluation"]["paired_bootstrap_seed"])
    gates = science_gates(
        nfe100,
        references,
        config,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    convergence = sampler_convergence(nfe50, nfe100, config)
    seed = expected_seed
    seed_pass = bool(gates["pass"] and convergence["pass"])
    frozen_inputs = {
        "contract_sha256": sha256(args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"),
        "contract_digest": contract["frozen_digest"],
        "nfe50_evaluation": str(args.nfe50_evaluation.resolve()),
        "nfe50_evaluation_sha256": sha256(args.nfe50_evaluation),
        "nfe100_evaluation": str(args.nfe100_evaluation.resolve()),
        "nfe100_evaluation_sha256": sha256(args.nfe100_evaluation),
        "matched_reference_marker": str(reference_marker_path.resolve()),
        "matched_reference_marker_sha256": sha256(reference_marker_path),
        "common_evaluator_seed": common_seed,
        "paired_bootstrap_repeats": bootstrap_repeats,
        "paired_bootstrap_seed": bootstrap_seed,
        "seed": seed,
        "seed_role": args.seed_role,
    }
    output = args.output_root / f"D2_SEED{seed}_PH006_DECISION.json"
    marker = {
        "schema_version": DECISION_SCHEMA,
        "created_utc": utc_now(),
        "pass": True,
        "seed_pass": seed_pass,
        "seed": seed,
        "seed_role": args.seed_role,
        "selected_arm": nfe100[0]["selected_arm"],
        "selected_presentations": int(nfe100[0]["selected_presentations"]),
        "selected_weights": str(nfe100[0]["selected_weights"]),
        "science_gates_nfe100": gates,
        "sampler_convergence_nfe50_nfe100": convergence,
        "frozen_inputs": frozen_inputs,
        "frozen_digest": digest(frozen_inputs),
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
    }
    figure = args.output_root / f"d2_seed{seed}_ph006_decision"
    if output.exists():
        existing = json.loads(output.read_text())
        if (
            existing.get("schema_version") != DECISION_SCHEMA
            or existing.get("frozen_digest") != marker["frozen_digest"]
            or existing.get("seed_pass") != marker["seed_pass"]
            or existing.get("science_gates_nfe100") != marker["science_gates_nfe100"]
            or existing.get("sampler_convergence_nfe50_nfe100")
            != marker["sampler_convergence_nfe50_nfe100"]
            or existing.get("ph001_opened")
            or sha256(Path(existing["figures"]["png"]))
            != existing["figures"]["png_sha256"]
            or sha256(Path(existing["figures"]["pdf"]))
            != existing["figures"]["pdf_sha256"]
        ):
            raise RuntimeError("existing D2 seed decision changed")
        marker = existing
    else:
        _plot_decision(gates, convergence, figure)
        marker["figures"] = {
            "png": str(figure.with_suffix(".png").resolve()),
            "png_sha256": sha256(figure.with_suffix(".png")),
            "pdf": str(figure.with_suffix(".pdf").resolve()),
            "pdf_sha256": sha256(figure.with_suffix(".pdf")),
        }
        atomic_json(output, marker)
    if args.seed_role == "primary":
        final_selection = args.output_root / "D2_FINAL_SELECTION.json"
        confirmation = args.output_root / "D2_INTERNAL_CONFIRMATION.json"
        license_path = args.output_root / "D2_SECOND_SEED_LICENSE.json"
        license_frozen = {
            "contract_digest": contract["frozen_digest"],
            "contract_sha256": frozen_inputs["contract_sha256"],
            "final_selection": str(final_selection.resolve()),
            "final_selection_sha256": sha256(final_selection),
            "internal_confirmation": str(confirmation.resolve()),
            "internal_confirmation_sha256": sha256(confirmation),
            "seed42_decision": str(output.resolve()),
            "seed42_decision_sha256": sha256(output),
            "selected_arm": marker["selected_arm"],
            "selected_presentations": marker["selected_presentations"],
            "selected_weights": marker["selected_weights"],
            "ph001_opened": False,
        }
        license_marker = {
            "schema_version": LICENSE_SCHEMA,
            "created_utc": utc_now(),
            "pass": True,
            "licensed": seed_pass,
            **license_frozen,
            "frozen_digest": digest(license_frozen),
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        }
        if license_path.exists():
            existing_license = json.loads(license_path.read_text())
            if any(
                existing_license.get(key) != license_marker.get(key)
                for key in (
                    "schema_version",
                    "pass",
                    "licensed",
                    "contract_digest",
                    "contract_sha256",
                    "final_selection",
                    "final_selection_sha256",
                    "internal_confirmation",
                    "internal_confirmation_sha256",
                    "seed42_decision",
                    "seed42_decision_sha256",
                    "selected_arm",
                    "selected_presentations",
                    "selected_weights",
                    "frozen_digest",
                    "ph001_opened",
                )
            ):
                raise RuntimeError("existing D2 second-seed licence changed")
        else:
            atomic_json(license_path, license_marker)
        stochastic_path = args.output_root / "D2_STOCHASTIC_CONTROL_LICENSE.json"
        stochastic_frozen = {
            "role": "diagnostic_only_never_promotable",
            "reason": "run eta=1 NFE100 only when deterministic NFE50-to-100 convergence fails",
            "seed_decision": str(output.resolve()),
            "seed_decision_sha256": sha256(output),
            "contract_digest": contract["frozen_digest"],
            "seed": seed,
            "selected_arm": marker["selected_arm"],
            "selected_presentations": marker["selected_presentations"],
            "selected_weights": marker["selected_weights"],
            "ph001_opened": False,
        }
        stochastic = {
            "schema_version": STOCHASTIC_LICENSE_SCHEMA,
            "created_utc": utc_now(),
            "pass": True,
            "licensed": not convergence["pass"],
            **stochastic_frozen,
            "frozen_digest": digest(stochastic_frozen),
        }
        if stochastic_path.exists():
            existing_stochastic = json.loads(stochastic_path.read_text())
            if any(
                existing_stochastic.get(key) != stochastic.get(key)
                for key in (
                    "schema_version",
                    "pass",
                    "licensed",
                    "role",
                    "reason",
                    "seed_decision",
                    "seed_decision_sha256",
                    "contract_digest",
                    "seed",
                    "selected_arm",
                    "selected_presentations",
                    "selected_weights",
                    "frozen_digest",
                    "ph001_opened",
                )
            ):
                raise RuntimeError("existing D2 stochastic-control licence changed")
        else:
            atomic_json(stochastic_path, stochastic)
    print(json.dumps(marker, indent=2, sort_keys=True))


def _combined_stage(args, contract, config) -> None:
    if args.primary_decision is None or args.replication_decision is None:
        raise ValueError("combined decision requires primary and replication decisions")
    primary = json.loads(args.primary_decision.read_text())
    replication = json.loads(args.replication_decision.read_text())
    if any(
        row.get("schema_version") != DECISION_SCHEMA
        or not row.get("pass")
        or row.get("ph001_opened")
        or row.get("frozen_inputs", {}).get("contract_digest") != contract["frozen_digest"]
        or row.get("frozen_digest") != digest(row.get("frozen_inputs", {}))
        for row in (primary, replication)
    ):
        raise RuntimeError("unsafe D2 combined-seed inputs")
    if (
        primary.get("seed_role") != "primary"
        or int(primary.get("seed", -1)) != int(config["funnel"]["seed"])
        or replication.get("seed_role") != "replication"
        or int(replication.get("seed", -1))
        != int(config["funnel"]["replication_seed"])
        or any(
            primary.get(name) != replication.get(name)
            for name in ("selected_arm", "selected_presentations", "selected_weights")
        )
    ):
        raise RuntimeError("D2 replication did not preserve the seed-42 freeze")
    license_path = args.output_root / "D2_SECOND_SEED_LICENSE.json"
    license_marker = json.loads(license_path.read_text())
    if (
        license_marker.get("schema_version") != LICENSE_SCHEMA
        or not license_marker.get("pass")
        or not license_marker.get("licensed")
        or license_marker.get("seed42_decision") != str(args.primary_decision.resolve())
        or license_marker.get("seed42_decision_sha256") != sha256(args.primary_decision)
        or license_marker.get("contract_digest") != contract["frozen_digest"]
        or license_marker.get("frozen_digest")
        != digest(
            {
                key: license_marker[key]
                for key in (
                    "contract_digest",
                    "contract_sha256",
                    "final_selection",
                    "final_selection_sha256",
                    "internal_confirmation",
                    "internal_confirmation_sha256",
                    "seed42_decision",
                    "seed42_decision_sha256",
                    "selected_arm",
                    "selected_presentations",
                    "selected_weights",
                    "ph001_opened",
                )
            }
        )
        or any(
            license_marker.get(name) != primary.get(name)
            for name in ("selected_arm", "selected_presentations", "selected_weights")
        )
        or license_marker.get("ph001_opened")
    ):
        raise RuntimeError("D2 combined decision lacks its exact second-seed licence")
    replication_eval = json.loads(
        Path(replication["frozen_inputs"]["nfe100_evaluation"]).read_text()
    )
    replication_archive = json.loads(Path(replication_eval["archive"]).read_text())
    if (
        sha256(Path(replication["frozen_inputs"]["nfe100_evaluation"]))
        != replication["frozen_inputs"]["nfe100_evaluation_sha256"]
        or replication_archive.get("second_seed_license_sha256")
        != sha256(license_path)
        or replication_archive.get("seed_role") != "replication"
        or int(replication_archive.get("seed", -1))
        != int(config["funnel"]["replication_seed"])
    ):
        raise RuntimeError("D2 replication decision lost its licensed archive linkage")
    frozen = {
        "contract_digest": contract["frozen_digest"],
        "primary": str(args.primary_decision.resolve()),
        "primary_sha256": sha256(args.primary_decision),
        "replication": str(args.replication_decision.resolve()),
        "replication_sha256": sha256(args.replication_decision),
        "second_seed_license": str(license_path.resolve()),
        "second_seed_license_sha256": sha256(license_path),
        "ph001_opened": False,
    }
    marker = {
        "schema_version": COMBINED_SCHEMA,
        "created_utc": utc_now(),
        "pass": True,
        "promoted": bool(primary["seed_pass"] and replication["seed_pass"]),
        "selected_arm": primary["selected_arm"],
        "selected_presentations": primary["selected_presentations"],
        "selected_weights": primary["selected_weights"],
        "seed_results": {
            str(primary["seed"]): primary["seed_pass"],
            str(replication["seed"]): replication["seed_pass"],
        },
        "failure_action": None
        if primary["seed_pass"] and replication["seed_pass"]
        else "D2_not_promoted_no_further_successor_registered",
        "frozen_inputs": frozen,
        "frozen_digest": digest(frozen),
        "truth_files_read": ["ph006"],
        "ph001_opened": False,
    }
    output = args.output_root / "D2_COMBINED_SEED_DECISION.json"
    figure_base = args.output_root / "d2_combined_seed_decision"
    if output.exists():
        existing = json.loads(output.read_text())
        immutable = tuple(key for key in marker if key != "created_utc")
        if (
            any(existing.get(key) != marker.get(key) for key in immutable)
            or sha256(Path(existing["figures"]["png"]))
            != existing["figures"]["png_sha256"]
            or sha256(Path(existing["figures"]["pdf"]))
            != existing["figures"]["pdf_sha256"]
        ):
            raise RuntimeError("existing D2 combined-seed decision changed")
        marker = existing
    else:
        _plot_combined(primary, replication, figure_base)
        marker["figures"] = {
            "png": str(figure_base.with_suffix(".png").resolve()),
            "png_sha256": sha256(figure_base.with_suffix(".png")),
            "pdf": str(figure_base.with_suffix(".pdf").resolve()),
            "pdf_sha256": sha256(figure_base.with_suffix(".pdf")),
        }
        atomic_json(output, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    args.contract = contract_path
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    if args.stage == "seed":
        _seed_stage(args, contract, config)
    else:
        _combined_stage(args, contract, config)


if __name__ == "__main__":
    main()
