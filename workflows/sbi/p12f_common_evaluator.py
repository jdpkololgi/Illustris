#!/usr/bin/env python3
"""Evaluate standardized ph006 P12-F field samples under one frozen ladder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from workflows.abacus_tweb.p6_field_patch_utils import trilinear_sample
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_calibration_diagnostics import tarp_diagnostic
from workflows.sbi.p12f_challenger_common import (
    FieldSampleContract,
    core_joint_scores,
    haar_coarse,
    paired_core_bootstrap,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    conditional_reports,
    fixed_tidal_eigenvalues,
    physics_closure_report,
    quantile_labels,
    scalar_posterior_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--panel-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def efficient_crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> float:
    """Exact scalar ensemble CRPS without an O(draws^2 * rows) allocation."""
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if draws.ndim != 2 or target.shape != (draws.shape[1],):
        raise ValueError("CRPS expects [draws,rows] and [rows]")
    first = np.mean(np.abs(draws - target[None]), axis=0)
    ordered = np.sort(draws, axis=0)
    m = draws.shape[0]
    coefficient = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    half_pairwise = np.sum(coefficient[:, None] * ordered, axis=0) / (m * m)
    return float(np.mean(first - half_pairwise))


def fourier_low_modes(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    maximum_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_k = np.fft.rfftn(samples, axes=(-3, -2, -1), norm="ortho")
    truth_k = np.fft.rfftn(truth, axes=(-3, -2, -1), norm="ortho")
    nx, ny, nz = truth.shape
    kx = np.fft.fftfreq(nx)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kz = np.fft.rfftfreq(nz)[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    order = np.argsort(kmag.ravel())
    order = order[kmag.ravel()[order] > 0][:maximum_modes]
    draw = sample_k.reshape(samples.shape[0], -1)[:, order]
    target = truth_k.ravel()[order]
    return (
        np.concatenate((draw.real, draw.imag), axis=1).astype(np.float32),
        np.concatenate((target.real, target.imag), axis=0).astype(np.float32),
    )


def sample_eigenvalues_at_galaxies(
    eigen_samples: np.ndarray,
    eigen_truth: np.ndarray,
    frac_index_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(frac_index_local, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("galaxy coordinates must have shape [rows,3]")
    if len(coordinates) == 0:
        return (
            np.empty((eigen_samples.shape[0], 0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    draws = eigen_samples.shape[0]
    sample_channels = np.moveaxis(eigen_samples, -1, 1).reshape(
        draws * 3, *eigen_samples.shape[1:4]
    )
    sampled = trilinear_sample(sample_channels, coordinates)
    sampled = sampled.reshape(len(coordinates), draws, 3).transpose(1, 0, 2)
    truth_channels = np.moveaxis(eigen_truth, -1, 0)
    sampled_truth = trilinear_sample(truth_channels, coordinates)
    return sampled.astype(np.float32), sampled_truth.astype(np.float32)


def maximum_conditional_coverage_error(rows: dict) -> float:
    values = []
    for variable in rows.values():
        for report in variable.values():
            for level in ("0.68", "0.90"):
                coverage = report.get("coverage", {}).get(level)
                if coverage is not None:
                    values.append(float(coverage["absolute_error"]))
    return max(values, default=float("inf"))


def validate_archive_manifest(
    archive: dict,
    *,
    archive_path: Path,
    panel: dict,
    panel_path: Path,
    config: dict,
) -> list[dict]:
    if archive.get("schema_version") != "p12f-sample-archive-v1":
        raise RuntimeError("unsupported P12-F sample archive")
    if archive.get("phase") != "ph006" or archive.get("ph001_opened"):
        raise PermissionError("common evaluator accepts ph006 only")
    if archive.get("truth_files_read") not in (["ph006"], ["ph006 density/T-web"]):
        raise RuntimeError("sample archive truth provenance is not explicit")
    if int(archive.get("draws", -1)) != int(
        config["matched_contract"]["posterior_draws"]
    ):
        raise RuntimeError("sample archive draw count is not the matched contract")
    if archive.get("panel_sha256") != sha256(panel_path):
        raise RuntimeError("sample archive panel hash mismatch")
    selected = [int(value) for value in panel["selected_core_id"]]
    entries = archive.get("entries", [])
    found = [int(row["core_id"]) for row in entries]
    if found != selected or len(set(found)) != len(found):
        raise RuntimeError("sample archive does not exactly follow the frozen panel")
    for row in entries:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"sample archive core artifact changed: {path}")
        if "ph001" in str(path).lower():
            raise PermissionError("ph001 path appeared in ph006 sample archive")
    if archive.get("manifest_sha256") not in (None, sha256(archive_path)):
        raise RuntimeError("self-hash must be absent or current")
    return entries


def load_core_record(entry: dict, draws: int) -> dict:
    with np.load(entry["path"], allow_pickle=False) as values:
        required = {
            "delta_samples",
            "delta_truth",
            "support",
            "angular_response",
            "boundary_distance_mpc",
            "tracer_density",
            "core_bounds",
            "galaxy_frac_index_local",
        }
        if not required.issubset(values.files):
            raise RuntimeError(f"sample core is missing {sorted(required-set(values.files))}")
        record = {name: np.asarray(values[name]) for name in required}
    if record["delta_samples"].shape[0] != draws:
        raise RuntimeError("sample core draw count mismatch")
    return record


def evaluate_records(
    records: list[tuple[dict, dict]],
    *,
    method: str,
    seed: int,
    device: str,
) -> dict:
    voxel_samples: list[np.ndarray] = []
    voxel_truth: list[np.ndarray] = []
    shell_parts: list[np.ndarray] = []
    response_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    tracer_parts: list[np.ndarray] = []
    environment_parts: list[np.ndarray] = []
    core_parts: list[np.ndarray] = []
    fourier_samples: list[np.ndarray] = []
    fourier_truth: list[np.ndarray] = []
    wavelet_samples: list[np.ndarray] = []
    wavelet_truth: list[np.ndarray] = []
    lambda_samples: list[np.ndarray] = []
    lambda_truth: list[np.ndarray] = []
    lambda_core: list[np.ndarray] = []
    per_core: list[dict] = []
    physics_rows: list[dict] = []

    for ordinal, (metadata, record) in enumerate(records):
        samples = np.asarray(record["delta_samples"], dtype=np.float32)
        truth = np.asarray(record["delta_truth"], dtype=np.float32)
        support = np.asarray(record["support"], dtype=bool)
        FieldSampleContract(
            method=method,
            core_id=int(metadata["core_id"]),
            samples=samples,
            truth=truth,
            support=support,
        ).validate()
        core_bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        if core_bounds.shape != (2, 3):
            raise RuntimeError("core bounds must have shape [2,3]")
        core = tuple(
            slice(int(left), int(right))
            for left, right in zip(core_bounds[0], core_bounds[1], strict=True)
        )
        support_core = support[core]
        valid = np.flatnonzero(support_core.ravel())
        if len(valid) == 0:
            raise RuntimeError("frozen ph006 panel core has no exact support")
        if len(valid) > 2048:
            valid = valid[
                np.linspace(0, len(valid) - 1, 2048, dtype=np.int64)
            ]
        sample_core = samples[(slice(None),) + core]
        truth_core = truth[core]
        selected_samples = sample_core.reshape(samples.shape[0], -1)[:, valid]
        selected_truth = truth_core.ravel()[valid]
        voxel_samples.append(selected_samples)
        voxel_truth.append(selected_truth)
        core_parts.append(np.full(len(valid), int(metadata["core_id"]), dtype=np.int64))
        shell_parts.append(np.full(len(valid), int(metadata["shell"]), dtype=np.int8))
        response_parts.append(
            np.asarray(record["angular_response"])[core].ravel()[valid]
        )
        boundary_parts.append(
            np.asarray(record["boundary_distance_mpc"])[core].ravel()[valid]
        )
        tracer_parts.append(np.asarray(record["tracer_density"])[core].ravel()[valid])
        environment_parts.append(selected_truth)

        joint = core_joint_scores(
            sample_core,
            truth_core,
            support_core,
            seed=seed + int(metadata["core_id"]),
        )
        joint["core_id"] = int(metadata["core_id"])
        per_core.append(joint)
        fs, ft = fourier_low_modes(sample_core, truth_core, maximum_modes=256)
        fourier_samples.append(fs)
        fourier_truth.append(ft)
        coarse_sample = haar_coarse(sample_core, levels=2)
        coarse_truth = haar_coarse(truth_core, levels=2)
        coarse_count = min(256, coarse_truth.size)
        coarse_index = np.linspace(
            0, coarse_truth.size - 1, coarse_count, dtype=np.int64
        )
        wavelet_samples.append(
            coarse_sample.reshape(samples.shape[0], -1)[:, coarse_index]
        )
        wavelet_truth.append(coarse_truth.ravel()[coarse_index])

        sample_tensor = torch.from_numpy(samples).to(device)
        truth_tensor = torch.from_numpy(truth).to(device)
        # eigvalsh already matches the frozen catalogue convention lambda1<=lambda2<=lambda3.
        sample_eigen = (
            fixed_tidal_eigenvalues(sample_tensor).detach().cpu().numpy()
        )
        truth_eigen = (
            fixed_tidal_eigenvalues(truth_tensor).detach().cpu().numpy()
        )
        physics_rows.append(physics_closure_report(truth_tensor))
        sampled_lambda, sampled_truth_lambda = sample_eigenvalues_at_galaxies(
            sample_eigen,
            truth_eigen,
            record["galaxy_frac_index_local"],
        )
        if sampled_truth_lambda.size:
            lambda_samples.append(sampled_lambda)
            lambda_truth.append(sampled_truth_lambda)
            lambda_core.append(
                np.full(
                    len(sampled_truth_lambda),
                    int(metadata["core_id"]),
                    dtype=np.int64,
                )
            )
        del sample_tensor, truth_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"evaluate {ordinal+1}/{len(records)} core={metadata['core_id']}", flush=True)

    samples_all = np.concatenate(voxel_samples, axis=1)
    truth_all = np.concatenate(voxel_truth)
    shell_all = np.concatenate(shell_parts)
    response_all = np.concatenate(response_parts)
    boundary_all = np.concatenate(boundary_parts)
    tracer_all = np.concatenate(tracer_parts)
    environment_all = np.concatenate(environment_parts)
    core_all = np.concatenate(core_parts)
    fourier_samples_all = np.concatenate(fourier_samples, axis=1)
    fourier_truth_all = np.concatenate(fourier_truth)
    wavelet_samples_all = np.concatenate(wavelet_samples, axis=1)
    wavelet_truth_all = np.concatenate(wavelet_truth)

    voxel_report = scalar_posterior_report(samples_all, truth_all, seed=seed + 1)
    voxel_report["crps"] = efficient_crps_ensemble(samples_all, truth_all)
    fourier_report = scalar_posterior_report(
        fourier_samples_all, fourier_truth_all, seed=seed + 2
    )
    wavelet_report = scalar_posterior_report(
        wavelet_samples_all, wavelet_truth_all, seed=seed + 3
    )
    conditional = {
        "shell": conditional_reports(
            samples_all, truth_all, shell_all, seed=seed + 10
        ),
        "random_response": conditional_reports(
            samples_all,
            truth_all,
            quantile_labels(response_all),
            seed=seed + 11,
        ),
        "boundary_distance": conditional_reports(
            samples_all,
            truth_all,
            quantile_labels(boundary_all),
            seed=seed + 12,
        ),
        "tracer_density": conditional_reports(
            samples_all,
            truth_all,
            quantile_labels(tracer_all),
            seed=seed + 13,
        ),
        "true_environment": conditional_reports(
            samples_all,
            truth_all,
            quantile_labels(environment_all),
            seed=seed + 14,
        ),
    }

    if not lambda_samples:
        raise RuntimeError("no authoritative galaxies were available for derived physics")
    lambda_draw = np.concatenate(lambda_samples, axis=1)
    lambda_target = np.concatenate(lambda_truth, axis=0)
    lambda_group = np.concatenate(lambda_core)
    lambda_report = {
        f"lambda{index+1}": scalar_posterior_report(
            lambda_draw[..., index],
            lambda_target[..., index],
            seed=seed + 20 + index,
        )
        for index in range(3)
    }
    gap_draw = lambda_draw[..., 1:] - lambda_draw[..., :-1]
    gap_target = lambda_target[..., 1:] - lambda_target[..., :-1]
    gap_report = {
        f"gap{index+1}{index+2}": scalar_posterior_report(
            gap_draw[..., index],
            gap_target[..., index],
            seed=seed + 24 + index,
        )
        for index in range(2)
    }
    class_draw = np.sum(lambda_draw > 0.2, axis=-1)
    class_truth = np.sum(lambda_target > 0.2, axis=-1)
    probability = np.stack(
        [np.mean(class_draw == value, axis=0) for value in range(4)], axis=1
    )
    one_hot = np.eye(4, dtype=np.float64)[class_truth]
    class_report = {
        "threshold": 0.2,
        "probability_normalization_max_abs": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "brier_score": float(np.mean(np.sum(np.square(probability - one_hot), axis=1))),
        "truth_counts": np.bincount(class_truth, minlength=4).tolist(),
    }
    tarp_lambda = tarp_diagnostic(
        np.transpose(lambda_draw, (1, 0, 2)),
        lambda_target,
        lambda_group,
        seed=seed + 30,
        bootstrap_repeats=100,
        bootstrap_rows=min(20000, len(lambda_target)),
    )
    tarp_gap = tarp_diagnostic(
        np.transpose(gap_draw, (1, 0, 2)),
        gap_target,
        lambda_group,
        seed=seed + 31,
        bootstrap_repeats=100,
        bootstrap_rows=min(20000, len(gap_target)),
    )
    tarp_values = [
        float(row["full_max_abs_ecp_minus_alpha"])
        for row in (tarp_lambda, tarp_gap)
        if row.get("available")
    ]
    tarp_maximum = max(tarp_values, default=float("inf"))

    global_errors = {
        level: max(
            [float(voxel_report["coverage"][level]["absolute_error"])]
            + [
                float(row["coverage"][level]["absolute_error"])
                for row in (*lambda_report.values(), *gap_report.values())
            ]
        )
        for level in ("0.68", "0.90")
    }
    proper = {
        "primary_joint": float(np.mean([row["energy"] for row in per_core])),
        "energy": float(np.mean([row["energy"] for row in per_core])),
        "variogram_p0p5": float(
            np.mean([row["variogram_p0p5"] for row in per_core])
        ),
        "coarse_energy": float(
            np.mean([row["coarse_energy"] for row in per_core])
        ),
        "marginal_crps": float(voxel_report["crps"]),
    }
    return {
        "schema_version": "p12f-common-evaluation-report-v1",
        "created_utc": utc_now(),
        "method": method,
        "phase": "ph006",
        "cores": len(records),
        "draws": int(samples_all.shape[0]),
        "finite_non_degenerate": True,
        "voxel": voxel_report,
        "low_frequency_fourier": fourier_report,
        "wavelet_coarse": wavelet_report,
        "derived_ordered_eigenvalues": lambda_report,
        "derived_eigengaps": gap_report,
        "web_class": class_report,
        "tarp": {"ordered_eigenvalues": tarp_lambda, "eigengaps": tarp_gap},
        "tarp_maximum_deviation": tarp_maximum,
        "global_coverage_error": global_errors,
        "conditional_voxel_coverage": conditional,
        "maximum_conditional_coverage_error": maximum_conditional_coverage_error(
            conditional
        ),
        "proper_scores": proper,
        "per_core_proper_scores": per_core,
        "posterior_mean_diagnostics_only": {
            "voxel_r2": voxel_report["posterior_mean_r2_diagnostic"],
            "fourier_r2": fourier_report["posterior_mean_r2_diagnostic"],
            "wavelet_r2": wavelet_report["posterior_mean_r2_diagnostic"],
        },
        "physics_closure": {
            "maximum_trace_max_abs": max(row["trace_max_abs"] for row in physics_rows),
            "maximum_trace_rmse": max(row["trace_rmse"] for row in physics_rows),
            "all_finite": all(row["all_finite"] for row in physics_rows),
            "all_ordered": all(row["ordered"] for row in physics_rows),
            "catalogue_order": "lambda1<=lambda2<=lambda3",
            "target_already_smoothed_mpc_h": 7.0,
            "additional_gaussian_smoothing": False,
        },
        "resampling_unit": "authoritative patch core",
        "voxel_independent_resampling": False,
        "local_patch_posterior_only": True,
        "full_cap_coherence_established": False,
        "ph001_opened": False,
    }


def attach_g1_comparison(candidate: dict, reference: dict, *, seed: int) -> None:
    candidate_rows = {
        int(row["core_id"]): float(row["energy"])
        for row in candidate["per_core_proper_scores"]
    }
    reference_rows = {
        int(row["core_id"]): float(row["energy"])
        for row in reference["per_core_proper_scores"]
    }
    if candidate_rows.keys() != reference_rows.keys():
        raise RuntimeError("candidate and G1 core identities differ")
    ids = sorted(candidate_rows)
    candidate["joint_score_vs_g1_bootstrap"] = paired_core_bootstrap(
        np.asarray([candidate_rows[value] for value in ids]),
        np.asarray([reference_rows[value] for value in ids]),
        replicates=4000,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F common evaluation requires a compute GPU")
    config = json.loads(args.config.read_text())
    panel = json.loads(args.panel_marker.read_text())
    archive = json.loads(args.archive_manifest.read_text())
    if panel.get("schema_version") != "p12f-truth-free-selection-panel-v1":
        raise RuntimeError("unsupported P12-F panel marker")
    if (
        not panel.get("pass")
        or panel.get("selection_uses_truth")
        or panel.get("truth_files_read")
        or panel.get("ph001_opened")
    ):
        raise RuntimeError("P12-F panel is not truth-free and passing")
    entries = validate_archive_manifest(
        archive,
        archive_path=args.archive_manifest,
        panel=panel,
        panel_path=args.panel_marker,
        config=config,
    )
    metadata = {
        int(row["core_id"]): row for row in panel["selected_core_metadata"]
    }
    records = [
        (metadata[int(row["core_id"])], load_core_record(row, int(archive["draws"])))
        for row in entries
    ]
    report = evaluate_records(
        records,
        method=str(archive["method"]),
        seed=42,
        device=args.device,
    )
    report.update(
        {
            "config_sha256": sha256(args.config),
            "panel_sha256": sha256(args.panel_marker),
            "archive_manifest": str(args.archive_manifest.resolve()),
            "archive_manifest_sha256": sha256(args.archive_manifest),
            "checkpoint_sha256": archive["checkpoint_sha256"],
            "conditioning_contract_sha256": archive[
                "conditioning_contract_sha256"
            ],
            "target_scaler_sha256": archive["target_scaler_sha256"],
            "truth_files_read": ["ph006 density/T-web"],
        }
    )
    if args.reference_report is not None:
        reference = json.loads(args.reference_report.read_text())
        if reference.get("method") != "gaussian_correlated_g1":
            raise RuntimeError("reference report must be correlated Gaussian G1")
        attach_g1_comparison(report, reference, seed=42)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
