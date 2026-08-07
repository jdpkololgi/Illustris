#!/usr/bin/env python3
"""Evaluate every registered MT3 classical control on frozen Bright targets.

The field products are label-free.  This evaluator is the only stage that reads
the tidal labels, and it uses them exactly as the frozen P8 classical benchmark:
per-eigenvalue affine response is fitted on training folds and all reported
scores use the complete authoritative Bright validation fold.

Evaluated rows
--------------
* Bright-only TSC;
* Bright+Faint combined-count and bias-aware TSC;
* three density-matched Bright+Faint CIC thinnings;
* Bright plus the Faint angular-position null in CIC.

Together with the already frozen Bright-only CIC and completed MT3a full
Bright+Faint CIC rows, these close the MT3 information-control gate.  They do
not select or promote a neural model.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_refit_fullcap_selection import build_cap_lookup
from workflows.abacus_tweb.p8_build_multitracer_control_fields import RED_SHIFT_STRATA
from workflows.abacus_tweb.p8_classical_fullcap import CAP_NAME, RSMOOTH_MPC, _sample_tidal_eigenvalues
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    fold_roles,
    sha256,
)
from workflows.abacus_tweb.p8_multitracer_classical import (
    bias_aware_contrast,
    combined_count_contrast,
    evaluate_estimator,
    fit_relative_bias,
    fold_block,
    gaussian_smooth,
    response_valid,
)


MT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P3_MANIFEST = Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json")
BRIGHT_SELECTION = Path(
    "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
    "fullcap_selection_v1/selection_manifest.json"
)
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
CORES = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/cores.npz")
P4_MANIFEST = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/spatial_manifest.json")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)


def stratum_scale(redshift: np.ndarray, factors: dict[str, float] | None) -> np.ndarray:
    """Return the registered cap/shell retention multiplier for every voxel."""
    redshift = np.asarray(redshift, dtype=np.float64)
    if factors is None:
        return np.ones(redshift.shape, dtype=np.float64)
    result = np.full(redshift.shape, np.nan, dtype=np.float64)
    for name, lower, upper in RED_SHIFT_STRATA:
        selected = (redshift >= lower) & (redshift < upper)
        result[selected] = float(factors[name])
    if np.any(~np.isfinite(result[(redshift >= 0.10) & (redshift < 0.60)])):
        raise RuntimeError("missing density-matched response factor inside support")
    # Voxels outside the catalogue's registered radial support are already masked
    # by zero exposure.  A zero factor avoids manufacturing expected counts there.
    result[~np.isfinite(result)] = 0.0
    return result


def thin_response_factors(control: dict, seed: int, cap_name: str) -> dict[str, dict[str, float]]:
    """Extract tracer-specific thinning rates from the immutable field audit."""
    cap_audit = control["density_matched_thinning"][str(seed)]["audit"][cap_name]
    result = {"bright": {}, "faint": {}}
    for shell_name, _, _ in RED_SHIFT_STRATA:
        row = cap_audit[shell_name]["retention_fraction_by_tracer"]
        result["bright"][shell_name] = float(row["bright"])
        result["faint"][shell_name] = float(row["faint"])
    return result


def load_control_pair(
    *,
    bright_counts_path: Path,
    bright_counts_dataset: str,
    faint_counts_path: Path,
    faint_counts_dataset: str,
    bright_exposure_path: Path,
    faint_exposure_path: Path,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    cell_mpc: float,
    bright_curve: dict,
    faint_curve: dict,
    cosmology: dict,
    lookup_row: dict,
    core_mpc: float,
    train_folds: tuple[int, ...],
    device: str,
    slab: int,
    bright_scale: dict[str, float] | None = None,
    faint_scale: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load control counts while retaining the frozen tracer-specific response."""
    tensors = {
        name: torch.empty(shape, dtype=torch.float32, device=device)
        for name in ("counts_b", "expected_b", "counts_f", "expected_f")
    }
    training = torch.empty(shape, dtype=torch.bool, device=device)
    radius_grid = np.asarray(cosmology["radius_grid_mpc"], dtype=np.float64)
    redshift_grid = np.asarray(cosmology["redshift_grid"], dtype=np.float64)
    y = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell_mpc
    z = origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell_mpc
    curves = {
        "b": (
            np.asarray(bright_curve["grid_z"], dtype=np.float64),
            np.asarray(bright_curve["ntilde"], dtype=np.float64),
            bright_scale,
        ),
        "f": (
            np.asarray(faint_curve["grid_z"], dtype=np.float64),
            np.asarray(faint_curve["ntilde"], dtype=np.float64),
            faint_scale,
        ),
    }
    sums = {"counts_b": 0.0, "expected_b": 0.0, "counts_f": 0.0, "expected_f": 0.0}
    with ExitStack() as stack:
        bright_counts = stack.enter_context(h5py.File(bright_counts_path, "r"))
        faint_counts = stack.enter_context(h5py.File(faint_counts_path, "r"))
        bright_exposure = stack.enter_context(h5py.File(bright_exposure_path, "r"))
        faint_exposure = stack.enter_context(h5py.File(faint_exposure_path, "r"))
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            x = origin[0] + (np.arange(left, right, dtype=np.float64) + 0.5) * cell_mpc
            radius = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2)
            redshift = np.interp(radius, radius_grid, redshift_grid)
            arrays = {
                "counts_b": np.asarray(bright_counts[bright_counts_dataset][left:right], dtype=np.float32),
                "counts_f": np.asarray(faint_counts[faint_counts_dataset][left:right], dtype=np.float32),
            }
            for key, handle in (("b", bright_exposure), ("f", faint_exposure)):
                exposure = np.asarray(handle["exposure_apodized"][left:right], dtype=np.float32)
                grid_z, ntilde, factors = curves[key]
                radial = np.interp(np.clip(redshift, grid_z[0], grid_z[-1]), grid_z, ntilde)
                expected = radial * cell_mpc**3 * exposure.astype(np.float64)
                expected *= stratum_scale(redshift, factors)
                arrays[f"expected_{key}"] = expected.astype(np.float32)
            fold = fold_block(
                left=left,
                right=right,
                shape=shape,
                origin=origin,
                cell_mpc=cell_mpc,
                base_mpc=np.asarray(lookup_row["base_mpc"], dtype=np.float64),
                core_mpc=core_mpc,
                lookup=np.asarray(lookup_row["lookup"], dtype=np.int8),
            )
            for name, value in arrays.items():
                tensors[name][left:right].copy_(torch.from_numpy(value).to(device))
                sums[name] += float(value.sum(dtype=np.float64))
            training[left:right].copy_(torch.from_numpy(np.isin(fold, train_folds)).to(device))
    return {**tensors, "training": training}, sums


def contrasts(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | float]:
    valid_b, floor_b = response_valid(fields["expected_b"])
    valid_f, floor_f = response_valid(fields["expected_f"])
    delta_b = torch.zeros_like(fields["expected_b"])
    delta_f = torch.zeros_like(fields["expected_f"])
    delta_b[valid_b] = fields["counts_b"][valid_b] / fields["expected_b"][valid_b] - 1.0
    delta_f[valid_f] = fields["counts_f"][valid_f] / fields["expected_f"][valid_f] - 1.0
    combined, valid_combined, floor_combined = combined_count_contrast(
        fields["counts_b"], fields["expected_b"], fields["counts_f"], fields["expected_f"]
    )
    return {
        "delta_b": delta_b,
        "delta_f": delta_f,
        "combined": combined,
        "valid_b": valid_b,
        "valid_f": valid_f,
        "valid_combined": valid_combined,
        "floor_b": floor_b,
        "floor_f": floor_f,
        "floor_combined": floor_combined,
    }


def run_rotation(rotation: int, args: argparse.Namespace, manifests: dict) -> dict:
    started = time.time()
    print(f"[MT3b eval] rotation={rotation} loading Bright ownership", flush=True)
    assignment = np.load(args.assignment, mmap_mode="r")
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    points = np.load(args.points, mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    train_folds, validation_fold, _ = fold_roles(rotations, rotation)
    authoritative = authoritative_mask(assignment)
    row_fold = np.asarray(assignment["fold"], dtype=np.int8)
    active_rows = np.flatnonzero(authoritative & np.isin(row_fold, (*train_folds, validation_fold)))
    parent = np.asarray(assignment["parent_node_id"][active_rows], dtype=np.int64)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("P8 authoritative parent IDs are not unique")
    train = np.isin(row_fold[active_rows], train_folds)
    validation = row_fold[active_rows] == validation_fold
    cap_id = np.asarray(points[parent, 3], dtype=np.int8)
    positions = np.asarray(points[parent, :3], dtype=np.float64)
    estimator_names = [
        "bright_tsc",
        "combined_tsc",
        "bias_aware_tsc",
        *(f"thin_seed{seed}_cic" for seed in args.thin_seeds),
        "faint_position_null_cic",
    ]
    raw = {name: np.empty((len(parent), 3), dtype=np.float32) for name in estimator_names}
    estimator_caps: dict[str, dict] = {name: {} for name in estimator_names}
    cores = np.load(args.cores, mmap_mode="r")
    p4_manifest = json.loads(args.p4_manifest.read_text())
    core_mpc = float(p4_manifest["unit_contract"]["core_mpc"])
    lookups = {cap: build_cap_lookup(cores, cap, core_mpc) for cap in (0, 1)}
    bright_rotation = manifests["bright_selection"]["rotations"][str(rotation)]
    faint_rotation = manifests["mt_selection"]["tracers"]["BGS_FAINT"]["rotations"][str(rotation)]

    def solve(
        *, name: str, delta: torch.Tensor, selected: np.ndarray, origin: np.ndarray,
        cell_mpc: float, cap_name: str, diagnostics: dict,
    ) -> None:
        print(f"[MT3b eval] rotation={rotation} cap={cap_name} estimator={name} FFT", flush=True)
        prediction, fft = _sample_tidal_eigenvalues(
            delta,
            positions=positions[selected],
            origin=origin,
            cell_mpc=cell_mpc,
            padding_voxels=args.padding_voxels,
            rsmooth_mpc=args.rsmooth_mpc,
        )
        raw[name][selected] = prediction
        estimator_caps[name][cap_name] = {**diagnostics, "fft": fft, "n_sampled": int(selected.sum())}

    control = manifests["control"]
    for cap in (0, 1):
        selected = cap_id == cap
        cap_name = CAP_NAME[cap]
        print(f"[MT3b eval] rotation={rotation} cap={cap_name} loading controls", flush=True)
        grid = manifests["p3"]["components"][cap_name]["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell_mpc = float(grid["cell_mpc"])
        bright_base = Path(manifests["p3"]["components"][cap_name]["file"])
        faint_base = Path(manifests["mt_fields"]["components"][cap_name]["file"])
        common = dict(
            bright_exposure_path=bright_base,
            faint_exposure_path=faint_base,
            shape=shape,
            origin=origin,
            cell_mpc=cell_mpc,
            bright_curve=bright_rotation["caps"][cap_name],
            faint_curve=faint_rotation["caps"][cap_name],
            cosmology=manifests["bright_selection"]["cosmology"],
            lookup_row=lookups[cap],
            core_mpc=core_mpc,
            train_folds=tuple(train_folds),
            device=args.device,
            slab=args.slab,
        )

        tsc_path = Path(control["products"]["original_tsc"]["components"][cap_name]["file"])
        fields, sums = load_control_pair(
            bright_counts_path=tsc_path,
            bright_counts_dataset="bright_counts",
            faint_counts_path=tsc_path,
            faint_counts_dataset="faint_counts",
            **common,
        )
        row = contrasts(fields)
        solve(
            name="bright_tsc", delta=row["delta_b"], selected=selected,
            origin=origin, cell_mpc=cell_mpc, cap_name=cap_name,
            diagnostics={"field_sums": sums, "assignment": "TSC", "response_floor": row["floor_b"]},
        )
        smooth_b = gaussian_smooth(
            row["delta_b"], cell_mpc=cell_mpc, rsmooth_mpc=args.rsmooth_mpc,
            padding_voxels=args.padding_voxels,
        )
        smooth_f = gaussian_smooth(
            row["delta_f"], cell_mpc=cell_mpc, rsmooth_mpc=args.rsmooth_mpc,
            padding_voxels=args.padding_voxels,
        )
        bias = fit_relative_bias(
            smooth_b, smooth_f, fields["training"] & row["valid_b"] & row["valid_f"]
        )
        del smooth_b, smooth_f
        bias_delta, valid_bias = bias_aware_contrast(
            row["delta_b"], fields["expected_b"], row["valid_b"],
            row["delta_f"], fields["expected_f"], row["valid_f"],
            bias["relative_bias_faint_over_bright"],
        )
        solve(
            name="combined_tsc", delta=row["combined"], selected=selected,
            origin=origin, cell_mpc=cell_mpc, cap_name=cap_name,
            diagnostics={"field_sums": sums, "assignment": "TSC", "response_floor": row["floor_combined"]},
        )
        solve(
            name="bias_aware_tsc", delta=bias_delta, selected=selected,
            origin=origin, cell_mpc=cell_mpc, cap_name=cap_name,
            diagnostics={
                "field_sums": sums, "assignment": "TSC", "relative_bias_fit": bias,
                "supported_voxels": int(valid_bias.sum().item()),
            },
        )
        del fields, row, bias_delta, valid_bias

        for seed in args.thin_seeds:
            name = f"thin_seed{seed}_cic"
            path = Path(control["products"][name]["components"][cap_name]["file"])
            scale = thin_response_factors(control, seed, cap_name)
            fields, sums = load_control_pair(
                bright_counts_path=path,
                bright_counts_dataset="bright_counts",
                faint_counts_path=path,
                faint_counts_dataset="faint_counts",
                bright_scale=scale["bright"],
                faint_scale=scale["faint"],
                **common,
            )
            row = contrasts(fields)
            solve(
                name=name, delta=row["combined"], selected=selected,
                origin=origin, cell_mpc=cell_mpc, cap_name=cap_name,
                diagnostics={
                    "field_sums": sums, "assignment": "CIC",
                    "response_floor": row["floor_combined"], "response_scaling": scale,
                },
            )
            del fields, row

        null_path = Path(
            control["products"]["faint_position_null_cic"]["components"][cap_name]["file"]
        )
        fields, sums = load_control_pair(
            bright_counts_path=bright_base,
            bright_counts_dataset="counts",
            faint_counts_path=null_path,
            faint_counts_dataset="faint_counts",
            **common,
        )
        row = contrasts(fields)
        solve(
            name="faint_position_null_cic", delta=row["combined"], selected=selected,
            origin=origin, cell_mpc=cell_mpc, cap_name=cap_name,
            diagnostics={
                "field_sums": sums, "assignment": "CIC",
                "response_floor": row["floor_combined"],
                "null_contract": control["faint_position_null"]["contract"],
            },
        )
        del fields, row

    runtime = {
        "elapsed_seconds": time.time() - started,
        "device": args.device,
        "padding_voxels": int(args.padding_voxels),
        "rsmooth_mpc": float(args.rsmooth_mpc),
        "screen_rotation": int(rotation),
    }
    output = args.root / "classical/mt3b_controls" / f"rotation_{rotation}"
    reports = {}
    for name in estimator_names:
        reports[name] = evaluate_estimator(
            name=name,
            raw=raw[name],
            parent=parent,
            train=train,
            validation=validation,
            truth=truth,
            assignment=assignment,
            validation_fold=validation_fold,
            output=output,
            runtime=runtime,
            extra={
                "schema_version": "p8-mt3b-estimator-v1",
                "caps": estimator_caps[name],
                "faint_labels_used": False,
                "authoritative_population": "frozen BGS_BRIGHT only",
            },
        )
    rotation_report = {
        "schema_version": "p8-mt3b-rotation-v1",
        "rotation": int(rotation),
        "train_folds": list(train_folds),
        "validation_fold": int(validation_fold),
        "estimators": {
            name: {
                "primary_macro_r2_lambda1": reports[name]["train_affine"]["primary_macro_r2_lambda1"],
                "diagnostic_first_three_shell_macro_r2_lambda1": reports[name]["train_affine"]["diagnostic_first_three_shell_macro_r2_lambda1"],
                "complete_core_coverage": reports[name]["train_affine"]["complete_core_coverage"],
                "ordering_violation_rate": reports[name]["train_affine"]["ordering_violation_rate"],
                "per_shell_r2_lambda1": {
                    shell: value["lambda1"]["r2"]
                    for shell, value in reports[name]["train_affine"]["per_shell"].items()
                },
            }
            for name in estimator_names
        },
        "runtime": runtime,
    }
    atomic_json(output / "rotation_report.json", rotation_report)
    print(f"[MT3b eval] rotation={rotation} complete elapsed={runtime['elapsed_seconds']:.1f}s", flush=True)
    return rotation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MT_ROOT)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p3-manifest", type=Path, default=P3_MANIFEST)
    parser.add_argument("--bright-selection", type=Path, default=BRIGHT_SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--cores", type=Path, default=CORES)
    parser.add_argument("--p4-manifest", type=Path, default=P4_MANIFEST)
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument("--screen-rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--thin-seeds", type=int, nargs="+", default=(17, 42, 2718))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MT3b full-cap controls require a CUDA allocation")
    control_path = args.root / "classical/control_fields" / args.product / "manifest.json"
    mt3a_path = args.root / "classical/mt3a_cic/summary.json"
    paths = {
        "p3_manifest": args.p3_manifest,
        "bright_selection": args.bright_selection,
        "mt_fields": args.root / "fields" / args.product / "manifest.json",
        "mt_selection": args.root / "selection" / args.product / "multitracer_selection_manifest.json",
        "control_fields": control_path,
        "mt3a_summary": mt3a_path,
        "assignment": args.assignment,
        "rotations": args.rotations,
        "cores": args.cores,
        "p4_manifest": args.p4_manifest,
        "points": args.points,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    manifests = {
        "p3": json.loads(args.p3_manifest.read_text()),
        "bright_selection": json.loads(args.bright_selection.read_text()),
        "mt_fields": json.loads(paths["mt_fields"].read_text()),
        "mt_selection": json.loads(paths["mt_selection"].read_text()),
        "control": json.loads(control_path.read_text()),
    }
    required_pass = ("mt_fields", "mt_selection", "control")
    if not all(manifests[name].get("pass") for name in required_pass):
        raise RuntimeError("passing field, selection, and control manifests required")
    reports = [run_rotation(rotation, args, manifests) for rotation in args.screen_rotations]
    bright_summary_path = args.p8_root / "classical/classical_summary.json"
    bright_summary = json.loads(bright_summary_path.read_text())
    mt3a = json.loads(mt3a_path.read_text())
    bright_by_rotation = dict(zip(bright_summary["screen_rotations"], bright_summary["primary_score_by_rotation"]))
    combined_cic_by_rotation = dict(zip(mt3a["screen_rotations"], mt3a["estimators"]["combined_cic"]["primary_score_by_rotation"]))
    estimator_names = list(reports[0]["estimators"])
    estimator_summary = {}
    for name in estimator_names:
        scores = [row["estimators"][name]["primary_macro_r2_lambda1"] for row in reports]
        estimator_summary[name] = {
            "primary_score_by_rotation": scores,
            "primary_score_mean": float(np.mean(scores)),
            "first_three_score_by_rotation": [
                row["estimators"][name]["diagnostic_first_three_shell_macro_r2_lambda1"]
                for row in reports
            ],
        }
    thin_names = [f"thin_seed{seed}_cic" for seed in args.thin_seeds]
    thin_matrix = np.asarray(
        [[reports[index]["estimators"][name]["primary_macro_r2_lambda1"] for name in thin_names]
         for index in range(len(reports))],
        dtype=np.float64,
    )
    bright_ref = np.asarray([bright_by_rotation[rotation] for rotation in args.screen_rotations])
    full_combined = np.asarray([combined_cic_by_rotation[rotation] for rotation in args.screen_rotations])
    null = np.asarray(estimator_summary["faint_position_null_cic"]["primary_score_by_rotation"])
    gates = {
        "two_registered_rotations": list(args.screen_rotations) == [0, 2],
        "three_density_matched_seeds": len(args.thin_seeds) == 3,
        "all_estimators_finite": all(
            np.all(np.isfinite(row["primary_score_by_rotation"]))
            for row in estimator_summary.values()
        ),
        "all_estimators_complete_bright_fold": all(
            estimator["complete_core_coverage"]
            for report in reports for estimator in report["estimators"].values()
        ),
        "zero_ordering_violations": all(
            estimator["ordering_violation_rate"] == 0.0
            for report in reports for estimator in report["estimators"].values()
        ),
        "faint_labels_never_used": True,
        "training_only_affine_and_bias_fit": True,
        "field_controls_passed": bool(manifests["control"]["pass"]),
    }
    diagnostics = {
        "full_combined_cic_minus_bright_cic_by_rotation": (full_combined - bright_ref).tolist(),
        "thin_score_mean_by_rotation": thin_matrix.mean(axis=1).tolist(),
        "thin_score_std_by_rotation": thin_matrix.std(axis=1, ddof=1).tolist(),
        "full_combined_cic_minus_thin_mean_by_rotation": (full_combined - thin_matrix.mean(axis=1)).tolist(),
        "faint_position_null_minus_bright_cic_by_rotation": (null - bright_ref).tolist(),
        "interpretation_contract": {
            "full_minus_thin": "increment attributable to extra sampling density beyond a density-matched mixed population",
            "thin_minus_bright": "population-mix effect at fixed total tracer count",
            "null_minus_bright": "shortcut/selection diagnostic; must not reproduce the real-Faint gain",
        },
    }
    summary = {
        "schema_version": "p8-mt3-complete-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "MT3 matched classical information controls",
        "status": "complete" if all(gates.values()) else "failed_gate",
        "pass": all(gates.values()),
        "product": args.product,
        "screen_rotations": list(args.screen_rotations),
        "bright_only_cic_reference": {
            "path": str(bright_summary_path),
            "sha256": sha256(bright_summary_path),
            "primary_score_by_rotation": bright_ref.tolist(),
        },
        "full_multitracer_cic_reference": {
            "path": str(mt3a_path),
            "sha256": sha256(mt3a_path),
            "primary_score_by_rotation": full_combined.tolist(),
        },
        "estimators": estimator_summary,
        "diagnostics": diagnostics,
        "gates": gates,
        "neural_training_unlocked": all(gates.values()),
        "input_records": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "rotation_reports": [
            str(args.root / "classical/mt3b_controls" / f"rotation_{rotation}" / "rotation_report.json")
            for rotation in args.screen_rotations
        ],
    }
    output = args.root / "classical/mt3_complete/summary.json"
    atomic_json(output, summary)
    marker = output.parent / "MT3_MULTITRACER_CLASSICAL_COMPLETE"
    if summary["pass"]:
        marker.write_text(
            f"summary_sha256={sha256(output)}\nneural_training_unlocked=true\n"
        )
    elif marker.exists():
        marker.unlink()
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pass"]:
        raise RuntimeError(f"MT3 evaluation gates failed: {gates}")


if __name__ == "__main__":
    main()
