#!/usr/bin/env python3
"""Fit cap-aware selection channels from P4 training folds.

This stage never mutates the immutable P3 fields.  It estimates one radial
selection curve per cap and spatial rotation from label-free observed galaxies
and the P3 effective exposure volume in the rotation's training folds.  It also
freezes field-channel normalizers using supported training voxels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import fitsio
import h5py
import numpy as np
from scipy.interpolate import LSQUnivariateSpline


CAPS = ("SGC", "NGC")
CAP_ID = {"SGC": 0, "NGC": 1}
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
ZSCORE_CHANNELS = ("counts", "expected_counts", "log_count_ratio", "ntilde_mpc3")
IDENTITY_CHANNELS = (
    "exposure_apodized",
    "exposure_binary",
    "los_x",
    "los_y",
    "los_z",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p3-manifest",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"),
    )
    parser.add_argument(
        "--p4-root",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest"),
    )
    parser.add_argument(
        "--p1-manifest",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
            "fullcap_selection_v1"
        ),
    )
    parser.add_argument("--z-min", type=float, default=0.10)
    parser.add_argument("--z-max", type=float, default=0.60)
    parser.add_argument("--bin-width", type=float, default=0.005)
    parser.add_argument("--curve-step", type=float, default=0.001)
    parser.add_argument("--knot-spacing", type=float, default=0.05)
    parser.add_argument(
        "--sensitivity-knot-spacing", type=float, nargs="+", default=(0.04, 0.06)
    )
    parser.add_argument("--fit-z-min", type=float, default=0.15)
    parser.add_argument("--fit-z-max", type=float, default=0.55)
    parser.add_argument("--minimum-exposure", type=float, default=1.0e-4)
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--closure-tolerance", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_cap_lookup(
    cores: np.lib.npyio.NpzFile, cap: int, core_mpc: float
) -> dict:
    chosen = np.asarray(cores["cap"] == cap)
    indices = np.asarray(cores["core_index"][chosen], dtype=np.int64)
    lower = np.asarray(cores["lower_mpc"][chosen], dtype=np.float64)
    folds = np.asarray(cores["fold"][chosen], dtype=np.int8)
    origins = lower - indices.astype(np.float64) * core_mpc
    base = np.median(origins, axis=0)
    origin_error = float(np.max(np.abs(origins - base)))
    if origin_error > 1.0e-6:
        raise RuntimeError(f"inconsistent P4 cap origin: {origin_error}")
    shape = tuple((indices.max(axis=0) + 1).tolist())
    lookup = np.full(shape, -1, dtype=np.int8)
    if np.any(lookup[tuple(indices.T)] >= 0):
        raise RuntimeError("duplicate P4 core index")
    lookup[tuple(indices.T)] = folds
    return {
        "base_mpc": base,
        "lookup": lookup,
        "origin_error_mpc": origin_error,
        "core_count": int(len(indices)),
    }


def radius_to_redshift_grid(z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    redshift = np.linspace(max(0.0, z_min - 0.05), z_max + 0.05, 20001)
    radius = Planck18.comoving_distance(redshift).value.astype(np.float64)
    return radius, redshift


def chunk_geometry(
    selection: tuple[slice, slice, slice],
    *,
    origin_mpc: np.ndarray,
    cell_mpc: float,
    base_mpc: np.ndarray,
    core_mpc: float,
    fold_lookup: np.ndarray,
    radius_grid_mpc: np.ndarray,
    redshift_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axes_index = [
        np.arange(part.start, part.stop, dtype=np.int64) for part in selection
    ]
    axes_mpc = [
        origin_mpc[axis] + (indices.astype(np.float64) + 0.5) * cell_mpc
        for axis, indices in enumerate(axes_index)
    ]
    core_axes = [
        np.floor((axis - base_mpc[d]) / core_mpc).astype(np.int64)
        for d, axis in enumerate(axes_mpc)
    ]
    valid_axes = [
        (axis >= 0) & (axis < fold_lookup.shape[d])
        for d, axis in enumerate(core_axes)
    ]
    clipped_axes = [
        np.clip(axis, 0, fold_lookup.shape[d] - 1)
        for d, axis in enumerate(core_axes)
    ]
    fold = fold_lookup[np.ix_(*clipped_axes)].copy()
    valid = (
        valid_axes[0][:, None, None]
        & valid_axes[1][None, :, None]
        & valid_axes[2][None, None, :]
    )
    fold[~valid] = -1
    radius = np.sqrt(
        axes_mpc[0][:, None, None] ** 2
        + axes_mpc[1][None, :, None] ** 2
        + axes_mpc[2][None, None, :] ** 2
    )
    redshift = np.interp(radius, radius_grid_mpc, redshift_grid)
    return fold, redshift


def histogram_counts(
    *,
    parent_path: Path,
    context_path: Path,
    edges: np.ndarray,
) -> tuple[np.ndarray, dict]:
    context = np.load(context_path, mmap_mode="r")
    parent_id = np.asarray(context["parent_node_id"], dtype=np.int64)
    cap = np.asarray(context["cap"], dtype=np.int8)
    fold = np.asarray(context["fold"], dtype=np.int8)
    z_all = fitsio.read(parent_path, columns=["Z"])["Z"]
    z = np.asarray(z_all[parent_id], dtype=np.float64)
    histogram = np.zeros((2, 5, len(edges) - 1), dtype=np.int64)
    for cap_id in range(2):
        for fold_id in range(5):
            mask = (cap == cap_id) & (fold == fold_id)
            histogram[cap_id, fold_id] = np.histogram(z[mask], bins=edges)[0]
    audit = {
        "context_rows": int(len(parent_id)),
        "unique_parent_rows": int(len(np.unique(parent_id))),
        "catalogue_rows": int(len(z_all)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }
    return histogram, audit


def histogram_effective_volume(
    *,
    p3: dict,
    lookups: dict,
    core_mpc: float,
    edges: np.ndarray,
    radius_grid_mpc: np.ndarray,
    redshift_grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    volume = np.zeros((2, 5, len(edges) - 1), dtype=np.float64)
    audit = {}
    for cap_name in CAPS:
        cap_id = CAP_ID[cap_name]
        component = p3["components"][cap_name]
        grid = component["grid"]
        cell_mpc = float(grid["cell_mpc"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        chunks = 0
        assigned_voxels = 0
        with h5py.File(component["file"], "r") as handle:
            exposure_dataset = handle["exposure_apodized"]
            for selection in exposure_dataset.iter_chunks():
                exposure = np.asarray(exposure_dataset[selection], dtype=np.float64)
                fold, redshift = chunk_geometry(
                    selection,
                    origin_mpc=origin,
                    cell_mpc=cell_mpc,
                    base_mpc=lookups[cap_name]["base_mpc"],
                    core_mpc=core_mpc,
                    fold_lookup=lookups[cap_name]["lookup"],
                    radius_grid_mpc=radius_grid_mpc,
                    redshift_grid=redshift_grid,
                )
                radial_bin = np.floor(
                    (redshift - edges[0]) / (edges[1] - edges[0])
                ).astype(np.int64)
                valid_z = (radial_bin >= 0) & (radial_bin < len(edges) - 1)
                for fold_id in range(5):
                    mask = (fold == fold_id) & valid_z & (exposure > 0)
                    if np.any(mask):
                        volume[cap_id, fold_id] += np.bincount(
                            radial_bin[mask],
                            weights=exposure[mask] * cell_mpc**3,
                            minlength=len(edges) - 1,
                        )[: len(edges) - 1]
                        assigned_voxels += int(np.count_nonzero(mask))
                chunks += 1
        audit[cap_name] = {
            "chunks": chunks,
            "assigned_exposed_voxels": assigned_voxels,
            "effective_volume_mpc3": float(volume[cap_id].sum()),
        }
    return volume, audit


def fit_log_spline(
    centers: np.ndarray,
    counts: np.ndarray,
    volume: np.ndarray,
    grid_z: np.ndarray,
    *,
    knot_spacing: float,
    fit_z_min: float,
    fit_z_max: float,
) -> tuple[np.ndarray, dict]:
    density = np.divide(
        counts.astype(np.float64),
        volume,
        out=np.full_like(volume, np.nan, dtype=np.float64),
        where=volume > 0,
    )
    valid = (
        np.isfinite(density)
        & (density > 0)
        & (counts >= 20)
        & (centers >= fit_z_min)
        & (centers < fit_z_max)
    )
    x = centers[valid]
    y = np.log(density[valid])
    weight = np.sqrt(counts[valid].astype(np.float64))
    knots = np.arange(fit_z_min + knot_spacing, fit_z_max, knot_spacing)
    knots = knots[(knots > x.min()) & (knots < x.max())]
    if len(x) <= len(knots) + 4:
        raise RuntimeError("insufficient populated radial bins for selection fit")
    spline = LSQUnivariateSpline(x, y, knots, w=weight, k=3)
    curve = np.exp(spline(np.clip(grid_z, x.min(), x.max())))
    in_fit = (centers >= fit_z_min) & (centers < fit_z_max)
    predicted = float(
        np.sum(np.interp(centers[in_fit], grid_z, curve) * volume[in_fit])
    )
    observed = float(np.sum(counts[in_fit]))
    amplitude = observed / predicted
    curve *= amplitude
    return curve, {
        "populated_bins": int(np.count_nonzero(valid)),
        "internal_knots": knots.tolist(),
        "amplitude_correction": float(amplitude),
        "observed_training_count": observed,
        "pre_amplitude_expected_count": predicted,
    }


def closure_by_role(
    *,
    counts: np.ndarray,
    volume: np.ndarray,
    centers: np.ndarray,
    grid_z: np.ndarray,
    curve: np.ndarray,
    rotation: dict,
) -> dict:
    roles = {
        "train": rotation["train_folds"],
        "validation": [rotation["validation_fold"]],
        "development_test": [rotation["development_test_fold"]],
    }
    output = {}
    predicted_density = np.interp(centers, grid_z, curve)
    for role, folds in roles.items():
        role_counts = counts[np.asarray(folds, dtype=np.int64)].sum(axis=0)
        role_volume = volume[np.asarray(folds, dtype=np.int64)].sum(axis=0)
        shell_rows = []
        for low, high in SHELLS:
            selected = (centers >= low) & (centers < high)
            observed = float(np.sum(role_counts[selected]))
            expected = float(np.sum(predicted_density[selected] * role_volume[selected]))
            shell_rows.append(
                {
                    "z_low": low,
                    "z_high": high,
                    "observed": observed,
                    "expected": expected,
                    "expected_over_observed": (
                        expected / observed if observed > 0 else None
                    ),
                    "fractional_error": (
                        expected / observed - 1.0 if observed > 0 else None
                    ),
                }
            )
        output[role] = shell_rows
    return output


def accumulate_moment(state: dict, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    slot = state.setdefault(name, {"count": 0, "sum": 0.0, "sum_sq": 0.0})
    slot["count"] += int(values.size)
    slot["sum"] += float(np.sum(values, dtype=np.float64))
    slot["sum_sq"] += float(np.sum(values * values, dtype=np.float64))


def fit_normalizers(
    *,
    p3: dict,
    lookups: dict,
    core_mpc: float,
    rotations: dict,
    rotation_curves: dict,
    radius_grid_mpc: np.ndarray,
    redshift_grid: np.ndarray,
    fit_z_min: float,
    fit_z_max: float,
    minimum_exposure: float,
    epsilon: float,
) -> dict:
    states = {str(index): {} for index in range(5)}
    for cap_name in CAPS:
        component = p3["components"][cap_name]
        cap_grid = component["grid"]
        cell_mpc = float(cap_grid["cell_mpc"])
        origin = np.asarray(cap_grid["origin_mpc"], dtype=np.float64)
        with h5py.File(component["file"], "r") as handle:
            counts_dataset = handle["counts"]
            exposure_dataset = handle["exposure_apodized"]
            for selection in exposure_dataset.iter_chunks():
                counts = np.asarray(counts_dataset[selection], dtype=np.float64)
                exposure = np.asarray(exposure_dataset[selection], dtype=np.float64)
                fold, redshift = chunk_geometry(
                    selection,
                    origin_mpc=origin,
                    cell_mpc=cell_mpc,
                    base_mpc=lookups[cap_name]["base_mpc"],
                    core_mpc=core_mpc,
                    fold_lookup=lookups[cap_name]["lookup"],
                    radius_grid_mpc=radius_grid_mpc,
                    redshift_grid=redshift_grid,
                )
                base_mask = (
                    (exposure > minimum_exposure)
                    & (redshift >= fit_z_min)
                    & (redshift < fit_z_max)
                )
                for rotation_id in range(5):
                    rotation = rotations[str(rotation_id)]
                    mask = base_mask & np.isin(fold, rotation["train_folds"])
                    if not np.any(mask):
                        continue
                    curve_spec = rotation_curves[str(rotation_id)]["caps"][cap_name]
                    ntilde = np.interp(
                        redshift[mask],
                        np.asarray(curve_spec["grid_z"], dtype=np.float64),
                        np.asarray(curve_spec["ntilde"], dtype=np.float64),
                    )
                    expected = ntilde * cell_mpc**3 * exposure[mask]
                    contrast = np.log(
                        (counts[mask] + epsilon) / (expected + epsilon)
                    )
                    accumulate_moment(
                        states[str(rotation_id)], "counts", np.log1p(counts[mask])
                    )
                    accumulate_moment(
                        states[str(rotation_id)],
                        "expected_counts",
                        np.log1p(expected),
                    )
                    accumulate_moment(
                        states[str(rotation_id)], "log_count_ratio", contrast
                    )
                    accumulate_moment(
                        states[str(rotation_id)], "ntilde_mpc3", np.log(ntilde)
                    )
    normalizers = {}
    for rotation_id, state in states.items():
        channels = {}
        count_set = set()
        for name in ZSCORE_CHANNELS:
            slot = state[name]
            mean = slot["sum"] / slot["count"]
            variance = max(slot["sum_sq"] / slot["count"] - mean**2, 0.0)
            std = variance**0.5
            if slot["count"] <= 0 or not np.isfinite(std) or std <= 0:
                raise RuntimeError(
                    f"invalid normalizer for rotation {rotation_id} channel {name}"
                )
            count_set.add(slot["count"])
            channels[name] = {
                "policy": "zscore",
                "pre_transform": (
                    "log1p_nonnegative"
                    if name in {"counts", "expected_counts"}
                    else ("log_floor_1e-12" if name == "ntilde_mpc3" else "identity")
                ),
                "mean": float(mean),
                "std": float(std),
                "fit_voxels": int(slot["count"]),
            }
        if len(count_set) != 1:
            raise RuntimeError(f"normalizer count mismatch for rotation {rotation_id}")
        for name in IDENTITY_CHANNELS:
            channels[name] = {"policy": "identity", "pre_transform": "identity"}
        normalizers[rotation_id] = {
            "scope": (
                "supported P4 training-core voxels across NGC+SGC; "
                f"{fit_z_min}<=z<{fit_z_max}"
            ),
            "channels": channels,
        }
    return normalizers


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "selection_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} exists; pass --force to replace")

    p3 = json.loads(args.p3_manifest.read_text())
    p4_manifest_path = args.p4_root / "spatial_manifest.json"
    rotations_path = args.p4_root / "rotations.json"
    p4 = json.loads(p4_manifest_path.read_text())
    rotations = json.loads(rotations_path.read_text())
    p1 = json.loads(args.p1_manifest.read_text())
    cores = np.load(args.p4_root / "cores.npz", mmap_mode="r")
    core_mpc = float(p4["unit_contract"]["core_mpc"])
    lookups = {
        cap_name: build_cap_lookup(cores, CAP_ID[cap_name], core_mpc)
        for cap_name in CAPS
    }

    edges = np.arange(
        args.z_min, args.z_max + 0.5 * args.bin_width, args.bin_width
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_z = np.arange(
        args.z_min, args.z_max + 0.5 * args.curve_step, args.curve_step
    )
    radius_grid_mpc, redshift_grid = radius_to_redshift_grid(
        args.z_min, args.z_max
    )
    counts, count_audit = histogram_counts(
        parent_path=Path(p1["parent"]),
        context_path=args.p4_root / "context_assignment.npz",
        edges=edges,
    )
    volume, volume_audit = histogram_effective_volume(
        p3=p3,
        lookups=lookups,
        core_mpc=core_mpc,
        edges=edges,
        radius_grid_mpc=radius_grid_mpc,
        redshift_grid=redshift_grid,
    )

    rotation_curves = {}
    primary_train_errors = []
    sensitivity_max = []
    for rotation_id in range(5):
        rotation = rotations[str(rotation_id)]
        rotation_entry = {
            "train_folds": rotation["train_folds"],
            "validation_fold": rotation["validation_fold"],
            "development_test_fold": rotation["development_test_fold"],
            "caps": {},
        }
        for cap_name in CAPS:
            cap_id = CAP_ID[cap_name]
            train = np.asarray(rotation["train_folds"], dtype=np.int64)
            train_counts = counts[cap_id, train].sum(axis=0)
            train_volume = volume[cap_id, train].sum(axis=0)
            curve, fit = fit_log_spline(
                centers,
                train_counts,
                train_volume,
                grid_z,
                knot_spacing=args.knot_spacing,
                fit_z_min=args.fit_z_min,
                fit_z_max=args.fit_z_max,
            )
            sensitivities = {}
            for spacing in args.sensitivity_knot_spacing:
                alternate, alternate_fit = fit_log_spline(
                    centers,
                    train_counts,
                    train_volume,
                    grid_z,
                    knot_spacing=float(spacing),
                    fit_z_min=args.fit_z_min,
                    fit_z_max=args.fit_z_max,
                )
                in_scope = (grid_z >= args.fit_z_min) & (grid_z < args.fit_z_max)
                maximum = float(
                    np.max(np.abs(alternate[in_scope] / curve[in_scope] - 1.0))
                )
                sensitivity_max.append(maximum)
                sensitivities[str(spacing)] = {
                    "max_fractional_curve_difference": maximum,
                    "fit": alternate_fit,
                }
            closure = closure_by_role(
                counts=counts[cap_id],
                volume=volume[cap_id],
                centers=centers,
                grid_z=grid_z,
                curve=curve,
                rotation=rotation,
            )
            for row in closure["train"]:
                if row["fractional_error"] is not None:
                    primary_train_errors.append(abs(row["fractional_error"]))
            rotation_entry["caps"][cap_name] = {
                "grid_z": grid_z.tolist(),
                "ntilde": curve.tolist(),
                "units": "Mpc^-3",
                "fit": fit,
                "closure": closure,
                "sensitivity": sensitivities,
            }
        rotation_curves[str(rotation_id)] = rotation_entry

    normalizers = fit_normalizers(
        p3=p3,
        lookups=lookups,
        core_mpc=core_mpc,
        rotations=rotations,
        rotation_curves=rotation_curves,
        radius_grid_mpc=radius_grid_mpc,
        redshift_grid=redshift_grid,
        fit_z_min=args.fit_z_min,
        fit_z_max=args.fit_z_max,
        minimum_exposure=args.minimum_exposure,
        epsilon=args.epsilon,
    )
    for rotation_id in range(5):
        rotation_curves[str(rotation_id)]["normalization"] = normalizers[
            str(rotation_id)
        ]

    histogram_path = args.output_root / "fold_radial_histograms.npz"
    np.savez_compressed(
        histogram_path,
        bin_edges_z=edges,
        bin_centers_z=centers,
        observed_counts=counts,
        effective_volume_mpc3=volume,
    )
    max_train_error = max(primary_train_errors)
    gates = {
        "fit_uses_training_folds_only": True,
        "separate_cap_curves": True,
        "curves_finite_positive": bool(
            all(
                np.all(np.isfinite(entry["ntilde"]))
                and np.all(np.asarray(entry["ntilde"]) > 0)
                for rotation in rotation_curves.values()
                for entry in rotation["caps"].values()
            )
        ),
        "train_shell_closure_within_tolerance": bool(
            max_train_error <= args.closure_tolerance
        ),
        "normalizers_training_only": True,
        "normalizers_finite": bool(
            all(
                np.isfinite(spec["mean"]) and np.isfinite(spec["std"])
                for normalizer in normalizers.values()
                for spec in normalizer["channels"].values()
                if spec["policy"] == "zscore"
            )
        ),
    }
    manifest = {
        "schema_version": "p6-fullcap-selection-v1",
        "stage": "P6_SELECTION_REFIT",
        "status": "complete" if all(gates.values()) else "failed_gate",
        "pass": bool(all(gates.values())),
        "git_sha": git_sha(),
        "elapsed_seconds": time.time() - start_time,
        "inputs": {
            "p1_manifest": str(args.p1_manifest),
            "p1_manifest_sha256": sha256(args.p1_manifest),
            "p3_manifest": str(args.p3_manifest),
            "p3_manifest_sha256": sha256(args.p3_manifest),
            "p4_spatial_manifest": str(p4_manifest_path),
            "p4_spatial_manifest_sha256": sha256(p4_manifest_path),
            "p4_rotations": str(rotations_path),
            "p4_rotations_sha256": sha256(rotations_path),
            "context_assignment": str(args.p4_root / "context_assignment.npz"),
            "parent_catalogue": p1["parent"],
        },
        "contract": {
            "p3_fields_are_immutable": True,
            "estimator": "weighted cubic LSQ spline in log number density",
            "primary_knot_spacing_z": args.knot_spacing,
            "sensitivity_knot_spacing_z": list(args.sensitivity_knot_spacing),
            "training_information": (
                "label-free observed galaxy redshifts and P3 apodized effective "
                "exposure volume in the rotation's three P4 training folds"
            ),
            "validation_and_development_truth_used_for_fit": False,
            "curve_scope": "one frozen curve per P4 rotation and Galactic cap",
            "normalization_scope": (
                "one frozen normalizer per P4 rotation, pooled over both caps, "
                "using supported training-core voxels only"
            ),
        },
        "contrast": {
            "epsilon": args.epsilon,
            "minimum_exposure": args.minimum_exposure,
            "formula": "log((counts+epsilon)/(ntilde*cell_mpc^3*exposure+epsilon))",
        },
        "cosmology": {
            "name": "Planck18",
            "radius_grid_mpc": radius_grid_mpc.tolist(),
            "redshift_grid": redshift_grid.tolist(),
        },
        "radial_fit": {
            "z_min": args.z_min,
            "z_max": args.z_max,
            "bin_width": args.bin_width,
            "fit_z_min": args.fit_z_min,
            "fit_z_max": args.fit_z_max,
            "curve_step": args.curve_step,
            "closure_tolerance": args.closure_tolerance,
        },
        "fold_radial_histograms": str(histogram_path),
        "fold_radial_histograms_sha256": sha256(histogram_path),
        "count_audit": count_audit,
        "volume_audit": volume_audit,
        "cap_lookup_audit": {
            cap_name: {
                "base_mpc": lookups[cap_name]["base_mpc"].tolist(),
                "shape": list(lookups[cap_name]["lookup"].shape),
                "origin_error_mpc": lookups[cap_name]["origin_error_mpc"],
                "core_count": lookups[cap_name]["core_count"],
            }
            for cap_name in CAPS
        },
        "maximum_primary_train_shell_fractional_error": max_train_error,
        "maximum_sensitivity_curve_fractional_difference": max(sensitivity_max),
        "rotations": rotation_curves,
        "gates": gates,
    }
    atomic_json(manifest_path, manifest)
    marker = args.output_root / "SELECTION_REFIT_COMPLETE"
    if manifest["pass"]:
        marker.write_text(
            json.dumps(
                {
                    "selection_manifest": str(manifest_path),
                    "selection_manifest_sha256": sha256(manifest_path),
                    "git_sha": manifest["git_sha"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    elif marker.exists():
        marker.unlink()
    print(json.dumps({
        "manifest": str(manifest_path),
        "pass": manifest["pass"],
        "gates": gates,
        "maximum_primary_train_shell_fractional_error": max_train_error,
        "maximum_sensitivity_curve_fractional_difference": max(sensitivity_max),
    }, indent=2))


if __name__ == "__main__":
    main()
