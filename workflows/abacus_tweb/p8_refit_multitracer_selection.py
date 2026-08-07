#!/usr/bin/env python3
"""Fit independent Faint radial selection channels for multitracer P8.

Bright continues to use the frozen P6 full-cap selection manifest.  This stage
assigns Faint context points to the unchanged P4 spatial folds, fits one Faint
number-density curve per cap and rotation from training folds only, and freezes
Faint field normalizers.  A combined manifest references both tracer contracts;
there is no merged selection function.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from workflows.abacus_tweb.p6_refit_fullcap_selection import (
    CAPS,
    CAP_ID,
    atomic_json,
    build_cap_lookup,
    closure_by_role,
    fit_log_spline,
    fit_normalizers,
    git_sha,
    histogram_effective_volume,
    radius_to_redshift_grid,
    sha256,
)


DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
DEFAULT_P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")
DEFAULT_BRIGHT_SELECTION = Path(
    "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
    "fullcap_selection_v1/selection_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--p4-root", type=Path, default=DEFAULT_P4)
    parser.add_argument(
        "--bright-selection", type=Path, default=DEFAULT_BRIGHT_SELECTION
    )
    parser.add_argument(
        "--products", nargs="+",
        default=("bf_oracle_assigned_v1", "bf_proxy_response_v1"),
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


def point_folds(
    xyz: np.ndarray, *, base_mpc: np.ndarray, core_mpc: float,
    fold_lookup: np.ndarray,
) -> np.ndarray:
    indices = np.floor(
        (np.asarray(xyz, dtype=np.float64) - base_mpc[None, :]) / core_mpc
    ).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(fold_lookup.shape)), axis=1)
    folds = np.full(len(indices), -1, dtype=np.int8)
    folds[valid] = fold_lookup[tuple(indices[valid].T)]
    return folds


def faint_histograms(
    *, points_path: Path, index_path: Path, bright_rows: int,
    lookups: dict, core_mpc: float, edges: np.ndarray,
    radius_grid_mpc: np.ndarray, redshift_grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    points = np.load(points_path, mmap_mode="r")
    index = np.load(index_path)
    tracer = np.asarray(index["tracer_type"][bright_rows:], dtype=np.uint8)
    context = np.asarray(index["context"][bright_rows:], dtype=bool)
    chosen = (tracer == 1) & context
    faint = np.asarray(points[bright_rows:][chosen], dtype=np.float64)
    cap = np.asarray(faint[:, 3], dtype=np.uint8)
    radius = np.linalg.norm(faint[:, :3], axis=1)
    redshift = np.interp(radius, radius_grid_mpc, redshift_grid)
    histogram = np.zeros((2, 5, len(edges) - 1), dtype=np.int64)
    unassigned = {}
    for cap_name in CAPS:
        cap_id = CAP_ID[cap_name]
        cap_mask = cap == cap_id
        folds = point_folds(
            faint[cap_mask, :3],
            base_mpc=lookups[cap_name]["base_mpc"],
            core_mpc=core_mpc,
            fold_lookup=lookups[cap_name]["lookup"],
        )
        cap_redshift = redshift[cap_mask]
        for fold_id in range(5):
            histogram[cap_id, fold_id] = np.histogram(
                cap_redshift[folds == fold_id], bins=edges
            )[0]
        unassigned[cap_name] = int(np.count_nonzero(folds < 0))
    audit = {
        "faint_context_rows": int(len(faint)),
        "assigned_rows": int(histogram.sum()),
        "unassigned_by_cap": unassigned,
        "unassigned_fraction": float(
            (len(faint) - histogram.sum()) / max(len(faint), 1)
        ),
        "redshift_min": float(redshift.min()),
        "redshift_max": float(redshift.max()),
    }
    return histogram, audit


def overlay_as_p3(field_manifest: dict) -> dict:
    return {
        "schema_version": "p8-faint-overlay-as-p3-v1",
        "components": {
            cap: {
                "file": component["file"],
                "grid": component["grid"],
            }
            for cap, component in field_manifest["components"].items()
        },
    }


def fit_product(args: argparse.Namespace, product: str) -> dict:
    started = time.time()
    output = args.root / "selection" / product
    manifest_path = output / "multitracer_selection_manifest.json"
    marker = output / "MULTITRACER_SELECTION_COMPLETE"
    if marker.exists() and manifest_path.exists() and not args.force:
        return json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)

    catalogue_manifest_path = args.root / "catalogues" / product / "manifest.json"
    field_manifest_path = args.root / "fields" / product / "manifest.json"
    catalogue = json.loads(catalogue_manifest_path.read_text())
    fields = json.loads(field_manifest_path.read_text())
    p4_manifest_path = args.p4_root / "spatial_manifest.json"
    rotations_path = args.p4_root / "rotations.json"
    p4 = json.loads(p4_manifest_path.read_text())
    rotations = json.loads(rotations_path.read_text())
    cores = np.load(args.p4_root / "cores.npz", mmap_mode="r")
    core_mpc = float(p4["unit_contract"]["core_mpc"])
    lookups = {
        cap: build_cap_lookup(cores, CAP_ID[cap], core_mpc) for cap in CAPS
    }
    overlay = overlay_as_p3(fields)

    edges = np.arange(args.z_min, args.z_max + 0.5 * args.bin_width, args.bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_z = np.arange(args.z_min, args.z_max + 0.5 * args.curve_step, args.curve_step)
    radius_grid, redshift_grid = radius_to_redshift_grid(args.z_min, args.z_max)
    counts, count_audit = faint_histograms(
        points_path=Path(catalogue["points"]),
        index_path=Path(catalogue["index"]),
        bright_rows=int(catalogue["bright_prefix_rows"]),
        lookups=lookups,
        core_mpc=core_mpc,
        edges=edges,
        radius_grid_mpc=radius_grid,
        redshift_grid=redshift_grid,
    )
    volume, volume_audit = histogram_effective_volume(
        p3=overlay,
        lookups=lookups,
        core_mpc=core_mpc,
        edges=edges,
        radius_grid_mpc=radius_grid,
        redshift_grid=redshift_grid,
    )

    rotation_curves, train_errors, sensitivity_max = {}, [], []
    for rotation_id in range(5):
        rotation = rotations[str(rotation_id)]
        entry = {
            "train_folds": rotation["train_folds"],
            "validation_fold": rotation["validation_fold"],
            "development_test_fold": rotation["development_test_fold"],
            "caps": {},
        }
        for cap_name in CAPS:
            cap_id = CAP_ID[cap_name]
            train = np.asarray(rotation["train_folds"], dtype=np.int64)
            curve, fit = fit_log_spline(
                centers,
                counts[cap_id, train].sum(axis=0),
                volume[cap_id, train].sum(axis=0),
                grid_z,
                knot_spacing=args.knot_spacing,
                fit_z_min=args.fit_z_min,
                fit_z_max=args.fit_z_max,
            )
            sensitivities = {}
            for spacing in args.sensitivity_knot_spacing:
                alternate, alternate_fit = fit_log_spline(
                    centers,
                    counts[cap_id, train].sum(axis=0),
                    volume[cap_id, train].sum(axis=0),
                    grid_z,
                    knot_spacing=float(spacing),
                    fit_z_min=args.fit_z_min,
                    fit_z_max=args.fit_z_max,
                )
                in_scope = (grid_z >= args.fit_z_min) & (grid_z < args.fit_z_max)
                maximum = float(np.max(np.abs(alternate[in_scope] / curve[in_scope] - 1)))
                sensitivity_max.append(maximum)
                sensitivities[str(spacing)] = {
                    "max_fractional_curve_difference": maximum,
                    "fit": alternate_fit,
                }
            closure = closure_by_role(
                counts=counts[cap_id], volume=volume[cap_id], centers=centers,
                grid_z=grid_z, curve=curve, rotation=rotation,
            )
            train_errors.extend(
                abs(row["fractional_error"])
                for row in closure["train"] if row["fractional_error"] is not None
            )
            entry["caps"][cap_name] = {
                "grid_z": grid_z.tolist(),
                "ntilde": curve.tolist(),
                "units": "Mpc^-3",
                "fit": fit,
                "closure": closure,
                "sensitivity": sensitivities,
            }
        rotation_curves[str(rotation_id)] = entry

    normalizers = fit_normalizers(
        p3=overlay,
        lookups=lookups,
        core_mpc=core_mpc,
        rotations=rotations,
        rotation_curves=rotation_curves,
        radius_grid_mpc=radius_grid,
        redshift_grid=redshift_grid,
        fit_z_min=args.fit_z_min,
        fit_z_max=args.fit_z_max,
        minimum_exposure=args.minimum_exposure,
        epsilon=args.epsilon,
    )
    for rotation_id in range(5):
        rotation_curves[str(rotation_id)]["normalization"] = normalizers[str(rotation_id)]

    histogram_path = output / "faint_fold_radial_histograms.npz"
    np.savez_compressed(
        histogram_path,
        bin_edges_z=edges,
        bin_centers_z=centers,
        observed_counts=counts,
        effective_volume_mpc3=volume,
    )
    maximum_train_error = max(train_errors)
    gates = {
        "bright_selection_frozen_and_separate": args.bright_selection.exists(),
        "faint_fit_uses_training_folds_only": True,
        "faint_separate_cap_curves": True,
        "faint_curves_finite_positive": all(
            np.all(np.isfinite(cap["ntilde"])) and np.all(np.asarray(cap["ntilde"]) > 0)
            for rotation in rotation_curves.values() for cap in rotation["caps"].values()
        ),
        "faint_train_shell_closure_within_tolerance": (
            maximum_train_error <= args.closure_tolerance
        ),
        "faint_normalizers_training_only": True,
        "faint_point_fold_assignment_above_99pct": count_audit["unassigned_fraction"] < 0.01,
    }
    manifest = {
        "schema_version": "p8-multitracer-selection-v1",
        "product": product,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "status": "complete" if all(gates.values()) else "failed_gate",
        "pass": bool(all(gates.values())),
        "elapsed_seconds": time.time() - started,
        "tracers": {
            "BGS_BRIGHT": {
                "selection_manifest": str(args.bright_selection),
                "selection_manifest_sha256": sha256(args.bright_selection),
                "policy": "frozen P6 contract",
            },
            "BGS_FAINT": {
                "policy": "independent per-cap, per-rotation training-fold fit",
                "rotations": rotation_curves,
            },
        },
        "contract": {
            "combined_selection_function": False,
            "combined_count_channel": False,
            "labels_or_eigenvalues_used": False,
            "supervised_population": "BGS_BRIGHT only",
            "context_population": "BGS_BRIGHT plus BGS_FAINT",
        },
        "contrast": {
            "epsilon": args.epsilon,
            "minimum_exposure": args.minimum_exposure,
            "formula": "log((N_F+epsilon)/(ntilde_F*cell_mpc^3*exposure_F+epsilon))",
        },
        "inputs": {
            "catalogue_manifest": str(catalogue_manifest_path),
            "catalogue_manifest_sha256": sha256(catalogue_manifest_path),
            "field_manifest": str(field_manifest_path),
            "field_manifest_sha256": sha256(field_manifest_path),
            "p4_manifest": str(p4_manifest_path),
            "p4_manifest_sha256": sha256(p4_manifest_path),
            "rotations": str(rotations_path),
            "rotations_sha256": sha256(rotations_path),
        },
        "radial_fit": {
            "z_min": args.z_min, "z_max": args.z_max,
            "fit_z_min": args.fit_z_min, "fit_z_max": args.fit_z_max,
            "bin_width": args.bin_width, "curve_step": args.curve_step,
            "knot_spacing": args.knot_spacing,
            "closure_tolerance": args.closure_tolerance,
        },
        "faint_count_audit": count_audit,
        "faint_volume_audit": volume_audit,
        "maximum_faint_train_shell_fractional_error": maximum_train_error,
        "maximum_faint_sensitivity_curve_fractional_difference": max(sensitivity_max),
        "faint_fold_radial_histograms": str(histogram_path),
        "faint_fold_radial_histograms_sha256": sha256(histogram_path),
        "gates": gates,
    }
    atomic_json(manifest_path, manifest)
    if manifest["pass"]:
        marker.write_text(
            f"product={product}\nmanifest_sha256={sha256(manifest_path)}\n"
        )
    elif marker.exists():
        marker.unlink()
    return manifest


def main() -> None:
    args = parse_args()
    for path in (args.root, args.p4_root, args.bright_selection):
        if not path.exists():
            raise FileNotFoundError(path)
    result = {product: fit_product(args, product) for product in args.products}
    summary = {
        "schema_version": "p8-multitracer-selection-build-v1",
        "products": {
            product: {
                "manifest": str(
                    args.root / "selection" / product / "multitracer_selection_manifest.json"
                ),
                "pass": manifest["pass"],
            }
            for product, manifest in result.items()
        },
        "all_pass": all(manifest["pass"] for manifest in result.values()),
    }
    atomic_json(args.root / "selection" / "selection_build_summary.json", summary)
    if not summary["all_pass"]:
        failed = [name for name, manifest in result.items() if not manifest["pass"]]
        raise RuntimeError(f"multitracer selection gates failed: {failed}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
