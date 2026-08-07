#!/usr/bin/env python3
"""Quantify the information added by context-only BGS_FAINT tracers.

This MT2 audit is deliberately estimator-free.  It compares the unchanged
BGS_BRIGHT observation with the response-explicit Bright+Faint observation on
the frozen P3 grids and P4 target population.  It never reads tidal labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.spatial import cKDTree

from workflows.abacus_tweb.p6_field_patch_utils import derive_selection_channels
from workflows.abacus_tweb.p6_refit_fullcap_selection import radius_to_redshift_grid


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P3 = Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json")
P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
SHELLS = (
    ("0p15_0p25", 0.15, 0.25),
    ("0p25_0p35", 0.25, 0.35),
    ("0p35_0p45", 0.35, 0.45),
    ("0p45_0p55", 0.45, 0.55),
)
CAP_ID = {"NGC": 1, "SGC": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p3-manifest", type=Path, default=P3)
    parser.add_argument("--assignment", type=Path, default=P4)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--query-chunk", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def shell_index(redshift: np.ndarray) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=np.float64)
    result = np.full(redshift.shape, -1, dtype=np.int8)
    for index, (_, lower, upper) in enumerate(SHELLS):
        result[(redshift >= lower) & (redshift < upper)] = index
    return result


def correlation_from_moments(moment: np.ndarray) -> float | None:
    """Pearson correlation from [n, sx, sy, sxx, syy, sxy]."""
    n, sx, sy, sxx, syy, sxy = np.asarray(moment, dtype=np.float64)
    if n < 2:
        return None
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    if variance_x <= 0 or variance_y <= 0:
        return None
    return float(covariance / np.sqrt(variance_x * variance_y))


def update_moments(moment: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    moment += np.array(
        [len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(y, y), np.dot(x, y)],
        dtype=np.float64,
    )


def nearest_other_distance(
    tree: cKDTree,
    points: np.ndarray,
    *,
    workers: int,
    self_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Distance to another point when the query may itself be in ``tree``."""
    distance, _ = tree.query(points, k=2, workers=workers)
    distance = np.asarray(distance, dtype=np.float64)
    return np.where(distance[:, 0] <= self_tolerance, distance[:, 1], distance[:, 0])


def percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "p10_mpc": None, "median_mpc": None, "p90_mpc": None}
    q10, q50, q90 = np.quantile(values, (0.1, 0.5, 0.9))
    return {
        "n": int(len(values)),
        "p10_mpc": float(q10),
        "median_mpc": float(q50),
        "p90_mpc": float(q90),
    }


def grid_redshift(
    selection: tuple[slice, slice, slice],
    *,
    origin: np.ndarray,
    cell_mpc: float,
    radius_grid: np.ndarray,
    redshift_grid: np.ndarray,
) -> np.ndarray:
    axes = [
        origin[axis]
        + (np.arange(part.start, part.stop, dtype=np.float64) + 0.5) * cell_mpc
        for axis, part in enumerate(selection)
    ]
    radius = np.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    return np.interp(radius, radius_grid, redshift_grid)


def field_information(
    *,
    p3: dict,
    fields: dict,
    selection: dict,
    rotation: int,
) -> dict:
    radius_grid, redshift_grid = radius_to_redshift_grid(0.10, 0.60)
    output = {}
    for cap_name, cap_id in CAP_ID.items():
        bright_component = p3["components"][cap_name]
        faint_component = fields["components"][cap_name]
        curve = selection["tracers"]["BGS_FAINT"]["rotations"][str(rotation)][
            "caps"
        ][cap_name]
        epsilon = float(selection["contrast"]["epsilon"])
        minimum_exposure = float(selection["contrast"]["minimum_exposure"])
        shape = tuple(int(value) for value in faint_component["grid"]["shape"])
        origin = np.asarray(faint_component["grid"]["origin_mpc"], dtype=np.float64)
        cell_mpc = float(faint_component["grid"]["cell_mpc"])
        accumulators = [
            {
                "support": 0,
                "bright_occupied": 0,
                "faint_occupied": 0,
                "combined_occupied": 0,
                "bright_empty_filled": 0,
                "moments": np.zeros(6, dtype=np.float64),
            }
            for _ in SHELLS
        ]
        with h5py.File(bright_component["file"], "r") as bright, h5py.File(
            faint_component["file"], "r"
        ) as faint:
            if tuple(bright["counts"].shape) != shape or tuple(faint["counts"].shape) != shape:
                raise RuntimeError(f"{cap_name} field shapes do not match")
            for block in bright["counts"].iter_chunks():
                redshift = grid_redshift(
                    block,
                    origin=origin,
                    cell_mpc=cell_mpc,
                    radius_grid=radius_grid,
                    redshift_grid=redshift_grid,
                )
                shell = shell_index(redshift)
                bright_counts = np.asarray(bright["counts"][block], dtype=np.float32)
                bright_exposure = np.asarray(
                    bright["exposure_apodized"][block], dtype=np.float32
                )
                bright_contrast = np.asarray(
                    bright["log_count_ratio"][block], dtype=np.float32
                )
                faint_counts = np.asarray(faint["counts"][block], dtype=np.float32)
                faint_exposure = np.asarray(
                    faint["exposure_apodized"][block], dtype=np.float32
                )
                faint_derived = derive_selection_channels(
                    faint_counts,
                    faint_exposure,
                    redshift,
                    cell_mpc=cell_mpc,
                    grid_z=np.asarray(curve["grid_z"], dtype=np.float64),
                    ntilde=np.asarray(curve["ntilde"], dtype=np.float64),
                    epsilon=epsilon,
                    minimum_exposure=minimum_exposure,
                )
                common_support = (
                    (bright_exposure > minimum_exposure)
                    & (faint_exposure > minimum_exposure)
                )
                for shell_id in range(len(SHELLS)):
                    inside = shell == shell_id
                    supported = inside & common_support
                    row = accumulators[shell_id]
                    row["support"] += int(np.count_nonzero(supported))
                    row["bright_occupied"] += int(
                        np.count_nonzero(supported & (bright_counts > 0))
                    )
                    row["faint_occupied"] += int(
                        np.count_nonzero(supported & (faint_counts > 0))
                    )
                    row["combined_occupied"] += int(
                        np.count_nonzero(supported & ((bright_counts + faint_counts) > 0))
                    )
                    row["bright_empty_filled"] += int(
                        np.count_nonzero(
                            supported & (bright_counts <= 0) & (faint_counts > 0)
                        )
                    )
                    update_moments(
                        row["moments"],
                        bright_contrast[supported],
                        faint_derived["log_count_ratio"][supported],
                    )
        output[cap_name] = {}
        for shell_id, (shell_name, _, _) in enumerate(SHELLS):
            row = accumulators[shell_id]
            support = max(row["support"], 1)
            bright_empty = max(row["support"] - row["bright_occupied"], 1)
            output[cap_name][shell_name] = {
                "common_response_supported_voxels": row["support"],
                "bright_occupied_fraction": row["bright_occupied"] / support,
                "faint_occupied_fraction": row["faint_occupied"] / support,
                "combined_occupied_fraction": row["combined_occupied"] / support,
                "bright_empty_voxels_filled": row["bright_empty_filled"],
                "fraction_of_bright_empty_voxels_filled": (
                    row["bright_empty_filled"] / bright_empty
                ),
                "bright_faint_log_contrast_pearson": correlation_from_moments(
                    row["moments"]
                ),
            }
    return output


def catalogue_information(
    *,
    points_path: Path,
    index_path: Path,
    bright_rows: int,
    assignment_path: Path,
    query_chunk: int,
    workers: int,
) -> dict:
    points = np.load(points_path, mmap_mode="r")
    index = np.load(index_path, mmap_mode="r")
    tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    if len(points) != len(tracer) or bright_rows <= 0:
        raise RuntimeError("catalogue/index length contract failed")
    radius_grid, redshift_grid = radius_to_redshift_grid(0.10, 0.60)
    radius = np.linalg.norm(np.asarray(points[:, :3], dtype=np.float64), axis=1)
    redshift = np.interp(radius, radius_grid, redshift_grid)
    shell = shell_index(redshift)
    active = np.load(assignment_path, mmap_mode="r")
    authoritative_parent = np.unique(np.asarray(active["parent_node_id"], dtype=np.int64))
    if np.any((authoritative_parent < 0) | (authoritative_parent >= bright_rows)):
        raise RuntimeError("P4 authoritative parent is outside the Bright prefix")
    output = {
        "bright_rows": int(bright_rows),
        "faint_rows": int(np.count_nonzero(tracer == 1)),
        "bright_authoritative_rows": int(len(authoritative_parent)),
        "bright_authoritative_fraction": float(len(authoritative_parent) / bright_rows),
        "caps": {},
    }
    for cap_name, cap_id in CAP_ID.items():
        bright_context = (tracer == 0) & context & (cap == cap_id)
        faint_context = (tracer == 1) & context & (cap == cap_id)
        bright_tree = cKDTree(np.asarray(points[bright_context, :3], dtype=np.float64))
        faint_tree = cKDTree(np.asarray(points[faint_context, :3], dtype=np.float64))
        cap_result = {}
        for shell_id, (shell_name, _, _) in enumerate(SHELLS):
            target = np.flatnonzero(
                (cap[:bright_rows] == cap_id) & (shell[:bright_rows] == shell_id)
            )
            nearest_bright, nearest_faint = [], []
            for start in range(0, len(target), query_chunk):
                xyz = np.asarray(
                    points[target[start : start + query_chunk], :3], dtype=np.float64
                )
                nearest_bright.append(
                    nearest_other_distance(bright_tree, xyz, workers=workers)
                )
                nearest_faint.append(faint_tree.query(xyz, k=1, workers=workers)[0])
            bright_distance = np.concatenate(nearest_bright) if nearest_bright else np.empty(0)
            faint_distance = np.concatenate(nearest_faint) if nearest_faint else np.empty(0)
            n_bright = int(np.count_nonzero(bright_context & (shell == shell_id)))
            n_faint = int(np.count_nonzero(faint_context & (shell == shell_id)))
            cap_result[shell_name] = {
                "bright_target_rows": int(len(target)),
                "bright_context_rows": n_bright,
                "faint_context_rows": n_faint,
                "raw_poisson_noise_ratio_bright_plus_faint_over_bright": (
                    n_bright / max(n_bright + n_faint, 1)
                ),
                "nearest_other_bright": percentiles(bright_distance),
                "nearest_faint": percentiles(faint_distance),
            }
        output["caps"][cap_name] = cap_result
    return output


def add_sampling_density(report: dict, p3: dict) -> None:
    for cap_name in CAP_ID:
        atlas = p3["components"][cap_name]["support_atlas"]
        for shell_id, (shell_name, lower, upper) in enumerate(SHELLS):
            source_name = f"{lower:.2f}_{upper:.2f}"
            volume = float(atlas[source_name]["support_voxels"]) * float(
                p3["components"][cap_name]["grid"]["cell_mpc"]
            ) ** 3
            row = report["catalogue"]["caps"][cap_name][shell_name]
            n_bright = row["bright_context_rows"] / volume
            n_faint = row["faint_context_rows"] / volume
            row["reference_support_volume_mpc3"] = volume
            row["bright_number_density_mpc3"] = n_bright
            row["faint_number_density_mpc3"] = n_faint
            row["combined_number_density_mpc3"] = n_bright + n_faint
            row["bright_mean_separation_mpc"] = (
                n_bright ** (-1.0 / 3.0) if n_bright > 0 else None
            )
            row["combined_mean_separation_mpc"] = (
                (n_bright + n_faint) ** (-1.0 / 3.0)
                if n_bright + n_faint > 0
                else None
            )


def main() -> None:
    args = parse_args()
    catalogue_manifest_path = args.root / "catalogues" / args.product / "manifest.json"
    field_manifest_path = args.root / "fields" / args.product / "manifest.json"
    selection_manifest_path = (
        args.root / "selection" / args.product / "multitracer_selection_manifest.json"
    )
    catalogue = json.loads(catalogue_manifest_path.read_text())
    fields = json.loads(field_manifest_path.read_text())
    selection = json.loads(selection_manifest_path.read_text())
    p3 = json.loads(args.p3_manifest.read_text())
    if not (catalogue["pass"] and fields["pass"] and selection["pass"]):
        raise RuntimeError("multitracer dependencies have not passed")
    report = {
        "schema_version": "p8-multitracer-information-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "product": args.product,
        "rotation": args.rotation,
        "scope": "truth-free information audit; BGS_BRIGHT targets and BGS_FAINT context",
        "units": "comoving Mpc and Mpc^-3",
        "inputs": {
            "catalogue_manifest": str(catalogue_manifest_path),
            "field_manifest": str(field_manifest_path),
            "selection_manifest": str(selection_manifest_path),
            "bright_p3_manifest": str(args.p3_manifest),
            "p4_assignment": str(args.assignment),
        },
    }
    report["catalogue"] = catalogue_information(
        points_path=Path(catalogue["points"]),
        index_path=Path(catalogue["index"]),
        bright_rows=int(catalogue["bright_prefix_rows"]),
        assignment_path=args.assignment,
        query_chunk=args.query_chunk,
        workers=args.workers,
    )
    add_sampling_density(report, p3)
    report["fields"] = field_information(
        p3=p3,
        fields=fields,
        selection=selection,
        rotation=args.rotation,
    )
    correlations = [
        row["bright_faint_log_contrast_pearson"]
        for cap in report["fields"].values()
        for row in cap.values()
    ]
    report["information_signals"] = {
        "all_shells_have_positive_bright_faint_correlation": all(
            value is not None and value > 0 for value in correlations
        ),
        "all_shells_fill_bright_empty_voxels": all(
            row["bright_empty_voxels_filled"] > 0
            for cap in report["fields"].values()
            for row in cap.values()
        ),
        "all_shells_reduce_raw_poisson_noise": all(
            row["raw_poisson_noise_ratio_bright_plus_faint_over_bright"] < 1.0
            for cap in report["catalogue"]["caps"].values()
            for row in cap.values()
        ),
    }
    report["gates"] = {
        "bright_targets_unchanged": (
            report["catalogue"]["bright_rows"] == int(catalogue["bright_prefix_rows"])
        ),
        "faint_context_nonempty": report["catalogue"]["faint_rows"] > 0,
        "all_shells_have_faint_context": all(
            row["faint_context_rows"] > 0
            for cap in report["catalogue"]["caps"].values()
            for row in cap.values()
        ),
        "all_field_correlations_defined": all(
            value is not None and np.isfinite(value) for value in correlations
        ),
    }
    report["pass"] = all(report["gates"].values())
    output = args.output or args.root / "diagnostics" / args.product / "information_audit.json"
    atomic_json(output, report)
    marker = output.parent / "MT2_INFORMATION_AUDIT_COMPLETE"
    if not report["pass"]:
        raise RuntimeError(f"multitracer information gates failed: {report['gates']}")
    marker.write_text(f"product={args.product} report={output}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
