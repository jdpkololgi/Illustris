#!/usr/bin/env python3
"""Build P8.9 cap-aligned R=7 Mpc/h matter-density targets.

The canonical T-web slabs already contain eigenvalues of the R=7 smoothed
periodic tidal tensor. Their trace is therefore the exact smoothed matter
contrast under the frozen convention. This builder samples that trace at the
centres of the immutable P3 NGC/SGC voxels using the coordinate mapping proven
by ``p8_density_target_alignment.py``.

The output is privileged supervision. It is never a model input. The script
also freezes ``science_support`` and nominal P4 ``core_coverage`` masks. Their
intersection is the only admissible D0 density-loss support. Uncovered
supported voxels are reported explicitly and are never silently zero-filled
or replaced by a classical field.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import (
    build_slab_maps,
    discover_slabs,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
P3_MANIFEST = ROOT / "p3_full_footprint/field_manifest.json"
P6_ROOT = ROOT / "p6_unet_patch_adapter"
P6_SELECTION = P6_ROOT / "fullcap_selection_v1/selection_manifest.json"
ALIGNMENT = ROOT / "p8_density_phys_v1/preflight/coordinate_alignment.json"
TWEB = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/"
    "tweb_rank_outputs_fullgrid_v3/"
    "dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/"
    "backend_optimized_ngrid_2048_rsmooth_7"
)
OUTPUT = ROOT / "p8_density_phys_v1/targets"
CAP_ID = {"SGC": 0, "NGC": 1}
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-manifest", type=Path, default=P3_MANIFEST)
    parser.add_argument("--p6-root", type=Path, default=P6_ROOT)
    parser.add_argument("--p6-selection", type=Path, default=P6_SELECTION)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--tweb-dir", type=Path, default=TWEB)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--observer-origin-mpc-h", type=float, default=-1000.0)
    parser.add_argument("--z-min", type=float, default=0.15)
    parser.add_argument("--z-max", type=float, default=0.55)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--skip-source-hashes", action="store_true")
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def periodic_axis_indices(
    origin_mpc: float,
    size: int,
    output_cell_mpc: float,
    coordinate_h: float,
    observer_origin_mpc_h: float,
    boxsize_mpc_h: float,
    ngrid: int,
) -> np.ndarray:
    """Map observer-Mpc voxel centres to periodic Mpc/h source-grid cells."""
    centres_mpc = origin_mpc + (np.arange(size, dtype=np.float64) + 0.5) * output_cell_mpc
    centres_mpc_h = centres_mpc * float(coordinate_h)
    periodic = np.mod(centres_mpc_h + observer_origin_mpc_h, boxsize_mpc_h)
    indices = np.floor(periodic / (boxsize_mpc_h / float(ngrid))).astype(np.int64)
    return np.clip(indices, 0, ngrid - 1)


def trace_from_eigen_slab(
    eigenvalues: np.ndarray,
    local_x: np.ndarray,
    y_index: np.ndarray,
    z_index: np.ndarray,
) -> np.ndarray:
    """Sample and sum the three eigenvalue volumes without a second smoothing."""
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim != 4 or eigenvalues.shape[0] != 3:
        raise ValueError(f"expected [3,nx,ny,nz] eigenvalues, got {eigenvalues.shape}")
    local_x = np.asarray(local_x, dtype=np.int64)
    y_index = np.asarray(y_index, dtype=np.int64)
    z_index = np.asarray(z_index, dtype=np.int64)
    result = np.zeros((len(local_x), len(y_index), len(z_index)), dtype=np.float32)
    selection = np.ix_(local_x, y_index, z_index)
    for axis in range(3):
        result += np.asarray(eigenvalues[axis][selection], dtype=np.float32)
    return result


def build_core_coverage(
    shape: tuple[int, int, int],
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Return exact union of nominal P4 voxel cores on one cap lattice."""
    coverage = np.zeros(shape, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    if starts.shape != stops.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("core starts/stops must have shape [N,3]")
    for start, stop in zip(starts, stops, strict=True):
        left = np.maximum(start, 0)
        right = np.minimum(stop, np.asarray(shape, dtype=np.int64))
        if np.any(right <= left):
            continue
        coverage[
            left[0]:right[0],
            left[1]:right[1],
            left[2]:right[2],
        ] = True
    return coverage


def radius_squared(
    origin: np.ndarray,
    shape: tuple[int, int, int],
    cell: float,
) -> np.ndarray:
    axes = [
        origin[axis] + (np.arange(shape[axis], dtype=np.float64) + 0.5) * cell
        for axis in range(3)
    ]
    return (
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )


def shell_distance_bounds() -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(Planck18.comoving_distance(low).value),
            float(Planck18.comoving_distance(high).value),
        )
        for low, high in SHELLS
    )


def mask_summary(
    science_support: np.ndarray,
    core_coverage: np.ndarray,
    r2: np.ndarray,
) -> dict:
    supported = np.asarray(science_support, dtype=bool)
    covered = np.asarray(core_coverage, dtype=bool)
    total = int(np.sum(supported))
    intersection = int(np.sum(supported & covered))
    uncovered = total - intersection
    output = {
        "science_supported_voxels": total,
        "core_covered_voxels": int(np.sum(covered)),
        "covered_science_voxels": intersection,
        "uncovered_science_voxels": uncovered,
        "science_coverage_fraction": float(intersection / total) if total else float("nan"),
        "shells": {},
    }
    for shell_id, ((z_low, z_high), (r_low, r_high)) in enumerate(
        zip(SHELLS, shell_distance_bounds(), strict=True)
    ):
        radial = (r2 >= r_low**2) & (r2 < r_high**2)
        shell_support = supported & radial
        n_shell = int(np.sum(shell_support))
        n_covered = int(np.sum(shell_support & covered))
        output["shells"][str(shell_id)] = {
            "z_low": z_low,
            "z_high": z_high,
            "science_supported_voxels": n_shell,
            "covered_science_voxels": n_covered,
            "uncovered_science_voxels": n_shell - n_covered,
            "coverage_fraction": float(n_covered / n_shell) if n_shell else float("nan"),
        }
    return output


def hdf5_options(args: argparse.Namespace) -> dict:
    if args.compression == "none":
        return {}
    if args.compression == "gzip":
        return {"compression": "gzip", "compression_opts": int(args.gzip_level)}
    return {"compression": "lzf"}


def write_cap(
    path: Path,
    *,
    delta_r7: np.ndarray,
    science_support: np.ndarray,
    core_coverage: np.ndarray,
    component: dict,
    observer_origin: float,
    coordinate_h: float,
    source_ngrid: int,
    source_boxsize: float,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    chunk = tuple(
        min(int(component["chunk_shape"][i]), int(delta_r7.shape[i]))
        for i in range(3)
    )
    options = hdf5_options(args)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("delta_r7", data=delta_r7, dtype="f4", chunks=chunk, **options)
        handle.create_dataset(
            "science_support", data=np.asarray(science_support, dtype=np.uint8),
            dtype="u1", chunks=chunk, **options,
        )
        handle.create_dataset(
            "core_coverage", data=np.asarray(core_coverage, dtype=np.uint8),
            dtype="u1", chunks=chunk, **options,
        )
        handle.create_dataset(
            "density_loss_support",
            data=np.asarray(science_support & core_coverage, dtype=np.uint8),
            dtype="u1", chunks=chunk, **options,
        )
        grid = component["grid"]
        handle.attrs["axis_order"] = "ix,iy,iz"
        handle.attrs["units"] = "dimensionless matter contrast"
        handle.attrs["target_epoch"] = 0.2
        handle.attrs["smoothing_mpc_h"] = 7.0
        handle.attrs["sampling"] = "floor source cell at P3 voxel centre"
        handle.attrs["origin_mpc"] = np.asarray(grid["origin_mpc"], dtype=np.float64)
        handle.attrs["cell_mpc"] = float(grid["cell_mpc"])
        handle.attrs["observer_coordinate_h"] = float(coordinate_h)
        handle.attrs["observer_origin_mpc_h"] = np.asarray(
            [observer_origin] * 3, dtype=np.float64
        )
        handle.attrs["periodic_mapping"] = (
            "(observer_xyz_mpc * h + origin_mpc_h) modulo box"
        )
        handle.attrs["source_ngrid"] = int(source_ngrid)
        handle.attrs["source_boxsize_mpc_h"] = float(source_boxsize)
        handle.attrs["double_smoothing_applied"] = False
        handle.flush()
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    alignment = json.loads(args.alignment.read_text())
    if not alignment.get("pass"):
        raise RuntimeError("density-target coordinate preflight did not pass")
    if alignment.get("best_sky_variant") != "z_cosmo_origin_m1000p0":
        raise RuntimeError(f"unexpected coordinate mapping: {alignment.get('best_sky_variant')}")
    if abs(float(args.observer_origin_mpc_h) + 1000.0) > 1.0e-9:
        raise RuntimeError("observer origin differs from the passing preflight")

    p3 = json.loads(args.p3_manifest.read_text())
    p6_selection = json.loads(args.p6_selection.read_text())
    minimum_exposure = float(p6_selection["contrast"]["minimum_exposure"])
    coordinate_h = float(Planck18.h)
    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    del ix_to_slab, slab_xstart
    rsmooth = float(slabs[0].path.parent.name.rsplit("_", 1)[-1])
    if abs(rsmooth - 7.0) > 1.0e-9:
        raise RuntimeError(f"expected already-smoothed R=7 slabs, got R={rsmooth}")

    core_start = np.load(args.p6_root / "core_voxel_start.npy", mmap_mode="r")
    core_stop = np.load(args.p6_root / "core_voxel_stop.npy", mmap_mode="r")
    core_cap = np.load(args.p6_root / "core_cap.npy", mmap_mode="r")
    cap_state: dict[str, dict] = {}
    for cap_name, component in p3["components"].items():
        grid = component["grid"]
        shape = tuple(int(value) for value in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        source_index = tuple(
            periodic_axis_indices(
                origin[axis], shape[axis], cell, coordinate_h,
                float(args.observer_origin_mpc_h),
                boxsize, ngrid,
            )
            for axis in range(3)
        )
        cap_state[cap_name] = {
            "component": component,
            "shape": shape,
            "origin": origin,
            "cell": cell,
            "source_index": source_index,
            "delta": np.empty(shape, dtype=np.float32),
            "written_x": np.zeros(shape[0], dtype=bool),
        }

    slab_hashes = []
    for ordinal, slab in enumerate(slabs):
        record = {
            "path": str(slab.path),
            "bytes": int(slab.path.stat().st_size),
            "x_start": int(slab.x_start),
            "x_end": int(slab.x_end),
        }
        if not args.skip_source_hashes:
            record["sha256"] = sha256(slab.path)
        slab_hashes.append(record)
        with np.load(slab.path) as archive:
            eigenvalues = archive["eig_vals"]
            for state in cap_state.values():
                x_index, y_index, z_index = state["source_index"]
                destination_x = np.flatnonzero(
                    (x_index >= slab.x_start) & (x_index < slab.x_end)
                )
                if not len(destination_x):
                    continue
                local_x = x_index[destination_x] - slab.x_start
                state["delta"][destination_x] = trace_from_eigen_slab(
                    eigenvalues, local_x, y_index, z_index
                )
                state["written_x"][destination_x] = True
        print(f"slab {ordinal + 1}/{len(slabs)} x=[{slab.x_start},{slab.x_end})", flush=True)

    cap_reports = {}
    for cap_name, state in cap_state.items():
        if not np.all(state["written_x"]):
            missing = np.flatnonzero(~state["written_x"])
            raise RuntimeError(f"{cap_name} has {len(missing)} unwritten target x planes")
        delta = state["delta"]
        if not np.isfinite(delta).all():
            raise RuntimeError(f"{cap_name} target contains non-finite values")
        component = state["component"]
        shape = state["shape"]
        cap_mask = np.asarray(core_cap) == CAP_ID[cap_name]
        coverage = build_core_coverage(
            shape, np.asarray(core_start[cap_mask]), np.asarray(core_stop[cap_mask])
        )
        with h5py.File(component["file"], "r") as source:
            exposure = np.asarray(source["exposure_apodized"], dtype=np.float32)
        r2 = radius_squared(state["origin"], shape, state["cell"])
        r_min = float(Planck18.comoving_distance(args.z_min).value)
        r_max = float(Planck18.comoving_distance(args.z_max).value)
        science_support = (
            (exposure > minimum_exposure)
            & (r2 >= r_min**2)
            & (r2 < r_max**2)
        )
        support = mask_summary(science_support, coverage, r2)
        output_path = args.output / f"{cap_name.lower()}_delta_r7.h5"
        write_cap(
            output_path, delta_r7=delta, science_support=science_support,
            core_coverage=coverage, component=component,
            observer_origin=float(args.observer_origin_mpc_h), coordinate_h=coordinate_h,
            source_ngrid=ngrid,
            source_boxsize=boxsize, args=args,
        )
        cap_reports[cap_name] = {
            "cap_id": CAP_ID[cap_name],
            "file": str(output_path),
            "file_bytes": int(output_path.stat().st_size),
            "file_sha256": sha256(output_path),
            "grid": component["grid"],
            "source_field": component["file"],
            "target": {
                "minimum": float(np.min(delta)),
                "maximum": float(np.max(delta)),
                "mean": float(np.mean(delta, dtype=np.float64)),
                "std": float(np.std(delta, dtype=np.float64)),
                "all_finite": True,
            },
            "support": support,
        }
        del coverage, exposure, r2, science_support

    all_supported = sum(
        cap["support"]["science_supported_voxels"] for cap in cap_reports.values()
    )
    all_covered = sum(
        cap["support"]["covered_science_voxels"] for cap in cap_reports.values()
    )
    coverage_fraction = float(all_covered / all_supported) if all_supported else float("nan")
    target_ready = all(cap["target"]["all_finite"] for cap in cap_reports.values())
    stitching_ready = all(
        cap["support"]["uncovered_science_voxels"] == 0 for cap in cap_reports.values()
    )
    source_digest = hashlib.sha256(
        json.dumps(slab_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "p8-density-target-field-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "P8.9 D0 privileged target and support construction",
        "git_sha": git_sha(),
        "inputs": {
            "p3_manifest": str(args.p3_manifest),
            "p3_manifest_sha256": sha256(args.p3_manifest),
            "p6_root": str(args.p6_root),
            "p6_selection": str(args.p6_selection),
            "p6_selection_sha256": sha256(args.p6_selection),
            "alignment": str(args.alignment),
            "alignment_sha256": sha256(args.alignment),
            "tweb_dir": str(args.tweb_dir),
            "tweb_slab_manifest_sha256": source_digest,
            "tweb_slabs": slab_hashes,
        },
        "contract": {
            "target": "delta_R7 = lambda1 + lambda2 + lambda3",
            "target_epoch": 0.2,
            "smoothing_mpc_h": 7.0,
            "double_smoothing_applied": False,
            "observer_origin_mpc_h": [float(args.observer_origin_mpc_h)] * 3,
            "observer_coordinate_units": "Mpc",
            "observer_coordinate_h": coordinate_h,
            "coordinate_mapping": (
                "(P3 voxel centre in Mpc * h + observer origin in Mpc/h) modulo box"
            ),
            "source_sampling": "floor to ngrid=2048 source cell",
            "science_redshift_range": [float(args.z_min), float(args.z_max)],
            "science_support_coordinate_units": "observer-frame Mpc",
            "minimum_exposure_apodized": minimum_exposure,
            "density_loss_support": "science_support AND nominal P4 core coverage",
            "privileged_target_is_model_input": False,
        },
        "components": cap_reports,
        "global_support": {
            "science_supported_voxels": int(all_supported),
            "covered_science_voxels": int(all_covered),
            "uncovered_science_voxels": int(all_supported - all_covered),
            "coverage_fraction": coverage_fraction,
        },
        "gates": {
            "coordinate_preflight_passed": True,
            "all_target_voxels_finite": bool(target_ready),
            "all_target_x_planes_written_once_or_more": True,
            "all_supported_voxels_have_core_coverage": bool(stitching_ready),
        },
        "target_fields_ready": bool(target_ready),
        "stitching_support_ready": bool(stitching_ready),
        "pass": bool(target_ready and stitching_ready),
        "interpretation": (
            "target_fields_ready authorizes target-closure testing. "
            "stitching_support_ready is independently required before complete-cap "
            "prediction and the global FFT. A support failure must not be repaired "
            "by silent zero fill or classical substitution. The target builder's "
            "intersecting P4 coverage is diagnostic; exact density-loss/output "
            "ownership is frozen separately by p8_build_field_output_tiling.py."
        ),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output / "target_manifest.json"
    atomic_json(manifest_path, manifest)
    (args.output / "DENSITY_TARGET_FIELDS_READY").write_text(
        f"manifest_sha256={sha256(manifest_path)}\n"
        f"target_fields_ready={target_ready}\n"
        f"stitching_support_ready={stitching_ready}\n"
    )
    if stitching_ready:
        (args.output / "DENSITY_TARGET_SUPPORT_READY").write_text(
            f"manifest_sha256={sha256(manifest_path)}\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
