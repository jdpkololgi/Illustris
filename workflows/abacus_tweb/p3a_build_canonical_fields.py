#!/usr/bin/env python3
"""Build canonical P3a count/response fields for the full ph000 NGC+SGC mock.

The two Galactic caps are independent Cartesian lattices.  Fields are deposited
once from the P1b context catalogue and stored as chunked HDF5 datasets.  Patch
loaders must read views of these fields; they must not re-voxelize or normalize
individual patches.

P3a deliberately uses a target-free HEALPix occupancy footprint because the
current graph-ready parent does not carry a random/exposure field.  The manifest
labels this as an approximation to be replaced by a versioned P3b response
product.  No tidal target or supervised split is used in any input channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import gaussian_filter


CAPS = ((1, "NGC"), (0, "SGC"))
ACTIVE_SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
DATASET_ORDER = (
    "counts",
    "exposure_binary",
    "exposure_apodized",
    "expected_counts",
    "log_count_ratio",
    "ntilde_mpc3",
    "los_x",
    "los_y",
    "los_z",
)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@dataclass(frozen=True)
class GridSpec:
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    cell_mpc: float
    padding_mpc: float

    def as_dict(self) -> dict:
        return {
            "origin_mpc": list(self.origin),
            "shape": list(self.shape),
            "cell_mpc": self.cell_mpc,
            "padding_mpc": self.padding_mpc,
            "voxel_count": int(np.prod(self.shape)),
        }


def grid_from_xyz(xyz: np.ndarray, cell_mpc: float, padding_mpc: float) -> GridSpec:
    lo = np.min(xyz, axis=0) - padding_mpc
    hi = np.max(xyz, axis=0) + padding_mpc
    shape = np.ceil((hi - lo) / cell_mpc).astype(np.int64)
    if (shape <= 0).any():
        raise RuntimeError(f"invalid grid shape {shape}")
    return GridSpec(tuple(float(v) for v in lo), tuple(int(v) for v in shape),
                    float(cell_mpc), float(padding_mpc))


def iter_chunks(shape: tuple[int, int, int],
                chunk: tuple[int, int, int]) -> Iterator[tuple[slice, slice, slice]]:
    for i in range(0, shape[0], chunk[0]):
        for j in range(0, shape[1], chunk[1]):
            for k in range(0, shape[2], chunk[2]):
                yield (
                    slice(i, min(i + chunk[0], shape[0])),
                    slice(j, min(j + chunk[1], shape[1])),
                    slice(k, min(k + chunk[2], shape[2])),
                )


def fractional_index(xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    return (np.asarray(xyz) - np.asarray(spec.origin)) / spec.cell_mpc - 0.5


def cic_deposit(xyz: np.ndarray, spec: GridSpec,
                out: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Mass-conserving CIC deposit in canonical (ix,iy,iz) order."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if out is None:
        out = np.zeros(spec.shape, dtype=np.float32)
    if out.shape != spec.shape or out.dtype != np.float32:
        raise ValueError("CIC destination must be float32 with the canonical shape")
    u = fractional_index(xyz, spec)
    i0 = np.floor(u).astype(np.int64)
    frac = u - i0
    deposited = 0.0
    lost = 0.0
    shape = np.asarray(spec.shape)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + np.array([dx, dy, dz], dtype=np.int64)
                weight = (
                    np.where(dx, frac[:, 0], 1.0 - frac[:, 0])
                    * np.where(dy, frac[:, 1], 1.0 - frac[:, 1])
                    * np.where(dz, frac[:, 2], 1.0 - frac[:, 2])
                )
                valid = np.all((idx >= 0) & (idx < shape), axis=1)
                np.add.at(
                    out,
                    (idx[valid, 0], idx[valid, 1], idx[valid, 2]),
                    weight[valid].astype(np.float32),
                )
                deposited += float(np.sum(weight[valid], dtype=np.float64))
                lost += float(np.sum(weight[~valid], dtype=np.float64))
    return out, {
        "input_points": int(len(xyz)),
        "deposited_weight": deposited,
        "lost_weight": lost,
    }


def cosmology_lookup() -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(0.0, 0.75, 15001, dtype=np.float64)
    r = Planck18.comoving_distance(z).value.astype(np.float64)
    return z, r


def radius_to_z(radius: np.ndarray, z_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    return np.interp(radius, r_grid, z_grid, left=z_grid[0], right=z_grid[-1])


def ntilde_at(z: np.ndarray, spline: dict) -> np.ndarray:
    grid_z = np.asarray(spline["grid_z"], dtype=np.float64)
    ntilde = np.asarray(spline["ntilde"], dtype=np.float64)
    value = np.interp(np.clip(z, grid_z[0], grid_z[-1]), grid_z, ntilde)
    return np.maximum(value, float(spline["ntilde_floor"]))


def log_count_ratio(counts: np.ndarray, expected: np.ndarray, exposure: np.ndarray,
                    epsilon: float, minimum_exposure: float) -> np.ndarray:
    result = np.zeros_like(expected, dtype=np.float32)
    valid = exposure > minimum_exposure
    result[valid] = np.log(
        (counts[valid].astype(np.float64) + epsilon)
        / (expected[valid].astype(np.float64) + epsilon)
    ).astype(np.float32)
    return result


def make_angular_support(points: np.ndarray, nside: int, min_count: int,
                         out_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int32)
    step = 1_000_000
    for start in range(0, len(points), step):
        xyz = np.asarray(points[start:start + step, :3], dtype=np.float64)
        radius = np.linalg.norm(xyz, axis=1)
        valid = radius > 0
        pix = hp.vec2pix(
            nside,
            xyz[valid, 0] / radius[valid],
            xyz[valid, 1] / radius[valid],
            xyz[valid, 2] / radius[valid],
            nest=False,
        )
        np.add.at(counts, pix, 1)
    support = counts >= min_count
    np.savez_compressed(out_path, counts=counts, support=support)
    metadata = {
        "nside": int(nside),
        "nest": False,
        "npix": int(npix),
        "supported_pixels": int(support.sum()),
        "supported_area_deg2": float(support.sum() * hp.nside2pixarea(nside, degrees=True)),
        "minimum_parent_rows_per_pixel": int(min_count),
        "parent_rows": int(len(points)),
        "median_count_supported": float(np.median(counts[support])),
        "minimum_count_supported": int(counts[support].min()),
        "maximum_count_supported": int(counts[support].max()),
        "path": str(out_path),
    }
    return support, counts, metadata


def coordinate_block(spec: GridSpec, slices: tuple[slice, slice, slice],
                     halo: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for axis, slc in enumerate(slices):
        idx = np.arange(slc.start - halo, slc.stop + halo, dtype=np.float64)
        axes.append(spec.origin[axis] + (idx + 0.5) * spec.cell_mpc)
    return np.meshgrid(*axes, indexing="ij", sparse=True)


def binary_support_block(spec: GridSpec, slices: tuple[slice, slice, slice],
                         angular_support: np.ndarray, nside: int,
                         z_grid: np.ndarray, r_grid: np.ndarray,
                         z_context: tuple[float, float],
                         sentinel: tuple[float, float],
                         halo: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy, gz = coordinate_block(spec, slices, halo=halo)
    ext_shape = (gx.shape[0], gy.shape[1], gz.shape[2])
    xx = np.broadcast_to(gx, ext_shape)
    yy = np.broadcast_to(gy, ext_shape)
    zz = np.broadcast_to(gz, ext_shape)
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    safe = np.maximum(radius, 1e-12)
    pix = hp.vec2pix(nside, xx / safe, yy / safe, zz / safe, nest=False)
    redshift = radius_to_z(radius, z_grid, r_grid)
    radial = (
        (redshift >= z_context[0])
        & (redshift < z_context[1])
        & ~((redshift >= sentinel[0]) & (redshift < sentinel[1]))
    )
    binary = radial & angular_support[pix]
    return binary, redshift, radius


def field_block(spec: GridSpec, slices: tuple[slice, slice, slice],
                counts: np.ndarray, angular_support: np.ndarray, nside: int,
                z_grid: np.ndarray, r_grid: np.ndarray, spline: dict,
                schema: dict, halo_override: int | None = None) -> dict[str, np.ndarray]:
    sigma_vox = float(schema["apodization"]["sigma_mpc"]) / spec.cell_mpc
    truncate = float(schema["apodization"]["truncate_sigma"])
    halo = int(math.ceil(sigma_vox * truncate)) if halo_override is None else halo_override
    z_context = tuple(float(v) for v in schema["radial_support"]["z_context"])
    sentinel = tuple(float(v) for v in schema["radial_support"]["sentinel_excluded"])
    binary_ext, redshift_ext, radius_ext = binary_support_block(
        spec, slices, angular_support, nside, z_grid, r_grid,
        z_context, sentinel, halo=halo,
    )
    apod_ext = gaussian_filter(
        binary_ext.astype(np.float32),
        sigma=sigma_vox,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    trim = tuple(slice(halo, halo + slc.stop - slc.start) for slc in slices)
    binary = binary_ext[trim]
    apod = apod_ext[trim].astype(np.float32)
    redshift = redshift_ext[trim]
    radius = radius_ext[trim]
    count_chunk = np.asarray(counts[slices], dtype=np.float32)
    ntilde = ntilde_at(redshift, spline).astype(np.float32)
    ntilde *= (apod > float(schema["contrast"]["minimum_exposure"]))
    expected = (
        ntilde.astype(np.float64) * spec.cell_mpc ** 3 * apod.astype(np.float64)
    ).astype(np.float32)
    contrast = log_count_ratio(
        count_chunk, expected, apod,
        float(schema["contrast"]["epsilon"]),
        float(schema["contrast"]["minimum_exposure"]),
    )
    gx, gy, gz = coordinate_block(spec, slices, halo=0)
    core_shape = count_chunk.shape
    safe = np.maximum(radius, 1e-12)
    valid = binary.astype(np.float32)
    los_x = (np.broadcast_to(gx, core_shape) / safe * valid).astype(np.float32)
    los_y = (np.broadcast_to(gy, core_shape) / safe * valid).astype(np.float32)
    los_z = (np.broadcast_to(gz, core_shape) / safe * valid).astype(np.float32)
    return {
        "counts": count_chunk,
        "exposure_binary": binary.astype(np.uint8),
        "exposure_apodized": apod,
        "expected_counts": expected,
        "log_count_ratio": contrast,
        "ntilde_mpc3": ntilde,
        "los_x": los_x,
        "los_y": los_y,
        "los_z": los_z,
        "_redshift": redshift,
    }


def create_datasets(handle: h5py.File, spec: GridSpec, schema: dict) -> dict[str, h5py.Dataset]:
    chunks = tuple(int(v) for v in schema["grid"]["chunk_shape"])
    result = {}
    for name in DATASET_ORDER:
        dtype = np.dtype(schema["datasets"][name]["dtype"])
        result[name] = handle.create_dataset(
            name,
            shape=spec.shape,
            dtype=dtype,
            chunks=chunks,
            compression=schema["grid"]["compression"],
            shuffle=bool(schema["grid"]["shuffle"]),
            fillvalue=0,
        )
        result[name].attrs["units"] = schema["datasets"][name]["units"]
    return result


def storage_probe(points: np.ndarray, cap: np.ndarray, context: np.ndarray,
                  schema: dict) -> dict:
    result = {"candidates": {}, "selected_cell_mpc": float(schema["grid"]["cell_mpc"])}
    dtype_bytes = sum(np.dtype(v["dtype"]).itemsize for v in schema["datasets"].values())
    for cell in schema["grid"]["candidate_cell_mpc_audited"]:
        cell_result = {}
        for cap_id, name in CAPS:
            xyz = np.asarray(points[context & (cap == cap_id), :3], dtype=np.float64)
            spec = grid_from_xyz(xyz, float(cell), float(schema["grid"]["padding_mpc"]))
            cell_result[name] = {
                **spec.as_dict(),
                "raw_all_channels_gb": float(np.prod(spec.shape) * dtype_bytes / 1e9),
                "context_galaxies": int(len(xyz)),
            }
        cell_result["both_caps_raw_all_channels_gb"] = float(
            sum(v["raw_all_channels_gb"] for v in cell_result.values())
        )
        result["candidates"][str(cell)] = cell_result
    return result


def shell_key(shell_id: int) -> str:
    if shell_id < 0:
        return "context_buffer"
    lo, hi = ACTIVE_SHELLS[shell_id]
    return f"{lo:.2f}_{hi:.2f}"


def build_cap(cap_id: int, name: str, points: np.ndarray, cap: np.ndarray,
              context: np.ndarray, shell: np.ndarray, angular_support: np.ndarray,
              schema: dict, spline: dict, z_grid: np.ndarray, r_grid: np.ndarray,
              out_dir: Path) -> dict:
    started = time.time()
    final_path = out_dir / f"{name.lower()}_fields.h5"
    cap_meta_path = out_dir / f"{name.lower()}_build.json"
    if final_path.exists() and cap_meta_path.exists():
        metadata = json.loads(cap_meta_path.read_text())
        if metadata.get("file_sha256") != sha256(final_path):
            raise RuntimeError(f"{name} resumable artifact checksum mismatch")
        print(f"[{name}] reusing validated cap artifact {final_path}", flush=True)
        return metadata
    partial_path = out_dir / f"{name.lower()}_fields.partial.h5"
    if final_path.exists() or partial_path.exists() or cap_meta_path.exists():
        raise RuntimeError(f"{name} has an incomplete/ambiguous prior artifact; review manually")

    ids = np.flatnonzero(context & (cap == cap_id)).astype(np.int64)
    xyz = np.asarray(points[ids, :3], dtype=np.float64)
    labels = np.asarray(shell[ids], dtype=np.int8)
    spec = grid_from_xyz(
        xyz, float(schema["grid"]["cell_mpc"]), float(schema["grid"]["padding_mpc"])
    )
    print(f"[{name}] context={len(ids):,} spec={spec.as_dict()}", flush=True)

    counts = np.zeros(spec.shape, dtype=np.float32)
    deposit_stats = {}
    for shell_id in (-1, 0, 1, 2, 3):
        selected = labels == shell_id
        if not selected.any():
            continue
        _, stat = cic_deposit(xyz[selected], spec, out=counts)
        deposit_stats[shell_key(shell_id)] = stat
    deposited_sum = float(np.sum(counts, dtype=np.float64))
    if abs(deposited_sum - len(ids)) > max(1e-3, 1e-7 * len(ids)):
        raise RuntimeError(f"{name} CIC is not conservative: {deposited_sum} vs {len(ids)}")

    chunk_shape = tuple(int(v) for v in schema["grid"]["chunk_shape"])
    shell_atlas = {
        f"{lo:.2f}_{hi:.2f}": {
            "support_voxels": 0,
            "occupied_voxels": 0,
            "expected_count_sum": 0.0,
            "input_galaxies": int(np.sum(labels == shell_id)),
        }
        for shell_id, (lo, hi) in enumerate(ACTIVE_SHELLS)
    }
    allocated_chunks = 0
    finite = True
    counts_outside_binary = 0.0
    first_materialized = None
    with h5py.File(partial_path, "w", libver="latest") as handle:
        handle.attrs["cap_id"] = cap_id
        handle.attrs["cap_name"] = name
        handle.attrs["origin_mpc"] = spec.origin
        handle.attrs["shape"] = spec.shape
        handle.attrs["cell_mpc"] = spec.cell_mpc
        handle.attrs["axis_order"] = "ix,iy,iz"
        datasets = create_datasets(handle, spec, schema)
        for chunk_index, slices in enumerate(iter_chunks(spec.shape, chunk_shape)):
            block = field_block(
                spec, slices, counts, angular_support,
                int(schema["angular_support"]["nside"]),
                z_grid, r_grid, spline, schema,
            )
            finite = finite and all(
                np.isfinite(block[key]).all()
                for key in DATASET_ORDER
                if key != "exposure_binary"
            )
            counts_outside_binary += float(
                np.sum(block["counts"][block["exposure_binary"] == 0], dtype=np.float64)
            )
            materialized = (
                np.any(block["counts"] != 0)
                or np.any(block["exposure_apodized"] > 0)
            )
            if materialized:
                for key in DATASET_ORDER:
                    datasets[key][slices] = block[key]
                allocated_chunks += 1
                if first_materialized is None:
                    first_materialized = slices
            redshift = block["_redshift"]
            for shell_id, (lo, hi) in enumerate(ACTIVE_SHELLS):
                selected = (redshift >= lo) & (redshift < hi)
                key = shell_key(shell_id)
                shell_atlas[key]["support_voxels"] += int(
                    np.sum(selected & (block["exposure_binary"] > 0))
                )
                shell_atlas[key]["occupied_voxels"] += int(
                    np.sum(selected & (block["counts"] > 0))
                )
                shell_atlas[key]["expected_count_sum"] += float(
                    np.sum(block["expected_counts"][selected], dtype=np.float64)
                )
            if chunk_index % 100 == 0:
                print(
                    f"[{name}] chunks={chunk_index:,} allocated={allocated_chunks:,} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        handle.flush()

    if not finite:
        raise RuntimeError(f"{name} generated non-finite field values")
    if first_materialized is None:
        raise RuntimeError(f"{name} produced no materialized field chunks")

    # Recompute the first materialized output with a larger analytic halo.  The
    # Gaussian kernel has fixed truncation, so the retained chunk must be stable.
    base = field_block(
        spec, first_materialized, counts, angular_support,
        int(schema["angular_support"]["nside"]), z_grid, r_grid, spline, schema,
    )["exposure_apodized"]
    sigma_vox = float(schema["apodization"]["sigma_mpc"]) / spec.cell_mpc
    halo = int(math.ceil(sigma_vox * float(schema["apodization"]["truncate_sigma"])))
    wider = field_block(
        spec, first_materialized, counts, angular_support,
        int(schema["angular_support"]["nside"]), z_grid, r_grid, spline, schema,
        halo_override=halo + 2,
    )["exposure_apodized"]
    overlap_max_abs = float(np.max(np.abs(base - wider)))

    partial_path.replace(final_path)
    readback_sum = 0.0
    with h5py.File(final_path, "r") as handle:
        for slices in iter_chunks(spec.shape, chunk_shape):
            readback_sum += float(np.sum(handle["counts"][slices], dtype=np.float64))
        dataset_shapes = {key: list(handle[key].shape) for key in DATASET_ORDER}
        dataset_dtypes = {key: str(handle[key].dtype) for key in DATASET_ORDER}

    outside_fraction = counts_outside_binary / max(deposited_sum, 1.0)
    gates = {
        "cic_conserved": abs(deposited_sum - len(ids)) <= max(1e-3, 1e-7 * len(ids)),
        "hdf5_counts_conserved": abs(readback_sum - len(ids)) <= max(1e-3, 1e-7 * len(ids)),
        "all_fields_finite": finite,
        "apodization_chunk_stable": overlap_max_abs <= 1e-6,
        "counts_outside_binary_below_2pct": outside_fraction < 0.02,
        "dataset_shapes_match": all(tuple(v) == spec.shape for v in dataset_shapes.values()),
    }
    if not all(gates.values()):
        raise RuntimeError(f"{name} P3 gates failed: {gates}")
    metadata = {
        "cap_id": cap_id,
        "cap_name": name,
        "context_galaxies": int(len(ids)),
        "grid": spec.as_dict(),
        "chunk_shape": list(chunk_shape),
        "allocated_chunks": allocated_chunks,
        "deposition_by_shell": deposit_stats,
        "counts_sum": deposited_sum,
        "counts_readback_sum": readback_sum,
        "counts_outside_binary": counts_outside_binary,
        "counts_outside_binary_fraction": outside_fraction,
        "apodization_overlap_max_abs": overlap_max_abs,
        "dataset_shapes": dataset_shapes,
        "dataset_dtypes": dataset_dtypes,
        "support_atlas": shell_atlas,
        "gates": gates,
        "file": str(final_path),
        "file_bytes": final_path.stat().st_size,
        "file_sha256": sha256(final_path),
        "elapsed_seconds": time.time() - started,
    }
    cap_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[{name}] COMPLETE {json.dumps(metadata, indent=2, sort_keys=True)}", flush=True)
    return metadata


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--points", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
                     "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"),
    )
    ap.add_argument(
        "--canonical-index", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"),
    )
    ap.add_argument(
        "--p1-manifest", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json"),
    )
    ap.add_argument(
        "--schema", type=Path,
        default=repo / "docs/evidence/p3/p3_field_schema_v1.json",
    )
    ap.add_argument(
        "--ntilde-spline", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/conditioning/"
                     "ntilde_spline_v1_frozen.json"),
    )
    ap.add_argument(
        "--unit-audit", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/"
                     "unit_audit.json"),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint"),
    )
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "field_manifest.json"
    complete_path = args.out_dir / "FIELD_COMPLETE"
    if complete_path.exists() and manifest_path.exists() and not args.probe_only:
        print(manifest_path.read_text())
        return

    schema = json.loads(args.schema.read_text())
    spline = json.loads(args.ntilde_spline.read_text())
    p1 = json.loads(args.p1_manifest.read_text())
    unit_audit = json.loads(args.unit_audit.read_text())
    if schema["catalogue_id"] != p1["catalogue_id"]:
        raise RuntimeError("P3 schema/P1 catalogue mismatch")
    if not unit_audit.get("pass", False):
        raise RuntimeError("P3 unit audit is absent or failing")
    if schema["coordinate_frame"]["units"] != "Mpc":
        raise RuntimeError("P3 coordinate frame must match passing Mpc unit audit")
    historical_cell = unit_audit["historical_unet"]["resolution"]["historical_cell_mpc"]
    if not np.isclose(float(schema["grid"]["cell_mpc"]), float(historical_cell)):
        raise RuntimeError("P3 cell does not match the audited historical U-Net cell")
    h = float(unit_audit["physical_scales"]["planck18_h"])
    if not np.isclose(float(schema["grid"]["cell_mpc_h"]), h * historical_cell):
        raise RuntimeError("P3 Mpc/Mpc-h cell conversion is inconsistent")
    points = np.load(args.points, mmap_mode="r")
    index = np.load(args.canonical_index)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    shell = np.asarray(index["shell"], dtype=np.int8)
    if points.shape != (len(cap), 4):
        raise RuntimeError(f"points/index mismatch {points.shape} versus {len(cap)}")
    if int(context.sum()) != int(p1["counts"]["context"]):
        raise RuntimeError("P1 context count does not match canonical index")
    if (np.asarray(points[:, 3], dtype=np.uint8) != cap).any():
        raise RuntimeError("P1 cap labels do not match canonical points")

    schema_runtime = args.out_dir / "p3_field_schema_v1.json"
    if schema_runtime.exists():
        if sha256(schema_runtime) != sha256(args.schema):
            raise RuntimeError("runtime P3 schema differs from frozen tracked schema")
    else:
        shutil.copy2(args.schema, schema_runtime)
    probe = storage_probe(points, cap, context, schema)
    probe.update({
        "schema": str(schema_runtime),
        "schema_sha256": sha256(schema_runtime),
        "points": str(args.points),
        "canonical_index": str(args.canonical_index),
    })
    probe_path = args.out_dir / "storage_probe.json"
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    print(json.dumps(probe, indent=2, sort_keys=True), flush=True)
    if args.probe_only:
        return

    angular_path = args.out_dir / "angular_support_nside256.npz"
    angular_meta_path = args.out_dir / "angular_support_metadata.json"
    if angular_path.exists() and angular_meta_path.exists():
        angular_data = np.load(angular_path)
        angular_support = np.asarray(angular_data["support"], dtype=bool)
        angular_meta = json.loads(angular_meta_path.read_text())
    elif angular_path.exists() or angular_meta_path.exists():
        raise RuntimeError("ambiguous partial angular support artifact")
    else:
        angular_support, _, angular_meta = make_angular_support(
            points,
            int(schema["angular_support"]["nside"]),
            int(schema["angular_support"]["minimum_parent_rows_per_pixel"]),
            angular_path,
        )
        angular_meta["sha256"] = sha256(angular_path)
        angular_meta_path.write_text(json.dumps(angular_meta, indent=2, sort_keys=True) + "\n")
    if len(angular_support) != hp.nside2npix(int(schema["angular_support"]["nside"])):
        raise RuntimeError("invalid angular support size")

    z_grid, r_grid = cosmology_lookup()
    components = {}
    for cap_id, name in CAPS:
        components[name] = build_cap(
            cap_id, name, points, cap, context, shell, angular_support,
            schema, spline, z_grid, r_grid, args.out_dir,
        )
    total_counts = sum(v["counts_readback_sum"] for v in components.values())
    gates = {
        "p1_context_count_match": abs(total_counts - int(context.sum())) <= 1e-2,
        "two_caps_present": set(components) == {"NGC", "SGC"},
        "zero_cross_cap_storage": True,
        "component_gates_pass": all(all(v["gates"].values()) for v in components.values()),
        "target_free_channels": (
            not schema["angular_support"]["target_columns_used"]
            and not schema["selection"]["target_columns_used"]
        ),
        "split_free_channels": (
            not schema["angular_support"]["split_ownership_used"]
            and not schema["selection"]["split_ownership_used"]
        ),
        "unit_audit_pass": bool(unit_audit["pass"]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"P3a global gates failed: {gates}")
    support_atlas = {
        name: metadata["support_atlas"] for name, metadata in components.items()
    }
    support_path = args.out_dir / "support_atlas.json"
    support_path.write_text(json.dumps(support_atlas, indent=2, sort_keys=True) + "\n")
    validation = {
        "gates": gates,
        "components": {
            name: {
                "gates": meta["gates"],
                "counts_sum": meta["counts_sum"],
                "counts_readback_sum": meta["counts_readback_sum"],
                "counts_outside_binary_fraction": meta["counts_outside_binary_fraction"],
                "apodization_overlap_max_abs": meta["apodization_overlap_max_abs"],
            }
            for name, meta in components.items()
        },
    }
    validation_path = args.out_dir / "validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema_version": "1.0",
        "stage": "P3a canonical full-cap fields",
        "catalogue_id": p1["catalogue_id"],
        "git_sha": git_sha(repo),
        "p1_manifest": str(args.p1_manifest),
        "p1_manifest_sha256": sha256(args.p1_manifest),
        "canonical_index": str(args.canonical_index),
        "canonical_index_sha256": sha256(args.canonical_index),
        "points": str(args.points),
        "frozen_schema": str(schema_runtime),
        "frozen_schema_sha256": sha256(schema_runtime),
        "ntilde_spline": str(args.ntilde_spline),
        "unit_audit": str(args.unit_audit),
        "unit_audit_sha256": sha256(args.unit_audit),
        "ntilde_spline_sha256": sha256(args.ntilde_spline),
        "angular_support": angular_meta,
        "components": components,
        "storage_probe": str(probe_path),
        "support_atlas": str(support_path),
        "validation_report": str(validation_path),
        "gates": gates,
        "channel_order": list(DATASET_ORDER),
        "p3a_limitations": [
            "angular exposure is inferred from parent galaxy occupancy, not randoms",
            "no per-object completeness channel",
            "no luminosity-weighted count channel",
        ],
        "elapsed_seconds": time.time() - started,
    }
    temporary_manifest = args.out_dir / "field_manifest.partial.json"
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    complete_path.write_text(
        f"P3a catalogue={p1['catalogue_id']} context={int(context.sum())} "
        f"caps=NGC,SGC cell_mpc={schema['grid']['cell_mpc']} "
        f"manifest_sha256={sha256(manifest_path)}\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
