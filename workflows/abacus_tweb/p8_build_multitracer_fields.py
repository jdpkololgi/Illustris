#!/usr/bin/env python3
"""Build tracer-separated BGS_FAINT field overlays on the frozen P3 grids.

The immutable P3 BGS_BRIGHT fields are referenced, never copied or modified.
For each multitracer catalogue this stage deposits Faint context galaxies on
the exact NGC/SGC P3 lattices and constructs a Faint-specific effective
exposure.  The exposure is the frozen geometric P3 exposure multiplied by a
regularized HEALPix estimate of the Faint selected/target response.

Expected counts and contrasts are deliberately not baked into this artifact:
the Faint radial selection curve is fitted separately for every P4 rotation
using training folds only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from desitarget.targetmask import bgs_mask
import fitsio
import h5py
import healpy as hp
import numpy as np

from workflows.abacus_tweb.p3a_build_canonical_fields import (
    GridSpec,
    cic_deposit,
    coordinate_block,
)


FAINT_BITS = int(
    bgs_mask.BGS_FAINT
    | bgs_mask.BGS_FAINT_HIP
    | bgs_mask.BGS_FAINT_NORTH
    | bgs_mask.BGS_FAINT_SOUTH
)
CAPS = ((1, "NGC"), (0, "SGC"))
Z_CONTEXT = (0.10, 0.60)
Z_SENTINEL = (0.585, 0.595)
DEFAULT_STAGE3 = Path(
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322"
)
DEFAULT_CATALOGUES = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/catalogues"
)
DEFAULT_P3 = Path(
    "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/fields"
)


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue-root", type=Path, default=DEFAULT_CATALOGUES)
    parser.add_argument("--p3-manifest", type=Path, default=DEFAULT_P3)
    parser.add_argument(
        "--target-input", type=Path, default=DEFAULT_STAGE3 / "inputs/targ.fits"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--products",
        nargs="+",
        default=("bf_oracle_assigned_v1", "bf_proxy_response_v1"),
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--response-prior-targets", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def context_redshift(redshift: np.ndarray) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=np.float64)
    return (
        np.isfinite(redshift)
        & (redshift >= Z_CONTEXT[0])
        & (redshift < Z_CONTEXT[1])
        & ~((redshift >= Z_SENTINEL[0]) & (redshift < Z_SENTINEL[1]))
    )


def pixel_from_radec(ra: np.ndarray, dec: np.ndarray, nside: int) -> np.ndarray:
    return hp.ang2pix(
        nside,
        np.asarray(ra, dtype=np.float64),
        np.asarray(dec, dtype=np.float64),
        lonlat=True,
        nest=False,
    )


def pixel_from_xyz(xyz: np.ndarray, nside: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    if np.any(radius <= 0):
        raise RuntimeError("non-positive Cartesian radius")
    return hp.vec2pix(
        nside,
        xyz[:, 0] / radius,
        xyz[:, 1] / radius,
        xyz[:, 2] / radius,
        nest=False,
    )


def estimate_angular_response(
    target_pixel: np.ndarray,
    selected_pixel: np.ndarray,
    *,
    nside: int,
    prior_targets: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return regularized selected/target response without using labels."""
    npix = hp.nside2npix(nside)
    target = np.bincount(target_pixel, minlength=npix).astype(np.int64)
    selected = np.bincount(selected_pixel, minlength=npix).astype(np.int64)
    if np.any(selected > target):
        offending = int(np.count_nonzero(selected > target))
        raise RuntimeError(f"selected Faint rows exceed target rows in {offending} pixels")
    if target.sum() == 0:
        raise RuntimeError("no Faint target support")
    mean = float(selected.sum() / target.sum())
    response = (selected + prior_targets * mean) / (target + prior_targets)
    response[target == 0] = 0.0
    response = np.clip(response, 0.0, 1.0).astype(np.float32)
    supported = target > 0
    summary = {
        "target_rows": int(target.sum()),
        "selected_rows": int(selected.sum()),
        "global_selected_over_target": mean,
        "supported_pixels": int(np.count_nonzero(supported)),
        "response_prior_targets": float(prior_targets),
        "response_supported_min": float(response[supported].min()),
        "response_supported_median": float(np.median(response[supported])),
        "response_supported_max": float(response[supported].max()),
    }
    return response, target, selected, summary


def read_faint_targets(path: Path, nside: int) -> tuple[np.ndarray, np.ndarray]:
    table = fitsio.read(
        str(path), columns=["BGS_TARGET", "RA", "DEC", "RSDZ"]
    )
    faint = (np.asarray(table["BGS_TARGET"], dtype=np.int64) & FAINT_BITS) != 0
    faint &= context_redshift(table["RSDZ"])
    pixel = pixel_from_radec(table["RA"][faint], table["DEC"][faint], nside)
    # Sign of Galactic latitude from the ICRS-to-Galactic rotation matrix.
    ra = np.deg2rad(np.asarray(table["RA"][faint], dtype=np.float64))
    dec = np.deg2rad(np.asarray(table["DEC"][faint], dtype=np.float64))
    cos_dec = np.cos(dec)
    galactic_z = (
        -0.8676661490190047 * cos_dec * np.cos(ra)
        - 0.1980763734312015 * cos_dec * np.sin(ra)
        + 0.4559837761750669 * np.sin(dec)
    )
    cap = (galactic_z > 0).astype(np.uint8)
    return pixel.astype(np.int64), cap


def grid_spec(component: dict) -> GridSpec:
    grid = component["grid"]
    return GridSpec(
        origin=tuple(float(value) for value in grid["origin_mpc"]),
        shape=tuple(int(value) for value in grid["shape"]),
        cell_mpc=float(grid["cell_mpc"]),
        padding_mpc=float(grid["padding_mpc"]),
    )


def complete_cic_support(xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Return points whose complete eight-cell CIC stencil lies on the grid."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    origin = np.asarray(spec.origin, dtype=np.float64)
    shape = np.asarray(spec.shape, dtype=np.int64)
    fractional = (xyz - origin) / spec.cell_mpc - 0.5
    lower = np.floor(fractional).astype(np.int64)
    return np.all((lower >= 0) & ((lower + 1) < shape), axis=1)


def build_cap(
    *,
    cap_id: int,
    cap_name: str,
    faint_points: np.ndarray,
    response: np.ndarray,
    p3_component: dict,
    nside: int,
    output: Path,
) -> dict:
    spec = grid_spec(p3_component)
    input_rows = int(len(faint_points))
    field_supported = complete_cic_support(faint_points[:, :3], spec)
    excluded_rows = int(np.count_nonzero(~field_supported))
    field_points = faint_points[field_supported]
    counts = np.zeros(spec.shape, dtype=np.float32)
    counts, deposition = cic_deposit(field_points[:, :3], spec, out=counts)
    path = output / f"{cap_name.lower()}_faint_overlay.h5"
    temporary = path.with_suffix(".partial.h5")
    if temporary.exists():
        temporary.unlink()
    chunks = tuple(min(64, size) for size in spec.shape)
    with h5py.File(p3_component["file"], "r") as base, h5py.File(temporary, "w") as handle:
        datasets = {
            "counts": handle.create_dataset(
                "counts", data=counts, chunks=chunks, compression="lzf", shuffle=True
            ),
            "response_angular": handle.create_dataset(
                "response_angular", shape=spec.shape, dtype="f4", chunks=chunks,
                compression="lzf", shuffle=True, fillvalue=0,
            ),
            "exposure_apodized": handle.create_dataset(
                "exposure_apodized", shape=spec.shape, dtype="f4", chunks=chunks,
                compression="lzf", shuffle=True, fillvalue=0,
            ),
            "exposure_binary": handle.create_dataset(
                "exposure_binary", shape=spec.shape, dtype="u1", chunks=chunks,
                compression="lzf", shuffle=True, fillvalue=0,
            ),
        }
        for selection in base["exposure_apodized"].iter_chunks():
            gx, gy, gz = coordinate_block(spec, selection)
            shape = (gx.shape[0], gy.shape[1], gz.shape[2])
            xx = np.broadcast_to(gx, shape)
            yy = np.broadcast_to(gy, shape)
            zz = np.broadcast_to(gz, shape)
            radius = np.sqrt(xx * xx + yy * yy + zz * zz)
            safe = np.maximum(radius, 1.0e-12)
            pixel = hp.vec2pix(
                nside, xx / safe, yy / safe, zz / safe, nest=False
            )
            angular = response[pixel]
            base_apodized = np.asarray(base["exposure_apodized"][selection], dtype=np.float32)
            base_binary = np.asarray(base["exposure_binary"][selection], dtype=bool)
            effective = base_apodized * angular
            datasets["response_angular"][selection] = angular
            datasets["exposure_apodized"][selection] = effective
            datasets["exposure_binary"][selection] = (
                base_binary & (angular > 0)
            ).astype(np.uint8)
        for name, dataset in datasets.items():
            dataset.attrs["tracer"] = "BGS_FAINT"
            dataset.attrs["cap"] = cap_name
            dataset.attrs["units"] = "galaxies" if name == "counts" else "dimensionless"
        handle.attrs["schema_version"] = "p8-faint-field-overlay-v1"
        handle.attrs["cap_id"] = cap_id
        handle.attrs["cell_mpc"] = spec.cell_mpc
        handle.attrs["origin_mpc"] = spec.origin
    temporary.replace(path)
    deposited = float(np.sum(counts, dtype=np.float64))
    gates = {
        "shape_matches_p3": tuple(counts.shape) == spec.shape,
        "cic_conserved": abs(deposited - len(field_points)) <= max(1.0e-3, len(field_points) * 2.0e-6),
        "no_cic_loss": float(deposition["lost_weight"]) <= 1.0e-6,
        "grid_edge_exclusion_below_5e5": (
            excluded_rows / max(input_rows, 1) <= 5.0e-5
        ),
        "finite_counts": bool(np.all(np.isfinite(counts))),
        "nonempty_exposure": False,
        "exposure_bounded": False,
    }
    with h5py.File(path, "r") as handle:
        maximum, total = 0.0, 0.0
        for selection in handle["exposure_apodized"].iter_chunks():
            block = np.asarray(handle["exposure_apodized"][selection])
            maximum = max(maximum, float(block.max(initial=0.0)))
            total += float(block.sum(dtype=np.float64))
        gates["nonempty_exposure"] = total > 0
        gates["exposure_bounded"] = maximum <= 1.0 + 1.0e-6
    if not all(gates.values()):
        raise RuntimeError(f"{cap_name} Faint field gates failed: {gates}")
    return {
        "cap_id": cap_id,
        "cap_name": cap_name,
        "file": str(path),
        "file_sha256": sha256(path),
        "grid": spec.as_dict(),
        "faint_context_rows": input_rows,
        "faint_field_rows": int(len(field_points)),
        "grid_edge_excluded_rows": excluded_rows,
        "grid_edge_excluded_fraction": excluded_rows / max(input_rows, 1),
        "counts_sum": deposited,
        "deposition": deposition,
        "effective_exposure_sum": total,
        "effective_exposure_max": maximum,
        "gates": gates,
    }


def build_product(
    *,
    product: str,
    catalogue_root: Path,
    target_pixel: np.ndarray,
    target_cap: np.ndarray,
    p3_manifest_path: Path,
    p3: dict,
    output_root: Path,
    nside: int,
    prior_targets: float,
    force: bool,
) -> dict:
    source = catalogue_root / product
    source_manifest = json.loads((source / "manifest.json").read_text())
    output = output_root / product
    marker = output / "FIELD_OVERLAY_COMPLETE"
    manifest_path = output / "manifest.json"
    if marker.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output.glob("*"):
            if path.is_file():
                path.unlink()

    points = np.load(source_manifest["points"], mmap_mode="r")
    index = np.load(source_manifest["index"])
    bright_rows = int(source_manifest["bright_prefix_rows"])
    faint_mask = (
        np.asarray(index["tracer_type"][bright_rows:], dtype=np.uint8) == 1
    ) & np.asarray(index["context"][bright_rows:], dtype=bool)
    faint_points = np.asarray(points[bright_rows:][faint_mask], dtype=np.float64)
    faint_cap = np.asarray(faint_points[:, 3], dtype=np.uint8)
    selected_pixel = pixel_from_xyz(faint_points[:, :3], nside)

    components, responses = {}, {}
    for cap_id, cap_name in CAPS:
        response, target_count, selected_count, summary = estimate_angular_response(
            target_pixel[target_cap == cap_id],
            selected_pixel[faint_cap == cap_id],
            nside=nside,
            prior_targets=prior_targets,
        )
        np.savez_compressed(
            output / f"{cap_name.lower()}_faint_response.npz",
            response=response,
            target_count=target_count,
            selected_count=selected_count,
        )
        responses[cap_name] = summary
        components[cap_name] = build_cap(
            cap_id=cap_id,
            cap_name=cap_name,
            faint_points=faint_points[faint_cap == cap_id],
            response=response,
            p3_component=p3["components"][cap_name],
            nside=nside,
            output=output,
        )
    gates = {
        "catalogue_complete": bool(source_manifest["pass"]),
        "bright_grid_reused": True,
        "all_cap_gates": all(
            all(component["gates"].values()) for component in components.values()
        ),
        "both_caps": set(components) == {"NGC", "SGC"},
    }
    manifest = {
        "schema_version": "p8-multitracer-field-v1",
        "product": product,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "catalogue_manifest": str(source / "manifest.json"),
        "catalogue_manifest_sha256": sha256(source / "manifest.json"),
        "bright_p3_manifest": str(p3_manifest_path),
        "bright_p3_manifest_sha256": sha256(p3_manifest_path),
        "tracer_contract": {
            "BGS_BRIGHT": "frozen P3 fields referenced unchanged",
            "BGS_FAINT": "separate counts and response exposure overlay",
            "combined_count_channel": False,
            "rotation_selection_fit_required": True,
        },
        "response": {
            "nside": nside,
            "nest": False,
            "estimator": "regularized selected-over-target HEALPix ratio",
            "prior_targets": prior_targets,
            "caps": responses,
        },
        "components": components,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    if not manifest["pass"]:
        raise RuntimeError(f"{product} field manifest failed: {gates}")
    atomic_json(manifest_path, manifest)
    marker.write_text(
        f"product={product}\nmanifest_sha256={sha256(manifest_path)}\n"
    )
    return manifest


def main() -> None:
    args = parse_args()
    for path in (args.catalogue_root, args.p3_manifest, args.target_input):
        if not path.exists():
            raise FileNotFoundError(path)
    p3 = json.loads(args.p3_manifest.read_text())
    target_pixel, target_cap = read_faint_targets(args.target_input, args.nside)
    summaries = {}
    for product in args.products:
        summaries[product] = build_product(
            product=product,
            catalogue_root=args.catalogue_root,
            target_pixel=target_pixel,
            target_cap=target_cap,
            p3_manifest_path=args.p3_manifest,
            p3=p3,
            output_root=args.output_root,
            nside=args.nside,
            prior_targets=args.response_prior_targets,
            force=args.force,
        )
    summary = {
        "schema_version": "p8-multitracer-field-build-v1",
        "products": {
            name: str(args.output_root / name / "manifest.json") for name in summaries
        },
        "all_pass": all(item["pass"] for item in summaries.values()),
    }
    atomic_json(args.output_root / "field_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
