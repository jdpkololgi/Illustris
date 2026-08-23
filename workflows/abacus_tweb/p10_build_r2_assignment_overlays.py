#!/usr/bin/env python3
"""Project frozen P10 R2 assignment response onto the canonical P3a grids.

The R1 random-response overlay remains the immutable source for BRIGHT counts,
mask/exposure, and log-count ratio.  This builder adds the audited assignment
response without copying those arrays.  Pixels with no target-derived response
retain the registered neutral value and an explicit ``c_fibre_defined=0`` flag.

Only ph000 and ph002--ph006 are accepted.  No blind ph001 product can be built
by this development-stage command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import healpy as hp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p3a_build_canonical_fields import (
    GridSpec,
    coordinate_block,
    git_sha,
    iter_chunks,
    sha256,
)
from workflows.abacus_tweb.p10_training_contract import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
ANGULAR_ROOT = ROOT / "r2_assignment_response_v1"
PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
CAPS = {"NGC": 1, "SGC": 0}
NSIDE = 256
R1_VIRTUAL_CHANNELS = (
    "counts",
    "los_x",
    "los_y",
    "los_z",
    "support_random",
    "angular_response",
    "exposure_apodized_random",
    "expected_counts_random",
    "log_count_ratio_random",
    "distance_to_support_boundary",
    "ntilde_mpc3",
    "exposure_binary",
    "exposure_apodized",
    "expected_counts",
    "log_count_ratio",
)
R2_STORED_CHANNELS = (
    "c_fibre_defined",
    "c_fibre_tileloc",
    "c_fibre_tiles",
    "c_fibre_product",
    "c_z",
    "c_z_informative",
)
R2_MODEL_CHANNELS = (
    "counts",
    "exposure_apodized",
    "log_count_ratio",
    "c_fibre_tileloc",
    "c_fibre_tiles",
    "c_fibre_defined",
)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def grid_spec(record: dict) -> GridSpec:
    grid = record["grid"]
    return GridSpec(
        origin=tuple(float(value) for value in grid["origin_mpc"]),
        shape=tuple(int(value) for value in grid["shape"]),
        cell_mpc=float(grid["cell_mpc"]),
        padding_mpc=float(grid["padding_mpc"]),
    )


def angular_pixels(
    spec: GridSpec, slices: tuple[slice, slice, slice], nside: int = NSIDE
) -> np.ndarray:
    gx, gy, gz = coordinate_block(spec, slices, halo=0)
    shape = (gx.shape[0], gy.shape[1], gz.shape[2])
    xx = np.broadcast_to(gx, shape)
    yy = np.broadcast_to(gy, shape)
    zz = np.broadcast_to(gz, shape)
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    safe = np.maximum(radius, 1.0e-12)
    return hp.vec2pix(nside, xx / safe, yy / safe, zz / safe, nest=False)


def virtual_dataset(
    output: h5py.File,
    source_path: Path,
    source: h5py.File,
    name: str,
) -> None:
    dataset = source[name]
    layout = h5py.VirtualLayout(shape=dataset.shape, dtype=dataset.dtype)
    layout[:] = h5py.VirtualSource(str(source_path), name, shape=dataset.shape)
    output.create_virtual_dataset(name, layout, fillvalue=0)


def build_component(root: Path, angular_root: Path, phase: str, cap: str) -> dict:
    if phase not in PHASES:
        raise ValueError(f"unsupported or sealed phase: {phase}")
    if cap not in CAPS:
        raise ValueError(cap)

    r1_manifest_path = root / phase / "p3b_random_response_v1/manifest.json"
    r1_manifest = load_json(r1_manifest_path)
    if not r1_manifest.get("pass") or r1_manifest.get("ph001_opened"):
        raise RuntimeError(f"{phase}: R1 response overlay is not freezeable")
    r1_record = r1_manifest["components"][cap]
    r1_path = Path(r1_record["file"])
    spec = grid_spec(r1_record)

    angular_path = angular_root / phase / "assignment_response_angular.npz"
    angular_report_path = angular_root / phase / "assignment_response_angular.json"
    angular_report = load_json(angular_report_path)
    if not angular_report.get("pass") or angular_report.get("blind_phase_opened"):
        raise RuntimeError(f"{phase}: angular assignment policy is not freezeable")
    with np.load(angular_path) as archive:
        maps = {name: np.asarray(archive[name]) for name in archive.files}
    if maps["support"].size != hp.nside2npix(NSIDE):
        raise ValueError(f"{phase}: unexpected angular-map size")

    cap_id = CAPS[cap]
    owned_cap = (maps["domain"] // 2) == cap_id
    output_dir = root / phase / "p3c_assignment_response_v1" / cap
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "response_overlay.h5"
    qa_path = output_dir / "qa.json"
    partial = output_path.with_suffix(".partial.h5")
    if output_path.exists() and qa_path.exists():
        existing = load_json(qa_path)
        if existing.get("pass"):
            return existing
        raise RuntimeError(f"refusing to replace failing completed artifact: {output_path}")
    if partial.exists():
        with h5py.File(partial, "r") as handle:
            identity = (
                handle.attrs.get("schema_version") == "p10-r2-assignment-overlay-v1"
                and handle.attrs.get("phase") == phase
                and handle.attrs.get("cap") == cap
            )
        if not identity:
            raise RuntimeError(f"ambiguous interrupted artifact: {partial}")
        partial.unlink()

    chunk_shape = None
    finite = True
    range_valid = True
    outside_zero = True
    undefined_neutral = True
    defined_voxels = 0
    supported_voxels = 0
    with h5py.File(r1_path, "r") as r1, h5py.File(partial, "w", libver="latest") as out:
        chunk_shape = tuple(int(value) for value in r1["support_random"].chunks)
        out.attrs.update(
            {
                "schema_version": "p10-r2-assignment-overlay-v1",
                "phase": phase,
                "cap": cap,
                "cap_id": cap_id,
                "origin_mpc": spec.origin,
                "shape": spec.shape,
                "cell_mpc": spec.cell_mpc,
                "axis_order": "ix,iy,iz",
                "angular_nside": NSIDE,
                "angular_ordering": "RING",
            }
        )
        for name in R1_VIRTUAL_CHANNELS:
            virtual_dataset(out, r1_path, r1, name)
        dtype_by_name = {
            "c_fibre_defined": "u1",
            "c_fibre_tileloc": "f4",
            "c_fibre_tiles": "f4",
            "c_fibre_product": "f4",
            "c_z": "f4",
            "c_z_informative": "u1",
        }
        datasets = {
            name: out.create_dataset(
                name,
                shape=spec.shape,
                dtype=dtype_by_name[name],
                chunks=chunk_shape,
                compression="lzf",
                shuffle=True,
                fillvalue=0,
            )
            for name in R2_STORED_CHANNELS
        }
        for slices in iter_chunks(spec.shape, chunk_shape):
            pix = angular_pixels(spec, slices)
            support = np.asarray(r1["support_random"][slices], dtype=bool)
            cap_ok = owned_cap[pix]
            if np.any(support & ~cap_ok):
                raise RuntimeError(f"{phase} {cap}: R1 support crosses angular cap ownership")
            values = {
                "c_fibre_defined": (
                    maps["c_fibre_defined"][pix].astype(bool) & support
                ).astype(np.uint8),
                "c_fibre_tileloc": (
                    maps["c_fibre_tileloc"][pix].astype(np.float32) * support
                ),
                "c_fibre_tiles": (
                    maps["c_fibre_tiles"][pix].astype(np.float32) * support
                ),
                "c_fibre_product": (
                    maps["c_fibre_product"][pix].astype(np.float32) * support
                ),
                "c_z": support.astype(np.float32),
                "c_z_informative": np.zeros(support.shape, dtype=np.uint8),
            }
            defined = values["c_fibre_defined"].astype(bool)
            undefined = support & ~defined
            supported_voxels += int(support.sum())
            defined_voxels += int(defined.sum())
            finite &= all(np.isfinite(value).all() for value in values.values())
            for name in ("c_fibre_tileloc", "c_fibre_tiles", "c_fibre_product", "c_z"):
                value = values[name]
                range_valid &= bool(np.all((value >= 0.0) & (value <= 1.0)))
                outside_zero &= bool(np.all(value[~support] == 0.0))
            for name in ("c_fibre_tileloc", "c_fibre_tiles", "c_fibre_product"):
                undefined_neutral &= bool(np.all(values[name][undefined] == 1.0))
            for name, value in values.items():
                if np.any(value):
                    datasets[name][slices] = value
        out.flush()
    partial.replace(output_path)

    with h5py.File(output_path, "r") as check:
        virtual_identity = all(check[name].is_virtual for name in R1_VIRTUAL_CHANNELS)
        schema_complete = set(R1_VIRTUAL_CHANNELS + R2_STORED_CHANNELS) <= set(check)
    gates = {
        "r1_channels_are_virtual_identity_views": bool(virtual_identity),
        "schema_complete": bool(schema_complete),
        "all_arrays_finite": bool(finite),
        "response_constrained_0_1": bool(range_valid),
        "response_zero_outside_support": bool(outside_zero),
        "undefined_response_is_neutral_with_flag": bool(undefined_neutral),
        "supported_voxels_nonzero": supported_voxels > 0,
        "defined_voxels_nonzero": defined_voxels > 0,
    }
    qa = {
        "schema_version": "p10-r2-assignment-overlay-qa-v1",
        "phase": phase,
        "cap": cap,
        "blind_phase_opened": False,
        "file": str(output_path),
        "file_sha256": sha256(output_path),
        "file_bytes": output_path.stat().st_size,
        "r1_overlay": str(r1_path),
        "r1_overlay_sha256": r1_record["file_sha256"],
        "r1_manifest": str(r1_manifest_path),
        "r1_manifest_sha256": sha256(r1_manifest_path),
        "angular_response": str(angular_path),
        "angular_response_sha256": sha256(angular_path),
        "angular_report_sha256": sha256(angular_report_path),
        "grid": spec.as_dict(),
        "model_channels": list(R2_MODEL_CHANNELS),
        "stored_response_channels": list(R2_STORED_CHANNELS),
        "supported_voxels": supported_voxels,
        "defined_voxels": defined_voxels,
        "defined_fraction": float(defined_voxels / max(supported_voxels, 1)),
        "undefined_policy": "neutral_no_competition_value_1_plus_defined_flag",
        "spatial_interpolation": "none",
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    atomic_json(qa_path, qa)
    if not qa["pass"]:
        raise RuntimeError(f"{phase} {cap}: R2 overlay gates failed: {gates}")
    return qa


def aggregate(root: Path, phases: tuple[str, ...]) -> dict:
    records = {}
    for phase in phases:
        records[phase] = {}
        for cap in CAPS:
            qa_path = root / phase / "p3c_assignment_response_v1" / cap / "qa.json"
            if not qa_path.is_file():
                raise FileNotFoundError(qa_path)
            qa = load_json(qa_path)
            if not qa.get("pass") or qa.get("blind_phase_opened"):
                raise RuntimeError(f"{phase} {cap}: R2 overlay is not freezeable")
            records[phase][cap] = {
                "file": qa["file"],
                "file_sha256": qa["file_sha256"],
                "qa": str(qa_path),
                "qa_sha256": sha256(qa_path),
                "defined_fraction": qa["defined_fraction"],
                "grid": qa["grid"],
            }
        manifest = {
            "schema_version": "p10-r2-assignment-overlay-manifest-v1",
            "phase": phase,
            "blind_phase_opened": False,
            "components": records[phase],
            "model_channels": list(R2_MODEL_CHANNELS),
            "stored_response_channels": list(R2_STORED_CHANNELS),
            "creation_commit": git_sha(REPO_ROOT),
            "pass": True,
        }
        manifest_path = root / phase / "p3c_assignment_response_v1/manifest.json"
        atomic_json(manifest_path, manifest)
        records[phase]["manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        }
    ready = {
        "schema_version": "p10-r2-assignment-overlays-ready-v1",
        "phases": list(phases),
        "blind_phase_opened": False,
        "model_channels": list(R2_MODEL_CHANNELS),
        "phase_records": records,
        "loader_ready": False,
        "throughput_canary_pass": False,
        "view_ladder_marker_written": False,
        "pass": True,
    }
    marker = root / "r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json"
    atomic_json(marker, ready)
    return {"path": str(marker), "sha256": sha256(marker), **ready}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    component = sub.add_parser("component")
    component.add_argument("--phase", required=True, choices=PHASES)
    component.add_argument("--cap", required=True, choices=tuple(CAPS))
    component.add_argument("--root", type=Path, default=ROOT)
    component.add_argument("--angular-root", type=Path, default=ANGULAR_ROOT)
    freeze = sub.add_parser("aggregate")
    freeze.add_argument("--root", type=Path, default=ROOT)
    freeze.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    args = parser.parse_args()

    if args.command == "component":
        result = build_component(args.root, args.angular_root, args.phase, args.cap)
    else:
        result = aggregate(args.root, tuple(args.phases))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
