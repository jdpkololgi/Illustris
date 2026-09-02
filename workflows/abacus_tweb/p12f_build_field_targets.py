#!/usr/bin/env python3
"""Build visible-phase cap-aligned ``delta_R7`` targets for P12-F.

The P10 T-web products already contain the eigenvalues of the R=7 Mpc/h
smoothed tidal tensor.  Their trace is therefore the correctly smoothed matter
contrast and must *not* be smoothed a second time.  This builder samples that
trace on each phase's immutable P3 lattice and freezes exact random-catalogue
support as the admissible training/evaluation mask.

The products are privileged targets.  They are never model inputs.  ph001 is
rejected unconditionally; it remains sealed until the field-posterior method,
response contract, diagnostics, and selection rules have all been frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import h5py
import numpy as np

from workflows.abacus_tweb.p8_build_density_targets import (
    CAP_ID,
    build_core_coverage,
    hdf5_options,
    periodic_axis_indices,
    radius_squared,
    trace_from_eigen_slab,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import (
    build_slab_maps,
    discover_slabs,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
ALIGNMENT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/"
    "preflight/coordinate_alignment.json"
)
VISIBLE_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
BLIND_PHASE = "ph001"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output")
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--observer-origin-mpc-h", type=float, default=-1000.0)
    parser.add_argument("--z-min", type=float, default=0.15)
    parser.add_argument("--z-max", type=float, default=0.55)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    return parser.parse_args()


def validate_visible_phase(phase: str) -> str:
    phase = str(phase)
    if phase == BLIND_PHASE or phase not in VISIBLE_PHASES:
        raise PermissionError(f"P12-F target build forbids phase {phase}")
    return phase


def _same_grid(left: dict, right: dict) -> bool:
    return (
        tuple(left["shape"]) == tuple(right["shape"])
        and np.allclose(left["origin_mpc"], right["origin_mpc"], rtol=0.0, atol=1e-9)
        and np.isclose(left["cell_mpc"], right["cell_mpc"], rtol=0.0, atol=1e-12)
    )


def _inventory_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _write_component(
    path: Path,
    *,
    delta: np.ndarray,
    support: np.ndarray,
    core_coverage: np.ndarray,
    grid: dict,
    source_tweb: Path,
    observer_origin: float,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite field target: {path}")
    chunk = tuple(min(64, int(size)) for size in delta.shape)
    options = hdf5_options(args)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("delta_r7", data=delta, dtype="f4", chunks=chunk, **options)
        handle.create_dataset(
            "science_support", data=support.astype(np.uint8), dtype="u1", chunks=chunk,
            **options,
        )
        handle.create_dataset(
            "core_coverage", data=core_coverage.astype(np.uint8), dtype="u1", chunks=chunk,
            **options,
        )
        handle.create_dataset(
            "density_loss_support", data=(support & core_coverage).astype(np.uint8),
            dtype="u1", chunks=chunk, **options,
        )
        handle.attrs["axis_order"] = "ix,iy,iz"
        handle.attrs["units"] = "dimensionless matter contrast"
        handle.attrs["target_epoch"] = 0.2
        handle.attrs["smoothing_mpc_h"] = 7.0
        handle.attrs["double_smoothing_applied"] = False
        handle.attrs["source"] = "trace of already-R7-smoothed T-web eigenvalues"
        handle.attrs["source_tweb_complete"] = str(source_tweb)
        handle.attrs["origin_mpc"] = np.asarray(grid["origin_mpc"], dtype=np.float64)
        handle.attrs["cell_mpc"] = float(grid["cell_mpc"])
        handle.attrs["observer_coordinate_h"] = float(Planck18.h)
        handle.attrs["observer_origin_mpc_h"] = np.asarray([observer_origin] * 3)
        handle.attrs["periodic_mapping"] = (
            "(observer_xyz_mpc * h + observer_origin_mpc_h) modulo 2000 Mpc/h"
        )
        handle.flush()
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    phase = validate_visible_phase(args.phase)
    phase_root = args.root / phase
    p3_path = phase_root / "p3_fields/field_manifest.json"
    response_path = phase_root / "p3b_random_response_v1/manifest.json"
    cores_path = phase_root / "p4_patches/cores.npz"
    tweb_dir = phase_root / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7"
    tweb_complete_path = tweb_dir / "TWEB_COMPLETE.json"
    output = (
        Path(args.output)
        if args.output
        else phase_root / "p12f_field_targets_v1"
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"non-empty output requires a new version: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    p3 = json.loads(p3_path.read_text())
    response = json.loads(response_path.read_text())
    tweb_complete = json.loads(tweb_complete_path.read_text())
    alignment = json.loads(args.alignment.read_text())
    if (
        not alignment.get("pass")
        or alignment.get("best_sky_variant") != "z_cosmo_origin_m1000p0"
        or not np.isclose(float(args.observer_origin_mpc_h), -1000.0)
    ):
        raise RuntimeError("frozen observer-to-box coordinate alignment did not pass")
    p3_gates = p3.get("gates", {})
    if not p3_gates or not all(bool(value) for value in p3_gates.values()):
        raise RuntimeError("P3 input contract gates do not pass")
    if not response.get("pass"):
        raise RuntimeError("P3b-R input contract does not pass")
    if response.get("ph001_opened") or response.get("phase") != phase:
        raise RuntimeError("response phase/seal contract failed")
    if tweb_complete.get("phase") != phase:
        raise RuntimeError("T-web phase mismatch")
    target_contract = tweb_complete["target_contract"]
    if (
        int(target_contract["grid_size"]) != 2048
        or float(target_contract["tidal_smoothing_mpc_h"]) != 7.0
        or target_contract["eigenvalue_order"] != "lambda1<=lambda2<=lambda3"
    ):
        raise RuntimeError("T-web target contract differs from P12-F")

    cores = np.load(cores_path, mmap_mode="r")
    core_cap = np.asarray(cores["cap"], dtype=np.int8)
    core_start = np.asarray(cores["voxel_start"], dtype=np.int64)
    core_stop = np.asarray(cores["voxel_stop"], dtype=np.int64)
    if not np.array_equal(np.asarray(cores["core_id"]), np.arange(len(core_cap))):
        raise RuntimeError("P4 core identity is not canonical row identity")

    slabs = discover_slabs(tweb_dir)
    _, _, ngrid, boxsize = build_slab_maps(slabs)
    if ngrid != 2048 or not np.isclose(boxsize, 2000.0):
        raise RuntimeError("unexpected T-web source geometry")
    slab_paths = [slab.path for slab in slabs]

    cap_state: dict[str, dict] = {}
    for cap_name, component in p3["components"].items():
        response_component = response["components"][cap_name]
        if not _same_grid(component["grid"], response_component["grid"]):
            raise RuntimeError(f"{phase} {cap_name}: P3/P3b-R grid mismatch")
        grid = component["grid"]
        shape = tuple(int(v) for v in grid["shape"])
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        source_index = tuple(
            periodic_axis_indices(
                origin[axis], shape[axis], float(grid["cell_mpc"]), float(Planck18.h),
                float(args.observer_origin_mpc_h), boxsize, ngrid,
            )
            for axis in range(3)
        )
        cap_state[cap_name] = {
            "component": component,
            "response": response_component,
            "shape": shape,
            "origin": origin,
            "source_index": source_index,
            "delta": np.empty(shape, dtype=np.float32),
            "written_x": np.zeros(shape[0], dtype=bool),
        }

    for ordinal, slab in enumerate(slabs):
        with np.load(slab.path) as archive:
            eigenvalues = archive["eig_vals"]
            for state in cap_state.values():
                x_index, y_index, z_index = state["source_index"]
                destination_x = np.flatnonzero(
                    (x_index >= slab.x_start) & (x_index < slab.x_end)
                )
                if len(destination_x) == 0:
                    continue
                local_x = x_index[destination_x] - slab.x_start
                state["delta"][destination_x] = trace_from_eigen_slab(
                    eigenvalues, local_x, y_index, z_index
                )
                state["written_x"][destination_x] = True
        print(
            f"{phase}: slab {ordinal + 1}/{len(slabs)} "
            f"x=[{slab.x_start},{slab.x_end})",
            flush=True,
        )

    r_min = float(Planck18.comoving_distance(args.z_min).value)
    r_max = float(Planck18.comoving_distance(args.z_max).value)
    components = {}
    for cap_name, state in cap_state.items():
        if not np.all(state["written_x"]):
            raise RuntimeError(f"{phase} {cap_name}: target x planes are incomplete")
        delta = state["delta"]
        if not np.all(np.isfinite(delta)):
            raise RuntimeError(f"{phase} {cap_name}: target contains non-finite values")
        grid = state["component"]["grid"]
        with h5py.File(state["response"]["file"], "r") as handle:
            support_random = np.asarray(handle["support_random"], dtype=bool)
        if support_random.shape != delta.shape:
            raise RuntimeError(f"{phase} {cap_name}: support/target shape mismatch")
        r2 = radius_squared(state["origin"], state["shape"], float(grid["cell_mpc"]))
        science_support = support_random & (r2 >= r_min**2) & (r2 < r_max**2)
        cap_mask = core_cap == CAP_ID[cap_name]
        coverage = build_core_coverage(
            state["shape"], core_start[cap_mask], core_stop[cap_mask]
        )
        path = output / f"{cap_name.lower()}_delta_r7.h5"
        _write_component(
            path,
            delta=delta,
            support=science_support,
            core_coverage=coverage,
            grid=grid,
            source_tweb=tweb_complete_path,
            observer_origin=float(args.observer_origin_mpc_h),
            args=args,
        )
        supported = int(science_support.sum())
        covered = int((science_support & coverage).sum())
        components[cap_name] = {
            "cap_id": CAP_ID[cap_name],
            "file": str(path.resolve()),
            "file_bytes": path.stat().st_size,
            "file_sha256": sha256(path),
            "grid": grid,
            "support_random_source": state["response"]["file"],
            "science_supported_voxels": supported,
            "covered_science_voxels": covered,
            "coverage_fraction": covered / supported if supported else None,
            "target": {
                "minimum": float(delta.min()),
                "maximum": float(delta.max()),
                "mean": float(delta.mean(dtype=np.float64)),
                "std": float(delta.std(dtype=np.float64)),
                "all_finite": True,
            },
        }

    manifest = {
        "schema_version": "p12f-visible-phase-field-target-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "phase": phase,
        "role": "training" if phase != "ph006" else "validation_and_selection",
        "sealed_blind_phase": BLIND_PHASE,
        "ph001_opened": False,
        "contract": {
            "target": "delta_R7 = lambda1 + lambda2 + lambda3",
            "target_epoch": 0.2,
            "smoothing_mpc_h": 7.0,
            "double_smoothing_applied": False,
            "physics_operator": "T_ij(k)=k_i*k_j/k^2*delta_R7(k); k=0 removed",
            "science_support": "support_random AND 0.15<=z<0.55",
            "M_zero_is_not_void": True,
            "privileged_target_is_model_input": False,
            "coordinate_mapping": (
                "(P3 voxel centre in Mpc * h - 1000 Mpc/h) modulo 2000 Mpc/h"
            ),
        },
        "inputs": {
            "p3_manifest": str(p3_path.resolve()),
            "p3_manifest_sha256": sha256(p3_path),
            "response_manifest": str(response_path.resolve()),
            "response_manifest_sha256": sha256(response_path),
            "p4_cores": str(cores_path.resolve()),
            "p4_cores_sha256": sha256(cores_path),
            "tweb_complete": str(tweb_complete_path.resolve()),
            "tweb_complete_sha256": sha256(tweb_complete_path),
            "coordinate_alignment": str(args.alignment.resolve()),
            "coordinate_alignment_sha256": sha256(args.alignment),
            "tweb_rank_inventory_fingerprint": _inventory_fingerprint(slab_paths),
            "tweb_rank_files": [
                {
                    "path": str(slab.path.resolve()),
                    "bytes": slab.path.stat().st_size,
                    "x_start": slab.x_start,
                    "x_end": slab.x_end,
                }
                for slab in slabs
            ],
        },
        "components": components,
        "gates": {
            "visible_registered_phase": True,
            "ph001_sealed": True,
            "p3_and_random_response_pass": True,
            "p3_response_grid_identity": True,
            "coordinate_alignment_preflight_passed": True,
            "tweb_r7_trace_no_double_smoothing": True,
            "all_targets_finite": True,
            "support_is_exact_random_support": True,
        },
        "pass": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = output / "FIELD_TARGET_READY.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
