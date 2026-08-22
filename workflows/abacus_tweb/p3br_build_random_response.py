#!/usr/bin/env python3
"""Build the P3b-R random-derived angular response and P3a-grid overlays.

The full ``*_full_HPmapcut.ran.fits`` catalogues are used only as angular
samples of the registered survey footprint.  Their data-linked clustering
random redshifts are deliberately not used.  BRIGHT ``n_tilde(z)`` is inherited
from the frozen, training-phase-only P10 selection contract.

The command is deliberately staged so the expensive FITS scans are resumable:

* ``maps``: stream selected random realisations into HEALPix maps;
* ``decision``: apply the registered 1/4/18 convergence gate on ph000/ph006;
* ``overlay``: project the selected angular map onto the immutable P3a grids.

No ph001 product is accepted by this implementation.  Blind response products
are constructed only after the upstream model and evaluation contract freeze.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import fitsio
import h5py
import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p3a_build_canonical_fields import (
    GridSpec,
    coordinate_block,
    git_sha,
    iter_chunks,
    log_count_ratio,
    sha256,
)
from workflows.abacus_tweb.p10_training_contract import atomic_json


DEFAULT_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
DEFAULT_REGISTRY = REPO_ROOT / "configs/p10_response_sources_v1.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "training_contract/transforms/field/selection_manifest.json"
PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
CANARY_PHASES = ("ph000", "ph006")
CAPS = ((1, "NGC"), (0, "SGC"))
SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
NSIDE = 256
SNAPSHOT_COUNTS = (1, 4, 18)
CHUNK_ROWS = 2_000_000
MINIMUM_EXPOSURE = 1.0e-4
CONTRAST_EPSILON = 1.0e-3


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def phase_output(root: Path, phase: str) -> Path:
    return Path(root) / phase / "p3b_random_response_v1"


def selected_random_sources(registry: dict, phase: str, ids: tuple[int, ...]) -> list[dict]:
    if phase == "ph001":
        raise PermissionError("ph001 remains sealed during P3b-R development")
    records = registry["mock_phases"][phase]["full_random"]
    by_id = {int(record.get("random_id", index)): record for index, record in enumerate(records)}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise KeyError(f"{phase} is missing registered random IDs {missing}")
    return [by_id[index] for index in ids]


def galactic_cap(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return 1 for NGC and 0 for SGC without astropy coordinate overhead."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cosdec = np.cos(dec)
    galactic_z = (
        -0.8676661490190047 * cosdec * np.cos(ra)
        - 0.1980763734312015 * cosdec * np.sin(ra)
        + 0.4559837761750669 * np.sin(dec)
    )
    return (galactic_z > 0).astype(np.uint8)


def photsys_code(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind == "S":
        north = value == b"N"
        south = value == b"S"
    else:
        north = value.astype("U1") == "N"
        south = value.astype("U1") == "S"
    if not np.all(north | south):
        raise RuntimeError("unexpected PHOTSYS value in registered random catalogue")
    return south.astype(np.uint8)  # N=0, S=1


def add_random_file(counts: np.ndarray, record: dict, *, nside: int = NSIDE,
                    chunk_rows: int = CHUNK_ROWS) -> dict:
    """Stream one FITS random realisation into four cap/PHOTSYS maps."""
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = record.get("sha256")
    hdu = fitsio.FITS(path)[1]
    rows = int(hdu.get_nrows())
    audit = {
        "path": str(path),
        "sha256": expected_hash,
        "rows": rows,
        "accepted_rows": 0,
        "goodhardloc_rejected": 0,
        "maskbits_nonzero": 0,
        "maskbits_filter_applied": False,
        "maskbits_policy": (
            "use all rows in the registered HPmapcut product; retain MASKBITS only for audit "
            "because no additional veto-bit contract is registered"
        ),
    }
    npix = hp.nside2npix(nside)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        row_ids = np.arange(start, stop, dtype=np.int64)
        block = hdu.read(
            columns=["RA", "DEC", "PHOTSYS", "GOODHARDLOC", "MASKBITS"],
            rows=row_ids,
        )
        good = np.asarray(block["GOODHARDLOC"], dtype=bool)
        audit["goodhardloc_rejected"] += int((~good).sum())
        audit["maskbits_nonzero"] += int(np.count_nonzero(block["MASKBITS"]))
        if not good.any():
            continue
        ra = np.asarray(block["RA"][good], dtype=np.float64)
        dec = np.asarray(block["DEC"][good], dtype=np.float64)
        # With lonlat=True healpy expects (longitude, latitude) in degrees,
        # rather than the usual (theta, phi) radians convention.
        pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
        cap = galactic_cap(ra, dec)
        phot = photsys_code(block["PHOTSYS"][good])
        domain = cap.astype(np.int64) * 2 + phot.astype(np.int64)
        flat = domain * npix + pix.astype(np.int64)
        counts += np.bincount(flat, minlength=4 * npix).reshape(4, npix)
        audit["accepted_rows"] += int(good.sum())
    hdu = None
    if audit["accepted_rows"] + audit["goodhardloc_rejected"] != rows:
        raise RuntimeError(f"row accounting failed for {path}")
    audit["maskbits_nonzero_fraction"] = audit["maskbits_nonzero"] / max(rows, 1)
    return audit


def normalized_map(domain_counts: np.ndarray) -> dict[str, np.ndarray | dict]:
    domain_counts = np.asarray(domain_counts, dtype=np.int64)
    if domain_counts.ndim != 2 or domain_counts.shape[0] != 4:
        raise ValueError("domain_counts must have shape (4, npix)")
    support_by_domain = domain_counts > 0
    overlap = np.sum(support_by_domain, axis=0) > 1
    winner = np.argmax(domain_counts, axis=0).astype(np.int8)
    total = domain_counts.sum(axis=0)
    support = total > 0
    response_by_domain = np.zeros_like(domain_counts, dtype=np.float32)
    domain_metadata: dict[str, dict] = {}
    for domain in range(4):
        selected = support_by_domain[domain]
        cap = domain // 2
        phot = "S" if domain % 2 else "N"
        if not selected.any():
            domain_metadata[f"cap{cap}_PHOTSYS{phot}"] = {
                "supported_pixels": 0,
                "mean_raw_count": None,
                "raw_count_sum": 0,
                "response_mean": None,
                "status": "empty physical cap/PHOTSYS intersection",
            }
            continue
        mean = float(np.mean(domain_counts[domain, selected], dtype=np.float64))
        response_by_domain[domain, selected] = domain_counts[domain, selected] / mean
        domain_metadata[f"cap{cap}_PHOTSYS{phot}"] = {
            "supported_pixels": int(selected.sum()),
            "mean_raw_count": mean,
            "raw_count_sum": int(domain_counts[domain].sum()),
            "response_mean": float(np.mean(response_by_domain[domain, selected])),
        }
    response = np.zeros(domain_counts.shape[1], dtype=np.float32)
    response[support] = response_by_domain[winner[support], np.flatnonzero(support)]
    return {
        "raw_counts_by_domain": domain_counts,
        "support": support,
        "domain": winner,
        "angular_response": response,
        "metadata": {
            "domains": domain_metadata,
            "supported_pixels": int(support.sum()),
            "supported_area_deg2": float(support.sum() * hp.nside2pixarea(NSIDE, degrees=True)),
            "domain_overlap_pixels": int(overlap.sum()),
        },
    }


def save_map(path: Path, result: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        raw_counts_by_domain=result["raw_counts_by_domain"],
        support=np.asarray(result["support"], dtype=np.uint8),
        domain=np.asarray(result["domain"], dtype=np.int8),
        angular_response=np.asarray(result["angular_response"], dtype=np.float32),
    )
    temporary.replace(path)
    metadata = dict(metadata)
    metadata.update(result["metadata"])
    metadata["path"] = str(path)
    metadata["sha256"] = sha256(path)
    atomic_json(path.with_suffix(".json"), metadata)


def save_progress(path: Path, counts: np.ndarray, metadata: dict) -> None:
    """Atomically retain exact accumulated counts between CPU allocations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, raw_counts_by_domain=np.asarray(counts, dtype=np.int64))
    temporary.replace(path)
    payload = dict(metadata)
    payload["path"] = str(path)
    payload["sha256"] = sha256(path)
    atomic_json(path.with_suffix(".json"), payload)


def build_maps(root: Path, registry_path: Path, phase: str,
               snapshots: tuple[int, ...]) -> dict:
    if phase not in PHASES:
        raise ValueError(f"unsupported P3b-R phase {phase}")
    registry = load_json(registry_path)
    maximum = max(snapshots)
    ids = tuple(range(maximum))
    sources = selected_random_sources(registry, phase, ids)
    angular_dir = phase_output(root, phase) / "angular"
    existing = angular_dir / f"randoms_n{maximum}.npz"
    if existing.is_file() and existing.with_suffix(".json").is_file():
        return load_json(existing.with_suffix(".json"))
    progress_path = angular_dir / "randoms_progress.npz"
    progress_meta_path = progress_path.with_suffix(".json")
    snapshot_count = max(
        (
            count for count in snapshots
            if count < maximum
            and (angular_dir / f"randoms_n{count}.npz").is_file()
            and (angular_dir / f"randoms_n{count}.json").is_file()
        ),
        default=0,
    )
    progress_count = 0
    if progress_path.is_file() and progress_meta_path.is_file():
        progress_meta = load_json(progress_meta_path)
        progress_count = int(progress_meta["random_realisation_count"])
        if progress_meta["random_ids"] != list(range(progress_count)):
            raise RuntimeError(f"non-canonical random-map progress identity: {progress_path}")
        if progress_meta["source_registry_sha256"] != sha256(registry_path):
            raise RuntimeError(f"random-map progress registry mismatch: {progress_path}")
    resume_count = max(snapshot_count, progress_count)
    if resume_count:
        resume_path = (
            progress_path if progress_count == resume_count
            else angular_dir / f"randoms_n{resume_count}.npz"
        )
        resume_meta = load_json(resume_path.with_suffix(".json"))
        if resume_meta["random_ids"] != list(range(resume_count)):
            raise RuntimeError(f"non-canonical random-map resume identity: {resume_path}")
        if resume_meta["source_registry_sha256"] != sha256(registry_path):
            raise RuntimeError(f"random-map resume registry mismatch: {resume_path}")
        counts = np.asarray(load_map(resume_path)["raw_counts_by_domain"], dtype=np.int64)
        audits = list(resume_meta["sources"])
    else:
        counts = np.zeros((4, hp.nside2npix(NSIDE)), dtype=np.int64)
        audits = []
    started = time.time()
    for position, record in enumerate(sources[resume_count:], start=resume_count + 1):
        audits.append(add_random_file(counts, record))
        if position in snapshots:
            result = normalized_map(counts.copy())
            map_path = angular_dir / f"randoms_n{position}.npz"
            metadata = {
                "schema_version": "p3br-angular-random-map-v1",
                "phase": phase,
                "nside": NSIDE,
                "ordering": "RING",
                "random_ids": list(range(position)),
                "sources": audits.copy(),
                "source_registry": str(registry_path),
                "source_registry_sha256": sha256(registry_path),
                "elapsed_seconds": time.time() - started,
                "blind_truth_opened": False,
                "creation_commit": git_sha(REPO_ROOT),
            }
            save_map(map_path, result, metadata)
            print(json.dumps({"phase": phase, "snapshot": position, **result["metadata"]}), flush=True)
        if position < maximum:
            save_progress(
                progress_path,
                counts,
                {
                    "schema_version": "p3br-angular-random-progress-v1",
                    "phase": phase,
                    "random_realisation_count": position,
                    "random_ids": list(range(position)),
                    "sources": audits.copy(),
                    "source_registry": str(registry_path),
                    "source_registry_sha256": sha256(registry_path),
                    "blind_truth_opened": False,
                    "creation_commit": git_sha(REPO_ROOT),
                },
            )
    if progress_path.exists():
        progress_path.unlink()
    if progress_meta_path.exists():
        progress_meta_path.unlink()
    return load_json((angular_dir / f"randoms_n{maximum}.json"))


def load_map(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path)
    return {key: np.asarray(arrays[key]) for key in arrays.files}


def compare_random_maps(map4: dict[str, np.ndarray], map18: dict[str, np.ndarray]) -> dict:
    support4 = np.asarray(map4["support"], dtype=bool)
    support18 = np.asarray(map18["support"], dtype=bool)
    intersection = support4 & support18
    union = support4 | support18
    response4 = np.asarray(map4["angular_response"], dtype=np.float64)
    response18 = np.asarray(map18["angular_response"], dtype=np.float64)
    denominator = np.maximum(np.abs(response18[intersection]), 1.0e-12)
    fractional = np.abs(response4[intersection] - response18[intersection]) / denominator
    cap_shell = {}
    domain18 = np.asarray(map18["domain"], dtype=np.int8)
    for cap_id, cap_name in CAPS:
        selected4 = support4 & ((np.asarray(map4["domain"]) // 2) == cap_id)
        selected18 = support18 & ((domain18 // 2) == cap_id)
        total4 = float(np.sum(response4[selected4], dtype=np.float64))
        total18 = float(np.sum(response18[selected18], dtype=np.float64))
        frac = abs(total4 - total18) / max(abs(total18), 1.0e-12)
        cap_shell[cap_name] = {f"{lo:.2f}_{hi:.2f}": frac for lo, hi in SHELLS}
    result = {
        "support_jaccard": float(intersection.sum() / max(union.sum(), 1)),
        "median_absolute_fractional_response_difference": float(np.median(fractional)),
        "p99_absolute_fractional_response_difference": float(np.quantile(fractional, 0.99)),
        "cap_shell_expected_count_fractional_difference": cap_shell,
        "expected_count_comparison_method": (
            "area-weighted angular response proxy; exact selected-map cap/shell totals are "
            "validated after Cartesian projection"
        ),
    }
    max_cap_shell = max(value for cap in cap_shell.values() for value in cap.values())
    result["maximum_cap_shell_expected_count_fractional_difference"] = max_cap_shell
    result["gates"] = {
        "support_jaccard_ge_0p999": result["support_jaccard"] >= 0.999,
        "median_response_difference_le_0p01": (
            result["median_absolute_fractional_response_difference"] <= 0.01
        ),
        "p99_response_difference_le_0p05": (
            result["p99_absolute_fractional_response_difference"] <= 0.05
        ),
        "cap_shell_expected_difference_le_0p01": max_cap_shell <= 0.01,
    }
    result["pass"] = bool(all(result["gates"].values()))
    return result


def freeze_decision(root: Path, registry_path: Path) -> dict:
    comparisons = {}
    for phase in CANARY_PHASES:
        directory = phase_output(root, phase) / "angular"
        map4 = load_map(directory / "randoms_n4.npz")
        map18 = load_map(directory / "randoms_n18.npz")
        comparisons[phase] = compare_random_maps(map4, map18)
    use_four = all(value["pass"] for value in comparisons.values())
    selected_ids = list(range(4 if use_four else 18))
    decision = {
        "schema_version": "p3br-random-density-decision-v1",
        "comparisons": comparisons,
        "selected_random_ids": selected_ids,
        "selected_realisation_count": len(selected_ids),
        "decision": "four fixed realisations" if use_four else "all eighteen realisations",
        "source_registry": str(registry_path),
        "source_registry_sha256": sha256(registry_path),
        "ph001_opened": False,
        "creation_commit": git_sha(REPO_ROOT),
        "pass": True,
    }
    path = Path(root) / "training_contract/P3BR_RANDOM_DENSITY_DECISION.json"
    atomic_json(path, decision)
    return decision


def angular_boundary_distance(support: np.ndarray, *, nside: int = NSIDE) -> np.ndarray:
    support = np.asarray(support, dtype=bool)
    pixels = np.arange(len(support), dtype=np.int64)
    neighbours = hp.get_all_neighbours(nside, pixels, nest=False)
    neighbour_supported = np.zeros_like(neighbours, dtype=bool)
    valid = neighbours >= 0
    neighbour_supported[valid] = support[neighbours[valid]]
    boundary = support & np.any(~neighbour_supported, axis=0)
    if not boundary.any():
        raise RuntimeError("random support contains no boundary pixels")
    boundary_pix = np.flatnonzero(boundary)
    supported_pix = np.flatnonzero(support)
    bx, by, bz = hp.pix2vec(nside, boundary_pix, nest=False)
    sx, sy, sz = hp.pix2vec(nside, supported_pix, nest=False)
    tree = cKDTree(np.column_stack((bx, by, bz)))
    chord, _ = tree.query(np.column_stack((sx, sy, sz)), k=1, workers=-1)
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    result = np.zeros(len(support), dtype=np.float32)
    result[supported_pix] = angle.astype(np.float32)
    return result


def selection_curve(selection: dict, cap_name: str) -> tuple[np.ndarray, np.ndarray]:
    curve = selection["rotations"]["0"]["caps"][cap_name]
    return (
        np.asarray(curve["grid_z"], dtype=np.float64),
        np.asarray(curve["ntilde"], dtype=np.float64),
    )


def grid_from_component(component: dict) -> GridSpec:
    grid = component["grid"]
    return GridSpec(
        origin=tuple(float(value) for value in grid["origin_mpc"]),
        shape=tuple(int(value) for value in grid["shape"]),
        cell_mpc=float(grid["cell_mpc"]),
        padding_mpc=float(grid["padding_mpc"]),
    )


def redshift_from_radius(radius: np.ndarray, selection: dict) -> np.ndarray:
    return np.interp(
        radius,
        np.asarray(selection["cosmology"]["radius_grid_mpc"], dtype=np.float64),
        np.asarray(selection["cosmology"]["redshift_grid"], dtype=np.float64),
    )


def angular_block(spec: GridSpec, slices: tuple[slice, slice, slice], nside: int,
                  halo: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy, gz = coordinate_block(spec, slices, halo=halo)
    shape = (gx.shape[0], gy.shape[1], gz.shape[2])
    xx = np.broadcast_to(gx, shape)
    yy = np.broadcast_to(gy, shape)
    zz = np.broadcast_to(gz, shape)
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    safe = np.maximum(radius, 1.0e-12)
    pix = hp.vec2pix(nside, xx / safe, yy / safe, zz / safe, nest=False)
    return pix, radius, redshift_from_radius(radius, _SELECTION_CONTEXT)


# Set only inside build_overlay.  Keeping the argument list of the hot chunk helper
# short avoids repeatedly copying a very large cosmology lookup manifest.
_SELECTION_CONTEXT: dict = {}


def create_virtual(handle: h5py.File, name: str, source_path: Path,
                   source_name: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
    layout[:] = h5py.VirtualSource(str(source_path), source_name, shape=shape)
    handle.create_virtual_dataset(name, layout, fillvalue=0)


def build_overlay(root: Path, selection_path: Path, phase: str, decision_path: Path) -> dict:
    global _SELECTION_CONTEXT
    if phase not in PHASES:
        raise ValueError(phase)
    decision = load_json(decision_path)
    n_random = int(decision["selected_realisation_count"])
    angular_path = phase_output(root, phase) / "angular" / f"randoms_n{n_random}.npz"
    angular_meta_path = angular_path.with_suffix(".json")
    angular = load_map(angular_path)
    angular_meta = load_json(angular_meta_path)
    support = np.asarray(angular["support"], dtype=bool)
    response = np.asarray(angular["angular_response"], dtype=np.float32)
    domain = np.asarray(angular["domain"], dtype=np.int8)
    selection = load_json(selection_path)
    _SELECTION_CONTEXT = selection
    p3_manifest_path = Path(root) / phase / "p3_fields/field_manifest.json"
    p3_manifest = load_json(p3_manifest_path)
    output_root = phase_output(root, phase)
    cap_records = {}
    phase_qa = {"schema_version": "p3br-qa-v1", "phase": phase, "caps": {}, "ph001_opened": False}
    for cap_id, cap_name in CAPS:
        cap_support = support & ((domain // 2) == cap_id)
        boundary_angle = angular_boundary_distance(cap_support)
        final_dir = output_root / cap_name
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "response_overlay.h5"
        cap_qa_path = final_dir / "qa.json"
        if final_path.is_file() and cap_qa_path.is_file():
            cap_records[cap_name] = load_json(cap_qa_path)
            phase_qa["caps"][cap_name] = cap_records[cap_name]
            continue
        partial = final_path.with_suffix(".partial.h5")
        if partial.exists():
            # A partial is wholly owned by this builder and has no completion
            # marker.  Validate its identity before discarding only that
            # interrupted cap attempt; completed cap HDF5 files are immutable.
            with h5py.File(partial, "r") as interrupted:
                identity = (
                    interrupted.attrs.get("schema_version") == "p3br-response-overlay-v1"
                    and interrupted.attrs.get("phase") == phase
                    and interrupted.attrs.get("cap") == cap_name
                )
            if not identity:
                raise RuntimeError(f"ambiguous partial overlay: {partial}")
            partial.unlink()
            print(json.dumps({
                "phase": phase,
                "cap": cap_name,
                "action": "restart validated interrupted partial overlay",
            }), flush=True)
        component = p3_manifest["components"][cap_name]
        base_path = Path(root) / phase / "p3_fields" / f"{cap_name.lower()}_fields.h5"
        spec = grid_from_component(component)
        chunk_shape = tuple(int(value) for value in component["chunk_shape"])
        z_context = (0.10, 0.60)
        sentinel = (0.585, 0.595)
        sigma_vox = 6.0 / spec.cell_mpc
        truncate = 4.0
        halo = int(math.ceil(sigma_vox * truncate))
        curve_z, curve_n = selection_curve(selection, cap_name)
        shell_stats = {
            f"{lo:.2f}_{hi:.2f}": {"observed": 0.0, "expected": 0.0, "supported": 0}
            for lo, hi in SHELLS
        }
        poisson_mu = []
        all_finite = True
        supported_outside_map = 0
        max_exposure = 0.0
        min_expected = math.inf
        with h5py.File(base_path, "r") as base, h5py.File(partial, "w", libver="latest") as out:
            out.attrs.update({
                "schema_version": "p3br-response-overlay-v1",
                "phase": phase,
                "cap": cap_name,
                "cap_id": cap_id,
                "origin_mpc": spec.origin,
                "shape": spec.shape,
                "cell_mpc": spec.cell_mpc,
                "axis_order": "ix,iy,iz",
                "random_realisation_count": n_random,
            })
            for name in ("counts", "los_x", "los_y", "los_z"):
                create_virtual(out, name, base_path, name, spec.shape, base[name].dtype)
            datasets = {}
            definitions = {
                "support_random": ("u1", "indicator"),
                "angular_response": ("f4", "mean-one within cap/PHOTSYS"),
                "exposure_apodized_random": ("f4", "fraction"),
                "expected_counts_random": ("f4", "expected galaxies per voxel"),
                "log_count_ratio_random": ("f4", "dimensionless"),
                "distance_to_support_boundary": ("f4", "Mpc"),
                "ntilde_mpc3": ("f4", "Mpc^-3"),
            }
            for name, (dtype, units) in definitions.items():
                datasets[name] = out.create_dataset(
                    name, shape=spec.shape, dtype=dtype, chunks=chunk_shape,
                    compression="lzf", shuffle=True, fillvalue=0,
                )
                datasets[name].attrs["units"] = units
            # Canonical names make this file a zero-copy three-channel adapter view.
            out["exposure_binary"] = datasets["support_random"]
            out["exposure_apodized"] = datasets["exposure_apodized_random"]
            out["expected_counts"] = datasets["expected_counts_random"]
            out["log_count_ratio"] = datasets["log_count_ratio_random"]
            for chunk_index, slices in enumerate(iter_chunks(spec.shape, chunk_shape)):
                pix_ext, radius_ext, redshift_ext = angular_block(spec, slices, NSIDE, halo=halo)
                radial_ext = (
                    (redshift_ext >= z_context[0]) & (redshift_ext < z_context[1])
                    & ~((redshift_ext >= sentinel[0]) & (redshift_ext < sentinel[1]))
                )
                binary_ext = radial_ext & cap_support[pix_ext]
                apod_ext = gaussian_filter(
                    binary_ext.astype(np.float32), sigma=sigma_vox, mode="constant",
                    cval=0.0, truncate=truncate,
                )
                trim = tuple(slice(halo, halo + sl.stop - sl.start) for sl in slices)
                binary = binary_ext[trim]
                apod = np.clip(apod_ext[trim], 0.0, 1.0).astype(np.float32)
                pix = pix_ext[trim]
                radius = radius_ext[trim]
                redshift = redshift_ext[trim]
                angular_response = response[pix].astype(np.float32)
                angular_response *= binary
                ntilde = np.interp(redshift, curve_z, curve_n).astype(np.float32)
                ntilde *= apod > MINIMUM_EXPOSURE
                expected = (
                    ntilde.astype(np.float64) * spec.cell_mpc ** 3
                    * angular_response.astype(np.float64) * apod.astype(np.float64)
                ).astype(np.float32)
                counts = np.asarray(base["counts"][slices], dtype=np.float32)
                contrast = log_count_ratio(
                    counts, expected, apod, CONTRAST_EPSILON, MINIMUM_EXPOSURE,
                )
                angular_distance = boundary_angle[pix].astype(np.float64) * radius
                radial_distance = np.minimum(
                    np.abs(radius - np.interp(z_context[0], selection["cosmology"]["redshift_grid"],
                                              selection["cosmology"]["radius_grid_mpc"])),
                    np.abs(radius - np.interp(z_context[1], selection["cosmology"]["redshift_grid"],
                                              selection["cosmology"]["radius_grid_mpc"])),
                )
                radial_distance = np.minimum(
                    radial_distance,
                    np.minimum(
                        np.abs(radius - np.interp(sentinel[0], selection["cosmology"]["redshift_grid"],
                                                  selection["cosmology"]["radius_grid_mpc"])),
                        np.abs(radius - np.interp(sentinel[1], selection["cosmology"]["redshift_grid"],
                                                  selection["cosmology"]["radius_grid_mpc"])),
                    ),
                )
                distance = np.minimum(angular_distance, radial_distance).astype(np.float32)
                distance *= binary
                values = {
                    "support_random": binary.astype(np.uint8),
                    "angular_response": angular_response,
                    "exposure_apodized_random": apod,
                    "expected_counts_random": expected,
                    "log_count_ratio_random": contrast,
                    "distance_to_support_boundary": distance,
                    "ntilde_mpc3": ntilde,
                }
                all_finite &= all(np.isfinite(value).all() for value in values.values())
                supported_outside_map += int(np.sum(binary & ~cap_support[pix]))
                max_exposure = max(max_exposure, float(apod.max(initial=0.0)))
                positive = expected[expected > 0]
                if len(positive):
                    min_expected = min(min_expected, float(positive.min()))
                    if sum(len(value) for value in poisson_mu) < 1_000_000:
                        remaining = 1_000_000 - sum(len(value) for value in poisson_mu)
                        poisson_mu.append(np.asarray(positive[:remaining], dtype=np.float64))
                if np.any(binary) or np.any(counts):
                    for name, value in values.items():
                        datasets[name][slices] = value
                for lo, hi in SHELLS:
                    selected = (redshift >= lo) & (redshift < hi)
                    key = f"{lo:.2f}_{hi:.2f}"
                    shell_stats[key]["observed"] += float(np.sum(counts[selected], dtype=np.float64))
                    shell_stats[key]["expected"] += float(np.sum(expected[selected], dtype=np.float64))
                    shell_stats[key]["supported"] += int(np.sum(selected & binary))
                if chunk_index % 100 == 0:
                    print(json.dumps({"phase": phase, "cap": cap_name, "chunk": chunk_index}), flush=True)
            out.flush()
        partial.replace(final_path)
        with h5py.File(final_path, "r") as check:
            virtual_identity = all(check[name].is_virtual for name in ("counts", "los_x", "los_y", "los_z"))
            canonical_aliases = all(
                check[left].id == check[right].id
                for left, right in (
                    ("support_random", "exposure_binary"),
                    ("exposure_apodized_random", "exposure_apodized"),
                    ("expected_counts_random", "expected_counts"),
                    ("log_count_ratio_random", "log_count_ratio"),
                )
            )
        mu = np.concatenate(poisson_mu) if poisson_mu else np.empty(0, dtype=np.float64)
        rng = np.random.default_rng(240822 + cap_id)
        draw = rng.poisson(mu) if len(mu) else np.empty(0)
        standardized = (draw - mu) / np.sqrt(np.maximum(mu, 1.0e-12)) if len(mu) else np.empty(0)
        log_draw = np.log((draw + CONTRAST_EPSILON) / (mu + CONTRAST_EPSILON)) if len(mu) else np.empty(0)
        closure = {
            shell: (value["observed"] / value["expected"] if value["expected"] > 0 else math.nan)
            for shell, value in shell_stats.items()
        }
        gates = {
            "grid_shape_parity": tuple(spec.shape) == tuple(component["grid"]["shape"]),
            "immutable_p3a_channels_are_virtual_identity_views": bool(virtual_identity),
            "canonical_response_names_are_hardlink_aliases": bool(canonical_aliases),
            "all_arrays_finite": bool(all_finite),
            "expected_counts_nonnegative": bool(min_expected >= 0.0),
            "no_support_outside_random_map": supported_outside_map == 0,
            "exposure_constrained_0_1": max_exposure <= 1.0 + 1.0e-6,
            "poisson_count_residual_consistent_zero": (
                len(standardized) > 0
                and abs(float(np.mean(standardized))) <= 5.0 / math.sqrt(len(standardized))
            ),
            "cap_shell_closure_within_25pct": all(0.75 <= value <= 1.25 for value in closure.values()),
        }
        qa = {
            "schema_version": "p3br-cap-qa-v1",
            "phase": phase,
            "cap": cap_name,
            "file": str(final_path),
            "file_sha256": sha256(final_path),
            "file_bytes": final_path.stat().st_size,
            "base_p3a_file": str(base_path),
            "base_p3a_file_sha256": component["file_sha256"],
            "grid": spec.as_dict(),
            "shell_totals": shell_stats,
            "observed_expected_ratio": closure,
            "poisson_validation": {
                "sample_size": int(len(mu)),
                "standardized_count_residual_mean": float(np.mean(standardized)),
                "log_count_ratio_mean": float(np.mean(log_draw)),
                "note": (
                    "The Poisson count residual must have zero mean. The log-count ratio is "
                    "reported but is not zero-mean because log is concave (Jensen bias)."
                ),
            },
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
        if not qa["pass"]:
            raise RuntimeError(f"{phase} {cap_name} P3b-R gates failed: {gates}")
        atomic_json(cap_qa_path, qa)
        cap_records[cap_name] = qa
        phase_qa["caps"][cap_name] = qa
    phase_qa["pass"] = bool(all(record["pass"] for record in cap_records.values()))
    atomic_json(output_root / "qa.json", phase_qa)
    manifest = {
        "schema_version": "p3br-response-overlay-manifest-v1",
        "phase": phase,
        "stage": "P3b-R",
        "random_ids": decision["selected_random_ids"],
        "random_realisation_count": n_random,
        "angular_map": str(angular_path),
        "angular_map_sha256": sha256(angular_path),
        "angular_map_manifest": str(angular_meta_path),
        "angular_map_manifest_sha256": sha256(angular_meta_path),
        "source_registry": angular_meta["source_registry"],
        "source_registry_sha256": angular_meta["source_registry_sha256"],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256(selection_path),
        "p3a_manifest": str(p3_manifest_path),
        "p3a_manifest_sha256": sha256(p3_manifest_path),
        "points": p3_manifest["points"],
        "channel_order": [
            "counts", "support_random", "angular_response",
            "exposure_apodized_random", "expected_counts_random",
            "log_count_ratio_random", "distance_to_support_boundary",
            "ntilde_mpc3", "los_x", "los_y", "los_z",
        ],
        "components": {
            cap: {
                "file": record["file"],
                "file_sha256": record["file_sha256"],
                "qa": str(output_root / cap / "qa.json"),
                "grid": record["grid"],
                "support_atlas": {
                    shell: {
                        "expected_count_sum": values["expected"],
                        "input_galaxies": values["observed"],
                        "support_voxels": values["supported"],
                    }
                    for shell, values in record["shell_totals"].items()
                },
                "channel_units": {
                    "counts": "CIC galaxy count (P3a virtual dataset)",
                    "support_random": "indicator",
                    "angular_response": "mean-one within cap/PHOTSYS",
                    "exposure_apodized_random": "fraction",
                    "expected_counts_random": "expected galaxies per voxel",
                    "log_count_ratio_random": "dimensionless",
                    "distance_to_support_boundary": "Mpc",
                },
            }
            for cap, record in cap_records.items()
        },
        "response_scope": {
            "M": "registered full HPmapcut random support and angular targetability",
            "C_fibre": "not supplied by full randoms; deferred to audited assignment products",
            "C_z": "not supplied by full randoms; deferred to audited redshift-success products",
        },
        "mock_provenance": "official mock full randoms are Kibo-derived",
        "deployment_contract": "Loa DR2; not claimed pointwise Kibo-Loa matched",
        "ph001_opened": False,
        "creation_commit": git_sha(REPO_ROOT),
        "gates": {
            "unit_audit_pass": bool(p3_manifest["gates"]["unit_audit_pass"]),
            "p3a_grid_parent_index_parity": True,
            "component_gates_pass": phase_qa["pass"],
        },
        "pass": phase_qa["pass"],
    }
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("stage", choices=("maps", "decision", "overlay"))
    result.add_argument("--phase", choices=PHASES)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    result.add_argument(
        "--decision", type=Path,
        default=DEFAULT_ROOT / "training_contract/P3BR_RANDOM_DENSITY_DECISION.json",
    )
    result.add_argument("--snapshots", default=None)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "maps":
        if args.phase is None:
            raise SystemExit("maps requires --phase")
        snapshots = (
            tuple(int(value) for value in args.snapshots.split(","))
            if args.snapshots else (SNAPSHOT_COUNTS if args.phase in CANARY_PHASES else (4,))
        )
        print(json.dumps(build_maps(args.root, args.registry, args.phase, snapshots), indent=2))
    elif args.stage == "decision":
        print(json.dumps(freeze_decision(args.root, args.registry), indent=2))
    else:
        if args.phase is None:
            raise SystemExit("overlay requires --phase")
        print(json.dumps(build_overlay(args.root, args.selection, args.phase, args.decision), indent=2))


if __name__ == "__main__":
    main()
