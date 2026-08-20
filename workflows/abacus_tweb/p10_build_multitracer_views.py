#!/usr/bin/env python3
"""Build P10 phase-matched BRIGHT-target/FAINT-context Proxy and Null views.

Only ph000/ph002--ph006 are visible.  BRIGHT supervision and P4 ownership are
unchanged.  Fibre-assigned FAINT galaxies are deposited on each phase's exact
P3 lattice as a separate tracer, with an independently fitted selection curve.
The Null retains every FAINT radius and angular-direction multiset but permutes
their pairing within Galactic cap and Delta-z=0.01 strata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import fitsio
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_refit_fullcap_selection import (
    build_cap_lookup,
    fit_log_spline,
    histogram_effective_volume,
    radius_to_redshift_grid,
)
from workflows.abacus_tweb.p8_build_multitracer_catalogues import sky_to_points
from workflows.abacus_tweb.p8_build_multitracer_control_fields import (
    angular_null,
    write_fields,
)
from workflows.abacus_tweb.p8_build_multitracer_fields import (
    build_product,
    context_redshift,
    pixel_from_radec,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p10_multitracer_source_audit import (
    BRIGHT_BITS,
    FAINT_BITS,
    paths_for_phase,
    target_counts,
)


VISIBLE = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
TRAIN = ("ph000", "ph002", "ph003", "ph004", "ph005")
CAP_NAME = {0: "SGC", 1: "NGC"}
PRODUCT = "bf_proxy_response_v1"
ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
OUTPUT = ROOT / "multitracer/v1"
REGISTRY = REPO_ROOT / "configs/p10_phase_registry_v1.json"
SOURCE_AUDIT = ROOT / "multitracer/source_audit_v1.json"
Z_MIN, Z_MAX = 0.10, 0.60
BIN_WIDTH, CURVE_STEP = 0.005, 0.001
KNOT_SPACING = 0.05
FIT_Z_MIN, FIT_Z_MAX = 0.15, 0.55
MINIMUM_EXPOSURE = 1.0e-4
CONTRAST_EPSILON = 1.0e-3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with fitsio.FITS(str(path), "r") as handle:
        return handle[1].read(columns=list(columns))


def unique_forfa_faint(table: np.ndarray) -> np.ndarray:
    bits = np.asarray(table["BGS_TARGET"], dtype=np.int64)
    faint = (bits & FAINT_BITS) != 0
    bright = (bits & BRIGHT_BITS) != 0
    if np.any(faint & bright):
        raise RuntimeError("forFA contains ambiguous BRIGHT+FAINT target rows")
    table = table[faint]
    targetid = np.asarray(table["TARGETID"], dtype=np.int64)
    order = np.argsort(targetid, kind="mergesort")
    table = table[order]
    targetid = targetid[order]
    first = np.r_[True, targetid[1:] != targetid[:-1]]
    if np.any(~first):
        duplicate = targetid[~first]
        raise RuntimeError(
            f"forFA FAINT truth is not unique ({len(duplicate)} duplicate rows)"
        )
    return table


def assigned_faint_ids(table: np.ndarray) -> np.ndarray:
    counts = target_counts(table["TARGETID"], table["BGS_TARGET"])
    return np.asarray(counts["targetid"][counts["faint"]], dtype=np.int64)


def cap_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    # z row of the fixed ICRS-to-Galactic rotation matrix.
    ra_rad = np.deg2rad(np.asarray(ra, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    galactic_z = (
        -0.8676661490190047 * cos_dec * np.cos(ra_rad)
        - 0.1980763734312015 * cos_dec * np.sin(ra_rad)
        + 0.4559837761750669 * np.sin(dec_rad)
    )
    return (galactic_z > 0).astype(np.uint8)


def join_assigned_faint(forfa: np.ndarray, assigned_id: np.ndarray) -> np.ndarray:
    targetid = np.asarray(forfa["TARGETID"], dtype=np.int64)
    position = np.searchsorted(targetid, assigned_id)
    matched = (position < len(targetid)) & (
        targetid[np.minimum(position, len(targetid) - 1)] == assigned_id
    )
    if not np.all(matched):
        raise RuntimeError(
            f"{int(np.count_nonzero(~matched))} assigned FAINT IDs lack forFA truth"
        )
    result = forfa[position]
    keep = context_redshift(result["RSDZ"])
    return result[keep]


def phase_p3(root: Path, phase: str) -> Path:
    return root / phase / "p3_fields/field_manifest.json"


def build_phase_catalogue(
    *, phase: str, registry: dict, source_audit: dict, output: Path, force: bool,
    nside: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    catalogue_root = output / "phases" / phase / "catalogues"
    product_root = catalogue_root / PRODUCT
    manifest_path = product_root / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["response_nside"]) != nside:
            raise RuntimeError(f"{phase} cached response NSIDE differs from requested NSIDE")
        points = np.load(manifest["points"], mmap_mode="r")
        index = np.load(manifest["index"])
        return manifest, np.asarray(points), np.asarray(index["redshift"])
    product_root.mkdir(parents=True, exist_ok=True)
    forfa_path, assigned_path, resolution = paths_for_phase(registry, phase)
    audit = source_audit["phases"][phase]
    if str(forfa_path) != audit["forfa"] or str(assigned_path) != audit["assigned"]:
        raise RuntimeError(f"{phase} source audit/path registry mismatch")
    forfa = unique_forfa_faint(
        _read(forfa_path, ("TARGETID", "BGS_TARGET", "RA", "DEC", "RSDZ", "TRUEZ"))
    )
    targetable = context_redshift(forfa["RSDZ"])
    target_pixel = pixel_from_radec(forfa["RA"][targetable], forfa["DEC"][targetable], nside)
    target_cap = cap_from_radec(forfa["RA"][targetable], forfa["DEC"][targetable])
    assigned = _read(assigned_path, ("TARGETID", "BGS_TARGET"))
    selected = join_assigned_faint(forfa, assigned_faint_ids(assigned))
    points = sky_to_points(selected["RA"], selected["DEC"], selected["RSDZ"])
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    direct_cap = cap_from_radec(selected["RA"], selected["DEC"])
    if not np.array_equal(cap, direct_cap):
        raise RuntimeError(f"{phase} FAINT Galactic-cap calculation mismatch")
    points_path = product_root / "faint_points.npy"
    index_path = product_root / "faint_index.npz"
    target_pixel_path = product_root / "targetable_pixel.npy"
    target_cap_path = product_root / "targetable_cap.npy"
    np.save(points_path, points, allow_pickle=False)
    np.save(target_pixel_path, target_pixel, allow_pickle=False)
    np.save(target_cap_path, target_cap, allow_pickle=False)
    np.savez_compressed(
        index_path,
        targetid=np.asarray(selected["TARGETID"], dtype=np.int64),
        tracer_type=np.ones(len(selected), dtype=np.uint8),
        context=np.ones(len(selected), dtype=np.uint8),
        cap=cap,
        redshift=np.asarray(selected["RSDZ"], dtype=np.float32),
    )
    manifest = {
        "schema_version": "p10-multitracer-catalogue-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "product": PRODUCT,
        "bright_prefix_rows": 0,
        "bright_supervision_unchanged": True,
        "faint_context_only": True,
        "faint_rows": int(len(selected)),
        "targetable_faint_rows": int(len(target_pixel)),
        "response_nside": int(nside),
        "points": str(points_path),
        "points_sha256": sha256(points_path),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "targetable_pixel": str(target_pixel_path),
        "targetable_pixel_sha256": sha256(target_pixel_path),
        "targetable_cap": str(target_cap_path),
        "targetable_cap_sha256": sha256(target_cap_path),
        "forfa": str(forfa_path),
        "forfa_sha256": audit["forfa_sha256"],
        "assigned": str(assigned_path),
        "assigned_sha256": audit["assigned_sha256"],
        "path_resolution": resolution,
        "pass": bool(
            len(selected) > 0
            and np.all(np.isfinite(points))
            and len(np.unique(selected["TARGETID"])) == len(selected)
        ),
    }
    if not manifest["pass"]:
        raise RuntimeError(f"{phase} multitracer catalogue gates failed")
    atomic_json(manifest_path, manifest)
    return manifest, points, np.asarray(selected["RSDZ"], dtype=np.float64)


def build_phase_fields(
    *, phase: str, registry: dict, source_audit: dict, output: Path,
    catalogue: dict, points: np.ndarray, redshift: np.ndarray, force: bool,
    nside: int, prior_targets: float, null_seed: int,
) -> dict:
    phase_root = output / "phases" / phase
    if int(catalogue["response_nside"]) != nside:
        raise RuntimeError(f"{phase} catalogue/field response NSIDE mismatch")
    target_pixel = np.load(catalogue["targetable_pixel"], mmap_mode="r")
    target_cap = np.load(catalogue["targetable_cap"], mmap_mode="r")
    p3_path = phase_p3(ROOT, phase)
    p3 = json.loads(p3_path.read_text())
    proxy = build_product(
        product=PRODUCT,
        catalogue_root=phase_root / "catalogues",
        target_pixel=target_pixel,
        target_cap=target_cap,
        p3_manifest_path=p3_path,
        p3=p3,
        output_root=phase_root / "fields",
        nside=nside,
        prior_targets=prior_targets,
        force=force,
    )
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    stratum = np.floor((redshift - Z_MIN) / 0.01).astype(np.int16)
    null_xyz, donor, audit = angular_null(
        np.asarray(points[:, :3], dtype=np.float64),
        cap=cap,
        stratum=stratum,
        seed=null_seed + int(phase[-3:]),
    )
    null_points = np.column_stack((null_xyz, cap)).astype(np.float64)
    controls = phase_root / "controls"
    null = write_fields(
        output=controls,
        name="faint_position_null_cic",
        scheme="cic",
        points=null_points,
        tracer=np.ones(len(null_points), dtype=np.uint8),
        cap=cap,
        selected=np.arange(len(null_points), dtype=np.int64),
        p3=p3,
        chunk=250_000,
        include_bright=False,
        include_faint=True,
        force=force,
    )
    donor_path = controls / "faint_position_null_direction_donor.npy"
    np.save(donor_path, donor, allow_pickle=False)
    record = {
        "schema_version": "p10-multitracer-phase-views-v1",
        "phase": phase,
        "catalogue": catalogue,
        "proxy": proxy,
        "null": null,
        "null_audit": audit,
        "null_seed": null_seed + int(phase[-3:]),
        "null_donor": str(donor_path),
        "null_donor_sha256": sha256(donor_path),
        "null_contract": "radii and direction multiset fixed within cap and Delta-z=0.01",
        "pass": bool(proxy["pass"] and null["pass"] and len(donor) == len(points)),
    }
    if not record["pass"]:
        raise RuntimeError(f"{phase} Proxy/Null field gates failed")
    atomic_json(phase_root / "PHASE_MULTITRACER_VIEWS_READY.json", record)
    return record


def fit_shared_selection(*, output: Path, records: dict[str, dict]) -> dict:
    edges = np.arange(Z_MIN, Z_MAX + 0.5 * BIN_WIDTH, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_z = np.arange(Z_MIN, Z_MAX + 0.5 * CURVE_STEP, CURVE_STEP)
    radius_grid, redshift_grid = radius_to_redshift_grid(Z_MIN, Z_MAX)
    counts_total = np.zeros((2, len(centers)), dtype=np.int64)
    volume_total = np.zeros((2, len(centers)), dtype=np.float64)
    sources = {}
    per_phase_moments = {}
    for phase in TRAIN:
        row = records[phase]
        index = np.load(row["catalogue"]["index"])
        redshift = np.asarray(index["redshift"], dtype=np.float64)
        cap = np.asarray(index["cap"], dtype=np.uint8)
        for cap_id in (0, 1):
            counts_total[cap_id] += np.histogram(redshift[cap == cap_id], bins=edges)[0]
        p4 = ROOT / phase / "p4_patches"
        cores = np.load(p4 / "cores.npz", mmap_mode="r")
        p4_manifest = json.loads((p4 / "spatial_manifest.json").read_text())
        core_mpc = float(p4_manifest["unit_contract"]["core_mpc"])
        lookups = {
            name: build_cap_lookup(cores, cap_id, core_mpc)
            for cap_id, name in CAP_NAME.items()
        }
        overlay = {
            "components": {
                name: {
                    "file": row["proxy"]["components"][name]["file"],
                    "grid": row["proxy"]["components"][name]["grid"],
                }
                for name in ("NGC", "SGC")
            }
        }
        volume, volume_audit = histogram_effective_volume(
            p3=overlay,
            lookups=lookups,
            core_mpc=core_mpc,
            edges=edges,
            radius_grid_mpc=radius_grid,
            redshift_grid=redshift_grid,
        )
        volume_total += volume.sum(axis=1)
        value_sum = 0.0
        value_square_sum = 0.0
        value_count = 0
        for name in ("NGC", "SGC"):
            with h5py.File(row["proxy"]["components"][name]["file"], "r") as handle:
                for selection in handle["counts"].iter_chunks():
                    count = np.asarray(handle["counts"][selection], dtype=np.float64)
                    exposure = np.asarray(handle["exposure_apodized"][selection], dtype=np.float64)
                    values = np.log1p(count[exposure > MINIMUM_EXPOSURE])
                    value_sum += float(values.sum(dtype=np.float64))
                    value_square_sum += float(np.square(values).sum(dtype=np.float64))
                    value_count += int(len(values))
        if value_count <= 0:
            raise RuntimeError(f"{phase} has no exposed FAINT voxels")
        per_phase_moments[phase] = {
            "mean": value_sum / value_count,
            "second_moment": value_square_sum / value_count,
            "count": value_count,
        }
        sources[phase] = {"effective_volume": volume_audit}
    caps = {}
    for cap_id, name in CAP_NAME.items():
        curve, fit = fit_log_spline(
            centers,
            counts_total[cap_id],
            volume_total[cap_id],
            grid_z,
            knot_spacing=KNOT_SPACING,
            fit_z_min=FIT_Z_MIN,
            fit_z_max=FIT_Z_MAX,
        )
        caps[name] = {"grid_z": grid_z.tolist(), "ntilde": curve.tolist(), "fit": fit}
    mean = float(np.mean([per_phase_moments[p]["mean"] for p in TRAIN]))
    second = float(np.mean([per_phase_moments[p]["second_moment"] for p in TRAIN]))
    std = float(np.sqrt(max(second - mean * mean, 0.0)))
    manifest = {
        "schema_version": "p10-multitracer-selection-v1",
        "created_utc": utc_now(),
        "product": PRODUCT,
        "fit_phases": list(TRAIN),
        "application_phases": ["ph006"],
        "bright_supervision_unchanged": True,
        "faint_context_only": True,
        "caps": caps,
        "cosmology": {
            "radius_grid_mpc": radius_grid.tolist(),
            "redshift_grid": redshift_grid.tolist(),
        },
        "contrast": {"epsilon": CONTRAST_EPSILON, "minimum_exposure": MINIMUM_EXPOSURE},
        "faint_count_normalization": {"policy": "log1p_zscore", "mean": mean, "std": std},
        "per_phase_count_moments": per_phase_moments,
        "sources": sources,
        "pass": bool(
            std > 0
            and all(np.all(np.asarray(row["ntilde"]) > 0) for row in caps.values())
        ),
    }
    if not manifest["pass"]:
        raise RuntimeError("shared multitracer selection gates failed")
    atomic_json(output / "selection_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--response-prior-targets", type=float, default=20.0)
    parser.add_argument("--null-seed", type=int, default=314159)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    source_audit = json.loads(args.source_audit.read_text())
    if not source_audit.get("all_visible_phases_pass") or source_audit.get("sealed_phase_opened"):
        raise RuntimeError("passing truth-free visible-phase source audit is required")
    records = {}
    for phase in VISIBLE:
        catalogue, points, redshift = build_phase_catalogue(
            phase=phase,
            registry=registry,
            source_audit=source_audit,
            output=args.root,
            force=args.force,
            nside=args.nside,
        )
        records[phase] = build_phase_fields(
            phase=phase,
            registry=registry,
            source_audit=source_audit,
            output=args.root,
            catalogue=catalogue,
            points=points,
            redshift=redshift,
            force=args.force,
            nside=args.nside,
            prior_targets=args.response_prior_targets,
            null_seed=args.null_seed,
        )
    selection = fit_shared_selection(output=args.root, records=records)
    final = {
        "schema_version": "p10-multitracer-views-ready-v1",
        "created_utc": utc_now(),
        "visible_phases": list(VISIBLE),
        "training_phases": list(TRAIN),
        "validation_phase": "ph006",
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "bright_supervision_only": True,
        "proxy_minus_null_is_spatial_information_estimand": True,
        "phase_records": {
            phase: str(args.root / "phases" / phase / "PHASE_MULTITRACER_VIEWS_READY.json")
            for phase in VISIBLE
        },
        "selection_manifest": str(args.root / "selection_manifest.json"),
        "selection_manifest_sha256": sha256(args.root / "selection_manifest.json"),
        "pass": bool(selection["pass"] and all(row["pass"] for row in records.values())),
    }
    atomic_json(args.root / "P10_MULTITRACER_VIEWS_READY.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
