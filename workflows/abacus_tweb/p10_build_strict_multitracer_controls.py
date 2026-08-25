#!/usr/bin/env python3
"""Build strict P10 sparse-random and cross-phase FAINT control views.

The exported roots deliberately implement the existing P10 multitracer schema,
so the frozen six-channel U-PATCH trainer can consume them with
``--model unet_multitracer --multitracer-view proxy``.  No tidal labels are read.

``r3_rf_dm``
    Draw an unclustered angular point process from the all-18 random-response
    map, retain the exact assigned-FAINT cap/redshift multiset, and deposit the
    resulting synthetic catalogue on the recipient phase's immutable P3 grid.

``bf_xphase``
    Deposit real FAINT points from a deranged donor phase on the recipient
    phase's immutable grid.  The recipient BRIGHT inputs, targets and exposure
    remain unchanged in the downstream trainer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import Planck18

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_build_multitracer_views import cap_from_radec
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, cic_deposit
from workflows.abacus_tweb.p8_build_multitracer_fields import complete_cic_support
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
BASE = ROOT / "multitracer/v1"
OUTPUT = ROOT / "multitracer/strict_controls"
VISIBLE = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
TRAIN = VISIBLE[:-1]
CAPS = ((0, "SGC"), (1, "NGC"))
Z_MIN, Z_MAX, DZ = 0.10, 0.60, 0.01
NSIDE_RANDOM = 256
NSIDE_FAINT = 64
SUBPIXEL_FACTOR = 16
CHUNK = 250_000
DM_SEEDS = (1701, 2718)
DONOR_MAPS = {
    "forward": {
        "ph000": "ph002", "ph002": "ph003", "ph003": "ph004",
        "ph004": "ph005", "ph005": "ph000", "ph006": "ph005",
    },
    "reverse": {
        "ph000": "ph005", "ph002": "ph000", "ph003": "ph002",
        "ph004": "ph003", "ph005": "ph004", "ph006": "ph004",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def grid_spec(component: dict) -> GridSpec:
    grid = component["grid"]
    return GridSpec(
        origin=tuple(float(value) for value in grid["origin_mpc"]),
        shape=tuple(int(value) for value in grid["shape"]),
        cell_mpc=float(grid["cell_mpc"]),
        padding_mpc=float(grid["padding_mpc"]),
    )


def control_name(control: str, seed: int, derangement: str) -> str:
    if control == "r3_rf_dm":
        return f"r3_rf_dm_seed{seed}_v1"
    return f"bf_xphase_{derangement}_v1"


def validate_donor_map(mapping: dict[str, str]) -> None:
    if set(mapping) != set(VISIBLE):
        raise ValueError("donor mapping must cover every visible phase")
    if any(recipient == donor for recipient, donor in mapping.items()):
        raise ValueError("cross-phase donor mapping contains a self-pair")
    if any(donor not in VISIBLE for donor in mapping.values()):
        raise ValueError("cross-phase donor mapping contains an unknown phase")


def fine_stratum(redshift: np.ndarray) -> np.ndarray:
    values = np.asarray(redshift, dtype=np.float64)
    # Stabilise exact decimal bin edges such as 0.11 against binary round-off.
    result = np.floor((values - Z_MIN) / DZ + 1.0e-10).astype(np.int16)
    result[(values < Z_MIN) | (values >= Z_MAX)] = -1
    return result


def subpixel_angles(
    parent_ring: np.ndarray, rng: np.random.Generator,
    *, nside: int = NSIDE_RANDOM, factor: int = SUBPIXEL_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fine nested subpixel centres inside RING-ordered parent pixels."""
    parent_ring = np.asarray(parent_ring, dtype=np.int64)
    parent_nest = hp.ring2nest(nside, parent_ring)
    child_count = factor * factor
    child_nest = parent_nest * child_count + rng.integers(
        0, child_count, size=len(parent_ring), dtype=np.int64
    )
    lon, lat = hp.pix2ang(nside * factor, child_nest, nest=True, lonlat=True)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def cartesian_from_radec_z(
    ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray
) -> np.ndarray:
    distance = Planck18.comoving_distance(np.asarray(redshift, dtype=np.float64)).value
    ra_rad = np.deg2rad(np.asarray(ra, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec, dtype=np.float64))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((
        distance * cos_dec * np.cos(ra_rad),
        distance * cos_dec * np.sin(ra_rad),
        distance * np.sin(dec_rad),
    ))


def angular_sampler(phase: str, cap_id: int, catalogue: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    angular_path = ROOT / phase / "p3b_random_response_v1/angular/randoms_n18.npz"
    angular = np.load(angular_path, mmap_mode="r")
    raw = np.asarray(angular["raw_counts_by_domain"], dtype=np.float64).sum(axis=0)
    supported = np.asarray(angular["support"], dtype=bool)
    pixels = np.arange(len(supported), dtype=np.int64)
    ra, dec = hp.pix2ang(NSIDE_RANDOM, pixels, nest=False, lonlat=True)
    cap = cap_from_radec(ra, dec)

    target_pixel = np.load(catalogue["targetable_pixel"], mmap_mode="r")
    target_cap = np.load(catalogue["targetable_cap"], mmap_mode="r")
    faint_support = np.zeros(hp.nside2npix(NSIDE_FAINT), dtype=bool)
    faint_support[np.asarray(target_pixel[target_cap == cap_id], dtype=np.int64)] = True
    coarse = hp.ang2pix(NSIDE_FAINT, ra, dec, nest=False, lonlat=True)
    valid = supported & (cap == cap_id) & faint_support[coarse] & (raw > 0)
    valid_pixel = pixels[valid]
    weight = raw[valid]
    weight /= weight.sum(dtype=np.float64)
    cdf = np.cumsum(weight, dtype=np.float64)
    cdf[-1] = 1.0
    return valid_pixel, cdf, {
        "angular_map": str(angular_path),
        "angular_map_sha256": sha256(angular_path),
        "valid_parent_pixels": int(len(valid_pixel)),
        "weight_sum_before_normalization": float(raw[valid].sum(dtype=np.float64)),
        "faint_support_pixels_nside64": int(np.count_nonzero(faint_support)),
    }


def sample_parent_pixels(
    valid_pixel: np.ndarray, cdf: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    return valid_pixel[np.searchsorted(cdf, rng.random(size), side="right")]


def deposit_supported(counts: np.ndarray, xyz: np.ndarray, spec: GridSpec) -> dict:
    supported = complete_cic_support(xyz, spec)
    if not np.all(supported):
        raise RuntimeError("deposit_supported received a point without complete CIC support")
    _, stats = cic_deposit(xyz, spec, out=counts)
    return stats


def deposit_dm_cap(
    *, phase: str, cap_id: int, redshift: np.ndarray, catalogue: dict,
    spec: GridSpec, seed: int,
) -> tuple[np.ndarray, dict]:
    valid_pixel, cdf, angular_audit = angular_sampler(phase, cap_id, catalogue)
    counts = np.zeros(spec.shape, dtype=np.float32)
    strata = fine_stratum(redshift)
    if np.any(strata < 0):
        raise RuntimeError(f"{phase} cap={cap_id}: FAINT redshift outside registered context")
    total_retry = 0
    deposited = 0
    stratum_rows = {}
    for stratum_id in np.unique(strata):
        source = np.asarray(redshift[strata == stratum_id], dtype=np.float64)
        rng = np.random.default_rng(seed + 100_003 * int(phase[-3:]) + 997 * cap_id + int(stratum_id))
        source = source[rng.permutation(len(source))]
        stratum_rows[str(int(stratum_id))] = int(len(source))
        for start in range(0, len(source), CHUNK):
            pending = source[start : start + CHUNK]
            while len(pending):
                parent = sample_parent_pixels(valid_pixel, cdf, len(pending), rng)
                ra, dec = subpixel_angles(parent, rng)
                xyz = cartesian_from_radec_z(ra, dec, pending)
                supported = complete_cic_support(xyz, spec)
                if np.any(supported):
                    deposit_supported(counts, xyz[supported], spec)
                    deposited += int(np.count_nonzero(supported))
                total_retry += int(np.count_nonzero(~supported))
                pending = pending[~supported]
    count_sum = float(counts.sum(dtype=np.float64))
    audit = {
        "input_redshifts": int(len(redshift)),
        "deposited_points": int(deposited),
        "angular_retries_for_grid_support": int(total_retry),
        "count_sum": count_sum,
        "fine_stratum_rows": stratum_rows,
        "redshift_multiset_sum": float(np.sum(redshift, dtype=np.float64)),
        "redshift_multiset_square_sum": float(np.square(redshift).sum(dtype=np.float64)),
        "angular": angular_audit,
        "pass": bool(
            deposited == len(redshift)
            and abs(count_sum - len(redshift)) <= max(1.0e-3, 2.0e-6 * len(redshift))
        ),
    }
    if not audit["pass"]:
        raise RuntimeError(f"{phase} cap={cap_id}: density-matched deposition failed")
    return counts, audit


def deposit_xphase_cap(
    *, donor_points: np.ndarray, cap_id: int, spec: GridSpec,
) -> tuple[np.ndarray, dict]:
    counts = np.zeros(spec.shape, dtype=np.float32)
    rows = np.flatnonzero(np.asarray(donor_points[:, 3], dtype=np.uint8) == cap_id)
    deposited = 0
    excluded = 0
    for start in range(0, len(rows), CHUNK):
        xyz = np.asarray(donor_points[rows[start : start + CHUNK], :3], dtype=np.float64)
        supported = complete_cic_support(xyz, spec)
        excluded += int(np.count_nonzero(~supported))
        if np.any(supported):
            deposit_supported(counts, xyz[supported], spec)
            deposited += int(np.count_nonzero(supported))
    count_sum = float(counts.sum(dtype=np.float64))
    audit = {
        "donor_rows_in_cap": int(len(rows)),
        "deposited_points": int(deposited),
        "grid_edge_excluded_rows": int(excluded),
        "grid_edge_excluded_fraction": excluded / max(len(rows), 1),
        "count_sum": count_sum,
        "pass": bool(
            deposited > 0
            and excluded / max(len(rows), 1) < 5.0e-3
            and abs(count_sum - deposited) <= max(1.0e-3, 2.0e-6 * deposited)
        ),
    }
    if not audit["pass"]:
        raise RuntimeError("cross-phase deposition failed")
    return counts, audit


def write_count_overlay(
    *, path: Path, counts: np.ndarray, exposure_source: Path,
    control: str, phase: str, donor: str | None, audit: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.h5")
    if temporary.exists():
        temporary.unlink()
    chunks = tuple(min(64, value) for value in counts.shape)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset(
            "counts", data=counts, chunks=chunks, compression="lzf", shuffle=True
        )
        relative = os.path.relpath(exposure_source, temporary.parent)
        for name in ("exposure_apodized", "exposure_binary", "response_angular"):
            handle[name] = h5py.ExternalLink(relative, f"/{name}")
        handle.attrs["schema_version"] = "p10-strict-multitracer-control-overlay-v1"
        handle.attrs["control"] = control
        handle.attrs["recipient_phase"] = phase
        handle.attrs["donor_phase"] = donor or "NONE"
        handle.attrs["count_sum"] = audit["count_sum"]
    temporary.replace(path)


def component_record(
    *, source: dict, path: Path, audit: dict, control: str,
    donor: str | None,
) -> dict:
    return {
        **source,
        "file": str(path),
        "file_sha256": sha256(path),
        "counts_sum": audit["count_sum"],
        "control": control,
        "donor_phase": donor,
        "strict_control_audit": audit,
        "exposure_source": source["file"],
        "exposure_source_sha256": source["file_sha256"],
        "gates": {**source.get("gates", {}), "strict_control_pass": audit["pass"]},
    }


def build_phase(
    *, control: str, phase: str, seed: int, derangement: str,
    output_root: Path, force: bool,
) -> dict:
    if phase not in VISIBLE:
        raise ValueError(f"phase must be one of {VISIBLE}")
    name = control_name(control, seed, derangement)
    root = output_root / name
    phase_root = root / "phases" / phase
    marker = phase_root / "PHASE_MULTITRACER_VIEWS_READY.json"
    if marker.exists() and not force:
        row = load(marker)
        if row.get("pass"):
            return row
    base_row = load(BASE / "phases" / phase / "PHASE_MULTITRACER_VIEWS_READY.json")
    if not base_row.get("pass"):
        raise RuntimeError(f"{phase}: base multitracer contract does not pass")
    donor = None
    donor_points = None
    if control == "bf_xphase":
        mapping = DONOR_MAPS[derangement]
        validate_donor_map(mapping)
        donor = mapping[phase]
        donor_row = load(BASE / "phases" / donor / "PHASE_MULTITRACER_VIEWS_READY.json")
        donor_points = np.load(donor_row["catalogue"]["points"], mmap_mode="r")
    elif control != "r3_rf_dm":
        raise ValueError("control must be r3_rf_dm or bf_xphase")

    index = np.load(base_row["catalogue"]["index"], mmap_mode="r")
    redshift = np.asarray(index["redshift"], dtype=np.float64)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    components = {}
    for cap_id, cap_name in CAPS:
        source = base_row["proxy"]["components"][cap_name]
        spec = grid_spec(source)
        if control == "r3_rf_dm":
            counts, audit = deposit_dm_cap(
                phase=phase, cap_id=cap_id, redshift=redshift[cap == cap_id],
                catalogue=base_row["catalogue"], spec=spec, seed=seed,
            )
        else:
            counts, audit = deposit_xphase_cap(
                donor_points=donor_points, cap_id=cap_id, spec=spec,
            )
        path = phase_root / "fields" / f"{cap_name.lower()}_{name}_overlay.h5"
        write_count_overlay(
            path=path, counts=counts, exposure_source=Path(source["file"]),
            control=control, phase=phase, donor=donor, audit=audit,
        )
        components[cap_name] = component_record(
            source=source, path=path, audit=audit, control=control, donor=donor,
        )
        del counts

    proxy = {
        **base_row["proxy"],
        "schema_version": "p10-strict-multitracer-proxy-v1",
        "product": name,
        "components": components,
        "pass": all(row["strict_control_audit"]["pass"] for row in components.values()),
    }
    record = {
        **base_row,
        "schema_version": "p10-strict-multitracer-phase-view-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "control": control,
        "control_name": name,
        "catalogue_seed": seed if control == "r3_rf_dm" else None,
        "derangement": derangement if control == "bf_xphase" else None,
        "donor_phase": donor,
        "proxy": proxy,
        "ph001_opened": False,
        "labels_read_by_builder": False,
        "pass": bool(proxy["pass"] and donor != phase),
    }
    if control == "r3_rf_dm":
        record["pass"] = bool(proxy["pass"])
    atomic_json(marker, record)
    if not record["pass"]:
        raise RuntimeError(f"{phase}: strict multitracer control failed")
    return record


def copy_or_validate_selection(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    source = BASE / "selection_manifest.json"
    destination = root / "selection_manifest.json"
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError("strict-control selection manifest differs from FAINT base")
        return
    shutil.copy2(source, destination)


def finalize(
    *, control: str, seed: int, derangement: str, output_root: Path,
) -> dict:
    name = control_name(control, seed, derangement)
    root = output_root / name
    copy_or_validate_selection(root)
    records = {}
    for phase in VISIBLE:
        path = root / "phases" / phase / "PHASE_MULTITRACER_VIEWS_READY.json"
        if not path.exists():
            raise RuntimeError(f"missing strict-control phase product: {path}")
        row = load(path)
        if not row.get("pass") or row.get("ph001_opened"):
            raise RuntimeError(f"{phase}: strict-control phase record does not pass")
        records[phase] = str(path)
    payload = {
        "schema_version": "p10-strict-multitracer-views-ready-v1",
        "created_utc": utc_now(),
        "creation_commit": git_revision(),
        "control": control,
        "control_name": name,
        "catalogue_seed": seed if control == "r3_rf_dm" else None,
        "derangement": derangement if control == "bf_xphase" else None,
        "donor_map": DONOR_MAPS[derangement] if control == "bf_xphase" else None,
        "training_phases": list(TRAIN),
        "validation_phase": "ph006",
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "bright_supervision_only": True,
        "labels_read_by_builder": False,
        "selection_manifest": str(root / "selection_manifest.json"),
        "selection_manifest_sha256": sha256(root / "selection_manifest.json"),
        "base_selection_manifest_sha256": sha256(BASE / "selection_manifest.json"),
        "phase_records": records,
        "trainer_contract": (
            "frozen unet_multitracer with --multitracer-view proxy; identical six-channel "
            "architecture, BRIGHT targets, phase sampler and evaluator"
        ),
        "pass": True,
    }
    atomic_json(root / "P10_MULTITRACER_VIEWS_READY.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("r3_rf_dm", "bf_xphase"), required=True)
    parser.add_argument("--phase", choices=VISIBLE)
    parser.add_argument("--seed", type=int, default=DM_SEEDS[0])
    parser.add_argument("--derangement", choices=tuple(DONOR_MAPS), default="forward")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.control == "bf_xphase":
        validate_donor_map(DONOR_MAPS[args.derangement])
    name = control_name(args.control, args.seed, args.derangement)
    copy_or_validate_selection(args.output_root / name)
    if args.finalize:
        result = finalize(
            control=args.control, seed=args.seed, derangement=args.derangement,
            output_root=args.output_root,
        )
    elif args.phase:
        result = build_phase(
            control=args.control, phase=args.phase, seed=args.seed,
            derangement=args.derangement, output_root=args.output_root,
            force=args.force,
        )
    else:
        records = {
            phase: build_phase(
                control=args.control, phase=phase, seed=args.seed,
                derangement=args.derangement, output_root=args.output_root,
                force=args.force,
            )
            for phase in VISIBLE
        }
        result = finalize(
            control=args.control, seed=args.seed, derangement=args.derangement,
            output_root=args.output_root,
        )
        result["phase_pass"] = {phase: row["pass"] for phase, row in records.items()}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
