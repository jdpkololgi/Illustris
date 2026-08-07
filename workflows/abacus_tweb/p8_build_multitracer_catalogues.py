#!/usr/bin/env python3
"""Build response-explicit Bright-target/Faint-context P8 catalogues.

The frozen Bright GraphWeb parent is always the first block of every output and
retains its exact row order.  BGS_FAINT rows are appended as context only.  Two
products are written from the same immutable staged-mock rows:

``BF_ORACLE_ASSIGNED_v1``
    Fibre-observed Faint rows with their simulated RSD redshift and no
    spectroscopic failure.  This is an information upper bound only.

``BF_PROXY_RESPONSE_v1``
    The same fibre-observed Faint rows after a deterministic, tracer-specific
    LOA redshift-success draw.  Bright rows are *not* redrawn: the established
    Bright parent is copied exactly.

Only Bright rows may be supervised.  ``BRIGHT_PARENT_ID`` is therefore the
identity map for the Bright prefix and ``-1`` for every Faint row.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
import astropy.units as u
import fitsio
import numpy as np
from numpy.lib import recfunctions as rfn
from desitarget.targetmask import bgs_mask


BRIGHT_BITS = int(
    bgs_mask.BGS_BRIGHT | bgs_mask.BGS_BRIGHT_NORTH | bgs_mask.BGS_BRIGHT_SOUTH
)
FAINT_BITS = int(
    bgs_mask.BGS_FAINT
    | bgs_mask.BGS_FAINT_HIP
    | bgs_mask.BGS_FAINT_NORTH
    | bgs_mask.BGS_FAINT_SOUTH
)
FAINT_NORTH_BITS = int(bgs_mask.BGS_FAINT_NORTH)
FAINT_SOUTH_BITS = int(bgs_mask.BGS_FAINT_SOUTH)
SENTINEL_ZWARN = 999999
DELTACHI2_MIN = 25.0
Z_CONTEXT = (0.10, 0.60)
Z_SENTINEL = (0.585, 0.595)

DEFAULT_STAGE3 = Path(
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322"
)
DEFAULT_BRIGHT_PARENT = Path(
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
    "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
    "ngrid2048_thr0p2_halo_xcom.fits"
)
DEFAULT_BRIGHT_POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)
DEFAULT_BRIGHT_INDEX = Path(
    "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"
)
DEFAULT_ZALL = Path(
    "/global/cfs/cdirs/desi/public/dr2/spectro/redux/loa/zcatalog/v1/"
    "zall-pix-loa.fits"
)
DEFAULT_OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/catalogues"
)


CATALOGUE_DTYPE = np.dtype(
    [
        ("TARGETID", "i8"),
        ("RA", "f8"),
        ("DEC", "f8"),
        ("Z", "f8"),
        ("Z_COSMO", "f8"),
        ("R_MAG_APP", "f4"),
        ("FILE_NUM", "i4"),
        ("BOX_INDEX", "i4"),
        ("HALO_INDEX", "i8"),
        ("BGS_TARGET", "i8"),
        ("TRACER_TYPE", "u1"),  # 0=Bright, 1=Faint
        ("ASSIGNED", "u1"),
        ("SPEC_SUCCESS", "u1"),
        ("CONTEXT", "u1"),
        ("BRIGHT_PARENT_ID", "i8"),
        ("SOURCE_ROW", "i8"),
    ]
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
    parser.add_argument(
        "--spectroscopic-join",
        type=Path,
        default=DEFAULT_STAGE3 / "loa-v1/mock0/datcomb_bright_tarspecwdup_zdone.fits",
    )
    parser.add_argument(
        "--target-input",
        type=Path,
        default=DEFAULT_STAGE3 / "inputs/targ.fits",
        help="Immutable unique target table supplying Faint truth/sky columns.",
    )
    parser.add_argument("--bright-parent", type=Path, default=DEFAULT_BRIGHT_PARENT)
    parser.add_argument("--bright-points", type=Path, default=DEFAULT_BRIGHT_POINTS)
    parser.add_argument("--bright-index", type=Path, default=DEFAULT_BRIGHT_INDEX)
    parser.add_argument("--zall", type=Path, default=DEFAULT_ZALL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=824731)
    parser.add_argument("--chunk-rows", type=int, default=2_000_000)
    parser.add_argument(
        "--repair-proxy-from-oracle",
        action="store_true",
        help=(
            "Rebuild only BF_PROXY_RESPONSE_v1 from the passed Oracle catalogue, "
            "applying the response by BGS target-selection bit."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _hdu(path: Path):
    handle = fitsio.FITS(str(path), "r")
    return handle, handle[1]


def _column_names(hdu) -> set[str]:
    return set(hdu.get_colnames())


def deterministic_uniform(targetid: np.ndarray, seed: int) -> np.ndarray:
    """Order-independent SplitMix64 variates keyed by TARGETID and seed."""
    value = np.asarray(targetid, dtype=np.uint64) + np.uint64(seed)
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return ((value >> np.uint64(11)).astype(np.float64) * (1.0 / 2.0**53))


def faint_response_probability(
    target_bits: np.ndarray, calibration: dict
) -> tuple[np.ndarray, dict]:
    """Map calibrated response by BGS imaging/target-selection bit.

    ``BGS_FAINT_NORTH`` and ``BGS_FAINT_SOUTH`` describe the target-selection
    system. They must not be inferred from Galactic NGC/SGC membership. Rows
    carrying neither regional bit use the explicitly recorded overall rate;
    carrying both is an invalid target definition and is a hard failure.
    """
    target = np.asarray(target_bits, dtype=np.int64)
    north = (target & FAINT_NORTH_BITS) != 0
    south = (target & FAINT_SOUTH_BITS) != 0
    ambiguous = north & south
    if np.any(ambiguous):
        raise RuntimeError(
            f"{int(np.count_nonzero(ambiguous))} Faint rows carry both NORTH and SOUTH bits"
        )
    fallback = ~(north | south)
    probability = np.full(
        len(target), calibration["all"]["pass_probability"], dtype=np.float64
    )
    probability[north] = calibration["north"]["pass_probability"]
    probability[south] = calibration["south"]["pass_probability"]
    regional_rows = int(np.count_nonzero(north | south))
    fallback_rows = int(np.count_nonzero(fallback))
    if regional_rows == 0:
        mapping = (
            "overall BGS_FAINT rate marginalized over DESI PHOTSYS; mock has "
            "no PHOTSYS or regional target-selection bits"
        )
    else:
        mapping = (
            "BGS_FAINT_NORTH/SOUTH target-selection bits where available; "
            "overall PHOTSYS-marginal rate otherwise; never Galactic cap"
        )
    audit = {
        "mapping": mapping,
        "north_rows": int(np.count_nonzero(north)),
        "south_rows": int(np.count_nonzero(south)),
        "overall_fallback_rows": fallback_rows,
        "ambiguous_rows": int(np.count_nonzero(ambiguous)),
    }
    return probability, audit


def sky_to_points(ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    distance = Planck18.comoving_distance(np.asarray(redshift, dtype=np.float64)).value
    sky = SkyCoord(
        ra=np.asarray(ra, dtype=np.float64) * u.deg,
        dec=np.asarray(dec, dtype=np.float64) * u.deg,
        distance=distance * u.Mpc,
        frame="icrs",
    )
    cart = sky.cartesian
    cap = (sky.galactic.b.deg > 0).astype(np.float64)
    return np.column_stack(
        [
            cart.x.to_value(u.Mpc),
            cart.y.to_value(u.Mpc),
            cart.z.to_value(u.Mpc),
            cap,
        ]
    ).astype(np.float64)


def calibrate_faint_response(zall: Path, chunk_rows: int) -> dict:
    """Fit independent Faint success rates overall and by north/south target bit."""
    counts = {"all": [0, 0], "north": [0, 0], "south": [0, 0]}
    with fitsio.FITS(str(zall), "r") as handle:
        hdu = handle["ZCATALOG"] if "ZCATALOG" in handle else handle[1]
        nrows = int(hdu.get_nrows())
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            table = hdu.read(
                rows=rows,
                columns=["BGS_TARGET", "ZWARN", "DELTACHI2", "SPECTYPE", "PHOTSYS"],
            )
            target = np.asarray(table["BGS_TARGET"], dtype=np.int64)
            faint = (target & FAINT_BITS) != 0
            if not np.any(faint):
                continue
            spectype = np.char.strip(np.asarray(table["SPECTYPE"]).astype("U16"))
            passed = (
                (np.asarray(table["ZWARN"]) == 0)
                & (np.asarray(table["DELTACHI2"], dtype=np.float64) >= DELTACHI2_MIN)
                & (spectype == "GALAXY")
            )
            photsys = np.char.upper(
                np.char.strip(np.asarray(table["PHOTSYS"]).astype("U1"))
            )
            masks = {
                "all": faint,
                "north": faint & (photsys == "N"),
                "south": faint & (photsys == "S"),
            }
            for name, selected in masks.items():
                counts[name][0] += int(np.count_nonzero(selected))
                counts[name][1] += int(np.count_nonzero(selected & passed))
    if counts["all"][0] == 0:
        raise RuntimeError("no BGS_FAINT rows found in the DESI zcatalog")
    overall = counts["all"][1] / counts["all"][0]
    output = {}
    for name, (total, passed) in counts.items():
        probability = passed / total if total >= 100 else overall
        output[name] = {
            "rows": total,
            "passing_rows": passed,
            "pass_probability": probability,
            "fallback_to_overall": total < 100,
            "calibration_basis": "DESI LOA PHOTSYS",
        }
    return output


def _read_faint_candidates(path: Path, target_input: Path) -> tuple[np.ndarray, dict]:
    """Join assignment state onto immutable unique Faint target rows.

    The repeated spectroscopic join is not a truth table: alternate-tile rows can
    carry fill values in immutable columns. It is used only to establish whether
    a TARGETID was fibre observed.
    """
    assignment_columns = ["TARGETID", "BGS_TARGET", "ZWARN", "TILELOCID"]
    with fitsio.FITS(str(path), "r") as handle:
        hdu = handle[1]
        missing = set(assignment_columns).difference(_column_names(hdu))
        if missing:
            raise KeyError(f"spectroscopic join lacks required columns: {sorted(missing)}")
        assignment = hdu.read(columns=assignment_columns)
    assignment_target = np.asarray(assignment["BGS_TARGET"], dtype=np.int64)
    assignment_faint = (assignment_target & FAINT_BITS) != 0
    assignment_bright = (assignment_target & BRIGHT_BITS) != 0
    if np.any(assignment_faint & assignment_bright):
        raise RuntimeError("spectroscopic join contains ambiguous Bright/Faint rows")
    assignment_source_row = np.flatnonzero(assignment_faint).astype(np.int64)
    assignment = assignment[assignment_faint]
    observed = np.asarray(assignment["ZWARN"], dtype=np.int64) != SENTINEL_ZWARN
    tileloc = np.asarray(assignment["TILELOCID"], dtype=np.int64)
    order = np.lexsort(
        (assignment_source_row, tileloc, ~observed, assignment["TARGETID"])
    )
    assignment = assignment[order]
    assignment_source_row = assignment_source_row[order]
    assignment_targetid = np.asarray(assignment["TARGETID"], dtype=np.int64)
    first = np.r_[True, assignment_targetid[1:] != assignment_targetid[:-1]]
    assignment = assignment[first]
    assignment_source_row = assignment_source_row[first]
    assignment_observed = (
        np.asarray(assignment["ZWARN"], dtype=np.int64) != SENTINEL_ZWARN
    )
    observed_id = np.asarray(assignment["TARGETID"][assignment_observed], dtype=np.int64)
    observed_source_row = assignment_source_row[assignment_observed]
    observed_response_bits = np.asarray(
        assignment["BGS_TARGET"][assignment_observed], dtype=np.int64
    )
    observed_order = np.argsort(observed_id)
    observed_id = observed_id[observed_order]
    observed_source_row = observed_source_row[observed_order]
    observed_response_bits = observed_response_bits[observed_order]

    truth_columns = [
        "TARGETID", "BGS_TARGET", "R_MAG_APP", "RA", "DEC", "Z_COSMO",
        "RSDZ", "ZWARN", "FILE_NUM", "BOX_INDEX", "HALO_INDEX",
    ]
    with fitsio.FITS(str(target_input), "r") as handle:
        hdu = handle[1]
        missing = set(truth_columns).difference(_column_names(hdu))
        if missing:
            raise KeyError(f"target input lacks required columns: {sorted(missing)}")
        table = hdu.read(columns=truth_columns)
    target = np.asarray(table["BGS_TARGET"], dtype=np.int64)
    faint = (target & FAINT_BITS) != 0
    bright = (target & BRIGHT_BITS) != 0
    if np.any(faint & bright):
        raise RuntimeError("target input contains ambiguous Bright/Faint rows")
    table = table[faint]
    targetid = np.asarray(table["TARGETID"], dtype=np.int64)
    if len(np.unique(targetid)) != len(targetid):
        raise RuntimeError("Faint target input TARGETID is not unique")
    target_order = np.argsort(targetid)
    targetid_sorted = targetid[target_order]
    position = np.searchsorted(targetid_sorted, observed_id)
    matched = (
        (position < len(targetid_sorted))
        & (targetid_sorted[np.minimum(position, len(targetid_sorted) - 1)] == observed_id)
    )
    unique = table[target_order[position[matched]]]
    unique_source = observed_source_row[matched]
    unique_response_bits = observed_response_bits[matched]
    finite = np.isfinite(np.asarray(unique["RSDZ"], dtype=np.float64)) & (
        np.asarray(unique["RSDZ"], dtype=np.float64) > 0
    )
    unique = unique[finite]
    unique_source = unique_source[finite]
    unique_response_bits = unique_response_bits[finite]
    unique["BGS_TARGET"] = unique_response_bits
    _, response_audit = faint_response_probability(
        unique_response_bits,
        {
            "all": {"pass_probability": 0.0},
            "north": {"pass_probability": 0.0},
            "south": {"pass_probability": 0.0},
        },
    )
    audit = {
        "spectroscopic_join_rows": int(len(assignment_target)),
        "faint_repeated_assignment_rows": int(np.count_nonzero(assignment_faint)),
        "faint_unique_assignment_ids": int(len(assignment)),
        "fibre_observed_unique_assignment_ids": int(len(observed_id)),
        "target_input_rows": int(len(target)),
        "faint_unique_target_rows": int(len(table)),
        "observed_ids_matched_to_target_input": int(np.count_nonzero(matched)),
        "observed_ids_unmatched_to_target_input": int(np.count_nonzero(~matched)),
        "finite_positive_rsdz_matched": int(len(unique)),
        "eligible_oracle_unique": int(len(unique)),
        "deduplication": (
            "assignment state only: TARGETID; prefer fibre-observed, then smallest "
            "TILELOCID, then earliest source row; immutable values come from target input"
        ),
        "response_target_bits": response_audit,
    }
    return rfn.append_fields(
        unique, "SOURCE_ROW", unique_source, usemask=False
    ), audit


def _read_bright(parent: Path, bright_index: Path) -> tuple[np.ndarray, np.ndarray]:
    requested = [
        "TARGETID", "RA", "DEC", "Z", "Z_COSMO", "R_MAG_APP",
        "FILE_NUM", "BOX_INDEX", "HALO_INDEX", "BGS_TARGET",
    ]
    with fitsio.FITS(str(parent), "r") as handle:
        hdu = handle[1]
        names = _column_names(hdu)
        required = {"TARGETID", "RA", "DEC", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"}
        if not required.issubset(names):
            raise KeyError(f"Bright parent lacks {sorted(required.difference(names))}")
        table = hdu.read(columns=[name for name in requested if name in names])
    index = np.load(bright_index)
    context = np.asarray(index["context"], dtype=bool)
    if len(table) != len(context):
        raise RuntimeError("Bright parent/index length mismatch")
    output = np.empty(len(table), dtype=CATALOGUE_DTYPE)
    for name in ("TARGETID", "RA", "DEC", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"):
        output[name] = table[name]
    output["Z_COSMO"] = table["Z_COSMO"] if "Z_COSMO" in table.dtype.names else np.nan
    output["R_MAG_APP"] = table["R_MAG_APP"] if "R_MAG_APP" in table.dtype.names else np.nan
    output["BGS_TARGET"] = table["BGS_TARGET"] if "BGS_TARGET" in table.dtype.names else BRIGHT_BITS
    output["TRACER_TYPE"] = 0
    output["ASSIGNED"] = 1
    output["SPEC_SUCCESS"] = 1
    output["CONTEXT"] = context.astype(np.uint8)
    output["BRIGHT_PARENT_ID"] = np.arange(len(table), dtype=np.int64)
    output["SOURCE_ROW"] = np.arange(len(table), dtype=np.int64)
    if len(np.unique(output["TARGETID"])) != len(output):
        raise RuntimeError("frozen Bright parent TARGETID is not unique")
    return output, context


def _make_faint_rows(source: np.ndarray, keep: np.ndarray) -> np.ndarray:
    source = source[keep]
    if source.dtype == CATALOGUE_DTYPE:
        output = np.array(source, copy=True)
        output["TRACER_TYPE"] = 1
        output["ASSIGNED"] = 1
        output["SPEC_SUCCESS"] = 1
        output["BRIGHT_PARENT_ID"] = -1
        return output
    output = np.empty(len(source), dtype=CATALOGUE_DTYPE)
    for name in (
        "TARGETID", "RA", "DEC", "Z_COSMO", "R_MAG_APP", "FILE_NUM",
        "BOX_INDEX", "HALO_INDEX", "BGS_TARGET", "SOURCE_ROW",
    ):
        output[name] = source[name]
    output["Z"] = source["RSDZ"]
    output["TRACER_TYPE"] = 1
    output["ASSIGNED"] = 1
    output["SPEC_SUCCESS"] = 1
    z = np.asarray(output["Z"], dtype=np.float64)
    output["CONTEXT"] = (
        (z >= Z_CONTEXT[0])
        & (z < Z_CONTEXT[1])
        & ~((z >= Z_SENTINEL[0]) & (z < Z_SENTINEL[1]))
    ).astype(np.uint8)
    output["BRIGHT_PARENT_ID"] = -1
    return output


def repair_proxy_from_oracle(args: argparse.Namespace) -> dict:
    """Repair only the Proxy response without rereading the full staged mock."""
    oracle_manifest_path = args.output_root / "bf_oracle_assigned_v1/manifest.json"
    proxy_manifest_path = args.output_root / "bf_proxy_response_v1/manifest.json"
    archive_path = args.output_root / "invalidated_proxy_capmapped_manifest.json"
    if not oracle_manifest_path.exists():
        raise FileNotFoundError("Proxy repair requires the passed Oracle manifest")
    oracle_manifest = json.loads(oracle_manifest_path.read_text())
    if proxy_manifest_path.exists():
        old_proxy_manifest = json.loads(proxy_manifest_path.read_text())
        archived = {
            "invalidated_utc": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "LOA north/south response was incorrectly mapped by Galactic cap; "
                "the calibration itself remains valid"
            ),
            "manifest_sha256": sha256(proxy_manifest_path),
            "manifest": old_proxy_manifest,
        }
        atomic_json(archive_path, archived)
    elif archive_path.exists():
        archived = json.loads(archive_path.read_text())
        old_proxy_manifest = archived["manifest"]
    else:
        raise FileNotFoundError("Proxy repair requires a live or archived Proxy manifest")
    if not oracle_manifest.get("pass"):
        raise RuntimeError("Oracle catalogue has not passed its frozen gates")
    calibration = calibrate_faint_response(args.zall, args.chunk_rows)

    oracle_catalogue = Path(oracle_manifest["catalogue"])
    with fitsio.FITS(str(oracle_catalogue), "r") as handle:
        rows = handle[1].read()
    bright_rows = int(oracle_manifest["bright_prefix_rows"])
    bright = np.asarray(rows[:bright_rows], dtype=CATALOGUE_DTYPE)
    faint_source = np.asarray(rows[bright_rows:], dtype=CATALOGUE_DTYPE)
    probability, response_audit = faint_response_probability(
        faint_source["BGS_TARGET"], calibration
    )
    if response_audit["overall_fallback_rows"]:
        raise RuntimeError(
            "Oracle lacks assignment-time regional response bits; run the full rebuild"
        )
    keep_proxy = (
        deterministic_uniform(faint_source["TARGETID"], args.seed) < probability
    )
    inputs = dict(oracle_manifest["inputs"])
    inputs.update(
        {
            "repair_source_oracle_manifest": str(oracle_manifest_path),
            "repair_source_oracle_manifest_sha256": sha256(oracle_manifest_path),
            "invalidated_proxy_manifest": str(archive_path),
            "invalidated_proxy_manifest_sha256": sha256(archive_path),
        }
    )
    proxy = write_product(
        name="BF_PROXY_RESPONSE_v1",
        scope="DEVELOPMENT_PROXY_RESPONSE",
        bright=bright,
        bright_points_path=args.bright_points,
        faint_source=faint_source,
        faint_keep=keep_proxy,
        output_root=args.output_root,
        inputs=inputs,
        response={
            "fibre_assignment": "ZWARN != 999999 in the staged spectroscopic join",
            "redshift_success": (
                "deterministic LOA BGS_FAINT draw calibrated by DESI PHOTSYS "
                "and applied using assignment-time NORTH/SOUTH target-selection bits"
            ),
            "calibration_basis": "DESI LOA PHOTSYS",
            "seed": args.seed,
            "calibration": calibration,
            "application_audit": response_audit,
            "magnitude_conditioning": (
                "unavailable in zall-pix zcatalog; marginalized over the DESI Faint population"
            ),
            "final_selection_photometry": (
                "not available in current CutSky; this remains a proxy, not final BGS_FAINT"
            ),
            "production_eligible": False,
        },
        force=True,
    )
    summary_path = args.output_root / "catalogue_build_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(
        {
            "oracle": str(oracle_manifest_path),
            "oracle_rows": int(oracle_manifest["total_rows"]),
            "proxy": str(args.output_root / "bf_proxy_response_v1/manifest.json"),
            "proxy_rows": int(proxy["total_rows"]),
            "faint_response_calibration": calibration,
            "proxy_response_application_audit": response_audit,
            "proxy_repaired_from_oracle": True,
        }
    )
    atomic_json(summary_path, summary)
    return summary


def write_product(
    *,
    name: str,
    scope: str,
    bright: np.ndarray,
    bright_points_path: Path,
    faint_source: np.ndarray,
    faint_keep: np.ndarray,
    output_root: Path,
    inputs: dict,
    response: dict,
    force: bool,
) -> dict:
    output = output_root / name.lower()
    manifest_path = output / "manifest.json"
    marker_path = output / "CATALOGUE_COMPLETE"
    if marker_path.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    if output.exists() and any(output.iterdir()) and not force:
        raise RuntimeError(f"ambiguous existing product: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if force and path.is_file():
            path.unlink()

    faint = _make_faint_rows(faint_source, faint_keep)
    if np.intersect1d(bright["TARGETID"], faint["TARGETID"], assume_unique=True).size:
        raise RuntimeError("Bright/Faint TARGETID overlap")
    catalogue_path = output / "catalogue.fits"
    handle = fitsio.FITS(str(catalogue_path), "rw", clobber=True)
    try:
        handle.write(bright)
        handle[1].append(faint)
    finally:
        handle.close()

    bright_points = np.load(bright_points_path, mmap_mode="r")
    if bright_points.shape != (len(bright), 4):
        raise RuntimeError("Bright points do not align to the frozen parent")
    points_path = output / "points.npy"
    points = np.lib.format.open_memmap(
        points_path, mode="w+", dtype=np.float64, shape=(len(bright) + len(faint), 4)
    )
    points[: len(bright)] = bright_points
    points[len(bright):] = sky_to_points(faint["RA"], faint["DEC"], faint["Z"])
    points.flush()
    del points
    points_check = np.load(points_path, mmap_mode="r")
    if not np.array_equal(points_check[: len(bright)], bright_points):
        raise RuntimeError("Bright point prefix changed")

    tracer = np.r_[
        np.zeros(len(bright), dtype=np.uint8), np.ones(len(faint), dtype=np.uint8)
    ]
    context = np.r_[bright["CONTEXT"], faint["CONTEXT"]].astype(bool)
    cap = np.asarray(points_check[:, 3], dtype=np.uint8)
    index_path = output / "catalogue_index.npz"
    np.savez_compressed(
        index_path,
        node_id=np.arange(len(tracer), dtype=np.int64),
        tracer_type=tracer,
        context=context,
        cap=cap,
        bright_parent_id=np.r_[
            np.arange(len(bright), dtype=np.int64),
            np.full(len(faint), -1, dtype=np.int64),
        ],
    )
    selected_ids_path = output / "faint_targetid.npy"
    np.save(selected_ids_path, np.asarray(faint["TARGETID"], dtype=np.int64))

    shell_edges = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55])
    counts = {}
    for tracer_id, tracer_name in ((0, "BRIGHT"), (1, "FAINT")):
        rows = bright if tracer_id == 0 else faint
        row_points = points_check[: len(bright)] if tracer_id == 0 else points_check[len(bright):]
        row_cap = np.asarray(row_points[:, 3], dtype=np.uint8)
        counts[tracer_name] = {
            "rows": int(len(rows)),
            "context": int(np.count_nonzero(rows["CONTEXT"])),
            "NGC": int(np.count_nonzero(row_cap == 1)),
            "SGC": int(np.count_nonzero(row_cap == 0)),
            "shells": {
                f"{shell_edges[i]:.2f}_{shell_edges[i + 1]:.2f}": int(
                    np.count_nonzero(
                        (rows["Z"] >= shell_edges[i])
                        & (rows["Z"] < shell_edges[i + 1])
                    )
                )
                for i in range(4)
            },
            "valid_truth_link": int(
                np.count_nonzero(
                    (rows["FILE_NUM"] >= 0)
                    & (rows["BOX_INDEX"] >= 0)
                    & (rows["HALO_INDEX"] >= 0)
                )
            ),
        }
    gates = {
        "targetid_unique": len(np.unique(np.r_[bright["TARGETID"], faint["TARGETID"]]))
        == len(bright) + len(faint),
        "bright_prefix_exact": bool(np.array_equal(points_check[: len(bright)], bright_points)),
        "bright_parent_identity": bool(
            np.array_equal(bright["BRIGHT_PARENT_ID"], np.arange(len(bright)))
        ),
        "faint_context_only": bool(np.all(faint["BRIGHT_PARENT_ID"] == -1)),
        "two_caps": bool(set(np.unique(cap)) == {0, 1}),
        "finite_points": bool(np.all(np.isfinite(points_check))),
        "finite_positive_redshift": bool(np.all(np.isfinite(faint["Z"])) and np.all(faint["Z"] > 0)),
    }
    if not all(gates.values()):
        raise RuntimeError(f"{name} catalogue gates failed: {gates}")
    manifest = {
        "schema_version": "p8-multitracer-catalogue-v1",
        "product": name,
        "scope": scope,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "catalogue": str(catalogue_path),
        "catalogue_sha256": sha256(catalogue_path),
        "points": str(points_path),
        "points_sha256": sha256(points_path),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "faint_targetid": str(selected_ids_path),
        "faint_targetid_sha256": sha256(selected_ids_path),
        "bright_prefix_rows": int(len(bright)),
        "faint_context_rows": int(len(faint)),
        "total_rows": int(len(bright) + len(faint)),
        "counts": counts,
        "response": response,
        "inputs": inputs,
        "supervision_contract": {
            "supervised_tracer": "BGS_BRIGHT only",
            "context_only_tracer": "BGS_FAINT",
            "authoritative_rows": "unchanged P4 Bright parent rows",
            "faint_predictions_released": False,
        },
        "coordinate_contract": {
            "frame": "ICRS Cartesian comoving",
            "units": "Mpc",
            "cosmology": "Planck18",
            "redshift": "RSDZ for Faint; frozen observed Z for Bright",
            "caps": {"0": "SGC", "1": "NGC"},
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    atomic_json(manifest_path, manifest)
    marker_path.write_text(
        f"product={name}\nmanifest_sha256={sha256(manifest_path)}\n"
        f"bright_rows={len(bright)}\nfaint_rows={len(faint)}\n"
    )
    return manifest


def main() -> None:
    args = parse_args()
    if args.repair_proxy_from_oracle:
        for path in (args.bright_points, args.output_root, args.zall):
            if not path.exists():
                raise FileNotFoundError(path)
        summary = repair_proxy_from_oracle(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    for path in (
        args.spectroscopic_join,
        args.target_input,
        args.bright_parent,
        args.bright_points,
        args.bright_index,
        args.zall,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    bright, _ = _read_bright(args.bright_parent, args.bright_index)
    faint_source, dedup_audit = _read_faint_candidates(
        args.spectroscopic_join, args.target_input
    )
    unmatched = dedup_audit["observed_ids_unmatched_to_target_input"]
    if unmatched:
        raise RuntimeError(
            f"{unmatched} observed BGS_FAINT TARGETIDs do not match inputs/targ.fits"
        )
    calibration = calibrate_faint_response(args.zall, args.chunk_rows)
    probability, response_audit = faint_response_probability(
        faint_source["BGS_TARGET"], calibration
    )
    if response_audit["overall_fallback_rows"]:
        raise RuntimeError(
            "observed Faint rows lack assignment-time regional target-selection bits"
        )
    keep_proxy = deterministic_uniform(faint_source["TARGETID"], args.seed) < probability
    inputs = {
        "spectroscopic_join": str(args.spectroscopic_join),
        "spectroscopic_join_sha256": sha256(args.spectroscopic_join),
        "target_input": str(args.target_input),
        "target_input_sha256": sha256(args.target_input),
        "bright_parent": str(args.bright_parent),
        "bright_parent_sha256": sha256(args.bright_parent),
        "bright_points": str(args.bright_points),
        "bright_points_sha256": sha256(args.bright_points),
        "bright_index": str(args.bright_index),
        "bright_index_sha256": sha256(args.bright_index),
        "desi_zcatalog": str(args.zall),
        "desi_zcatalog_sha256": sha256(args.zall),
        "faint_deduplication_audit": dedup_audit,
    }
    oracle = write_product(
        name="BF_ORACLE_ASSIGNED_v1",
        scope="ORACLE_REDSHIFT_INFORMATION_UPPER_BOUND",
        bright=bright,
        bright_points_path=args.bright_points,
        faint_source=faint_source,
        faint_keep=np.ones(len(faint_source), dtype=bool),
        output_root=args.output_root,
        inputs=inputs,
        response={
            "fibre_assignment": "ZWARN != 999999 in the staged spectroscopic join",
            "redshift_success": "oracle; no post-assignment failure",
            "production_eligible": False,
        },
        force=args.force,
    )
    proxy = write_product(
        name="BF_PROXY_RESPONSE_v1",
        scope="DEVELOPMENT_PROXY_RESPONSE",
        bright=bright,
        bright_points_path=args.bright_points,
        faint_source=faint_source,
        faint_keep=keep_proxy,
        output_root=args.output_root,
        inputs=inputs,
        response={
            "fibre_assignment": "ZWARN != 999999 in the staged spectroscopic join",
            "redshift_success": (
                "deterministic LOA BGS_FAINT draw calibrated by DESI PHOTSYS "
                "and applied using assignment-time NORTH/SOUTH target-selection bits"
            ),
            "seed": args.seed,
            "calibration": calibration,
            "application_audit": response_audit,
            "calibration_basis": "DESI LOA PHOTSYS",
            "magnitude_conditioning": (
                "unavailable in zall-pix zcatalog; marginalized over the DESI Faint population"
            ),
            "final_selection_photometry": (
                "not available in current CutSky; this remains a proxy, not final BGS_FAINT"
            ),
            "production_eligible": False,
        },
        force=args.force,
    )
    summary = {
        "oracle": str(args.output_root / "bf_oracle_assigned_v1/manifest.json"),
        "oracle_rows": oracle["total_rows"],
        "proxy": str(args.output_root / "bf_proxy_response_v1/manifest.json"),
        "proxy_rows": proxy["total_rows"],
        "faint_response_calibration": calibration,
    }
    atomic_json(args.output_root / "catalogue_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
