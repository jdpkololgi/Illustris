#!/usr/bin/env python3
"""Reconstruct the compact P10 BGS-BRIGHT parent catalogue with halo linkage.

The public ``forFA{mock}_nomask.fits`` products deliberately omit the CutSky
``(FILE_NUM, HALO_INDEX, BOX_INDEX)`` columns needed for T-web truth.  Their
TARGETIDs are, however, assigned sequentially *after* the deterministic
BRIGHT magnitude and DESI bright-tile footprint selection.  This command
replays that exact selection on the registered CutSky catalogue, retains the
linkage columns, and proves row-for-row identity against the public forFA
BRIGHT block before writing an atomic completion manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import fitsio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import (  # noqa: E402
    DEFAULT_REGISTRY,
    expand_phase,
    load_registry,
    sha256_file,
)


BRIGHT_BIT = 2
DEFAULT_R_LIMIT = 19.5
CUTSKY_COLUMNS = (
    "RA",
    "DEC",
    "Z",
    "Z_COSMO",
    "R_MAG_APP",
    "FILE_NUM",
    "HALO_INDEX",
    "BOX_INDEX",
)


class BrightParentError(RuntimeError):
    """The reconstructed parent catalogue differs from the public forFA contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def default_output(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/bright_parent" / f"{phase}_bgs_bright_parent_linkage.fits"


def compact_parent_block(selected: np.ndarray, target_start: int) -> np.ndarray:
    dtype = [
        ("TARGETID", "i8"),
        ("RA", "f8"),
        ("DEC", "f8"),
        ("Z", "f4"),
        ("Z_COSMO", "f4"),
        ("R_MAG_APP", "f4"),
        ("BGS_TARGET", "i8"),
        ("FILE_NUM", "i4"),
        ("HALO_INDEX", "i4"),
        ("BOX_INDEX", "i4"),
    ]
    block = np.empty(len(selected), dtype=dtype)
    block["TARGETID"] = np.arange(target_start, target_start + len(selected), dtype=np.int64)
    for column in ("RA", "DEC", "Z", "Z_COSMO", "R_MAG_APP", "FILE_NUM", "HALO_INDEX", "BOX_INDEX"):
        block[column] = selected[column]
    block["BGS_TARGET"] = BRIGHT_BIT
    return block


def select_bright_chunk(
    table: np.ndarray,
    *,
    r_limit: float,
    footprint_selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    bright = np.asarray(table["R_MAG_APP"] < r_limit)
    indices = np.flatnonzero(bright)
    if not len(indices):
        return table[:0]
    inside = np.asarray(
        footprint_selector(table["RA"][indices], table["DEC"][indices]),
        dtype=bool,
    )
    if inside.shape != (len(indices),):
        raise BrightParentError("footprint selector returned an invalid shape")
    return table[indices[inside]]


def scan_forfa_bright_contract(path: Path, chunk_size: int) -> dict[str, int]:
    n_bright = 0
    first_non_bright = None
    nonsequential = 0
    with fitsio.FITS(path) as fits:
        hdu = fits[1]
        nrows = int(hdu.get_nrows())
        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            data = hdu[start:stop][["TARGETID", "BGS_TARGET"]]
            target_expected = np.arange(start + 1, stop + 1, dtype=np.int64)
            nonsequential += int(np.count_nonzero(data["TARGETID"] != target_expected))
            is_bright = (data["BGS_TARGET"] & BRIGHT_BIT) != 0
            n_bright += int(is_bright.sum())
            if first_non_bright is None and np.any(~is_bright):
                first_non_bright = start + int(np.flatnonzero(~is_bright)[0])
            if first_non_bright is not None and np.any(is_bright & (target_expected > first_non_bright + 1)):
                raise BrightParentError("forFA BRIGHT rows are not one contiguous leading block")
    if nonsequential:
        raise BrightParentError(f"forFA has {nonsequential} non-sequential TARGETIDs")
    if first_non_bright is not None and first_non_bright != n_bright:
        raise BrightParentError("forFA BRIGHT block boundary/count mismatch")
    return {"forfa_rows": nrows, "bright_rows": n_bright}


def validate_parent_against_forfa(parent: Path, forfa: Path, n_bright: int, chunk_size: int) -> dict[str, Any]:
    maxima = {"RA": 0.0, "DEC": 0.0, "Z": 0.0, "Z_COSMO": 0.0, "R_MAG_APP": 0.0}
    with fitsio.FITS(parent) as pf, fitsio.FITS(forfa) as ff:
        if int(pf[1].get_nrows()) != n_bright:
            raise BrightParentError("parent row count does not equal forFA BRIGHT count")
        for start in range(0, n_bright, chunk_size):
            stop = min(start + chunk_size, n_bright)
            p = pf[1][start:stop]
            f = ff[1][start:stop][["TARGETID", "RA", "DEC", "TRUEZ", "RSDZ", "R_MAG_APP", "BGS_TARGET"]]
            expected = np.arange(start + 1, stop + 1, dtype=np.int64)
            if not np.array_equal(p["TARGETID"], expected) or not np.array_equal(f["TARGETID"], expected):
                raise BrightParentError(f"TARGETID mismatch at rows [{start},{stop})")
            if np.any((f["BGS_TARGET"] & BRIGHT_BIT) == 0):
                raise BrightParentError(f"non-BRIGHT forFA row inside leading block [{start},{stop})")
            pairs = {
                "RA": (p["RA"], f["RA"]),
                "DEC": (p["DEC"], f["DEC"]),
                "Z": (p["Z"], f["RSDZ"]),
                "Z_COSMO": (p["Z_COSMO"], f["TRUEZ"]),
                "R_MAG_APP": (p["R_MAG_APP"], f["R_MAG_APP"]),
            }
            for name, (left, right) in pairs.items():
                maxima[name] = max(maxima[name], float(np.max(np.abs(left - right), initial=0.0)))
    if maxima["RA"] != 0.0 or maxima["DEC"] != 0.0:
        raise BrightParentError(f"parent sky coordinates differ from forFA: {maxima}")
    if max(maxima["Z"], maxima["Z_COSMO"], maxima["R_MAG_APP"]) > 2e-7:
        raise BrightParentError(f"parent float32 quantities differ from forFA: {maxima}")
    return {"row_identity_verified": True, "max_abs_differences": maxima}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--r-limit", type=float, default=DEFAULT_R_LIMIT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise BrightParentError(f"unregistered phase: {args.phase}")
    phase_cfg = registry["phases"][args.phase]
    if phase_cfg["role"] == "sealed_blind":
        raise BrightParentError("refusing to construct a ph001 truth-linkage product before unsealing")
    assets = expand_phase(registry, args.phase)["assets"]
    cutsky = Path(assets["cutsky"])
    forfa = Path(assets["forfa"])
    output = args.output or default_output(registry, args.phase)
    marker = output.with_suffix(output.suffix + ".complete.json")
    if output.exists() or marker.exists():
        raise BrightParentError(f"refusing to overwrite existing parent artifact: {output} / {marker}")

    from astropy.table import Table
    from desimodel.footprint import is_point_in_desi

    tiles_path = Path("/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits")
    tiles = Table.read(tiles_path)
    contract = scan_forfa_bright_contract(forfa, args.chunk_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer = fitsio.FITS(temporary, "rw", clobber=True)
    first = True
    next_targetid = 1
    source_rows = 0
    try:
        with fitsio.FITS(cutsky) as fits:
            hdu = fits[1]
            source_rows = int(hdu.get_nrows())
            for start in range(0, source_rows, args.chunk_size):
                stop = min(start + args.chunk_size, source_rows)
                chunk = hdu[start:stop][list(CUTSKY_COLUMNS)]
                selected = select_bright_chunk(
                    chunk,
                    r_limit=args.r_limit,
                    footprint_selector=lambda ra, dec: is_point_in_desi(tiles, ra, dec),
                )
                if len(selected):
                    block = compact_parent_block(selected, next_targetid)
                    if first:
                        writer.write(block, extname="PARENT")
                        first = False
                    else:
                        writer[-1].append(block)
                    next_targetid += len(block)
                print(f"rows {start:,}-{stop:,}; selected={next_targetid - 1:,}", flush=True)
    finally:
        writer.close()
    if first:
        temporary.unlink(missing_ok=True)
        raise BrightParentError("selection produced no BRIGHT parent rows")
    selected_rows = next_targetid - 1
    if selected_rows != contract["bright_rows"]:
        temporary.unlink(missing_ok=True)
        raise BrightParentError(
            f"CutSky selection produced {selected_rows:,} rows, forFA has "
            f"{contract['bright_rows']:,} BRIGHT rows"
        )
    try:
        parity = validate_parent_against_forfa(temporary, forfa, selected_rows, args.chunk_size)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, output)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    payload = {
        "schema_version": "p10-bright-parent-complete-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "role": phase_cfg["role"],
        "git_sha": git_sha,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "selection": {"r_mag_app_lt": args.r_limit, "footprint": str(tiles_path)},
        "cutsky": {"path": str(cutsky), "rows": source_rows},
        "forfa": {"path": str(forfa), **contract},
        "output": {
            "path": str(output.resolve()),
            "rows": selected_rows,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "parity": parity,
    }
    atomic_write_json(marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BrightParentError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
