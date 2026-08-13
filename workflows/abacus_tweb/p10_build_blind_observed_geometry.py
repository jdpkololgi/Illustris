#!/usr/bin/env python3
"""Join sealed-phase final LSS observations to geometry/host linkage only.

This is the ph001 counterpart of ``p10_build_observed_truth.py``.  It never opens
T-web, density, or annotated-parent products and its output dtype deliberately has
no target columns.  Stable TARGETID and host keys allow predictions to be frozen
now and truth to be joined only after the registered blind unsealing decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import fitsio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file  # noqa: E402
from p10_build_observed_truth import LSS_COLUMNS, default_lss, observed_success_mask  # noqa: E402


class BlindObservedError(RuntimeError):
    """A blind-geometry or linkage invariant failed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def default_parent(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/blind_parent" / f"{phase}_bgs_bright_parent_linkage.fits"


def default_output(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/blind_observed" / f"{phase}_bgs_bright_full_observed_geometry.fits"


def output_dtype() -> np.dtype:
    return np.dtype([
        ("TARGETID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("Z", "f8"),
        ("ZWARN", "i8"), ("BGS_TARGET", "i8"),
        ("R_MAG_APP", "f4"), ("R_MAG_ABS", "f4"),
        ("G_R_REST", "f4"), ("G_R_OBS", "f4"),
        ("NTILE", "i4"), ("COMP_TILE", "f8"),
        ("FRACZ_TILELOCID", "f8"), ("FRAC_TLOBS_TILES", "f8"),
        ("PHOTSYS", "S1"), ("MASKBITS", "i8"),
        ("NOBS_G", "i4"), ("NOBS_R", "i4"), ("NOBS_Z", "i4"),
        ("GOODPRI", "i1"), ("GOODHARDLOC", "i1"), ("LOCATION_ASSIGNED", "i1"),
        ("FILE_NUM", "i4"), ("HALO_INDEX", "i4"), ("BOX_INDEX", "i4"),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", default="ph001")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--lss", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    cfg = registry["phases"].get(args.phase)
    if not cfg or cfg["role"] != "sealed_blind":
        raise BlindObservedError("this truth-free builder is restricted to sealed ph001")
    parent_path = args.parent or default_parent(registry, args.phase)
    lss_path = args.lss or default_lss(registry, args.phase)
    output = args.output or default_output(registry, args.phase)
    marker = output.with_suffix(output.suffix + ".complete.json")
    if output.exists() or marker.exists():
        raise BlindObservedError(f"refusing to overwrite {output} / {marker}")

    parent_columns = [
        "TARGETID", "RA", "DEC", "R_MAG_APP", "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
    ]
    parent = fitsio.read(parent_path, columns=parent_columns)
    expected = np.arange(1, len(parent) + 1, dtype=np.int64)
    if not np.array_equal(parent["TARGETID"], expected):
        raise BlindObservedError("blind parent TARGETIDs are not sequential/index-aligned")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer = fitsio.FITS(temporary, "rw", clobber=True)
    first, output_rows = True, 0
    seen: list[np.ndarray] = []
    maxima = {"RA": 0.0, "DEC": 0.0}
    try:
        with fitsio.FITS(lss_path) as fits:
            hdu = fits[1]
            input_rows = int(hdu.get_nrows())
            missing = sorted(set(LSS_COLUMNS).difference(hdu.get_colnames()))
            if missing:
                raise BlindObservedError(f"LSS lacks required columns: {missing}")
            for start in range(0, input_rows, args.chunk_size):
                stop = min(start + args.chunk_size, input_rows)
                source = hdu[start:stop][list(LSS_COLUMNS)]
                source = source[observed_success_mask(source)]
                if not len(source):
                    continue
                ids = np.asarray(source["TARGETID"], dtype=np.int64)
                if np.any(ids <= 0) or np.any(ids > len(parent)):
                    raise BlindObservedError("LSS TARGETID outside blind parent")
                link = parent[ids - 1]
                if not np.array_equal(link["TARGETID"], ids):
                    raise BlindObservedError("TARGETID join lost index alignment")
                ra_diff = float(np.max(np.abs(source["RA"] - link["RA"]), initial=0.0))
                dec_diff = float(np.max(np.abs(source["DEC"] - link["DEC"]), initial=0.0))
                maxima["RA"], maxima["DEC"] = max(maxima["RA"], ra_diff), max(maxima["DEC"], dec_diff)
                if ra_diff > 1e-10 or dec_diff > 1e-10:
                    raise BlindObservedError("LSS/blind-parent sky mismatch")
                block = np.empty(len(source), dtype=output_dtype())
                for name in (
                    "TARGETID", "RA", "DEC", "ZWARN", "BGS_TARGET", "R_MAG_ABS",
                    "G_R_REST", "G_R_OBS", "NTILE", "COMP_TILE", "FRACZ_TILELOCID",
                    "FRAC_TLOBS_TILES", "PHOTSYS", "MASKBITS", "NOBS_G", "NOBS_R",
                    "NOBS_Z", "GOODPRI", "GOODHARDLOC", "LOCATION_ASSIGNED",
                ):
                    block[name] = source[name]
                block["Z"] = source["Z_not4clus"]
                block["R_MAG_APP"] = link["R_MAG_APP"]
                for name in ("FILE_NUM", "HALO_INDEX", "BOX_INDEX"):
                    block[name] = link[name]
                if first:
                    writer.write(block, extname="BLIND_OBSERVED")
                    first = False
                else:
                    writer[-1].append(block)
                output_rows += len(block)
                seen.append(ids)
                print(f"rows {start:,}-{stop:,}; successful={output_rows:,}", flush=True)
    finally:
        writer.close()
    if first:
        temporary.unlink(missing_ok=True)
        raise BlindObservedError("no successful observed rows")
    ids = np.concatenate(seen)
    if len(np.unique(ids)) != len(ids):
        temporary.unlink(missing_ok=True)
        raise BlindObservedError("blind observed TARGETIDs are not unique")
    os.replace(temporary, output)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "schema_version": "p10-blind-observed-geometry-v1",
        "created_utc": utc_now(), "phase": args.phase, "role": cfg["role"],
        "git_sha": git_sha, "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "parent": str(parent_path.resolve()), "lss": str(lss_path.resolve()),
        "output": {"path": str(output.resolve()), "rows": output_rows,
                   "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "parity": {"unique_targetids": True, "max_abs_sky_difference": maxima},
        "blind_contract": {
            "sealed": True, "truth_files_read": [], "target_columns_written": [],
            "allowed_content": "observed inputs plus host keys for duplicate control/deferred join",
        },
        "pass": True,
    }
    atomic_json(marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BlindObservedError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
