#!/usr/bin/env python3
"""Join compact P10 parent T-web truth to the frozen LSS observed-galaxy view."""

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


LABEL_COLUMNS = (
    "CWEB",
    "LAMBDA1", "LAMBDA2", "LAMBDA3",
    "DLAM1_DX", "DLAM1_DY", "DLAM1_DZ",
    "DLAM2_DX", "DLAM2_DY", "DLAM2_DZ",
    "DLAM3_DX", "DLAM3_DY", "DLAM3_DZ",
    "LAP_LAM1", "LAP_LAM2", "LAP_LAM3",
)
LSS_COLUMNS = (
    "TARGETID", "RA", "DEC", "Z_not4clus", "ZWARN", "BGS_TARGET",
    "R_MAG_ABS", "G_R_REST", "G_R_OBS", "NTILE", "COMP_TILE",
    "FRACZ_TILELOCID", "FRAC_TLOBS_TILES", "PHOTSYS", "MASKBITS",
    "NOBS_G", "NOBS_R", "NOBS_Z", "GOODPRI", "GOODHARDLOC",
    "LOCATION_ASSIGNED",
)


class ObservedTruthError(RuntimeError):
    """The final observation view or TARGETID truth join failed its contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def default_annotated_parent(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/annotated_parent" / (
        f"{phase}_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits"
    )


def default_lss(registry: dict[str, Any], phase: str) -> Path:
    mock = int(registry["phases"][phase]["mock"])
    root = Path(registry["path_templates"]["lss"].format(mock=mock, phase=phase))
    # The enclosing mock{N}/LSScats directory already selects the mock/phase.
    # Names such as BGS_BRIGHT-02 are restricted tracer products and must not be
    # inferred from the mock number.  The unsuffixed BGS_BRIGHT file is the full
    # final-view Bright catalogue used by the production observation contract.
    return root / "BGS_BRIGHT_full_HPmapcut.dat.fits"


def default_output(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/observed" / f"{phase}_bgs_bright_full_observed_with_tweb.fits"


def observed_success_mask(table: np.ndarray) -> np.ndarray:
    z = np.asarray(table["Z_not4clus"], dtype=np.float64)
    return np.isfinite(z) & (z > 0.0) & (np.asarray(table["ZWARN"]) == 0)


def output_dtype() -> np.dtype:
    return np.dtype(
        [
            ("TARGETID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("Z", "f8"),
            ("ZWARN", "i8"), ("BGS_TARGET", "i8"),
            ("R_MAG_APP", "f4"), ("R_MAG_ABS", "f4"),
            ("G_R_REST", "f4"), ("G_R_OBS", "f4"),
            ("NTILE", "i4"), ("COMP_TILE", "f8"),
            ("FRACZ_TILELOCID", "f8"), ("FRAC_TLOBS_TILES", "f8"),
            ("PHOTSYS", "S1"), ("MASKBITS", "i8"),
            ("NOBS_G", "i4"), ("NOBS_R", "i4"), ("NOBS_Z", "i4"),
            ("GOODPRI", "i1"), ("GOODHARDLOC", "i1"),
            ("LOCATION_ASSIGNED", "i1"),
            ("FILE_NUM", "i4"), ("HALO_INDEX", "i4"), ("BOX_INDEX", "i4"),
            ("HAS_LABEL", "i1"),
            ("CWEB", "i1"),
            ("LAMBDA1", "f4"), ("LAMBDA2", "f4"), ("LAMBDA3", "f4"),
            ("DLAM1_DX", "f4"), ("DLAM1_DY", "f4"), ("DLAM1_DZ", "f4"),
            ("DLAM2_DX", "f4"), ("DLAM2_DY", "f4"), ("DLAM2_DZ", "f4"),
            ("DLAM3_DX", "f4"), ("DLAM3_DY", "f4"), ("DLAM3_DZ", "f4"),
            ("LAP_LAM1", "f4"), ("LAP_LAM2", "f4"), ("LAP_LAM3", "f4"),
        ]
    )


def join_successful_chunk(lss: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    selected = lss[observed_success_mask(lss)]
    if not len(selected):
        return np.empty(0, dtype=output_dtype()), {"max_ra_diff": 0.0, "max_dec_diff": 0.0}
    targetids = np.asarray(selected["TARGETID"], dtype=np.int64)
    if np.any(targetids <= 0) or np.any(targetids > len(parent)):
        raise ObservedTruthError("LSS TARGETID lies outside the annotated parent range")
    truth = parent[targetids - 1]
    if not np.array_equal(np.asarray(truth["TARGETID"], dtype=np.int64), targetids):
        raise ObservedTruthError("annotated parent is not TARGETID-index aligned")
    ra_diff = float(np.max(np.abs(selected["RA"] - truth["RA"]), initial=0.0))
    dec_diff = float(np.max(np.abs(selected["DEC"] - truth["DEC"]), initial=0.0))
    if ra_diff > 1e-10 or dec_diff > 1e-10:
        raise ObservedTruthError(f"LSS/parent sky mismatch: RA={ra_diff}, DEC={dec_diff}")

    out = np.empty(len(selected), dtype=output_dtype())
    for column in (
        "TARGETID", "RA", "DEC", "ZWARN", "BGS_TARGET", "R_MAG_ABS",
        "G_R_REST", "G_R_OBS", "NTILE", "COMP_TILE", "FRACZ_TILELOCID",
        "FRAC_TLOBS_TILES", "PHOTSYS", "MASKBITS", "NOBS_G", "NOBS_R",
        "NOBS_Z", "GOODPRI", "GOODHARDLOC", "LOCATION_ASSIGNED",
    ):
        out[column] = selected[column]
    out["Z"] = selected["Z_not4clus"]
    out["R_MAG_APP"] = truth["R_MAG_APP"]
    for column in ("FILE_NUM", "HALO_INDEX", "BOX_INDEX") + LABEL_COLUMNS:
        out[column] = truth[column]
    finite = np.isfinite(out["LAMBDA1"]) & np.isfinite(out["LAMBDA2"]) & np.isfinite(out["LAMBDA3"])
    ordered = (out["LAMBDA1"] <= out["LAMBDA2"]) & (out["LAMBDA2"] <= out["LAMBDA3"])
    out["HAS_LABEL"] = finite & ordered
    if not np.all(out["HAS_LABEL"]):
        raise ObservedTruthError("successful observed rows have missing or unordered T-web truth")
    cweb = (
        (out["LAMBDA1"] > 0.2).astype(np.int8)
        + (out["LAMBDA2"] > 0.2).astype(np.int8)
        + (out["LAMBDA3"] > 0.2).astype(np.int8)
    )
    if not np.array_equal(out["CWEB"], cweb):
        raise ObservedTruthError("CWEB class disagrees with thresholded ordered eigenvalues")
    return out, {"max_ra_diff": ra_diff, "max_dec_diff": dec_diff}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--annotated-parent", type=Path)
    parser.add_argument("--lss", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise ObservedTruthError(f"unregistered phase: {args.phase}")
    cfg = registry["phases"][args.phase]
    if cfg["role"] == "sealed_blind":
        raise ObservedTruthError("refusing to construct ph001 observed truth before unsealing")
    parent_path = args.annotated_parent or default_annotated_parent(registry, args.phase)
    lss_path = args.lss or default_lss(registry, args.phase)
    output = args.output or default_output(registry, args.phase)
    marker = output.with_suffix(output.suffix + ".complete.json")
    if output.exists() or marker.exists():
        raise ObservedTruthError(f"refusing to overwrite observed artifact: {output} / {marker}")

    parent_required = [
        "TARGETID", "RA", "DEC", "R_MAG_APP", "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
        *LABEL_COLUMNS,
    ]
    parent = fitsio.read(parent_path, columns=parent_required)
    expected_targetids = np.arange(1, len(parent) + 1, dtype=np.int64)
    if not np.array_equal(parent["TARGETID"], expected_targetids):
        raise ObservedTruthError("annotated parent TARGETIDs are not sequential/index aligned")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer = fitsio.FITS(temporary, "rw", clobber=True)
    first = True
    input_rows = 0
    output_rows = 0
    seen_ids: list[np.ndarray] = []
    maxima = {"max_ra_diff": 0.0, "max_dec_diff": 0.0}
    try:
        with fitsio.FITS(lss_path) as fits:
            hdu = fits[1]
            input_rows = int(hdu.get_nrows())
            columns = set(hdu.get_colnames())
            missing = sorted(set(LSS_COLUMNS).difference(columns))
            if missing:
                raise ObservedTruthError(f"LSS input lacks required columns: {missing}")
            for start in range(0, input_rows, args.chunk_size):
                stop = min(start + args.chunk_size, input_rows)
                lss = hdu[start:stop][list(LSS_COLUMNS)]
                block, diagnostics = join_successful_chunk(lss, parent)
                maxima = {key: max(maxima[key], diagnostics[key]) for key in maxima}
                if len(block):
                    if first:
                        writer.write(block, extname="OBSERVED")
                        first = False
                    else:
                        writer[-1].append(block)
                    output_rows += len(block)
                    seen_ids.append(np.asarray(block["TARGETID"], dtype=np.int64))
                print(f"rows {start:,}-{stop:,}; successful={output_rows:,}", flush=True)
    finally:
        writer.close()
    if first:
        temporary.unlink(missing_ok=True)
        raise ObservedTruthError("no successful observed LSS rows")
    ids = np.concatenate(seen_ids)
    if len(np.unique(ids)) != len(ids):
        temporary.unlink(missing_ok=True)
        raise ObservedTruthError("successful observed view contains duplicate TARGETIDs")
    os.replace(temporary, output)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    with fitsio.FITS(output) as fits:
        z = fits[1].read(columns=["Z"])["Z"]
    payload = {
        "schema_version": "p10-observed-truth-complete-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "role": cfg["role"],
        "git_sha": git_sha,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "annotated_parent": {"path": str(parent_path.resolve()), "rows": len(parent)},
        "lss": {"path": str(lss_path.resolve()), "rows": input_rows},
        "selection": "isfinite(Z_not4clus) and Z_not4clus>0 and ZWARN==0",
        "output": {
            "path": str(output.resolve()), "rows": output_rows,
            "bytes": output.stat().st_size, "sha256": sha256_file(output),
        },
        "redshift_counts": {
            "all_successful": output_rows,
            "z_0p15_0p55": int(np.count_nonzero((z >= 0.15) & (z < 0.55))),
        },
        "join": {"targetid_unique": True, "label_complete": True, **maxima},
    }
    atomic_write_json(marker, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ObservedTruthError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
