#!/usr/bin/env python3
"""Build a DESI-parity BGS mag-lim catalog from a mock stage-3 FITS table.

Applies the same selection as ``GraphWeb_DESI/workflows/catalog/build_bgs_maglim_catalog.py``:

- ``ZWARN == 0``
- ``DELTACHI2 >= 25``
- ``SPECTYPE == "GALAXY"``
- ``BGS_TARGET`` has at least one BGS_BRIGHT bit (north/south/unsplit), unless ``--no-bright-only``

No redshift cuts, no stellar-mass cuts.

If required spectroscopic columns are absent (typical for COMBD-only ``datcomb_brightwdup.fits``),
the script exits with a clear message listing missing columns and a suggested injection approach.

Default input: ``stage_3/fba0/datcomb_brightwdup.fits`` (has ``BGS_TARGET`` / ``ZWARN`` but usually
no ``DELTACHI2`` or ``SPECTYPE`` until ``joindspec`` + ``fulld`` or post-hoc mock injection).

Run with cosmic_env (not desi_environment required for this helper):

  env -u PYTHONPATH /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \\
    scripts/build_mock_bgs_maglim_catalog.py --out-path /path/to/mock_bgs_maglim.fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitsio
import numpy as np
from desitarget.targetmask import bgs_mask

BRIGHT_BITS = (
    bgs_mask.BGS_BRIGHT
    | bgs_mask.BGS_BRIGHT_NORTH
    | bgs_mask.BGS_BRIGHT_SOUTH
)

# Parity with build_bgs_maglim_catalog.py
REQUIRED_SPEC_COLS = ("ZWARN", "DELTACHI2", "SPECTYPE", "BGS_TARGET")
DELTACHI2_MIN = 25.0

DEFAULT_INPUT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/"
    "datcomb_brightwdup.fits"
)

DEFAULT_OUT_COLS = (
    "TARGETID",
    "RA",
    "DEC",
    "Z",
    "ZWARN",
    "DELTACHI2",
    "SPECTYPE",
    "BGS_TARGET",
    "FILE_NUM",
    "HALO_INDEX",
    "BOX_INDEX",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-fits",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Mock FITS table (default: {DEFAULT_INPUT})",
    )
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=2_000_000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--bright-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require BGS_BRIGHT (or north/south bright) in BGS_TARGET (default: True).",
    )
    return p.parse_args()


def _missing_columns(colnames: list[str]) -> list[str]:
    upper = {c.upper(): c for c in colnames}
    missing: list[str] = []
    for req in REQUIRED_SPEC_COLS:
        if req.upper() not in upper:
            missing.append(req)
    return missing


def _injection_help(missing: list[str], input_path: Path) -> str:
    lines = [
        f"Input FITS: {input_path}",
        f"Missing columns for DESI parity: {', '.join(missing)}",
        "",
        "Current ph000 COMBD-only products lack LOA spectroscopic fields. Options:",
        "",
        "1. Re-run mkCat with joindspec=y, fulld=y (see run_stage3_desi_aligned_mkcat.sh)",
        "   after resolving assignwdup blocker (STAGE3_DESI_ALIGNMENT.md).",
        "",
        "2. Post-hoc probabilistic injection on datcomb_brightwdup.fits:",
        "   - SPECTYPE: set all rows to 'GALAXY' (BGS mock targets).",
        "   - ZWARN: draw zfail from LOA BGS bright rates (~few percent); set ZWARN=0 otherwise.",
        "   - DELTACHI2: for ZWARN==0 rows, draw from LOA DELTACHI2 distribution truncated at 25;",
        "     for ZWARN!=0, assign low DELTACHI2 (<25) to mimic failed redshifts.",
        "   Join injected columns back by TARGETID before calling this script.",
        "",
        "Reference DESI builder:",
        "  GraphWeb_DESI/workflows/catalog/build_bgs_maglim_catalog.py",
    ]
    return "\n".join(lines)


def _resolve_out_cols(colnames: list[str]) -> list[str]:
    upper = {c.upper(): c for c in colnames}
    out: list[str] = []
    for c in DEFAULT_OUT_COLS:
        if c.upper() in upper:
            out.append(upper[c.upper()])
    return out


def main() -> None:
    args = parse_args()
    input_path = args.input_fits.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out = args.out_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if args.overwrite:
            out.unlink()
        else:
            raise FileExistsError(f"Output exists: {out} (use --overwrite)")

    with fitsio.FITS(str(input_path), "r") as f:
        hdu = f[1]
        colnames = hdu.get_colnames()
        missing = _missing_columns(colnames)
        if missing:
            print(_injection_help(missing, input_path), file=sys.stderr)
            sys.exit(2)

        read_cols = _resolve_out_cols(colnames)
        nrows = int(hdu.get_nrows())
        print(f"Input: {input_path}")
        print(f"Output: {out}")
        print(f"Rows: {nrows:,}; bright_only: {args.bright_only}")
        print(f"Parity cuts: ZWARN==0, DELTACHI2>={DELTACHI2_MIN}, SPECTYPE==GALAXY")

        fout = fitsio.FITS(str(out), "rw", clobber=True)
        try:
            wrote = 0
            kept = 0
            first = True
            for start in range(0, nrows, int(args.chunk_rows)):
                stop = min(start + int(args.chunk_rows), nrows)
                tab = hdu.read(rows=list(range(start, stop)), columns=read_cols)

                m = (
                    (tab["ZWARN"] == 0)
                    & (tab["DELTACHI2"] >= DELTACHI2_MIN)
                    & (np.char.strip(tab["SPECTYPE"].astype("S10")) == b"GALAXY")
                )
                if args.bright_only:
                    m &= (tab["BGS_TARGET"] & BRIGHT_BITS) != 0
                else:
                    m &= tab["BGS_TARGET"] != 0

                sub = tab[m]
                wrote += stop - start
                kept += int(sub.size)

                if sub.size:
                    if first:
                        fout.write(sub)
                        first = False
                    else:
                        fout[1].append(sub)

                if start == 0 or (start // int(args.chunk_rows) + 1) % 5 == 0 or stop == nrows:
                    frac = kept / max(wrote, 1)
                    print(f"  scanned {stop:,}/{nrows:,} rows, kept={kept:,} (frac={frac:.4f})")
        finally:
            fout.close()

    if kept == 0:
        if out.exists():
            out.unlink()
        raise RuntimeError(
            "No rows passed DESI parity cuts (ZWARN==0, DELTACHI2>=25, SPECTYPE==GALAXY, BGS bright). "
            "Check injected SPECTYPE values and BGS_TARGET on the input FITS."
        )

    with fitsio.FITS(str(out), "r") as fcheck:
        n_out = int(fcheck[1].get_nrows())
    print(f"Done. Output rows: {n_out:,}")


if __name__ == "__main__":
    main()
