#!/usr/bin/env python3
"""
Subset Abacus SecondGen BGS CutSky FITS for ph000-style workflows.

Applies R_MAG_APP cut, optional NGC/SGC sky split (same geometry as
GraphWeb_DESI/Y3-mocks-DisPerSE-runs-1.ipynb on Tb_sim), and writes the
full CutSky science column set with native CutSky names: observed/RSD
redshift ``Z`` and true/cosmological ``Z_COSMO``.

CutSky BGS extension 1 schema (fitsio on cosmosim ph000 file, 2026-05-13):
same names/dtypes as upstream_prepare_mocks_Y3_bright.CUTSKY_BGS_SECONDGEN_COLS
(RA/DEC f8; Z, Z_COSMO, mags/colors, HALO_MASS f4; linkage and footprint ints i4).
"""

from __future__ import annotations

import argparse
import os
import sys

import fitsio
import numpy as np
from astropy.table import Table

# Must match columns on SecondGen cosmosim CutSky BGS (see upstream_prepare_mocks_Y3_bright).
CUTSKY_BGS_SECONDGEN_COLS = (
    "RA",
    "DEC",
    "Z",
    "Z_COSMO",
    "R_MAG_APP",
    "R_MAG_ABS",
    "G_R_REST",
    "G_R_OBS",
    "HALO_MASS",
    "CEN",
    "RES",
    "FILE_NUM",
    "HALO_INDEX",
    "BOX_INDEX",
    "IN_Y1",
    "NGC_Y1",
    "SGC_Y1",
    "N_Y1",
    "S_Y1",
    "IN_Y5",
    "NGC_Y5",
    "SGC_Y5",
    "N_Y5",
    "S_Y5",
)


def _ngc_mask(
    ra: np.ndarray, dec: np.ndarray, r_mag: np.ndarray, rbandcut: float
) -> np.ndarray:
    return (
        (ra < 270.0)
        & (ra > 120.0)
        & (dec > -5.0)
        & (dec < 75.0)
        & (r_mag < rbandcut)
    )


def _sgc_mask(
    ra: np.ndarray, dec: np.ndarray, r_mag: np.ndarray, rbandcut: float
) -> np.ndarray:
    return (
        ((ra < 40.0) | (ra > 330.0))
        & (dec > -15.0)
        & (dec < 30.0)
        & (r_mag < rbandcut)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cutsky",
        default=(
            "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/"
            "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits"
        ),
        help=(
            "Input CutSky FITS (default: cosmosim SecondGenMocks CutSky ph000 BGS). "
            "Must include Z (RSD/observed) and Z_COSMO columns."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output FITS path (default: stage_1/ under this ph000 tree). "
            "Writes native Z, Z_COSMO, and the full SecondGen BGS column superset."
        ),
    )
    parser.add_argument(
        "--rbandcut",
        type=float,
        default=19.5,
        help="Bright cut on R_MAG_APP (default 19.5; notebook sim_* masks use the same edge).",
    )
    parser.add_argument(
        "--cap",
        choices=["ALL", "NGC", "SGC"],
        default="ALL",
        help="Sky cap for RA/Dec mask (same logic as notebook sim_NGC/sim_SGC). "
        "ALL applies only the magnitude cut. Output keeps native Z and Z_COSMO.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.cutsky):
        print(f"ERROR: cutsky file not found: {args.cutsky}", file=sys.stderr)
        sys.exit(2)

    colnames = list(fitsio.FITS(args.cutsky)[1].get_colnames())
    required = ["RA", "DEC", "Z", "Z_COSMO", "R_MAG_APP"]
    for c in required:
        if c not in colnames:
            print(f"ERROR: required column {c!r} not in FITS. Have: {colnames}", file=sys.stderr)
            sys.exit(2)

    missing = [c for c in CUTSKY_BGS_SECONDGEN_COLS if c not in colnames]
    if missing:
        print(
            f"ERROR: expected SecondGen BGS columns missing {missing}. Have: {colnames}",
            file=sys.stderr,
        )
        sys.exit(2)

    data = fitsio.read(args.cutsky, columns=list(CUTSKY_BGS_SECONDGEN_COLS))
    ra = np.asarray(data["RA"])
    dec = np.asarray(data["DEC"])
    r_mag = np.asarray(data["R_MAG_APP"])

    sel = r_mag < args.rbandcut
    if args.cap == "NGC":
        sel &= _ngc_mask(ra, dec, r_mag, args.rbandcut)
    elif args.cap == "SGC":
        sel &= _sgc_mask(ra, dec, r_mag, args.rbandcut)

    out = Table()
    for name in CUTSKY_BGS_SECONDGEN_COLS:
        out[name] = np.asarray(data[name])[sel]

    if args.out is None:
        ph000 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stage1 = os.path.join(ph000, "stage_1")
        os.makedirs(stage1, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.cutsky))[0]
        args.out = os.path.join(stage1, f"{base}_subset_{args.cap}_rmaglt{args.rbandcut}.fits")
    else:
        odir = os.path.dirname(os.path.abspath(args.out))
        if odir:
            os.makedirs(odir, exist_ok=True)

    out.write(args.out, format="fits", overwrite=True)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
