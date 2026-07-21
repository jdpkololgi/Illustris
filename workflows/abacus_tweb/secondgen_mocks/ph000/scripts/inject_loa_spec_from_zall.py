#!/usr/bin/env python3
"""Inject LOA-calibrated spectroscopic columns onto a mock COMBD FITS table.

Draws ``ZWARN``, ``DELTACHI2``, and ``SPECTYPE`` from the empirical distribution
of DESI LOA ``zall-pix-loa`` BGS-bright targets (same parent population as
``build_bgs_maglim_catalog.py``), then writes a new FITS table.

This implements pipeline component **D** when full mkCat ``fulld`` is blocked
(no ``datcomb_brightassignwdup.fits``). It matches LOA **marginal** failure and
Δχ² statistics, not a full spectroscopic forward model.

Run with cosmic_env::

  env -u PYTHONPATH /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \\
    scripts/inject_loa_spec_from_zall.py \\
    --input-fits .../datcomb_brightwdup.fits \\
    --out-fits .../datcomb_brightwdup_loa_spec.fits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitsio
import numpy as np
from numpy.lib import recfunctions as rfn
from desitarget.targetmask import bgs_mask

BRIGHT_BITS = (
    bgs_mask.BGS_BRIGHT
    | bgs_mask.BGS_BRIGHT_NORTH
    | bgs_mask.BGS_BRIGHT_SOUTH
)

DEFAULT_ZALL = (
    "/global/cfs/cdirs/desi/public/dr2/spectro/redux/loa/zcatalog/v1/zall-pix-loa.fits"
)
DELTACHI2_MIN = 25.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-fits", type=Path, required=True)
    p.add_argument("--out-fits", type=Path, required=True)
    p.add_argument("--zall-path", type=Path, default=Path(DEFAULT_ZALL))
    p.add_argument("--chunk-rows", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--zall-chunk-rows",
        type=int,
        default=2_000_000,
        help="Chunk size when scanning zall for calibration.",
    )
    return p.parse_args()


def _calibrate_from_zall(zall_path: Path, chunk_rows: int) -> dict:
    """Return LOA marginal stats for BGS-bright rows in zall."""
    cols = ("ZWARN", "DELTACHI2", "SPECTYPE", "BGS_TARGET")
    bright_mask = 0
    n_bright = 0
    n_pass = 0
    dchi2_pass: list[np.ndarray] = []
    dchi2_fail: list[np.ndarray] = []
    spectype_pass: list[bytes] = []

    with fitsio.FITS(str(zall_path), "r") as f:
        hdu = f["ZCATALOG"] if "ZCATALOG" in f else f[1]
        nrows = int(hdu.get_nrows())
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            tab = hdu.read(rows=list(range(start, stop)), columns=list(cols))
            m_b = (tab["BGS_TARGET"] & BRIGHT_BITS) != 0
            if not np.any(m_b):
                continue
            sub = tab[m_b]
            n_bright += int(sub.size)
            zwarn = sub["ZWARN"]
            dchi2 = sub["DELTACHI2"].astype(np.float64)
            st = sub["SPECTYPE"].astype("U10")

            pass_m = (
                (zwarn == 0)
                & (dchi2 >= DELTACHI2_MIN)
                & (st == "GALAXY")
            )
            n_pass += int(np.sum(pass_m))
            if np.any(pass_m):
                dchi2_pass.append(dchi2[pass_m])
                spectype_pass.append(st[pass_m].astype("S10"))
            fail_m = ~pass_m
            if np.any(fail_m):
                dchi2_fail.append(dchi2[fail_m])

    if n_bright == 0:
        raise RuntimeError("No BGS-bright rows found in zall for calibration")

    p_pass = n_pass / n_bright
    dchi2_pass_all = np.concatenate(dchi2_pass) if dchi2_pass else np.array([30.0])
    dchi2_fail_all = np.concatenate(dchi2_fail) if dchi2_fail else np.array([5.0])
    dchi2_pass_all = dchi2_pass_all[np.isfinite(dchi2_pass_all)]
    dchi2_fail_all = dchi2_fail_all[np.isfinite(dchi2_fail_all)]
    if dchi2_pass_all.size == 0:
        dchi2_pass_all = np.array([30.0])
    if dchi2_fail_all.size == 0:
        dchi2_fail_all = np.array([5.0])
    dchi2_fail_all = np.clip(dchi2_fail_all, 0.0, DELTACHI2_MIN - 1e-3)

    # SPECTYPE mode among passing rows
    if spectype_pass:
        st_all = np.concatenate(spectype_pass)
        unique, counts = np.unique(st_all, return_counts=True)
        ref = unique[np.argmax(counts)]
        if isinstance(ref, (bytes, np.bytes_)):
            spectype_ref = ref.decode("ascii").strip()
        else:
            spectype_ref = str(ref).strip()
    else:
        spectype_ref = "GALAXY"

    return {
        "n_bright": n_bright,
        "n_pass": n_pass,
        "p_pass": p_pass,
        "dchi2_pass": dchi2_pass_all,
        "dchi2_fail": dchi2_fail_all,
        "spectype_ref": spectype_ref,
    }


def _set_column(tab: np.ndarray, name: str, values: np.ndarray) -> np.ndarray:
    """Set or append a column on a structured FITS array."""
    if name in tab.dtype.names:
        out = tab.copy()
        out[name] = values
        return out
    return rfn.append_fields(tab, name, values, usemask=False)


def _inject_chunk(tab: np.ndarray, cal: dict, rng: np.random.Generator) -> np.ndarray:
    n = len(tab)
    # Fibre-UNOBSERVED rows (input ZWARN==999999) carry a sentinel redshift and no
    # real spectrum: never inject a pass onto them, and preserve ZWARN=999999 so the
    # downstream ZWARN==0 mag-lim cut drops them (sentinel-z fix, 2026-07-03).
    unobserved = np.asarray(tab["ZWARN"], dtype=np.int64) == 999999
    u = rng.random(n)
    is_pass = (u < cal["p_pass"]) & ~unobserved

    dchi2 = np.empty(n, dtype=np.float64)
    n_pass = int(np.sum(is_pass))
    n_fail = n - n_pass
    if n_pass:
        idx = np.where(is_pass)[0]
        dchi2[idx] = rng.choice(cal["dchi2_pass"], size=n_pass)
    if n_fail:
        idx = np.where(~is_pass)[0]
        dchi2[idx] = rng.choice(cal["dchi2_fail"], size=n_fail)
        dchi2[idx] = np.minimum(dchi2[idx], DELTACHI2_MIN - 1e-3)

    zwarn = np.zeros(n, dtype=np.int32)
    zwarn[~is_pass] = 1
    zwarn[unobserved] = 999999

    spectype = np.full(n, cal["spectype_ref"], dtype="S10")

    out = _set_column(tab, "ZWARN", zwarn)
    out = _set_column(out, "DELTACHI2", dchi2)
    out = _set_column(out, "SPECTYPE", spectype)
    return out


def main() -> None:
    args = parse_args()
    inp = args.input_fits.expanduser().resolve()
    out = args.out_fits.expanduser().resolve()
    zall = args.zall_path.expanduser().resolve()

    if not inp.exists():
        raise FileNotFoundError(inp)
    if not zall.exists():
        raise FileNotFoundError(zall)
    if out.exists():
        if args.overwrite:
            out.unlink()
        else:
            raise FileExistsError(f"Exists: {out} (use --overwrite)")

    print(f"Calibrating from zall: {zall}")
    cal = _calibrate_from_zall(zall, args.zall_chunk_rows)
    print(
        f"LOA BGS-bright: n={cal['n_bright']:,} pass={cal['n_pass']:,} "
        f"frac_pass={cal['p_pass']:.4f}"
    )

    rng = np.random.default_rng(args.seed)
    print(f"Injecting into: {inp}")
    print(f"Writing: {out}")

    with fitsio.FITS(str(inp), "r") as fin:
        hdu = fin[1]
        nrows = int(hdu.get_nrows())
        fout = fitsio.FITS(str(out), "rw", clobber=True)
        try:
            first = True
            for start in range(0, nrows, args.chunk_rows):
                stop = min(start + args.chunk_rows, nrows)
                tab = hdu.read(rows=list(range(start, stop)))
                tab_out = _inject_chunk(tab, cal, rng)
                if first:
                    fout.write(tab_out)
                    first = False
                else:
                    fout[1].append(tab_out)
                print(f"  wrote rows {stop:,}/{nrows:,}")
        finally:
            fout.close()

    with fitsio.FITS(str(out), "r") as f:
        n_out = int(f[1].get_nrows())
    print(f"Done. Output rows: {n_out:,}")


if __name__ == "__main__":
    main()
