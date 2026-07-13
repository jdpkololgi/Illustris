#!/usr/bin/env python3
"""S2 — extract a buffered shell catalog from the graph-ready path1 parent (roadmap §3b).

NO dilution (ñ-conditioning replaces it; S1(a) verdict). Sentinel window excluded
(phantoms live at z≈0.59; documented). DESI-like z-errors injected (σ_v=35 km/s,
Z_ORIG preserved) for consistency with the nzharm convention.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import numpy.lib.recfunctions as rfn
import fitsio

C_KMS = 299792.458
SENTINEL = (0.585, 0.595)
PARENT = ("/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
          "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zr", nargs=2, type=float, required=True, help="core shell z range")
    ap.add_argument("--zbuf", type=float, default=0.015)
    ap.add_argument("--ra", nargs=2, type=float, default=[118.0, 162.0])
    ap.add_argument("--dec", nargs=2, type=float, default=[12.5, 32.6])
    ap.add_argument("--sigma-v-kms", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-fits", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + int(args.zr[0] * 1000))

    t = fitsio.read(PARENT)
    z = np.asarray(t["Z"], np.float64)
    lo, hi = args.zr[0] - args.zbuf, args.zr[1] + args.zbuf
    sel = ((t["RA"] >= args.ra[0]) & (t["RA"] < args.ra[1]) &
           (t["DEC"] >= args.dec[0]) & (t["DEC"] < args.dec[1]) &
           (z >= lo) & (z < hi) & ~((z >= SENTINEL[0]) & (z < SENTINEL[1])))
    sub = t[sel]
    z0 = np.asarray(sub["Z"], np.float64)
    znew = z0 + args.sigma_v_kms * (1.0 + z0) / C_KMS * rng.standard_normal(len(z0))
    out = rfn.append_fields(sub, "Z_ORIG", z0, usemask=False)
    out["Z"] = znew
    args.out_fits.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(args.out_fits), out, clobber=True)
    print(f"shell {args.zr} (buffered {lo:.3f}-{hi:.3f}): {len(sub)} rows -> {args.out_fits}")


if __name__ == "__main__":
    main()
