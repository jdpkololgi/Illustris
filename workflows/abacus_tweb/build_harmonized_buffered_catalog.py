#!/usr/bin/env python3
"""A3 step 1 — buffered, n(z)-harmonized mock catalog for the wedge rebuild.

Dilutes the graph-ready parent per z-shell to match the DESI n(z) SHAPE (A2
ratios; same C as the core-wedge design so the core matches
path1_wedge_nzharm), over a BUFFERED box so the later Delaunay/feature build
has real neighbours beyond the final wedge — mirroring the original
subset-from-full semantics. Sentinel-z rows excluded; z-errors injected
(Z_ORIG preserved). Writes a FITS the abacus graph builder can consume.
"""
from __future__ import annotations
import argparse
import json
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
    ap.add_argument("--nz-json", type=Path, required=True)
    ap.add_argument("--ra", nargs=2, type=float, default=[118.0, 162.0])
    ap.add_argument("--dec", nargs=2, type=float, default=[12.5, 32.6])
    ap.add_argument("--zr", nargs=2, type=float, default=[0.185, 0.315])
    ap.add_argument("--core-zr", nargs=2, type=float, default=[0.2, 0.3],
                    help="C is computed on these shells (matches the core-wedge design)")
    ap.add_argument("--sigma-v-kms", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-fits", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    nz = json.loads(args.nz_json.read_text())
    zmid = np.asarray(nz["z_mid"]); cm = np.asarray(nz["count_mock"]); cd = np.asarray(nz["count_desi"])
    ok = (cm > 0) & (cd > 0)
    ratio_all = np.where(ok, cm / np.maximum(cd, 1), np.inf)
    core = ok & (zmid >= args.core_zr[0]) & (zmid < args.core_zr[1])
    C = ratio_all[core].min()
    dz_shell = zmid[1] - zmid[0]
    print(f"C (core {args.core_zr}) = {C:.3f}")

    t = fitsio.read(PARENT)
    ra, dec, z = t["RA"], t["DEC"], np.asarray(t["Z"], np.float64)
    sel = ((ra >= args.ra[0]) & (ra < args.ra[1]) & (dec >= args.dec[0]) & (dec < args.dec[1]) &
           (z >= args.zr[0]) & (z < args.zr[1]) & ~((z >= SENTINEL[0]) & (z < SENTINEL[1])))
    sub = t[sel]; zs = np.asarray(sub["Z"], np.float64)
    print(f"buffered box rows: {sel.sum()}")

    ishell = np.clip(((zs - (zmid[0] - dz_shell / 2)) / dz_shell).astype(int), 0, len(zmid) - 1)
    kf = np.clip(C / ratio_all[ishell], 0.0, 1.0)
    kf[~ok[ishell]] = np.median(np.clip(C / ratio_all[ok], 0, 1))   # unmeasured shells: median frac
    keep = rng.random(len(sub)) < kf
    kept = sub[keep]
    print(f"kept after shape-match dilution: {keep.sum()} ({keep.mean():.3f})")

    z0 = np.asarray(kept["Z"], np.float64)
    znew = z0 + args.sigma_v_kms * (1.0 + z0) / C_KMS * rng.standard_normal(len(z0))
    out = rfn.append_fields(kept, "Z_ORIG", z0, usemask=False)
    out["Z"] = znew
    args.out_fits.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(args.out_fits), out, clobber=True)
    print(f"Saved: {args.out_fits}  (sigma_v={args.sigma_v_kms} km/s injected; Z_ORIG kept)")


if __name__ == "__main__":
    main()
