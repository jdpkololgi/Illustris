#!/usr/bin/env python3
"""A3 — harmonize the training wedge's n(z) SHAPE to DESI + inject z-errors (roadmap v2).

Design (from the A2 measurement, nz_comparison_20260703):
  - mock/DESI shell ratio is Z-DEPENDENT (1.02 at z~0.20 -> 0.73 at z~0.275): a radial
    feature gradient one global SI median cannot absorb.
  - The mock cannot be upsampled, so we SHAPE-MATCH by dilution: keep fractions
    f_i = C * (DESI_i / mock_i) with C = min_i(mock_i / DESI_i), i.e. the most
    mock-deficient shell keeps 100% and denser-relative shells are randomly diluted.
    Result: mock n(z) shape == DESI shape exactly; a UNIFORM amplitude offset (=C)
    remains, which the scale-invariant per-graph-median features absorb by design.
  - z-ERRORS: Gaussian sigma_v (default 35 km/s, BGS-like) added to observed z before
    positions are recomputed downstream: z' = z + sigma_v(1+z)/c * N(0,1).

Outputs (originals untouched): kept-row mask npy, harmonized targets FITS (kept rows,
Z perturbed; original Z kept as Z_ORIG), selection JSON. The graph/feature rebuild on
the harmonized points is a separate (GPU/cuGraph) step — commands in the JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import numpy.lib.recfunctions as rfn
import fitsio

C_KMS = 299792.458


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wedge-targets", type=Path, required=True)
    ap.add_argument("--nz-json", type=Path, required=True, help="A2 output nz_mock_vs_desi.json")
    ap.add_argument("--zr", nargs=2, type=float, default=[0.2, 0.3])
    ap.add_argument("--sigma-v-kms", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    nz = json.loads(args.nz_json.read_text())
    zmid = np.asarray(nz["z_mid"]); cm = np.asarray(nz["count_mock"]); cd = np.asarray(nz["count_desi"])
    sel = (zmid >= args.zr[0]) & (zmid < args.zr[1]) & (cm > 0) & (cd > 0)
    zmid, cm, cd = zmid[sel], cm[sel], cd[sel]
    ratio = cm / cd
    C = ratio.min()
    keep_frac = np.clip(C / ratio, 0.0, 1.0)     # f_i = C * cd_i / cm_i
    print("shell  z_mid   mock/DESI   keep_frac")
    for z, r, f in zip(zmid, ratio, keep_frac):
        print(f"       {z:.3f}   {r:8.3f}   {f:8.3f}")
    print(f"uniform residual amplitude C = {C:.3f} (absorbed by SI features)")

    t = fitsio.read(args.wedge_targets)
    z = np.asarray(t["Z"], np.float64)
    edges = np.concatenate([zmid - (zmid[1]-zmid[0])/2, [zmid[-1] + (zmid[1]-zmid[0])/2]])
    ishell = np.digitize(z, edges) - 1
    keep = np.zeros(len(t), bool)
    inz = (ishell >= 0) & (ishell < len(zmid))
    keep[inz] = rng.random(inz.sum()) < keep_frac[ishell[inz]]
    print(f"wedge rows {len(t)} -> kept {keep.sum()} ({keep.mean():.3f}); "
          f"expected ~{(cm*keep_frac).sum()/cm.sum():.3f} of in-range rows")

    kept = t[keep]
    z0 = np.asarray(kept["Z"], np.float64)
    dz = args.sigma_v_kms * (1.0 + z0) / C_KMS * rng.standard_normal(len(z0))
    out = rfn.append_fields(kept, "Z_ORIG", z0, usemask=False)
    out["Z"] = z0 + dz
    print(f"z-error injected: sigma_v={args.sigma_v_kms} km/s "
          f"(median |dz| = {np.median(np.abs(dz)):.5f})")

    mask_npy = Path(str(args.out_prefix) + "_keepmask.npy")
    fits_out = Path(str(args.out_prefix) + "_targets.fits")
    np.save(mask_npy, keep)
    fitsio.write(str(fits_out), out, clobber=True)
    info = {
        "parent_targets": str(args.wedge_targets), "nz_json": str(args.nz_json),
        "keep_frac_per_shell": dict(zip(map(float, zmid), map(float, keep_frac))),
        "uniform_residual_amplitude_C": float(C), "sigma_v_kms": args.sigma_v_kms,
        "seed": args.seed, "n_parent": int(len(t)), "n_kept": int(keep.sum()),
        "next_steps": [
            "rebuild Delaunay graph + cuGraph features on kept points (GPU, rapids-gnn):"
            " export wedge points from harmonized FITS -> gudhi -> desi/abacus feature pipeline",
            "then build_abacus_sbi_cache.py --scale-invariant-features --power-scale-node-features"
            " --linear-increments --three-targets-only on the new arrays",
        ],
    }
    Path(str(args.out_prefix) + "_selection.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Saved: {mask_npy}, {fits_out}, {args.out_prefix}_selection.json")


if __name__ == "__main__":
    main()
