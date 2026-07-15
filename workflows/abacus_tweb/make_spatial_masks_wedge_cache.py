#!/usr/bin/env python3
"""Build a SPATIAL-HOLDOUT variant of the dense-wedge union cache (leakage audit).

WHY: the production dense-wedge anchor (union graph @3749: posterior-mean R2 0.8041/0.8461/0.8955)
was trained on the cache's RANDOM 70/21/9 node split -- train/test RA ranges overlap fully. The tidal
field is smooth on ~10 Mpc, so a random split leaves a test galaxy's neighbours in train and a model
can interpolate the label field. The same flaw was just shown to inflate S1(b)'s CNN (0.725 macro ->
0.461 under a spatial holdout). This script asks: how much of 0.804 is leakage?

METHOD: everything in the cache is kept byte-identical EXCEPT the masks, which are replaced by the
production spatial protocol (mirrors s3b): train RA<145 / val 145-150 / test RA>=150, assigned by
GLOBAL HALO CENTROID RA (halo-disjoint), with a 15 Mpc z-dependent gutter around both boundaries.
Same graph, same features, same targets, same trainer, same seed, same budget -- ONE variable: split.

RA per node is recovered by eigenvalue-triplet matching into the expanded-wedge targets FITS
(eigenvalues are halo-level, so duplicate triplets across rows are SAME-HALO and thus region-safe:
halo identity, which drives assignment, is recovered correctly even when the row is ambiguous).
Unmatched nodes (~0.14%) are excluded from ALL masks -- they stay in the graph (message passing) but
carry no loss, biasing nothing.

Emits <out-dir>/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl for use via
TNG_SBI_CACHE_DIR with jraph_sbi_flowjax.py --increment_mode linear.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo

_ZT = np.linspace(0.0, 0.75, 8000)
_DT = cosmo.comoving_distance(_ZT).value


def dcom(z):
    return np.interp(z, _ZT, _DT)


def deg_per_mpc(z):
    return np.degrees(1.0 / np.maximum(dcom(z), 1e-6))


def region_of(ra, ra_train_hi, ra_test_lo):
    r = np.ones(np.shape(ra), np.int8)
    r = np.where(ra < ra_train_hi, 0, r)
    r = np.where(ra >= ra_test_lo, 2, r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph/"
        "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"))
    ap.add_argument("--targets-fits", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "staged_mock_stage3_postcollision_full_rs7_wedge_v_limited_incomplete_expanded_"
        "ra120_160_dec14p5_30p6_z0p2_0p3_wedge_targets_halo_xcom.fits"))
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph_SPATIAL"))
    ap.add_argument("--ra-train-hi", type=float, default=145.0)
    ap.add_argument("--ra-test-lo", type=float, default=150.0)
    ap.add_argument("--gutter-mpc", type=float, default=15.0)
    args = ap.parse_args()

    c = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(c["eigenvalues_raw"], np.float64)
    tr0, va0, te0 = (np.asarray(m).astype(bool) for m in c["masks"])
    n = len(eig)
    print(f"cache nodes: {n:,}  original RANDOM split {tr0.sum():,}/{va0.sum():,}/{te0.sum():,}")

    t = fitsio.read(args.targets_fits,
                    columns=["RA", "Z", "LAMBDA1", "LAMBDA2", "LAMBDA3",
                             "FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
    lam_f = np.stack([t["LAMBDA1"], t["LAMBDA2"], t["LAMBDA3"]], 1).astype(np.float64)
    key_f = np.round(np.sort(lam_f, 1), 8)     # float64 BEFORE rounding: float32 keys never equal float64 keys
    key_c = np.round(np.sort(eig, 1), 8)
    lut = {tuple(r): i for i, r in enumerate(key_f)}
    hit = np.array([lut.get(tuple(r), -1) for r in key_c])
    matched = hit >= 0
    print(f"triplet-matched to FITS: {matched.sum():,}/{n:,} = {100*matched.mean():.2f}% "
          f"(unmatched are EXCLUDED from all masks)")

    ra = np.full(n, np.nan)
    z = np.full(n, np.nan)
    halo = np.full((n, 3), -1, np.int64)
    ra[matched] = t["RA"][hit[matched]]
    z[matched] = t["Z"][hit[matched]]
    halo[matched] = np.stack([t["FILE_NUM"], t["BOX_INDEX"], t["HALO_INDEX"]], 1)[hit[matched]]

    # global halo-centroid region (identical construction to s3b)
    hv = np.ascontiguousarray(halo[matched]).view([('', halo.dtype)] * 3).ravel()
    uk, inv = np.unique(hv, return_inverse=True)
    sum_ra = np.zeros(len(uk))
    cnt = np.zeros(len(uk))
    np.add.at(sum_ra, inv, ra[matched])
    np.add.at(cnt, inv, 1.0)
    halo_reg = region_of(sum_ra / cnt, args.ra_train_hi, args.ra_test_lo)
    reg = np.full(n, -1, np.int8)
    reg[matched] = halo_reg[inv]

    gut = args.gutter_mpc * deg_per_mpc(np.nan_to_num(z, nan=0.25))
    in_gutter = matched & ((np.abs(ra - args.ra_train_hi) < gut) | (np.abs(ra - args.ra_test_lo) < gut))
    active = matched & ~in_gutter

    train_m = active & (reg == 0)
    val_m = active & (reg == 1)
    test_m = active & (reg == 2)
    print(f"SPATIAL split: train {train_m.sum():,} / val {val_m.sum():,} / test {test_m.sum():,} "
          f"(gutter drops {int(in_gutter.sum()):,})")

    # ---- gates ----
    for nm, m in (("train", train_m), ("test", test_m)):
        print(f"  {nm:5s} RA [{np.nanmin(ra[m]):.2f},{np.nanmax(ra[m]):.2f}]")
    overlap = min(np.nanmax(ra[train_m]), np.nanmax(ra[test_m])) - max(np.nanmin(ra[train_m]), np.nanmin(ra[test_m]))
    if overlap > 0:
        raise RuntimeError(f"train/test RA ranges overlap by {overlap:.2f} deg -- split is not spatial")
    hv_tr = np.unique(np.ascontiguousarray(halo[train_m]).view([('', halo.dtype)] * 3).ravel())
    hv_te = np.unique(np.ascontiguousarray(halo[test_m]).view([('', halo.dtype)] * 3).ravel())
    shared = np.intersect1d(hv_tr, hv_te)
    if len(shared):
        raise RuntimeError(f"halo leak: {len(shared)} halos in both train and test")
    if not (np.isfinite(eig[train_m]).all() and np.isfinite(eig[test_m]).all()):
        raise RuntimeError("non-finite eigenvalues in active masks")
    if (train_m & val_m).any() or (train_m & test_m).any() or (val_m & test_m).any():
        raise RuntimeError("mask overlap")
    print("[PASS] gates: spatially disjoint RA, halo-disjoint, finite labels, no mask overlap")

    c["masks"] = (train_m, val_m, test_m)
    c["spatial_masks_provenance"] = {
        "built_by": "make_spatial_masks_wedge_cache.py", "source_cache": str(args.cache),
        "targets_fits": str(args.targets_fits), "ra_train_hi": args.ra_train_hi,
        "ra_test_lo": args.ra_test_lo, "gutter_mpc": args.gutter_mpc,
        "n_unmatched_excluded": int((~matched).sum()),
        "original_random_split": [int(tr0.sum()), int(va0.sum()), int(te0.sum())],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.cache.name
    with open(out, "wb") as f:
        pickle.dump(c, f)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
