#!/usr/bin/env python3
"""S1(a) — cutsky-truth shell-transfer matrix (roadmap §3b, amended 2026-07-12).

Decides pooled-ñ(z)-conditioned vs per-shell models BEFORE the Phase-B retrain.
Sample: master cutsky in the north wedge box, DOWNSAMPLED per shell to the DESI
ñ(z) spline (S0 atlas) so densities are DESI-realistic across z 0.05-0.55.
Features: aperture-density harness (log counts @ 3/7/10/14 Mpc/h) computed on the
DESI-realistic sample; conditioning feature = log ñ_DESI(z_i) from the S0 spline.

Configurations (GBM, matched settings):
  diag      : train+test within shell (5-fold CV)          -> the per-shell option
  transfer  : train on shell i, test on shell j (i != j)    -> cost of NOT handling n(z)
  pooled    : train on all shells, NO ñ feature             -> naive pooling
  pooled+ñ  : train on all shells WITH ñ feature            -> the conditioning option

GATE: pooled+ñ >= diag - 0.02 on EVERY shell  =>  single conditioned model confirmed.
CPU only. Restricted footprint per JDPK (volume/trainability concern).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import fitsio
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

MASTER = ("/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_23032026/"
          "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs.fits")
ATLAS = "/pscratch/sd/d/dkololgi/abacus/s0_selection_atlas/s0_atlas.json"
SHELLS = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55)]


def to_xyz(ra, dec, z):
    zt = np.linspace(0, 0.75, 6000)
    dt = cosmo.comoving_distance(zt).value
    d = np.interp(z, zt, dt)
    r, dd = np.deg2rad(ra), np.deg2rad(dec)
    return np.vstack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", nargs=2, type=float, default=[120, 160])
    ap.add_argument("--dec", nargs=2, type=float, default=[14.5, 30.6])
    ap.add_argument("--apertures-hmpc", type=float, nargs="+", default=[3, 7, 10, 14])
    ap.add_argument("--max-per-shell", type=int, default=60000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/s1_shell_transfer"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    h = cosmo.h
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atlas = json.loads(Path(ATLAS).read_text())
    zg = np.array(atlas["desi_ntilde_grid_z"]); nt = np.array(atlas["desi_ntilde"])
    ntilde = lambda z: np.interp(z, zg, nt)

    m = fitsio.read(MASTER, columns=["RA", "DEC", "Z", "LAMBDA1"])
    box = ((m["RA"] >= args.ra[0]) & (m["RA"] < args.ra[1]) &
           (m["DEC"] >= args.dec[0]) & (m["DEC"] < args.dec[1]) &
           (m["Z"] >= 0.03) & (m["Z"] < 0.60) & np.isfinite(m["LAMBDA1"]))
    sub = m[box]
    z = np.asarray(sub["Z"], np.float64)

    # per-shell downsample cutsky -> DESI ñ(z): keep prob = n_desi(z)/n_cutsky(z)
    dz = 0.01
    edges = np.arange(0.03, 0.60 + dz/2, dz)
    cnt = np.histogram(z, bins=edges)[0].astype(float)
    dcom = cosmo.comoving_distance(edges).value
    omega = np.deg2rad(args.ra[1]-args.ra[0]) * (np.sin(np.deg2rad(args.dec[1])) - np.sin(np.deg2rad(args.dec[0])))
    n_cut = cnt / ((omega/3.0) * (dcom[1:]**3 - dcom[:-1]**3))
    ish = np.clip(((z - 0.03) / dz).astype(int), 0, len(n_cut)-1)
    pkeep = np.clip(ntilde(z) / np.maximum(n_cut[ish], 1e-12), 0, 1)
    keep = rng.random(len(z)) < pkeep
    sub = sub[keep]; z = z[keep]
    print(f"cutsky box rows {box.sum()} -> DESI-realistic sample {len(sub)}")

    pos = to_xyz(sub["RA"], sub["DEC"], z)
    l1 = np.asarray(sub["LAMBDA1"], np.float64)
    tree = cKDTree(pos)
    feats = [np.log1p(tree.query_ball_point(pos, R/h, return_length=True))
             for R in args.apertures_hmpc]
    X = np.column_stack(feats)
    Xc = np.column_stack(feats + [np.log(ntilde(z))])          # + conditioning feature

    sid = np.full(len(z), -1)
    for k, (lo, hi) in enumerate(SHELLS):
        sid[(z >= lo) & (z < hi)] = k
    # cap per-shell for tractability (keeps class balance of shells honest in pooled)
    keep2 = np.zeros(len(z), bool)
    for k in range(len(SHELLS)):
        idx = np.where(sid == k)[0]
        if len(idx) > args.max_per_shell:
            idx = rng.permutation(idx)[: args.max_per_shell]
        keep2[idx] = True
    X, Xc, l1, z, sid = X[keep2], Xc[keep2], l1[keep2], z[keep2], sid[keep2]
    for k, (lo, hi) in enumerate(SHELLS):
        print(f"  shell {k} z {lo:.2f}-{hi:.2f}: n={np.sum(sid==k)}")

    def gbm():
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_depth=6)

    K = len(SHELLS)
    M = np.full((K, K), np.nan)          # transfer matrix (train i -> test j)
    diag = np.full(K, np.nan)
    for i in range(K):
        tri = sid == i
        # diagonal via CV within shell
        pred = cross_val_predict(gbm(), X[tri], l1[tri], cv=args.folds)
        diag[i] = r2_score(l1[tri], pred)
        mdl = gbm().fit(X[tri], l1[tri])
        for j in range(K):
            if j == i:
                M[i, j] = diag[i]; continue
            tj = sid == j
            M[i, j] = r2_score(l1[tj], mdl.predict(X[tj]))

    pooled = np.full(K, np.nan); pooledc = np.full(K, np.nan)
    predp = cross_val_predict(gbm(), X, l1, cv=args.folds)
    predc = cross_val_predict(gbm(), Xc, l1, cv=args.folds)
    for j in range(K):
        tj = sid == j
        pooled[j] = r2_score(l1[tj], predp[tj])
        pooledc[j] = r2_score(l1[tj], predc[tj])

    hdr = "".join(f"   S{j}({SHELLS[j][0]:.2f})" for j in range(K))
    print("\n=== S1(a) shell-transfer matrix, R^2(lambda1) ===")
    print("train\\test" + hdr)
    for i in range(K):
        print(f"  S{i} {SHELLS[i][0]:.2f}-{SHELLS[i][1]:.2f}" +
              "".join(f"  {M[i,j]:+.3f}" for j in range(K)))
    print("  pooled     " + "".join(f"  {pooled[j]:+.3f}" for j in range(K)))
    print("  pooled+n~  " + "".join(f"  {pooledc[j]:+.3f}" for j in range(K)))
    print("  diag(shell)" + "".join(f"  {diag[j]:+.3f}" for j in range(K)))

    ok = pooledc >= diag - 0.02
    print("\nGATE per shell (pooled+n~ >= diag-0.02): " +
          " ".join("PASS" if o else "FAIL" for o in ok))
    print("VERDICT: " + ("SINGLE conditioned model CONFIRMED" if ok.all()
                         else "per-shell fallback needed on failing shells"))
    json.dump({"shells": SHELLS, "transfer_matrix": M.tolist(), "diag": diag.tolist(),
               "pooled": pooled.tolist(), "pooled_cond": pooledc.tolist(),
               "gate_pass": ok.tolist()},
              open(args.out_dir / "s1_result.json", "w"), indent=1)
    print(f"Saved: {args.out_dir}/s1_result.json")


if __name__ == "__main__":
    main()
