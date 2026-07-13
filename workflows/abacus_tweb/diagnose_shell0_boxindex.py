#!/usr/bin/env python3
"""Confirm shell-0 label scrambling is driven by BOX_INDEX==-1 ("out-of-box"), NOT by z.

Disentangles the two (they're ~confounded: shell-0 is 99.9% BOX=-1):
  * BOX=-1 fraction vs z (fine bins) — where do out-of-box galaxies live?
  * R^2(λ1 | observer position) in z-bins x {BOX==-1, BOX>=0}. If BOX=-1 -> ~0 at EVERY z and
    BOX>=0 -> ~0.5 at every z, the failure variable is BOX_INDEX, not redshift.
  * permutation null: shuffle λ1 within shell-0 -> R^2 floor (== observed => fully scrambled).
Astrostat framing: R^2(pos)=0 with intact marginal P(λ1) == MI(position, label)=0 == a
closure-test failure isolated to the out-of-box subpopulation (labels drawn from the right
distribution but attached to the wrong galaxies).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import fitsio
from astropy.cosmology import Planck18 as cosmo
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

PARENT = ("/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
          "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits")


def to_xyz(ra, dec, z):
    zt = np.linspace(0, 0.75, 6000); dt = cosmo.comoving_distance(zt).value
    d = np.interp(z, zt, dt); r, dd = np.deg2rad(ra), np.deg2rad(dec)
    return np.vstack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)]).T


def cvr2(X, y):
    if len(y) < 400:
        return np.nan
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_depth=6)
    return r2_score(y, cross_val_predict(m, X, y, cv=3))


def main():
    rng = np.random.default_rng(0)
    d = fitsio.read(PARENT, columns=["RA", "DEC", "Z", "LAMBDA1", "BOX_INDEX", "HALO_INDEX"])
    ra, dec, z = d["RA"].astype(float), d["DEC"].astype(float), d["Z"].astype(float)
    l1 = d["LAMBDA1"].astype(float); bx = d["BOX_INDEX"]
    m = (ra >= 118) & (ra < 162) & (dec >= 12.5) & (dec < 32.6) & (z >= 0.03) & (z < 0.55) & np.isfinite(l1)
    ra, dec, z, l1, bx = ra[m], dec[m], z[m], l1[m], bx[m]
    print(f"rows={len(z):,}  BOX==-1 overall={100*(bx==-1).mean():.1f}%")

    print("\n== BOX==-1 fraction vs z ==")
    edges = np.arange(0.03, 0.55, 0.04)
    for a, b in zip(edges[:-1], edges[1:]):
        s = (z >= a) & (z < b)
        if s.sum():
            print(f"  z {a:.2f}-{b:.2f}: n={s.sum():7d}  BOX==-1 {100*(bx[s]==-1).mean():5.1f}%  "
                  f"λ1_std={l1[s].std():.3f}")

    print("\n== R2(λ1 | observer xyz), z-bin x BOX-stratum ==")
    print(f"{'zbin':12s} {'stratum':10s} {'n':>8s} {'R2_pos':>7s} {'λ1_std':>7s}")
    for a, b in [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35)]:
        for name, sel in [("BOX==-1", bx == -1), ("BOX>=0", bx >= 0)]:
            s = (z >= a) & (z < b) & sel
            n = int(s.sum())
            if n < 400:
                print(f"{a:.2f}-{b:.2f}   {name:10s} {n:8d}   (too few)")
                continue
            idx = np.where(s)[0]
            if n > 40000:
                idx = rng.permutation(idx)[:40000]
            r2 = cvr2(to_xyz(ra[idx], dec[idx], z[idx]), l1[idx])
            print(f"{a:.2f}-{b:.2f}   {name:10s} {n:8d} {r2:7.3f} {l1[idx].std():7.3f}")

    print("\n== permutation null (shell-0, shuffle λ1) ==")
    s = (z >= 0.05) & (z < 0.15)
    idx = np.where(s)[0]; idx = rng.permutation(idx)[:40000]
    X = to_xyz(ra[idx], dec[idx], z[idx]); y = l1[idx]
    yshuf = rng.permutation(y)
    print(f"  observed R2_pos={cvr2(X, y):.3f}   shuffled-label R2_pos={cvr2(X, yshuf):.3f}  "
          f"(equal => shell-0 labels are position-scrambled)")


if __name__ == "__main__":
    main()
