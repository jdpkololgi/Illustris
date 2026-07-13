#!/usr/bin/env python3
"""Diagnose why λ1 is unpredictable at the lowest shell (z0.05-0.15) — both GBM and GNN fail.

Discriminating tests, per shell, on the FULL final wedge targets (not just holdout):
  (1) λ1/λ2/λ3 dynamic range (std, IQR, percentiles) — is there variance to predict at all?
  (2) CV R^2(λ1) from OBSERVER POSITION (x,y,z) vs LOCAL DENSITY (aperture log-counts
      @3/7/10/14 Mpc/h) vs BOTH. Key discriminator:
        pos high, aperture ~0  -> field exists but local features can't see it (scale/feature)
        pos ~0 too             -> field genuinely flat/uninformative (intrinsic -> scope guard)
  (3) Spearman(aperture density, λ1) per shell — does local density track λ1?
  (4) CWEB class fractions + P(λ1>0.2) — is the shell degenerate (all one environment)?
  (5) comoving volume + N(7 Mpc/h cells) — large-scale-mode / cosmic-variance budget.
CPU only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import fitsio
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

SHELLS = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
ROOT = Path("/pscratch/sd/d/dkololgi/abacus/s2_shells")
CWEB_NAMES = {0: "void", 1: "sheet", 2: "filament", 3: "knot"}


def gbm():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, max_depth=6)


def cvr2(X, y, cv=3):
    return r2_score(y, cross_val_predict(gbm(), X, y, cv=cv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-shell", type=int, default=45000)
    ap.add_argument("--apertures-hmpc", type=float, nargs="+", default=[3, 7, 10, 14])
    args = ap.parse_args()
    h = cosmo.h
    rng = np.random.default_rng(0)

    print(f"{'shell':11s} {'n':>7s} {'l1_mean':>8s} {'l1_std':>7s} {'l1_IQR':>7s} "
          f"{'P(l1>.2)':>8s} | {'R2_pos':>7s} {'R2_aper':>7s} {'R2_both':>7s} | "
          f"{'sp_d7':>6s} {'sp_d14':>6s} | {'Ncell7':>8s}")
    for tag in SHELLS:
        tg = fitsio.read(ROOT / f"shell_{tag}_final_wedge_targets.fits",
                         columns=["LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB", "RA", "DEC", "Z"])
        pos = np.load(ROOT / f"shell_{tag}_final_points_xyz.npy").astype(np.float64)
        l1 = tg["LAMBDA1"].astype(np.float64); l2 = tg["LAMBDA2"].astype(np.float64); l3 = tg["LAMBDA3"].astype(np.float64)
        n_all = len(l1)
        # comoving volume + 7 Mpc/h cell budget (from this shell's footprint)
        zlo, zhi = [float(x.replace("p", ".")) for x in tag.split("_")]
        r_lo, r_hi = cosmo.comoving_distance([zlo, zhi]).value
        ra0, ra1 = tg["RA"].min(), tg["RA"].max(); d0, d1 = tg["DEC"].min(), tg["DEC"].max()
        omega = np.deg2rad(ra1 - ra0) * (np.sin(np.deg2rad(d1)) - np.sin(np.deg2rad(d0)))
        vol = (omega / 3.0) * (r_hi**3 - r_lo**3)
        ncell = vol / (7.0 / h) ** 3

        # subsample for tractable CV
        idx = np.arange(n_all)
        if n_all > args.max_per_shell:
            idx = rng.permutation(n_all)[: args.max_per_shell]
        p = pos[idx]; y1 = l1[idx]
        tree = cKDTree(p)
        aper = [np.log1p(tree.query_ball_point(p, R / h, return_length=True)) for R in args.apertures_hmpc]
        Xa = np.column_stack(aper); Xp = p; Xb = np.column_stack([p, Xa])

        r2p, r2a, r2b = cvr2(Xp, y1), cvr2(Xa, y1), cvr2(Xb, y1)
        sp7 = spearmanr(aper[1], y1).statistic       # density @7 Mpc/h vs λ1
        sp14 = spearmanr(aper[3], y1).statistic       # density @14 Mpc/h vs λ1
        iqr = np.subtract(*np.percentile(l1, [75, 25]))
        pcl = float((l1 > 0.2).mean())
        print(f"{tag:11s} {n_all:7d} {l1.mean():8.3f} {l1.std():7.3f} {iqr:7.3f} {pcl:8.3f} | "
              f"{r2p:7.3f} {r2a:7.3f} {r2b:7.3f} | {sp7:6.2f} {sp14:6.2f} | {ncell:8.0f}")

    print("\nread: (a) if l1_std small at shell0 -> low variance, nothing to predict (intrinsic).")
    print("      (b) R2_pos >> R2_aper at shell0 -> field EXISTS, local density can't see it")
    print("          (feature/smoothing-scale issue, possibly fixable).")
    print("      (c) R2_pos ~0 too -> field flat/unpredictable at low z -> scope-guard justified.")
    print("      (d) low Ncell7 / degenerate CWEB -> cosmic-variance-limited (few modes).")

    # CWEB composition per shell (separate loop, all rows)
    print(f"\n{'shell':11s}  " + "  ".join(f"{CWEB_NAMES[k]:>9s}" for k in range(4)))
    for tag in SHELLS:
        cw = fitsio.read(ROOT / f"shell_{tag}_final_wedge_targets.fits", columns=["CWEB"])["CWEB"]
        fr = [np.mean(cw == k) for k in range(4)]
        print(f"{tag:11s}  " + "  ".join(f"{f:9.3f}" for f in fr))


if __name__ == "__main__":
    main()
