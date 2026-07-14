#!/usr/bin/env python3
"""Validation plots for the held-out test region: (1) predicted-vs-true eigenvalue hexbins,
(2) cosmic-web 'fan' (wedge) plots of TRUE vs PREDICTED environment at lambda_th=0.2.
Reads the dump_predictions_positions.py npz; pure matplotlib (no compute)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from astropy.cosmology import Planck18 as cosmo
from sklearn.metrics import r2_score

CLASS_NAMES = ["void", "sheet", "filament", "knot"]
CLASS_COLORS = ["#4a6fa5", "#5aa469", "#e0902c", "#c0392b"]
_ZT = np.linspace(0, 0.75, 8000); _DT = cosmo.comoving_distance(_ZT).value


def tweb(lam, th):
    return ((lam[:, 0] > th).astype(int) + (lam[:, 1] > th).astype(int) + (lam[:, 2] > th).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lth", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)
    pred, true, ra, dec, z = d["pred"], d["true"], d["ra"], d["dec"], d["z"]
    m = z >= 0.15                                  # valid range (z<0.15 has corrupt labels)

    # ---- (1) predicted vs true eigenvalues (hexbin) --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, (ax, nm) in enumerate(zip(axes, ["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"])):
        t, p = true[m, k], pred[m, k]
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        hb = ax.hexbin(t, p, gridsize=50, bins="log", cmap="viridis", mincnt=1, extent=lim + lim)
        ax.plot(lim, lim, "r--", lw=1.2, label="1:1")
        if k == 0:
            ax.axvline(args.lth, color="k", ls=":", lw=1); ax.axhline(args.lth, color="k", ls=":", lw=1)
        r2 = r2_score(t, p)
        ax.set_title(f"{nm}   R$^2$={r2:.3f}   (n={m.sum():,})", fontsize=12)
        ax.set_xlabel(f"true {nm}"); ax.set_ylabel(f"predicted {nm}  (posterior mean)")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Predicted vs true tidal eigenvalues — held-out test region (RA$\\geq$150, z$\\geq$0.15)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "pred_vs_true_eigs.png", dpi=140)
    print(f"saved {out/'pred_vs_true_eigs.png'}")

    # ---- (2) cosmic-web fan (wedge) plot, true vs predicted -------------------------
    sl = m & (dec > 18) & (dec < 28)               # DEC slice for a clean 2D wedge
    r = np.interp(z[sl], _ZT, _DT)
    th = np.deg2rad(ra[sl])
    x, y = r * np.cos(th), r * np.sin(th)
    ct, cp = tweb(true[sl], args.lth), tweb(pred[sl], args.lth)
    agree = np.mean(ct == cp)
    cmap = ListedColormap(CLASS_COLORS); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    for ax, c, ttl in [(axes[0], ct, "TRUE"), (axes[1], cp, "PREDICTED (posterior mean)")]:
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=5, linewidths=0)
        ax.set_aspect("equal"); ax.set_title(f"{ttl}   T-web class ($\\lambda_{{th}}$={args.lth})", fontsize=12)
        ax.set_xlabel("comoving x  [Mpc]"); ax.set_ylabel("comoving y  [Mpc]")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
    axes[1].legend(handles=handles, loc="upper right", fontsize=10, title="environment")
    fig.suptitle(f"Cosmic-web environment on held-out sky (DEC 18-28° slice, {sl.sum():,} galaxies) "
                 f"— class agreement {agree*100:.1f}%", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "fan_true_vs_pred.png", dpi=140)
    print(f"saved {out/'fan_true_vs_pred.png'}")

    # class-composition + confusion summary (printed)
    print(f"\nclass agreement (z>=0.15, DEC slice): {agree*100:.1f}%")
    ct_all, cp_all = tweb(true[m], args.lth), tweb(pred[m], args.lth)
    print(f"overall class agreement (z>=0.15): {np.mean(ct_all==cp_all)*100:.1f}%")
    print("true class fractions:", np.round(np.bincount(ct_all, minlength=4) / len(ct_all), 3))
    print("pred class fractions:", np.round(np.bincount(cp_all, minlength=4) / len(cp_all), 3))
    print(f"P(lambda1>{args.lth}) knot: true {np.mean(true[m,0]>args.lth)*100:.1f}% "
          f"pred {np.mean(pred[m,0]>args.lth)*100:.1f}%")


if __name__ == "__main__":
    main()
