#!/usr/bin/env python3
"""True-vs-predicted parity for the P8 recovery best checkpoints (full validation fold).

  fig13_recovery_parity.png       3 eigenvalues x 3 methods, all 999,683 validation-core galaxies,
                                  annotated with R2, best-fit slope and amplitude ratio (pred/true sd)
  fig14_recovery_parity_shells.png lambda1 parity per reporting shell for both learned models + CIC
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("p8ev", REPO / "workflows/abacus_tweb/plot_p8_smoke_eval.py")
p8ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(p8ev)
REC = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1")
OUT = p8ev.OUT
SHELLS = ["0.15–0.25", "0.25–0.35", "0.35–0.45", "0.45–0.55"]
LIMS = {0: (-0.62, 1.05), 1: (-0.3, 1.5), 2: (-0.15, 2.0)}


def load():
    ids, truth, short, meta = p8ev.load_rotation(0)
    P = {"CIC (train-affine)": short["CIC (train-affine)"]}
    for m, nm in (("graph", "G-PATCH recovery"), ("unet", "U-PATCH recovery")):
        d = REC / m / "rotation_0/seed_42"
        rid = np.load(d / "best_validation_parent_node_id.npy")
        rp = np.load(d / "best_validation_eigenvalues.npy").astype(np.float64)
        o = np.argsort(rid); pos = np.searchsorted(rid[o], ids)
        assert (rid[o][pos] == ids).all()
        P[nm] = rp[o[pos]]
    return truth, P, meta


def panel(ax, y, p, lo, hi, title):
    hb = ax.hexbin(y, p, gridsize=95, bins="log", cmap="viridis", extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.1)
    slope = np.polyfit(y, p, 1)[0]
    ratio = p.std() / y.std()
    ax.axhline(0.2, color="gray", ls=":", lw=0.6); ax.axvline(0.2, color="gray", ls=":", lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(f"{title}\nR²={r2_score(y,p):.3f}  slope={slope:.2f}  σp/σt={ratio:.2f}", fontsize=9.5)
    return hb


def main():
    truth, P, meta = load()
    order = ["G-PATCH recovery", "U-PATCH recovery", "CIC (train-affine)"]

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15))
    for r, k in enumerate((0, 1, 2)):
        lo, hi = LIMS[k]
        for c, nm in enumerate(order):
            panel(axes[r, c], truth[:, k], P[nm][:, k], lo, hi, f"{nm} — λ{k+1}")
            axes[r, c].set_xlabel(f"true λ{k+1}")
            if c == 0: axes[r, c].set_ylabel(f"predicted λ{k+1}")
    fig.suptitle("P8 recovery best checkpoints — true vs predicted, complete validation fold "
                 "(999,683 authoritative cores, rotation 0)\n"
                 "slope<1 and σp/σt<1 are the expected conditional-mean shrinkage (≈√R²); "
                 "dotted lines mark λ_th=0.2", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "fig13_recovery_parity.png", dpi=125); plt.close(fig)
    print("fig13 done")

    # per-shell lambda1
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    lo, hi = LIMS[0]
    for r, nm in enumerate(order):
        for s in range(4):
            m = meta["shell"] == s
            panel(axes[r, s], truth[m, 0], P[nm][m, 0], lo, hi, f"{nm}\n{SHELLS[s]}  (n={m.sum():,})")
            axes[r, s].set_xlabel("true λ1")
            if s == 0: axes[r, s].set_ylabel("predicted λ1")
    fig.suptitle("λ1 parity per reporting shell — recovery checkpoints vs CIC (rotation-0 validation fold)\n"
                 "note CIC's sparse-shell panel: predictions spread but uncorrelated (R²<0), "
                 "while the learned models shrink toward the mean but stay correlated", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig14_recovery_parity_shells.png", dpi=118); plt.close(fig)
    print("fig14 done")


if __name__ == "__main__":
    main()
