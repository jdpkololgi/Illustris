#!/usr/bin/env python3
"""True-vs-predicted parity plots for the RA200-240 transfer tests (GraphNet vs 3-D U-Net).

Both models are scored on the same 95,220-galaxy test mask of the disjoint wedge; DTFE(cal) R2 is
annotated per panel as the no-ML reference. Inputs are the prediction dumps written by the transfer
eval runs (GraphNet: posterior_pred_eigs_seed_42.npz; U-Net: t2_transfer_pred_ra200_240.npz).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

GNET_NPZ = Path("/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_TRANSFER_ra200_240_eval/posterior_pred_eigs_seed_42.npz")
UNET_NPZ = Path("/pscratch/sd/d/dkololgi/abacus/field_level_tests/T2_transfer/t2_transfer_pred_ra200_240.npz")
DTFE_R2 = (0.534, 0.604, 0.634)          # cal, same wedge/test mask (classical_baseline/ra200_240_transfer)
LAM = (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$")


def panel(ax, y, p, name, k, extra=""):
    r2 = r2_score(y, p)
    lo = min(np.quantile(y, 0.001), np.quantile(p, 0.001))
    hi = max(np.quantile(y, 0.999), np.quantile(p, 0.999))
    hb = ax.hexbin(y, p, gridsize=90, bins="log", cmap="viridis",
                   extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
    ax.axvline(0.2, color="gray", ls=":", lw=0.8)
    ax.axhline(0.2, color="gray", ls=":", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(f"{name} — {LAM[k]}   $R^2$={r2:.3f}{extra}", fontsize=11)
    ax.set_xlabel(f"true {LAM[k]}"); ax.set_ylabel(f"predicted {LAM[k]}")
    return hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/transfer_plots/parity_graphnet_vs_unet_ra200_240.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    g = np.load(GNET_NPZ)
    gy, gp = g["true_raw"], g["pred_mean"]           # already test-subset rows

    u = np.load(UNET_NPZ)
    um = u["test_mask"].astype(bool)
    uy, up = u["true"][um], u["pred"][um]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
    for k in range(3):
        panel(axes[0, k], gy[:, k], gp[:, k], "GraphNet+NPE (posterior mean)", k,
              extra=f"  [DTFE {DTFE_R2[k]:.3f}]")
        panel(axes[1, k], uy[:, k], up[:, k], "3-D U-Net (T2)", k,
              extra=f"  [DTFE {DTFE_R2[k]:.3f}]")
    fig.suptitle("Pure-inductive transfer to the disjoint RA 200–240 wedge — true vs predicted "
                 f"(same {um.sum():,}-galaxy test mask; dotted = $\\lambda_{{th}}$=0.2; DTFE cal $R^2$ in brackets)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")

    # per-model R2 recap for the log
    for nm, (y, p) in (("GraphNet", (gy, gp)), ("U-Net", (uy, up))):
        print(nm, [round(float(r2_score(y[:, k], p[:, k])), 4) for k in range(3)])


if __name__ == "__main__":
    main()
