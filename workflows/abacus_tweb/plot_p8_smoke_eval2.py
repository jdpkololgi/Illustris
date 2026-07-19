#!/usr/bin/env python3
"""P8 short-screen visual evaluation, part 2: lambda2/lambda3 fields and T-web environment classes.

  fig5_visual_lambda23.png  validation super-block slab: TRUE vs G/U/CIC for lambda2 and lambda3
  fig6_environment_classes.png  same slab, classes at lambda_th=0.2: rows TRUE/G/U/CIC,
                                columns all-classes + void/wall/filament/knot separately,
                                with per-class recall/precision vs truth annotated
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("p8ev", REPO / "workflows/abacus_tweb/plot_p8_smoke_eval.py")
p8ev = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p8ev)

OUT = p8ev.OUT
BANNER = p8ev.BANNER
MODELS = p8ev.MODELS
LAM_TH = 0.2
CLS_NAMES = ["void", "wall", "filament", "knot"]
CLS_COLS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]


def slab_selection(rot=0):
    ids, truth, preds, meta = p8ev.load_rotation(rot)
    pts = np.load(p8ev.POINTS, mmap_mode="r")
    xyz = np.asarray(pts[ids][:, :3], np.float64)
    sb = np.load(p8ev.P4 / "super_blocks.npz")
    roles = json.load(open(p8ev.P8 / f"rotation_{rot}/roles.json"))
    cand = np.where((sb["cap"] == 1) & (sb["fold"] == roles["validation_fold"]))[0]
    sbi = cand[np.argmax(sb["active_count"][cand])]
    sbid = sb["superblock_id"][sbi]
    inb = meta["superblock_id"] == sbid
    z0 = np.median(xyz[inb, 2])
    slab = inb & (np.abs(xyz[:, 2] - z0) < 20)
    return ids, truth, preds, xyz, slab, sbid


def fig5():
    ids, truth, preds, xyz, slab, sbid = slab_selection()
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.2), sharex=True, sharey=True)
    clims = {1: (-0.25, 1.0), 2: (-0.1, 1.5)}
    for row, k in enumerate((1, 2)):
        vmin, vmax = clims[k]
        panels = [("TRUE", truth[:, k])] + [(t, preds[t][:, k]) for t in MODELS]
        for col, (nm, val) in enumerate(panels):
            ax = axes[row, col]
            s = ax.scatter(xyz[slab, 0], xyz[slab, 1], c=np.clip(val[slab], vmin, vmax),
                           s=7, cmap="RdBu_r", vmin=vmin, vmax=vmax, lw=0)
            if nm == "TRUE":
                ax.set_title(f"TRUE λ{k+1}", fontsize=11, fontweight="bold")
            else:
                ax.set_title(f"{nm} λ{k+1}\nslab R² = {r2_score(truth[slab, k], val[slab]):.3f}",
                             fontsize=10)
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlabel("X [Mpc]")
            if col == 0:
                ax.set_ylabel("Y [Mpc]")
        fig.colorbar(s, ax=axes[row, :], shrink=0.8, label=f"λ{k+1} (clipped)")
    fig.suptitle(f"Validation super-block {sbid} (rotation 0, NGC, 40 Mpc slab) — λ2 (top) and λ3 "
                 f"(bottom)\n{BANNER}", fontsize=11)
    fig.savefig(OUT / "fig5_visual_lambda23.png", dpi=130)
    plt.close(fig)
    print("fig5 done")


def classes_of(eig3):
    return (eig3 > LAM_TH).sum(axis=1)


def fig6():
    ids, truth, preds, xyz, slab, sbid = slab_selection()
    sources = [("TRUE", truth)] + [(t, preds[t]) for t in MODELS]
    cls = {nm: classes_of(v[slab]) for nm, v in sources}
    true_c = cls["TRUE"]
    x, y = xyz[slab, 0], xyz[slab, 1]

    fig, axes = plt.subplots(4, 5, figsize=(20, 16.5), sharex=True, sharey=True)
    cmap = ListedColormap(CLS_COLS)
    for r, (nm, _) in enumerate(sources):
        c = cls[nm]
        ax = axes[r, 0]
        ax.scatter(x, y, c=c, cmap=cmap, vmin=-0.5, vmax=3.5, s=6, lw=0)
        frac = " / ".join(f"{(c == k).mean()*100:.0f}%" for k in range(4))
        acc = "" if nm == "TRUE" else f"   acc={np.mean(c == true_c)*100:.0f}%"
        ax.set_title(f"{nm} — all classes ({frac}){acc}", fontsize=9.5,
                     fontweight="bold" if nm == "TRUE" else "normal")
        ax.set_aspect("equal")
        for k in range(4):
            ax = axes[r, k + 1]
            m = c == k
            ax.scatter(x[~m], y[~m], c="0.88", s=3, lw=0)
            ax.scatter(x[m], y[m], c=CLS_COLS[k], s=6, lw=0)
            if nm == "TRUE":
                ax.set_title(f"{CLS_NAMES[k]}  (n={m.sum():,})", fontsize=9.5, fontweight="bold")
            else:
                tm = true_c == k
                rec = (m & tm).sum() / max(tm.sum(), 1)
                prec = (m & tm).sum() / max(m.sum(), 1)
                ax.set_title(f"{CLS_NAMES[k]}  n={m.sum():,}\nrecall {rec*100:.0f}% / prec {prec*100:.0f}%",
                             fontsize=8.5)
            ax.set_aspect("equal")
        axes[r, 0].set_ylabel(f"{nm}\nY [Mpc]", fontsize=10)
    for ax in axes[3]:
        ax.set_xlabel("X [Mpc]")
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLS_COLS[k], label=CLS_NAMES[k]) for k in range(4)]
    fig.legend(handles=handles, loc="upper right", fontsize=10, ncol=4)
    fig.suptitle(f"T-web environments at λ_th=0.2 — validation super-block {sbid} (rotation 0, 40 Mpc slab)\n"
                 f"rows: TRUE / G-PATCH / U-PATCH / CIC — amplitude compression turns into class-occupancy "
                 f"bias (few predicted knots/voids)\n{BANNER}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "fig6_environment_classes.png", dpi=120)
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    fig5()
    fig6()
    print("part-2 figures ->", OUT)
