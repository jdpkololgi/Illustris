#!/usr/bin/env python3
"""Visual predictions for the P8 recovery best checkpoints (same slab as the short-screen figures).

  fig10_recovery_lambda1_beforeafter.png  TRUE / short-screen / recovery / CIC for lambda1
  fig11_recovery_lambda23.png             TRUE vs recovery G,U and CIC for lambda2 and lambda3
  fig12_recovery_classes.png              T-web classes at lambda_th=0.2 for the recovery models
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("p8ev", REPO / "workflows/abacus_tweb/plot_p8_smoke_eval.py")
p8ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(p8ev)

REC = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1")
OUT = p8ev.OUT
CLS_NAMES = ["void", "wall", "filament", "knot"]
CLS_COLS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]
LAM_TH = 0.2


def load_all(rot=0):
    """Short-screen preds + CIC (from p8ev) plus recovery best-checkpoint preds, all on one id order."""
    ids, truth, short, meta = p8ev.load_rotation(rot)
    out = {}
    for m, nm in (("graph", "G-PATCH"), ("unet", "U-PATCH")):
        d = REC / m / f"rotation_{rot}/seed_42"
        rid = np.load(d / "best_validation_parent_node_id.npy")
        rp = np.load(d / "best_validation_eigenvalues.npy").astype(np.float64)
        o = np.argsort(rid); pos = np.searchsorted(rid[o], ids)
        assert (rid[o][pos] == ids).all(), f"{nm} id mismatch"
        out[nm] = rp[o[pos]]
        out[nm + "_epoch"] = json.load(open(d / "best_validation_report.json"))
    return ids, truth, short, out, meta


def slab_of(ids, meta, rot=0):
    pts = np.load(p8ev.POINTS, mmap_mode="r")
    xyz = np.asarray(pts[ids][:, :3], np.float64)
    sb = np.load(p8ev.P4 / "super_blocks.npz")
    vf = json.load(open(p8ev.P8 / f"rotation_{rot}/roles.json"))["validation_fold"]
    cand = np.where((sb["cap"] == 1) & (sb["fold"] == vf))[0]
    sbid = sb["superblock_id"][cand[np.argmax(sb["active_count"][cand])]]
    inb = meta["superblock_id"] == sbid
    z0 = np.median(xyz[inb, 2])
    return xyz, inb & (np.abs(xyz[:, 2] - z0) < 20), sbid


def main():
    ids, truth, short, rec, meta = load_all()
    xyz, slab, sbid = slab_of(ids, meta)
    x, y = xyz[slab, 0], xyz[slab, 1]
    gm = rec["G-PATCH_epoch"]["primary_macro_r2_lambda1"]
    um = rec["U-PATCH_epoch"]["primary_macro_r2_lambda1"]
    banner = (f"P8 RECOVERY best checkpoints (runs IN FLIGHT): G macro {gm:.4f}, U macro {um:.4f} "
              f"| validation super-block {sbid}, 40 Mpc slab, never trained on")

    # ---- fig10: lambda1 before/after
    panels = [("TRUE λ1", truth[:, 0], None),
              ("G short-screen", short["G-PATCH"][:, 0], None),
              ("G RECOVERY", rec["G-PATCH"][:, 0], None),
              ("U short-screen", short["U-PATCH"][:, 0], None),
              ("U RECOVERY", rec["U-PATCH"][:, 0], None),
              ("CIC (train-affine)", short["CIC (train-affine)"][:, 0], None)]
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.8), sharex=True, sharey=True)
    for ax, (nm, val, _) in zip(axes, panels):
        s = ax.scatter(x, y, c=np.clip(val[slab], -0.4, 0.8), s=6, cmap="RdBu_r", vmin=-0.4, vmax=0.8, lw=0)
        t = nm if nm.startswith("TRUE") else f"{nm}\nslab R²={r2_score(truth[slab,0], val[slab]):.3f}"
        ax.set_title(t, fontsize=10, fontweight="bold" if nm.startswith("TRUE") else "normal")
        ax.set_aspect("equal"); ax.set_xlabel("X [Mpc]")
    axes[0].set_ylabel("Y [Mpc]")
    fig.colorbar(s, ax=axes, shrink=0.85, label="λ1 (clipped)")
    fig.suptitle("λ1 — short screen vs exposure-aware recovery.  " + banner, fontsize=11)
    fig.savefig(OUT / "fig10_recovery_lambda1_beforeafter.png", dpi=130); plt.close(fig)
    print("fig10 done")

    # ---- fig11: lambda2 / lambda3
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.2), sharex=True, sharey=True)
    clim = {1: (-0.25, 1.0), 2: (-0.1, 1.5)}
    for row, k in enumerate((1, 2)):
        vmin, vmax = clim[k]
        cols = [("TRUE", truth[:, k]), ("G-PATCH RECOVERY", rec["G-PATCH"][:, k]),
                ("U-PATCH RECOVERY", rec["U-PATCH"][:, k]), ("CIC", short["CIC (train-affine)"][:, k])]
        for col, (nm, val) in enumerate(cols):
            ax = axes[row, col]
            s = ax.scatter(x, y, c=np.clip(val[slab], vmin, vmax), s=6, cmap="RdBu_r", vmin=vmin, vmax=vmax, lw=0)
            ax.set_title(f"{nm} λ{k+1}" + ("" if nm == "TRUE" else
                         f"\nslab R²={r2_score(truth[slab,k], val[slab]):.3f}"),
                         fontsize=10, fontweight="bold" if nm == "TRUE" else "normal")
            ax.set_aspect("equal")
            if row == 1: ax.set_xlabel("X [Mpc]")
            if col == 0: ax.set_ylabel("Y [Mpc]")
        fig.colorbar(s, ax=axes[row, :], shrink=0.8, label=f"λ{k+1}")
    fig.suptitle("λ2 (top) and λ3 (bottom), recovery checkpoints.  " + banner, fontsize=11)
    fig.savefig(OUT / "fig11_recovery_lambda23.png", dpi=125); plt.close(fig)
    print("fig11 done")

    # ---- fig12: classes
    srcs = [("TRUE", truth), ("G-PATCH RECOVERY", rec["G-PATCH"]),
            ("U-PATCH RECOVERY", rec["U-PATCH"]), ("CIC", short["CIC (train-affine)"])]
    cls = {nm: (v[slab] > LAM_TH).sum(axis=1) for nm, v in srcs}
    tc = cls["TRUE"]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16.5), sharex=True, sharey=True)
    cmap = ListedColormap(CLS_COLS)
    for r, (nm, _) in enumerate(srcs):
        c = cls[nm]
        ax = axes[r, 0]
        ax.scatter(x, y, c=c, cmap=cmap, vmin=-0.5, vmax=3.5, s=6, lw=0)
        frac = " / ".join(f"{(c==k).mean()*100:.0f}%" for k in range(4))
        acc = "" if nm == "TRUE" else f"   acc={np.mean(c==tc)*100:.0f}%"
        ax.set_title(f"{nm} — all ({frac}){acc}", fontsize=9.5, fontweight="bold" if nm=="TRUE" else "normal")
        ax.set_aspect("equal"); ax.set_ylabel(f"{nm}\nY [Mpc]", fontsize=9)
        for k in range(4):
            ax = axes[r, k+1]; m = c == k
            ax.scatter(x[~m], y[~m], c="0.88", s=3, lw=0)
            ax.scatter(x[m], y[m], c=CLS_COLS[k], s=6, lw=0)
            if nm == "TRUE":
                ax.set_title(f"{CLS_NAMES[k]} (n={m.sum():,})", fontsize=9.5, fontweight="bold")
            else:
                tm = tc == k
                ax.set_title(f"{CLS_NAMES[k]} n={m.sum():,}\nrecall {100*(m&tm).sum()/max(tm.sum(),1):.0f}% / "
                             f"prec {100*(m&tm).sum()/max(m.sum(),1):.0f}%", fontsize=8.5)
            ax.set_aspect("equal")
    for ax in axes[3]: ax.set_xlabel("X [Mpc]")
    fig.legend(handles=[plt.Line2D([],[],marker="o",ls="",color=CLS_COLS[k],label=CLS_NAMES[k]) for k in range(4)],
               loc="upper right", ncol=4, fontsize=10)
    fig.suptitle("T-web environments at λ_th=0.2 — recovery checkpoints.  " + banner, fontsize=11.5)
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig(OUT / "fig12_recovery_classes.png", dpi=118); plt.close(fig)
    print("fig12 done")


if __name__ == "__main__":
    main()
