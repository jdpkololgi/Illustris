#!/usr/bin/env python3
"""Rotation-2 evaluation + rotation-0/2 replication for the P8 recovery runs.

  fig15_replication.png   matched-epoch curves, per-shell replication scatter, best-epoch bars
  fig16_rot2_parity.png   true-vs-predicted, 3 eigenvalues x 3 methods, rotation-2 validation fold
  fig17_rot2_visual.png   lambda1 field + T-web classes on a rotation-2 validation super-block (fold 3)
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
SH = ["0.15–0.25", "0.25–0.35", "0.35–0.45", "0.45–0.55"]
SHK = ["0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
CLS_NAMES = ["void", "wall", "filament", "knot"]; CLS_COLS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]
MC = {"graph": ("G-PATCH", "#2166ac"), "unet": ("U-PATCH", "#7b3294")}


def hist(m, rot):
    f = REC / m / f"rotation_{rot}/seed_42/epoch_history.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def preds_for(rot):
    ids, truth, short, meta = p8ev.load_rotation(rot)
    P = {"CIC (train-affine)": short["CIC (train-affine)"]}
    for m, nm in (("graph", "G-PATCH recovery"), ("unet", "U-PATCH recovery")):
        d = REC / m / f"rotation_{rot}/seed_42"
        rid = np.load(d / "best_validation_parent_node_id.npy")
        rp = np.load(d / "best_validation_eigenvalues.npy").astype(np.float64)
        o = np.argsort(rid); pos = np.searchsorted(rid[o], ids)
        assert (rid[o][pos] == ids).all()
        P[nm] = rp[o[pos]]
    return ids, truth, P, meta


def fig15():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for m, (nm, col) in MC.items():
        for rot, ls, mk in ((0, "-", "o"), (2, "--", "s")):
            h = hist(m, rot)
            if not h: continue
            axes[0].plot([d["epoch"] for d in h], [d["primary_macro_r2_lambda1"] for d in h],
                         ls, color=col, marker=mk, ms=4, label=f"{nm} rot{rot}")
    axes[0].axhline(0.440, color="k", ls=":", lw=1); axes[0].text(0.6, 0.446, "frozen R0 0.440", fontsize=7.5)
    axes[0].axhline(0.470, color="g", ls=":", lw=1); axes[0].text(0.6, 0.476, "promotion gate 0.470", fontsize=7.5, color="g")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val macro R²(λ1)")
    axes[0].set_title("Matched-epoch replication: rotation 0 vs 2"); axes[0].legend(fontsize=8, loc="lower right")

    # per-shell replication scatter at each rotation's BEST epoch
    ax = axes[1]
    for m, (nm, col) in MC.items():
        h0, h2 = hist(m, 0), hist(m, 2)
        if not (h0 and h2): continue
        b0 = max(h0, key=lambda d: d["primary_macro_r2_lambda1"])["per_shell_lambda1_r2"]
        b2 = max(h2, key=lambda d: d["primary_macro_r2_lambda1"])["per_shell_lambda1_r2"]
        ax.scatter([b0[k] for k in SHK], [b2[k] for k in SHK], s=80, c=col, label=nm)
        for k, s in zip(SHK, SH):
            ax.annotate(s.split("–")[0], (b0[k], b2[k]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = (0.25, 0.65); ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("per-shell R²(λ1), rotation 0 best"); ax.set_ylabel("rotation 2 best")
    ax.set_title("Per-shell replication (on-diagonal = geography-independent)"); ax.legend(fontsize=8)

    # best-epoch bars
    ax = axes[2]; x = np.arange(2); w = 0.35
    for i, (m, (nm, col)) in enumerate(MC.items()):
        vals = []
        for rot in (0, 2):
            h = hist(m, rot)
            vals.append(max(d["primary_macro_r2_lambda1"] for d in h) if h else 0)
        b = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=nm)
        for xx, v, rot in zip(x + (i - 0.5) * w, vals, (0, 2)):
            n = len(hist(m, rot))
            ax.text(xx, v + 0.006, f"{v:.3f}\n({n} ep)", ha="center", fontsize=7.5)
    ax.axhline(0.470, color="g", ls=":", lw=1.2); ax.text(1.35, 0.474, "promotion gate", fontsize=7.5, color="g")
    ax.axhline(0.440, color="k", ls=":", lw=1)
    ax.set_xticks(x, ["rotation 0", "rotation 2"]); ax.set_ylabel("best val macro R²(λ1)")
    ax.set_ylim(0, 0.58); ax.set_title("Best so far (rot 2 still running — fewer epochs)")
    ax.legend(fontsize=8, loc="lower right")
    for a in axes: a.grid(alpha=0.3)
    fig.suptitle("P8 recovery — rotation-0 vs rotation-2 replication (rot2 trains folds {0,1,4}, "
                 "validates fold 3: different sky, different galaxies)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig15_replication.png", dpi=130); plt.close(fig)
    print("fig15 done")


def fig16(ids, truth, P, meta):
    order = ["G-PATCH recovery", "U-PATCH recovery", "CIC (train-affine)"]
    lims = {0: (-0.62, 1.05), 1: (-0.3, 1.5), 2: (-0.15, 2.0)}
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15))
    for r, k in enumerate((0, 1, 2)):
        lo, hi = lims[k]
        for c, nm in enumerate(order):
            y, p = truth[:, k], P[nm][:, k]
            axes[r, c].hexbin(y, p, gridsize=95, bins="log", cmap="viridis", extent=(lo, hi, lo, hi), mincnt=1)
            axes[r, c].plot([lo, hi], [lo, hi], "r--", lw=1.1)
            axes[r, c].set_xlim(lo, hi); axes[r, c].set_ylim(lo, hi); axes[r, c].set_aspect("equal")
            axes[r, c].set_title(f"{nm} — λ{k+1}\nR²={r2_score(y,p):.3f}  "
                                 f"slope={np.polyfit(y,p,1)[0]:.2f}  σp/σt={p.std()/y.std():.2f}", fontsize=9.5)
            axes[r, c].set_xlabel(f"true λ{k+1}")
            if c == 0: axes[r, c].set_ylabel(f"predicted λ{k+1}")
    fig.suptitle("ROTATION 2 — true vs predicted, complete validation fold 3 "
                 f"({len(truth):,} authoritative cores). Recovery runs still in progress.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig16_rot2_parity.png", dpi=125); plt.close(fig)
    print("fig16 done")


def fig17(ids, truth, P, meta):
    pts = np.load(p8ev.POINTS, mmap_mode="r"); xyz = np.asarray(pts[ids][:, :3], np.float64)
    sb = np.load(p8ev.P4 / "super_blocks.npz")
    vf = json.load(open(p8ev.P8 / "rotation_2/roles.json"))["validation_fold"]
    cand = np.where((sb["cap"] == 1) & (sb["fold"] == vf))[0]
    sbid = sb["superblock_id"][cand[np.argmax(sb["active_count"][cand])]]
    inb = meta["superblock_id"] == sbid; z0 = np.median(xyz[inb, 2])
    slab = inb & (np.abs(xyz[:, 2] - z0) < 20)
    x, y = xyz[slab, 0], xyz[slab, 1]
    srcs = [("TRUE", truth), ("G-PATCH recovery", P["G-PATCH recovery"]),
            ("U-PATCH recovery", P["U-PATCH recovery"]), ("CIC", P["CIC (train-affine)"])]

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    for c, (nm, v) in enumerate(srcs):
        ax = axes[0, c]
        s = ax.scatter(x, y, c=np.clip(v[slab, 0], -0.4, 0.8), s=6, cmap="RdBu_r", vmin=-0.4, vmax=0.8, lw=0)
        ax.set_title(f"{nm} λ1" + ("" if nm == "TRUE" else f"\nslab R²={r2_score(truth[slab,0], v[slab,0]):.3f}"),
                     fontsize=10, fontweight="bold" if nm == "TRUE" else "normal")
        ax.set_aspect("equal"); ax.set_xlabel("X [Mpc]")
        if c == 0: ax.set_ylabel("Y [Mpc]")
        cl = (v[slab] > 0.2).sum(axis=1)
        ax2 = axes[1, c]
        ax2.scatter(x, y, c=cl, cmap=ListedColormap(CLS_COLS), vmin=-0.5, vmax=3.5, s=6, lw=0)
        frac = " / ".join(f"{(cl==k).mean()*100:.0f}%" for k in range(4))
        tc = (truth[slab] > 0.2).sum(axis=1)
        acc = "" if nm == "TRUE" else f"  acc={np.mean(cl==tc)*100:.0f}%"
        ax2.set_title(f"{nm} classes ({frac}){acc}", fontsize=9.5)
        ax2.set_aspect("equal"); ax2.set_xlabel("X [Mpc]")
        if c == 0: ax2.set_ylabel("Y [Mpc]")
    fig.colorbar(s, ax=axes[0, :], shrink=0.8, label="λ1")
    fig.legend(handles=[plt.Line2D([],[],marker="o",ls="",color=CLS_COLS[k],label=CLS_NAMES[k]) for k in range(4)],
               loc="lower right", ncol=4, fontsize=9)
    fig.suptitle(f"ROTATION 2 — validation super-block {sbid} (fold 3, NGC, 40 Mpc slab): λ1 field (top) "
                 f"and T-web classes at λ_th=0.2 (bottom).  Different sky region from rotation 0.", fontsize=12)
    fig.savefig(OUT / "fig17_rot2_visual.png", dpi=125); plt.close(fig)
    print("fig17 done | super-block", sbid, "slab", int(slab.sum()))


if __name__ == "__main__":
    fig15()
    ids, truth, P, meta = preds_for(2)
    fig16(ids, truth, P, meta)
    fig17(ids, truth, P, meta)
