#!/usr/bin/env python3
"""Evaluation figures for the P8 rotation-0/2 short screens (G-PATCH, U-PATCH, CIC).

All joins are by parent_node_id against the P4 manifest and P8 frozen arrays. Short screens are
NOT converged (2,000 replacement-sampled steps, ~15% of training cores exposed, single validation
evaluation) — every figure carries that banner.

  fig1_roles_rotation0.png   which cores trained / validated(scored) / dev-test(sealed), both caps
  fig2_visual_predictions.png one validation super-block, thin slab: true λ1 vs G / U / CIC fields
  fig3_parity_and_shells.png  λ1 parity per model + per-shell R2 bars (rot 0 and rot 2)
  fig4_diagnostics.png        |error| vs fold-boundary distance, variance ratio (regression-to-mean),
                              rot0-vs-rot2 per-shell consistency
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

P8 = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")
POINTS = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions/path1_fiberassign_mock_bgs_maglim_rs7_points.npy")
OUT = Path("/pscratch/sd/d/dkololgi/abacus/figures/p8_smoke_eval")
BANNER = ("SHORT SCREEN (NOT CONVERGED): 2,000 replacement-sampled steps, ~15% of training cores "
          "exposed, one validation pass — plumbing/transfer smoke, not a model verdict")
MODELS = {"G-PATCH": "#2166ac", "U-PATCH": "#7b3294", "CIC (train-affine)": "#d95f02"}
SHELLS = ["0.15–0.25", "0.25–0.35", "0.35–0.45", "0.45–0.55"]


def load_rotation(rot: int):
    aa = np.load(P4 / "active_assignment.npz")
    order = np.argsort(aa["parent_node_id"])
    aap = aa["parent_node_id"][order]

    def join(ids):
        pos = np.searchsorted(aap, ids)
        assert (aap[pos] == ids).all()
        j = order[pos]
        return {k: aa[k][j] for k in ("core_id", "superblock_id", "fold", "cap", "shell",
                                      "distance_to_conservative_fold_boundary_mpc")}

    truth_all = np.load(P8 / "parent_eigenvalues.npy", mmap_mode="r")
    out = {}
    for tag, path in (("G-PATCH", P8 / f"g_patch/rotation_{rot}/seed_42"),
                      ("U-PATCH", P8 / f"u_patch/rotation_{rot}/seed_42")):
        ids = np.load(path / "best_validation_parent_node_id.npy")
        out[tag] = (ids, np.load(path / "best_validation_eigenvalues.npy").astype(np.float64))
    cid = np.load(P8 / f"classical/rotation_{rot}/validation_parent_node_id.npy")
    out["CIC (train-affine)"] = (cid, np.load(P8 / f"classical/rotation_{rot}/cic_train_affine_eigenvalues.npy").astype(np.float64))

    ids0 = out["G-PATCH"][0]
    for tag, (ids, _) in out.items():
        assert np.array_equal(np.sort(ids), np.sort(ids0)), f"{tag} id set differs"
    # bring all to the G-PATCH id order
    aligned = {}
    for tag, (ids, pred) in out.items():
        if np.array_equal(ids, ids0):
            aligned[tag] = pred
        else:
            o = np.argsort(ids); pos = np.searchsorted(ids[o], ids0)
            aligned[tag] = pred[o[pos]]
    meta = join(ids0)
    truth = np.asarray(truth_all[ids0], np.float64)
    return ids0, truth, aligned, meta


def fig1(rot=0):
    roles = json.load(open(P8 / f"rotation_{rot}/roles.json"))
    cores = np.load(P4 / "cores.npz")
    cen, fold, cap, act = cores["centroid_mpc"], cores["fold"], cores["cap"], cores["active_count"]
    occ = act > 0
    train_f = set(roles["train_folds"]); val_f = roles["validation_fold"]; test_f = roles["development_test_fold"]
    color = np.where(np.isin(fold, list(train_f)), 0, np.where(fold == val_f, 1, np.where(fold == test_f, 2, 3)))
    cmap = {0: ("#66c2a5", f"TRAIN folds {sorted(train_f)}  ({roles['train_authoritative_rows']:,} gal)"),
            1: ("#d95f02", f"VALIDATION fold {val_f} = scored ({roles['validation_authoritative_rows']:,} gal)"),
            2: ("#7b3294", f"DEV-TEST fold {test_f} (SEALED this rotation)")}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, capv, nm in ((axes[0], 1, "NGC"), (axes[1], 0, "SGC")):
        for cv, (col, lab) in cmap.items():
            m = occ & (cap == capv) & (color == cv)
            ax.scatter(cen[m, 0], cen[m, 1], s=13 if capv == 1 else 20, c=col,
                       label=lab if capv == 1 else None, edgecolors="none")
        ax.set_aspect("equal"); ax.set_title(f"{nm} cores — rotation {rot}")
        ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(f"P8 rotation {rot}: which patches trained vs scored — blocked super-block folds\n{BANNER}",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig1_roles_rotation0.png", dpi=130); plt.close(fig)
    print("fig1 done")


def fig2(rot=0):
    ids, truth, preds, meta = load_rotation(rot)
    pts = np.load(POINTS, mmap_mode="r")
    xyz = np.asarray(pts[ids][:, :3], np.float64)
    sb = np.load(P4 / "super_blocks.npz")
    # most-populated NGC validation super-block
    cand = np.where((sb["cap"] == 1) & (sb["fold"] == json.load(open(P8 / f"rotation_{rot}/roles.json"))["validation_fold"]))[0]
    sbi = cand[np.argmax(sb["active_count"][cand])]
    sbid = sb["superblock_id"][sbi]
    inb = meta["superblock_id"] == sbid
    z0 = np.median(xyz[inb, 2])
    slab = inb & (np.abs(xyz[:, 2] - z0) < 20)
    print(f"fig2 super-block {sbid}: {inb.sum():,} val galaxies, slab {slab.sum():,}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), sharex=True, sharey=True)
    vmin, vmax = -0.4, 0.8
    panels = [("TRUE λ1", truth[:, 0])] + [(t, preds[t][:, 0]) for t in MODELS]
    for ax, (nm, val) in zip(axes, panels):
        s = ax.scatter(xyz[slab, 0], xyz[slab, 1], c=np.clip(val[slab], vmin, vmax),
                       s=7, cmap="RdBu_r", vmin=vmin, vmax=vmax, lw=0)
        if nm != "TRUE λ1":
            r2 = r2_score(truth[slab, 0], val[slab])
            ax.set_title(f"{nm}\nslab R²(λ1) = {r2:.3f}", fontsize=10.5)
        else:
            ax.set_title(nm, fontsize=11, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xlabel("X [Mpc]")
    axes[0].set_ylabel("Y [Mpc]")
    fig.colorbar(s, ax=axes, shrink=0.85, label="λ1 (clipped)")
    fig.suptitle(f"Validation super-block {sbid} (rotation {rot}, NGC, 40 Mpc slab) — the models never "
                 f"saw this volume in training\n{BANNER}", fontsize=11)
    fig.savefig(OUT / "fig2_visual_predictions.png", dpi=135); plt.close(fig)
    print("fig2 done")


def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
    ids, truth, preds, meta = load_rotation(0)
    for ax, (tag, col) in zip(axes[0], MODELS.items()):
        y, p = truth[:, 0], preds[tag][:, 0]
        lo, hi = -0.6, 1.1
        ax.hexbin(y, p, gridsize=85, bins="log", cmap="viridis", extent=(lo, hi, lo, hi), mincnt=1)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.1)
        ax.axhline(0.2, color="gray", ls=":", lw=0.7); ax.axvline(0.2, color="gray", ls=":", lw=0.7)
        ax.set_title(f"{tag} — rot0 val λ1   R²={r2_score(y, p):.3f}", fontsize=11)
        ax.set_xlabel("true λ1"); ax.set_ylabel("predicted λ1"); ax.set_aspect("equal")
    # per-shell bars rot0 + rot2
    per = {t: {0: [], 2: []} for t in MODELS}
    for rot in (0, 2):
        _, tr, pr, meta_r = load_rotation(rot)
        for t in MODELS:
            for s in range(4):
                m = meta_r["shell"] == s
                per[t][rot].append(r2_score(tr[m, 0], pr[t][m, 0]))
    xs = np.arange(4)
    for ax, (tag, col) in zip(axes[1], MODELS.items()):
        ax.bar(xs - 0.18, per[tag][0], 0.36, color=col, label="rotation 0")
        ax.bar(xs + 0.18, per[tag][2], 0.36, color=col, alpha=0.45, label="rotation 2")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs, SHELLS); ax.set_ylim(-1.05, 0.75)
        ax.set_title(f"{tag}: per-shell val λ1 R²", fontsize=10.5)
        ax.set_ylabel("R²"); ax.legend(fontsize=8)
        for x, v in zip(xs, per[tag][0]):
            ax.text(x - 0.18, v + (0.03 if v > 0 else -0.09), f"{v:.2f}", ha="center", fontsize=7.5)
    fig.suptitle(f"P8 short-screen parity and per-shell transfer (validation folds, never trained on)\n{BANNER}",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig3_parity_and_shells.png", dpi=130); plt.close(fig)
    print("fig3 done")
    return per


def fig4(per):
    ids, truth, preds, meta = load_rotation(0)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    ax = axes[0]
    d = meta["distance_to_conservative_fold_boundary_mpc"]
    bins = np.array([0, 5, 10.4, 20, 35, 55, 80, 120, 400])
    mids = 0.5 * (bins[1:] + bins[:-1])
    for tag, col in MODELS.items():
        err = np.abs(preds[tag][:, 0] - truth[:, 0])
        med = [np.median(err[(d >= a) & (d < b)]) for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mids, med, "o-", color=col, label=tag, ms=4)
    ax.axvline(10.4, color="r", ls=":", lw=1, label="smoothing 10.4 Mpc")
    ax.set_xscale("log"); ax.set_xlabel("distance to fold boundary [Mpc]")
    ax.set_ylabel("median |λ1 error|")
    ax.set_title("Boundary trend check (flat = no fold-boundary artifact)")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.arange(4)
    for k, (tag, col) in enumerate(MODELS.items()):
        ratio = [preds[tag][meta["shell"] == s, 0].std() / truth[meta["shell"] == s, 0].std()
                 for s in range(4)]
        ax.bar(xs + (k - 1) * 0.26, ratio, 0.26, color=col, label=tag)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(xs, SHELLS); ax.set_ylabel("pred std / true std")
    ax.set_title("Amplitude ratio (<< 1 = regression to the mean)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for tag, col in MODELS.items():
        ax.scatter(per[tag][0], per[tag][2], s=70, c=col, label=tag)
        for s in range(4):
            ax.annotate(SHELLS[s].split("–")[0], (per[tag][0][s], per[tag][2][s]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = (-1.05, 0.75)
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("per-shell R² (rotation 0)"); ax.set_ylabel("per-shell R² (rotation 2)")
    ax.set_title("Rotation consistency (on-diagonal = stable across folds)")
    ax.legend(fontsize=8)

    fig.suptitle(f"P8 short-screen diagnostics — validation fold, rotation 0 unless noted\n{BANNER}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig4_diagnostics.png", dpi=130); plt.close(fig)
    print("fig4 done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    per = fig3()
    fig4(per)
    print("all ->", OUT)


if __name__ == "__main__":
    main()
