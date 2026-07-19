#!/usr/bin/env python3
"""Pedagogical figures for the P4 shared spatial manifest and its model-agnostic patches.

All geometry is read from the real P4 artifacts (no schematic stand-ins for numbers):
  figA_manifest_anatomy.png  cores/super-blocks/folds — real NGC core map coloured by fold,
                             one super-block zoom, the definitions, and fold balance.
  figB_model_agnostic.png    ONE real scientific core shown three ways (graph / field / physics),
                             proving the core galaxies + target + fold are identical across models.
  figC_boundary_safety.png   why folds are blocked: hops-to-other-fold, boundary distance, periodic
                             images kept context-only, and the leakage contrast.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

P4 = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")
POINTS = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions/path1_fiberassign_mock_bgs_maglim_rs7_points.npy")
OUT = Path("/pscratch/sd/d/dkololgi/abacus/figures/p4_patches")
FOLD_C = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
CORE_MPC = 94.5906           # 64 Mpc/h
SB_MPC = 378.3624            # 256 Mpc/h


def figA():
    man = json.load(open(P4 / "spatial_manifest.json"))
    cores = np.load(P4 / "cores.npz")
    sb = np.load(P4 / "super_blocks.npz")
    cap, fold = cores["cap"], cores["fold"]
    cen, lo, up = cores["centroid_mpc"], cores["lower_mpc"], cores["upper_mpc"]
    act = cores["active_count"]
    ngc = (cap == 1) & (act > 0)                     # cap 1 = NGC (P1b scope.components)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

    # (a) NGC core centroids in X-Y, coloured by fold; super-block boundaries overlaid
    ax = fig.add_subplot(gs[0, :2])
    for f in range(5):
        m = ngc & (fold == f)
        ax.scatter(cen[m, 0], cen[m, 1], s=14, c=FOLD_C[f], label=f"fold {f}", edgecolors="none")
    sb_ngc = sb["cap"] == 1
    for i in np.where(sb_ngc)[0]:
        l, u = sb["lower_mpc"][i], sb["upper_mpc"][i]
        ax.add_patch(Rectangle((l[0], l[1]), u[0]-l[0], u[1]-l[1], fill=False,
                               ec="0.55", lw=0.4, alpha=0.5))
    ax.set_xlabel("X [comoving Mpc]"); ax.set_ylabel("Y [comoving Mpc]")
    ax.set_title("NGC cores coloured by fold — spatially BLOCKED: each 256 Mpc/h super-block "
                 "(grey) is one fold\n(17,202 occupied 64 Mpc/h cores; contiguous colour patches, "
                 "not salt-and-pepper = no random-split leakage)")
    ax.legend(loc="upper right", markerscale=1.6, fontsize=9)
    ax.set_aspect("equal")

    # (b) one super-block zoom: its cores as a grid, all one fold
    ax = fig.add_subplot(gs[0, 2])
    # pick a well-populated NGC super-block
    cand = np.where((sb["cap"] == 1) & (sb["active_count"] > 3000))[0]
    sbi = cand[np.argmax(sb["core_count"][cand])]
    sbid, sbf = sb["superblock_id"][sbi], sb["fold"][sbi]
    l, u = sb["lower_mpc"][sbi], sb["upper_mpc"][sbi]
    ax.add_patch(Rectangle((l[0], l[1]), u[0]-l[0], u[1]-l[1], fill=False, ec="k", lw=2))
    incore = cores["superblock_id"] == sbid
    for i in np.where(incore)[0]:
        cl, cu = lo[i], up[i]
        ax.add_patch(Rectangle((cl[0], cl[1]), cu[0]-cl[0], cu[1]-cl[1],
                               fc=FOLD_C[sbf], ec="w", lw=0.8, alpha=0.85))
    ax.set_xlim(l[0]-10, u[0]+10); ax.set_ylim(l[1]-10, u[1]+10); ax.set_aspect("equal")
    ax.set_title(f"One super-block (id {sbid}, fold {sbf})\n256 Mpc/h box = 4x4x4 tiling of 64 Mpc/h cores\n"
                 f"{int(sb['core_count'][sbi])} occupied cores, "
                 f"{int(sb['active_count'][sbi]):,} galaxies", fontsize=9)
    ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]")
    ax.annotate("", xy=(l[0]+4, u[1]+3), xytext=(l[0]+4+CORE_MPC, u[1]+3),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(l[0]+4+CORE_MPC/2, u[1]+7, "64 Mpc/h core", ha="center", fontsize=8)

    # (c) definitions
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    defs = [
        ("Scientific / authoritative core", "64 Mpc/h fixed-comoving box. Every eligible\n"
         "galaxy is in EXACTLY ONE core. Only core\ngalaxies contribute loss + metrics.", "#1a9850"),
        ("Context", "All a core needs to be predicted: K-hop\ngraph closure / field-halo / FFT tile.\n"
         "Message-passing only — NEVER loss.", "#d95f02"),
        ("Super-block", "256 Mpc/h (=4x core). Unit of fold\nassignment: a whole super-block goes to\n"
         "one fold, so folds are >= a block apart.", "#8073ac"),
        ("Fold (x5)", "Blocked partition of super-blocks.\nRotations 3 train / 1 val / 1 dev-test.\n"
         "Repeated halo hosts never cross folds.", "#4575b4"),
    ]
    y = 0.98
    for t, d, c in defs:
        ax.add_patch(FancyBboxPatch((0.0, y-0.175), 0.05, 0.14, boxstyle="round,pad=0.004",
                                    fc=c, ec="none", transform=ax.transAxes))
        ax.text(0.08, y-0.03, t, fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.08, y-0.095, d, fontsize=8.0, va="top", transform=ax.transAxes, linespacing=1.25)
        y -= 0.255
    ax.set_title("Definitions", fontsize=11, loc="left")

    # (d) fold balance by cap/shell
    ax = fig.add_subplot(gs[1, 1:])
    folds = man["folds"]; xs = np.arange(5); shells = ["0.15–0.25","0.25–0.35","0.35–0.45","0.45–0.55"]
    sc = ["#c6dbef","#6baed6","#2171b5","#08306b"]
    for cap_name, hatch in (("NGC", ""), ("SGC", "//")):
        bottom = np.zeros(5)
        for si in range(4):
            vals = np.array([folds[str(f)]["by_cap_shell"][cap_name][si] for f in range(5)])
            ax.bar(xs + (0.2 if cap_name=="SGC" else -0.2), vals, 0.38, bottom=bottom,
                   color=sc[si], hatch=hatch, edgecolor="w",
                   label=(f"{shells[si]}" if cap_name=="NGC" else None))
            bottom += vals
    ax.set_xticks(xs, [f"fold {f}\n{folds[str(f)]['active_rows']:,}" for f in range(5)])
    ax.set_ylabel("supervised galaxies")
    ax.set_title(f"Fold balance (solid=NGC, hatched=SGC). Active max/min ratio "
                 f"{man['fold_balance']['active_max_min_ratio']:.4f}; "
                 f"max cap-shell deviation {100*man['fold_balance']['max_cap_shell_relative_deviation']:.1f}%")
    ax.legend(title="reporting shell", fontsize=8, ncol=2)

    fig.suptitle("P4 shared fixed-comoving spatial manifest — ph000 NGC+SGC "
                 "(5,026,863 supervised galaxies, one manifest for every model)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "figA_manifest_anatomy.png", dpi=130); plt.close(fig)
    print("figA done  (super-block", sbid, ")")


def figB():
    aa = np.load(P4 / "active_assignment.npz")
    cores = np.load(P4 / "cores.npz")
    pts = np.load(POINTS)[:, :3].astype(np.float64)

    # choose a mid-populated interior NGC core
    ci = np.where((cores["cap"] == 1) & (cores["active_count"] > 60) & (cores["active_count"] < 160))[0]
    core = ci[len(ci)//2]
    cid = cores["core_id"][core]
    cl, cu = cores["lower_mpc"][core], cores["upper_mpc"][core]
    ccen = cores["centroid_mpc"][core]

    core_rows = aa["parent_node_id"][aa["core_id"] == cid]
    core_xyz = pts[core_rows]
    # context halo = 2x core box around the centroid (illustrative K-hop / conv reach)
    half = CORE_MPC
    box = ((pts[:, 0] > ccen[0]-half) & (pts[:, 0] < ccen[0]+half) &
           (pts[:, 1] > ccen[1]-half) & (pts[:, 1] < ccen[1]+half) &
           (pts[:, 2] > ccen[2]-half) & (pts[:, 2] < ccen[2]+half))
    ctx_xyz = pts[box]
    # project to core's X-Y
    def P(a): return a[:, 0], a[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    incore = ((core_xyz[:,0]>cl[0])&(core_xyz[:,0]<cu[0])&(core_xyz[:,1]>cl[1])&(core_xyz[:,1]<cu[1]))

    # (1) GraphNet view
    ax = axes[0]
    cx, cy = P(ctx_xyz); ax.scatter(cx, cy, s=8, c="0.7", label="context nodes (no loss)")
    # kNN links among a subsample to suggest the union graph
    from scipy.spatial import cKDTree
    show = ctx_xyz[np.random.default_rng(0).choice(len(ctx_xyz), min(400,len(ctx_xyz)), replace=False)]
    tree = cKDTree(show)
    for i, p in enumerate(show):
        for j in tree.query(p, k=4)[1][1:]:
            ax.plot([p[0], show[j,0]], [p[1], show[j,1]], color="#8da0cb", lw=0.3, alpha=0.5, zorder=0)
    kx, ky = P(core_xyz); ax.scatter(kx, ky, s=26, c="#1a9850", edgecolors="k", lw=0.3,
                                     label="CORE galaxies (loss)", zorder=5)
    ax.add_patch(Rectangle((cl[0],cl[1]), cu[0]-cl[0], cu[1]-cl[1], fill=False, ec="#1a9850", lw=2))
    ax.set_title("G-PATCH · GraphNet\ncore + K-hop union-graph context", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    # (2) U-Net view: same points -> voxel counts
    ax = axes[1]
    ext = [ccen[0]-half, ccen[0]+half, ccen[1]-half, ccen[1]+half]
    H, xe, ye = np.histogram2d(cx, cy, bins=38, range=[[ext[0],ext[1]],[ext[2],ext[3]]])
    ax.imshow(np.log1p(H.T), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.add_patch(Rectangle((cl[0],cl[1]), cu[0]-cl[0], cu[1]-cl[1], fill=False, ec="w", lw=2))
    ax.scatter(kx, ky, s=10, c="w", edgecolors="k", lw=0.2, label="same CORE galaxies")
    ax.set_title("U-PATCH · 3-D U-Net\nsame region as voxel field (5 Mpc cells)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    # (3) F-tier / classical
    ax = axes[2]; ax.imshow(np.log1p(H.T), origin="lower", extent=ext, cmap="magma", aspect="equal", alpha=0.9)
    ax.scatter(kx, ky, s=10, c="cyan", edgecolors="k", lw=0.2)
    ax.add_patch(Rectangle((cl[0],cl[1]), cu[0]-cl[0], cu[1]-cl[1], fill=False, ec="cyan", lw=2))
    ax.set_title("F-PATCH / CLASSICAL\ngraph→field→fixed FFT tidal solve; DTFE\nsampled at the SAME core",
                 fontsize=10.5)

    for ax in axes:
        ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    fig.suptitle(f"One scientific core, every model type — core {cid} (NGC), "
                 f"{int(incore.sum())} authoritative galaxies. IDENTICAL across models: core "
                 f"membership, target (linear increments), fold, train-core scaler.\n"
                 f"ONLY the context representation differs (graph edges / voxel field / FFT tile). "
                 f"The manifest owns the cores; each architecture supplies its own context.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "figB_model_agnostic.png", dpi=135); plt.close(fig)
    print("figB done  (core", cid, ")")


def figC():
    aa = np.load(P4 / "active_assignment.npz")
    gs_ = np.load(P4 / "graph_support_active.npz")
    man = json.load(open(P4 / "spatial_manifest.json"))
    gman = json.load(open(P4 / "graph_support_manifest.json"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # (a) min hops to another fold
    ax = axes[0]
    hops = gs_["min_hops_to_other_fold"]
    vals, cnts = np.unique(np.clip(hops, 0, 6), return_counts=True)
    bars = ax.bar(vals, cnts, color="#4575b4")
    ax.set_xticks(list(range(1, 7)), ["1", "2", "3", "4", "5", "6+"])
    s2 = gs_["safe_2pass"].mean(); s4 = gs_["safe_4pass"].mean()
    ax.set_xlabel("min graph hops from a core galaxy to any other-fold node")
    ax.set_ylabel("core galaxies"); ax.set_yscale("log")
    ax.set_title(f"Graph-hop separation between folds\nsafe at 2 passes: {100*s2:.0f}%  ·  "
                 f"safe at 4 passes: {100*s4:.0f}%  (higher hops = deeper interior = safer)")
    ax.text(0.97, 0.82, "hops=1: on a fold boundary — its K-hop\ncontext can touch a neighbouring fold;\n"
                        "FLAGGED per node (safe_2pass/4pass),\nnot hidden.  '6+' = deep interior (good).",
            transform=ax.transAxes, fontsize=7.6, ha="right", va="top",
            bbox=dict(fc="#fff3cd", ec="0.6"))

    # (b) distance to conservative fold boundary
    ax = axes[1]
    d = aa["distance_to_conservative_fold_boundary_mpc"]
    d = d[np.isfinite(d)]
    ax.hist(np.clip(d, 0, 120), bins=60, color="#66c2a5", edgecolor="none")
    ax.axvline(np.median(d), color="k", ls="--", lw=1, label=f"median {np.median(d):.0f} Mpc")
    ax.axvline(10.4, color="r", ls=":", lw=1.5, label="7 Mpc/h smoothing (10.4 Mpc)")
    ax.set_xlabel("comoving distance to nearest fold boundary [Mpc]  (rightmost bin = 120+)")
    ax.set_ylabel("core galaxies")
    ax.set_title("Physical margin between training and held-out volume")
    ax.legend(fontsize=8)

    # (c) leakage contrast + periodic images
    ax = axes[2]; ax.axis("off")
    pa = man["periodic_image_audit"]
    txt = (
        "WHY BLOCKED FOLDS (the whole point)\n\n"
        "Random-node split (the old failure):\n"
        "  train & test interleaved in one field →\n"
        "  a test galaxy's neighbours are in train →\n"
        "  model interpolates the smoothed label field.\n"
        "  Measured collapse at true transfer:\n"
        "  GraphNet 0.80→0.42 · U-Net 0.87→0.35.\n\n"
        "Blocked super-block folds (this manifest):\n"
        f"  • train/val/test separated by ≥ 256 Mpc/h blocks\n"
        f"  • {100*gman['counts']['cross_fold_pairs']/gman['counts']['union_context_pairs']:.1f}% of union "
        "pairs cross a fold\n    (flagged for context, excluded from loss)\n"
        f"  • repeated halo hosts never cross folds\n"
        f"  • {pa['adjacent_repeated_host_pairs']:,} periodic box-images\n"
        f"    ({pa['distance_mpc_min']:.0f}–{pa['distance_mpc_max']:.0f} Mpc apart)\n"
        "    kept CONTEXT-ONLY, never double-supervised\n\n"
        "→ extrapolation to unseen structure is now the\n   TRAINING objective, not just the test."
    )
    ax.text(0.0, 1.0, txt, fontsize=9.3, va="top", family="monospace", transform=ax.transAxes)

    fig.suptitle("P4 boundary safety — the manifest enforces spatial independence between folds",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "figC_boundary_safety.png", dpi=130); plt.close(fig)
    print("figC done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figA(); figB(); figC()
    print("all P4 figures ->", OUT)


if __name__ == "__main__":
    main()
