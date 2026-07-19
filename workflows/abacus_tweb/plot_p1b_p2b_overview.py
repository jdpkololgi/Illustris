#!/usr/bin/env python3
"""Overview figures for the authoritative P1b/P2b/P3a products and the generalisation plan.

Produces three PNGs under /pscratch/sd/d/dkololgi/abacus/figures/p1b_p2b_overview/:
  fig1_footprint_and_data.png   sky footprint (NGC+SGC, wedge canaries overlaid), N(z), shell counts,
                                canary-vs-authoritative scale comparison
  fig2_pipeline_status.png      work-package DAG with live statuses + pre/post-shutdown timeline
  fig3_patch_protocol.png       the generalised-model training/validation protocol schematic
"""
from __future__ import annotations

import json
from pathlib import Path

import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = Path("/pscratch/sd/d/dkololgi/abacus/figures/p1b_p2b_overview")
IDX = Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz")
P1B_MAN = Path("/global/homes/d/dkololgi/TNG/Illustris/docs/evidence/p1b_p2b/p1b_manifest.json")
P2B_MAN = Path("/global/homes/d/dkololgi/TNG/Illustris/docs/evidence/p1b_p2b/p2b_union_manifest.json")
PARENT = Path("/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
              "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits")

C_NGC, C_SGC, C_CAN = "#2166ac", "#b2182b", "#f4a582"
STATUS_C = {"COMPLETE": "#1a9850", "ACTIVE": "#fdae61", "GATED": "#bdbdbd",
            "DEFERRED": "#e0e0e0", "OPTIONAL": "#e0e0e0"}


def fig1():
    idx = np.load(IDX)
    man = json.load(open(P1B_MAN))
    p2b = json.load(open(P2B_MAN))
    par = fitsio.read(str(PARENT), columns=["RA", "DEC", "Z"])
    ra = np.asarray(par["RA"], np.float64)
    dec = np.asarray(par["DEC"], np.float64)
    z = np.asarray(par["Z"], np.float64)
    cap, act, ctx = idx["cap"], idx["active"], idx["context"]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    # (a) sky footprint
    ax = axes[0, 0]
    ra_p = np.where(ra > 300, ra - 360, ra)
    sub = np.random.default_rng(0).choice(np.where(ctx)[0], 1_500_000, replace=False)
    for capv, col, nm in ((1, C_NGC, "NGC"), (0, C_SGC, "SGC")):
        m = sub[cap[sub] == capv]
        ax.scatter(ra_p[m], dec[m], s=0.05, c=col, alpha=0.25, rasterized=True, lw=0)
        ax.scatter([], [], s=20, c=col, label=f"{nm}  ({man['counts'][nm]:,} ctx)")
    for (r0, r1, d0, d1, col, lab, ls) in (
            (118, 162, 12.5, 32.6, "k", "P1a/P2a wedge canary", "-"),
            (120, 160, 14.5, 30.6, "0.35", "legacy dense/full-range wedge", "--"),
            (200, 240, 14.5, 30.6, "#7b3294", "RA200-240 transfer wedge (dev)", ":")):
        ax.add_patch(Rectangle((r0, d0), r1 - r0, d1 - d0, fill=False, ec=col, ls=ls, lw=1.8))
        ax.plot([], [], color=col, ls=ls, label=lab)
    ax.set_xlabel("RA [deg] (RA>300 shown as RA-360)"); ax.set_ylabel("DEC [deg]")
    ax.set_title("P1b authoritative footprint: full ph000 NGC+SGC context galaxies "
                 "(1.5M-point subsample shown)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.invert_xaxis()

    # (b) N(z)
    ax = axes[0, 1]
    bins = np.linspace(0.10, 0.60, 101)
    for capv, col, nm in ((1, C_NGC, "NGC"), (0, C_SGC, "SGC")):
        ax.hist(z[ctx & (cap == capv)], bins=bins, histtype="step", lw=1.6, color=col, label=nm)
    ax.hist(z[act], bins=bins, histtype="stepfilled", color="#1a9850", alpha=0.15,
            label=f"ACTIVE (supervised) = {int(act.sum()):,}")
    for zb in (0.15, 0.25, 0.35, 0.45, 0.55):
        ax.axvline(zb, color="gray", ls=":", lw=0.8)
    ax.axvspan(0.585, 0.595, color="red", alpha=0.15, label="sentinel (excluded)")
    ax.set_xlabel("observed z"); ax.set_ylabel("galaxies / bin"); ax.set_yscale("log")
    ax.set_title("N(z): context [0.10,0.60) and active core [0.15,0.55); shell edges dotted")
    ax.legend(fontsize=8)

    # (c) per-shell active counts vs canary
    ax = axes[1, 0]
    shells = ["0.15_0.25", "0.25_0.35", "0.35_0.45", "0.45_0.55"]
    xs = np.arange(4)
    ngc = [man["counts"]["by_shell"][s]["NGC"] for s in shells]
    sgc = [man["counts"]["by_shell"][s]["SGC"] for s in shells]
    canary = [164222, 98584, 34645, 4532]
    ax.bar(xs - 0.25, ngc, 0.25, color=C_NGC, label="P1b NGC")
    ax.bar(xs, sgc, 0.25, color=C_SGC, label="P1b SGC")
    ax.bar(xs + 0.25, canary, 0.25, color=C_CAN, label="P1a wedge canary")
    for x, (a, b, c) in enumerate(zip(ngc, sgc, canary)):
        ax.text(x, (a + b) * 1.1, f"×{(a+b)/c:.0f}", ha="center", fontsize=9, color="k")
    ax.set_yscale("log"); ax.set_xticks(xs, [s.replace("_", "–") for s in shells])
    ax.set_ylabel("active galaxies"); ax.set_title("Supervised galaxies per shell (×N vs canary above bars)")
    ax.legend(fontsize=9)

    # (d) scale comparison
    ax = axes[1, 1]
    rows = [("active galaxies", 301_912, man["counts"]["active"]),
            ("context nodes", 374_537, p2b["counts"]["context_nodes"]),
            ("union pairs", 10_601_479, p2b["counts"]["union_pairs_context"]),
            ("high-z shell (0.45–0.55)", 4_532, man["counts"]["by_shell"]["0.45_0.55"]["all"])]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys + 0.18, [r[1] for r in rows], 0.34, color=C_CAN, label="wedge canary (P1a/P2a)")
    ax.barh(ys - 0.18, [r[2] for r in rows], 0.34, color="#1a9850", label="authoritative (P1b/P2b)")
    for y, r in zip(ys, rows):
        ax.text(r[2] * 1.15, y - 0.18, f"×{r[2]/r[1]:.0f}", va="center", fontsize=10)
    ax.set_xscale("log"); ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_title("Canary → authoritative scale-up"); ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(1e3, 1e9)

    fig.suptitle("GraphWeb-BGS canonical data products (2026-07-18): full NGC+SGC footprint, "
                 "zero cross-cap edges, parent-row-indexed", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig1_footprint_and_data.png", dpi=130)
    plt.close(fig)
    print("fig1 done")


def _box(ax, x, y, w, h, title, sub, status, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=STATUS_C[status], ec="k", lw=0.8,
                                alpha=0.95 if status in ("COMPLETE", "ACTIVE") else 0.65))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 1.6)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color="0.25", shrinkA=2, shrinkB=2))


def fig2():
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0, 17); ax.set_ylim(0, 10); ax.axis("off")

    _box(ax, 0.3, 8.6, 3.2, 1.0, "P0 evidence freeze ✓", "7 methods, 219,929 rows,\nblock bootstrap, checksummed", "COMPLETE")
    _box(ax, 0.3, 7.2, 3.2, 1.0, "P0S preservation", "scratch→CFS/HPSS manifest,\nenv exports (no moves yet)", "ACTIVE")
    _box(ax, 4.0, 8.6, 3.4, 1.0, "P1 catalogue ✓ (a+b)", "P1b: 9.54M parent rows,\n5.09M active, NGC+SGC", "COMPLETE")
    _box(ax, 4.0, 7.2, 3.4, 1.0, "P2 canonical graph ✓ (a+b)", "P2b: 190.6M union pairs,\n0 cross-cap, 99 s promote", "COMPLETE")
    _box(ax, 7.9, 8.6, 3.4, 1.0, "P3 fields ✓ (P3a)", "5 Mpc lattices per cap,\n8 channels, all gates", "COMPLETE")
    _box(ax, 7.9, 7.2, 3.4, 1.0, "P4 folds/manifest", "L_core eval, 5 blocked folds,\nmatched val/test geometry", "ACTIVE")
    _box(ax, 11.8, 8.6, 2.3, 1.0, "P5 G-patch", "K-hop core/context\n+ parity gate", "GATED")
    _box(ax, 11.8, 7.2, 2.3, 1.0, "P6 U-patch", "field patches\n+ parity gate", "GATED")
    _box(ax, 14.4, 7.9, 2.3, 1.0, "P7 F-tier", "graph→field→FFT\nconvergence", "GATED")
    _box(ax, 11.8, 5.4, 4.9, 1.2, "P8 protocol showdown", "G/U/F + DTFE on same folds;\nprimary = fold-mean macro R²(λ1); +0.03 gate", "GATED")
    _box(ax, 7.9, 5.4, 3.4, 1.0, "P10 phases", "ph002 benchmark → ph002-005 train,\nph006 cal, ph001 SEALED BLIND", "ACTIVE")
    _box(ax, 11.8, 3.6, 2.3, 1.0, "P9 hybrids", "only if residuals\ncomplementary", "GATED")
    _box(ax, 14.4, 3.6, 2.3, 1.0, "P11 JEPA", "optional,\n+0.03 bar", "DEFERRED")
    _box(ax, 9.3, 1.6, 3.4, 1.2, "P13 DESI canary + VAC", "deterministic first;\ngolden mock → canary → shards", "GATED")
    _box(ax, 13.3, 1.6, 3.4, 1.0, "P12 posterior (FMPE)", "only for posterior columns;\nSBC/TARP on ph006/ph001", "DEFERRED")

    for a, b in (((3.5, 9.1), (4.0, 9.1)), ((5.7, 8.6), (5.7, 8.2)), ((7.4, 9.1), (7.9, 9.1)),
                 ((9.6, 8.6), (9.6, 8.2)), ((7.4, 7.7), (7.9, 7.7)), ((11.3, 9.1), (11.8, 9.1)),
                 ((11.3, 7.7), (11.8, 7.7)), ((13.0, 8.6), (13.0, 8.2)), ((14.1, 8.9), (14.4, 8.5)),
                 ((13.0, 7.2), (13.3, 6.6)), ((15.5, 7.9), (14.6, 6.6)), ((11.3, 6.0), (11.3, 6.0)),
                 ((9.6, 7.2), (9.6, 6.4)), ((12.5, 5.4), (12.5, 4.6)), ((14.2, 6.0), (15.0, 4.6)),
                 ((11.0, 5.4), (11.0, 2.8)), ((13.3, 2.4), (12.7, 2.2))):
        _arrow(ax, *a, *b)

    ax.text(0.3, 6.3, "Legend:", fontsize=10, fontweight="bold")
    for i, (s, c) in enumerate([("COMPLETE", STATUS_C["COMPLETE"]), ("ACTIVE", STATUS_C["ACTIVE"]),
                                ("GATED", STATUS_C["GATED"]), ("DEFERRED/OPTIONAL", STATUS_C["DEFERRED"])]):
        ax.add_patch(Rectangle((0.3 + i * 1.9, 5.9), 0.35, 0.25, fc=c, ec="k", lw=0.6))
        ax.text(0.72 + i * 1.9, 6.02, s, fontsize=8, va="center")

    ax.add_patch(Rectangle((0.3, 0.3), 16.4, 0.9, fc="#f7f7f7", ec="k", lw=0.8))
    marks = [(0.02, "Jul 16-17\nP0 ✓"), (0.16, "Jul 18\nP1b/P2b/P3a ✓"), (0.30, "Jul 19\nP4 + P5/P6 parity"),
             (0.44, "Jul 20\n2-fold screen"), (0.55, "Jul 21\nFREEZE bundle"),
             (0.68, "Jul 22–Aug 3\nNERSC SHUTDOWN"), (0.85, "Aug 3+\nphases → ph001 blind"),
             (0.97, "→ DESI\ncanary/VAC")]
    for fx, lab in marks:
        x = 0.3 + fx * 16.4
        ax.plot([x, x], [0.3, 1.2], color="0.4", lw=0.7)
        ax.text(x, 0.75, lab, fontsize=7.5, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.add_patch(Rectangle((0.3 + 0.62 * 16.4, 0.3), 0.13 * 16.4, 0.9, fc="#fddbc7", alpha=0.5, ec="none"))

    ax.set_title("Generalisable GraphWeb VAC — work-package status (2026-07-18) and timeline",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_pipeline_status.png", dpi=130)
    plt.close(fig)
    print("fig2 done")


def fig3():
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.set_xlim(0, 17); ax.set_ylim(0, 9.5); ax.axis("off")

    # left: core/context patch anatomy
    ax.text(2.9, 9.0, "1 — Patch anatomy (per cap, canonical graph is NEVER rebuilt)",
            fontsize=11, fontweight="bold", ha="center")
    ax.add_patch(Rectangle((0.6, 4.6), 4.6, 3.9, fc="#f0f0f0", ec="k", lw=0.8))
    ax.add_patch(Rectangle((1.9, 5.9), 2.0, 1.4, fc="#1a9850", alpha=0.5, ec="k"))
    ax.text(2.9, 6.6, "CORE\nloss + metrics\n(each galaxy in exactly\nONE core)", ha="center",
            va="center", fontsize=8)
    ax.add_patch(Rectangle((1.15, 5.25), 3.5, 2.7, fill=False, ec="#d95f02", lw=1.8, ls="--"))
    ax.text(2.9, 5.02, "CONTEXT = exact K-hop dependency closure (message passing only, no loss)",
            ha="center", fontsize=7.4, color="#d95f02")
    ax.text(2.9, 4.35, "features copied from the ONE canonical graph; no per-patch normalisation;\n"
                       "no node/edge truncation — oversized cores are subdivided",
            ha="center", fontsize=7.6)

    # middle: blocked folds
    ax.text(8.5, 9.0, "2 — Spatially blocked 5-fold protocol", fontsize=11, fontweight="bold", ha="center")
    rng = np.random.default_rng(3)
    cols = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
    for i in range(6):
        for j in range(4):
            f = rng.integers(0, 5)
            ax.add_patch(Rectangle((6.4 + i * 0.7, 6.4 + j * 0.55), 0.66, 0.51, fc=cols[f], ec="w", lw=1))
    ax.text(8.5, 6.05, "cores → 4·L super-blocks → 5 folds; train 3 / val 1 / dev-test 1;\n"
                       "val and test MATCHED in volume + distance-from-training;\n"
                       "checkpoint on COMPLETE-fold macro R²(λ1) — never per-patch",
            ha="center", fontsize=7.8)
    ax.add_patch(Rectangle((6.4, 4.55), 4.2, 0.85, fc="#fddbc7", ec="k", lw=0.7))
    ax.text(8.5, 4.97, "sealed: dev-test opened per rotation only;\nph001 = SEALED BLIND PHASE (opened once, never tuned on)",
            ha="center", fontsize=7.6)

    # right: candidates + selection
    ax.text(14.3, 9.0, "3 — Matched candidates, one contract", fontsize=11, fontweight="bold", ha="center")
    for k, (nm, sub) in enumerate([("G-PATCH GraphNet", "8-feature schema,\nunion graph"),
                                   ("U-PATCH 3-D U-Net", "canonical field\nchannels"),
                                   ("F-PATCH F-tier", "graph→field→\nfixed FFT physics"),
                                   ("DTFE/CIC classical", "no ML —\nmandatory floor")]):
        ax.add_patch(FancyBboxPatch((12.2, 7.9 - k * 0.95), 4.2, 0.75, boxstyle="round,pad=0.01",
                                    fc="#deebf7" if k < 3 else "#fee0d2", ec="k", lw=0.7))
        ax.text(13.2, 8.27 - k * 0.95, nm, fontsize=8.4, fontweight="bold", va="center")
        ax.text(15.6, 8.27 - k * 0.95, sub, fontsize=7.2, va="center", ha="center")
    ax.text(14.3, 4.55, "identical: folds • cores • linear-increment target •\ntrain-core scalers • √N shell objective • no HPO",
            ha="center", fontsize=7.8)

    # bottom flow
    steps = [("canonical products\nP1b/P2b/P3a ✓", "COMPLETE"),
             ("P4 folds\n+ support atlas", "ACTIVE"),
             ("P5/P6/P7 adapters\n+ PARITY GATES", "GATED"),
             ("P8 blocked-fold\ntraining screen", "GATED"),
             ("freeze finalists\n(Jul 21 bundle)", "GATED"),
             ("multi-phase train\nph002-005", "GATED"),
             ("ph001 BLIND\ndeterministic test", "GATED"),
             ("DESI canary\n→ VAC", "GATED")]
    for k, (lab, st) in enumerate(steps):
        x = 0.6 + k * 2.05
        _box(ax, x, 2.2, 1.85, 1.05, "", "", st)
        ax.text(x + 0.925, 2.72, lab, ha="center", va="center", fontsize=7.8)
        if k:
            _arrow(ax, x - 0.2, 2.72, x, 2.72)
    ax.text(8.5, 1.45, "PASS RULES:  P8 protocol pass = +0.03 fold-mean macro R²(λ1) over the frozen baseline with no shell erased  •  "
                       "production claim REQUIRES the ph001 blind pass  •\n"
                       "if nothing beats source-calibrated DTFE, classical ships as primary and learned outputs are badged experimental",
            ha="center", fontsize=8.2)
    ax.text(8.5, 0.55, "Why patches: 4 transfer tests showed transductive random-split training produces interpolation, not physics\n"
                       "(0.80→0.42, 0.63→0.37, 0.87→0.35 vs DTFE 0.53). Patches make extrapolation the TRAINING task, not just the test.",
            ha="center", fontsize=8.2, style="italic")

    ax.set_title("The generalised-model protocol: patch training, blocked validation, blind phases",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_patch_protocol.png", dpi=130)
    plt.close(fig)
    print("fig3 done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    print(f"all figures -> {OUT}")


if __name__ == "__main__":
    main()
