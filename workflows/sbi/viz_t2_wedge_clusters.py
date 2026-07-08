#!/usr/bin/env python
"""
viz_t2_wedge_clusters.py

Sanity-check 3-D visualization for the T2 CNN-on-counts (3-D U-Net) per-galaxy
eigenvalue predictions on the path1 fiberassign wedge.

Headline product: P(lambda1 > lambda_th). With ascending eigenvalues
lambda1 <= lambda2 <= lambda3, lambda1 > 0.2 selects CLUSTERS (collapse on all
three axes). We plot the wedge galaxies in 3-D, coloring lambda1>0.2 galaxies
red (clusters) and the rest as faint grey context points, side-by-side:
(left) U-Net PREDICTED clusters, (right) TRUTH clusters.

All inputs are read-only and row-aligned in the same 100,935-galaxy order.
Outputs go under /pscratch/.../T2/viz/. Does NOT edit any shared file.
"""
import os
import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = "/pscratch/sd/d/dkololgi/abacus"
WEDGE = (f"{BASE}/graph_constructions/wedges/path1_fiberassign/"
         "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3")
PRED_NPY = f"{BASE}/field_level_tests/T2/t2_scores.pred_eigs.npy"
XYZ_NPY = f"{WEDGE}_points_xyz.npy"
FITS_TARGETS = f"{WEDGE}_wedge_targets.fits"
OUTDIR = f"{BASE}/field_level_tests/T2/viz"
LAMBDA_TH = 0.2

os.makedirs(OUTDIR, exist_ok=True)


def main():
    # ---- Load ----
    pred = np.load(PRED_NPY)            # (N,3) ascending lambda1<=lambda2<=lambda3
    xyz = np.load(XYZ_NPY)             # (N,3) observer-frame comoving Mpc
    with fits.open(FITS_TARGETS) as h:
        d = h[1].data
        L1 = np.asarray(d["LAMBDA1"], float)
        L2 = np.asarray(d["LAMBDA2"], float)
        L3 = np.asarray(d["LAMBDA3"], float)
    truth3 = np.stack([L1, L2, L3], axis=1)
    N = pred.shape[0]
    assert xyz.shape[0] == N == L1.shape[0], "row-count mismatch"

    pred_l1 = pred[:, 0]
    truth_l1 = L1

    # ---- Alignment sanity ----
    rho = spearmanr(pred_l1, truth_l1).correlation
    print(f"[align] N={N}  Spearman(pred l1, truth l1) = {rho:.4f}")

    # ---- Cluster masks ----
    pred_cl = pred_l1 > LAMBDA_TH
    truth_cl = truth_l1 > LAMBDA_TH
    f_pred = pred_cl.mean()
    f_truth = truth_cl.mean()

    # Confusion (truth cluster as ground truth for predicted cluster class)
    tp = int(np.sum(pred_cl & truth_cl))
    fp = int(np.sum(pred_cl & ~truth_cl))
    fn = int(np.sum(~pred_cl & truth_cl))
    tn = int(np.sum(~pred_cl & ~truth_cl))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))

    med_pred_in_truthcl = float(np.median(pred_l1[truth_cl])) if truth_cl.any() else float("nan")
    med_truth_in_truthcl = float(np.median(truth_l1[truth_cl])) if truth_cl.any() else float("nan")

    print("\n===== QUANTITATIVE READOUT =====")
    print(f"lambda_th = {LAMBDA_TH}")
    print(f"Fraction lambda1>{LAMBDA_TH}: predicted = {f_pred:.4f} ({pred_cl.sum()}),"
          f"  truth = {f_truth:.4f} ({truth_cl.sum()})")
    print(f"Cluster class (truth=GT): precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")
    print(f"Confusion  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Median lambda1 within TRUE clusters: pred={med_pred_in_truthcl:.4f}  truth={med_truth_in_truthcl:.4f}")
    print("================================\n")

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # ---- Figure 1: two-panel red-cluster / grey-context, a couple of angles ----
    angles = [(18, -60), (12, 30), (30, -110)]
    png_paths = []
    for (elev, azim) in angles:
        fig = plt.figure(figsize=(18, 8.5))
        for i, (mask, title) in enumerate([(pred_cl, "U-Net PREDICTED clusters"),
                                           (truth_cl, "TRUTH clusters")]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.scatter(x[~mask], y[~mask], z[~mask], s=0.5, c="0.72",
                       alpha=0.12, marker=".", linewidths=0, rasterized=True)
            ax.scatter(x[mask], y[mask], z[mask], s=6, c="red",
                       alpha=0.85, marker="o", linewidths=0, rasterized=True)
            ax.set_title(f"{title}\n(lambda1>{LAMBDA_TH}: {mask.sum()} gal, {mask.mean()*100:.1f}%)",
                         fontsize=13)
            ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]"); ax.set_zlabel("Z [Mpc]")
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        fig.suptitle(f"T2 3-D U-Net wedge clusters  |  Spearman(pred,truth) lambda1={rho:.3f}  "
                     f"|  F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  |  view elev={elev} azim={azim}",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = f"{OUTDIR}/t2_wedge_clusters_pred_vs_truth_elev{elev}_azim{azim}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(p)
        print(f"[png] wrote {p}")

    # ---- Figure 2 (bonus): continuous lambda1 pred vs truth, viridis, th marked ----
    vmin, vmax = -0.4, 0.8
    fig = plt.figure(figsize=(18, 8.5))
    elev, azim = angles[0]
    for i, (val, title) in enumerate([(pred_l1, "PREDICTED lambda1"),
                                      (truth_l1, "TRUTH lambda1")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(x, y, z, s=1.5, c=np.clip(val, vmin, vmax),
                        cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.5,
                        marker=".", linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]"); ax.set_zlabel("Z [Mpc]")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cb.ax.axhline(LAMBDA_TH, color="red", lw=2)
        cb.set_label(f"lambda1  (red line = cluster th {LAMBDA_TH})")
    fig.suptitle("T2 3-D U-Net continuous lambda1: predicted vs truth", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p_cont = f"{OUTDIR}/t2_wedge_lambda1_continuous_pred_vs_truth.png"
    fig.savefig(p_cont, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] wrote {p_cont}")

    # ---- Interactive plotly HTML (best effort) ----
    html_path = None
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                            subplot_titles=("U-Net PREDICTED clusters", "TRUTH clusters"))
        for col, mask in [(1, pred_cl), (2, truth_cl)]:
            fig.add_trace(go.Scatter3d(
                x=x[~mask], y=y[~mask], z=z[~mask], mode="markers",
                marker=dict(size=1.0, color="lightgrey", opacity=0.15),
                name="context", showlegend=(col == 1)), row=1, col=col)
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask], mode="markers",
                marker=dict(size=2.0, color="red", opacity=0.85),
                name="cluster (l1>0.2)", showlegend=(col == 1)), row=1, col=col)
        fig.update_layout(title=f"T2 3-D U-Net wedge clusters (F1={f1:.3f})",
                          scene=dict(aspectmode="data"), scene2=dict(aspectmode="data"))
        html_path = f"{OUTDIR}/t2_wedge_clusters_pred_vs_truth.html"
        fig.write_html(html_path, include_plotlyjs="inline")
        print(f"[html] wrote {html_path}")
    except Exception as e:
        print(f"[html] skipped ({e})")

    # ================================================================
    # FULL T-WEB ENVIRONMENT CLASSIFICATION (4 classes)
    # Count eigenvalues > lambda_th (ascending l1<=l2<=l3):
    #   3 above -> CLUSTER, 2 -> FILAMENT, 1 -> WALL, 0 -> VOID
    # class code: 0=VOID 1=WALL 2=FILAMENT 3=CLUSTER (= n_above)
    # ================================================================
    def tweb_class(eig3):
        return np.sum(eig3 > LAMBDA_TH, axis=1).astype(int)  # 0..3

    pred_cls = tweb_class(pred)
    truth_cls = tweb_class(truth3)
    CLASS_NAMES = ["VOID", "WALL", "FILAMENT", "CLUSTER"]
    # cosmic-web palette: void faint blue-grey, wall green, filament orange, cluster red
    CLASS_COLORS = ["#8ea3b0", "#2ca02c", "#ff7f0e", "#d62728"]
    CLASS_SIZE = [0.6, 3.5, 5.0, 6.0]
    CLASS_ALPHA = [0.10, 0.55, 0.75, 0.90]

    # --- fractions ---
    print("\n===== T-WEB 4-CLASS READOUT =====")
    frac_pred = np.array([(pred_cls == k).mean() for k in range(4)])
    frac_truth = np.array([(truth_cls == k).mean() for k in range(4)])
    print(f"{'class':<10}{'pred_frac':>12}{'truth_frac':>12}{'pred_n':>10}{'truth_n':>10}")
    for k in range(4):
        print(f"{CLASS_NAMES[k]:<10}{frac_pred[k]:>12.4f}{frac_truth[k]:>12.4f}"
              f"{int((pred_cls==k).sum()):>10}{int((truth_cls==k).sum()):>10}")

    # --- 4x4 confusion (truth rows, pred cols) ---
    conf = np.zeros((4, 4), dtype=int)
    for t in range(4):
        for p_ in range(4):
            conf[t, p_] = int(np.sum((truth_cls == t) & (pred_cls == p_)))
    print("\n4x4 confusion (rows=truth, cols=pred): order VOID,WALL,FILAMENT,CLUSTER")
    print("          " + "".join(f"{c[:4]:>10}" for c in CLASS_NAMES))
    for t in range(4):
        print(f"{CLASS_NAMES[t]:<10}" + "".join(f"{conf[t,p_]:>10}" for p_ in range(4)))

    # --- per-class precision/recall/F1 ---
    print("\nper-class (truth=GT):")
    print(f"{'class':<10}{'precision':>11}{'recall':>9}{'F1':>8}{'support':>9}")
    prf = {}
    for k in range(4):
        tp_ = conf[k, k]
        fp_ = int(conf[:, k].sum() - tp_)
        fn_ = int(conf[k, :].sum() - tp_)
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) else float("nan")
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else float("nan")
        f1_ = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
        prf[CLASS_NAMES[k]] = (prec, rec, f1_)
        print(f"{CLASS_NAMES[k]:<10}{prec:>11.4f}{rec:>9.4f}{f1_:>8.4f}{int(conf[k,:].sum()):>9}")
    accuracy = np.trace(conf) / conf.sum()
    print(f"\nOverall 4-class accuracy = {accuracy:.4f}")
    print("=================================\n")

    # ---- Figure 3 (MAIN NEW): combined 4-class pred vs truth ----
    elev, azim = 18, -60
    fig = plt.figure(figsize=(18, 8.5))
    for i, (cls, title) in enumerate([(pred_cls, "U-Net PREDICTED T-web"),
                                      (truth_cls, "TRUTH T-web")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # draw void->wall->filament->cluster so dense classes sit on top
        for k in range(4):
            m = cls == k
            if not m.any():
                continue
            ax.scatter(x[m], y[m], z[m], s=CLASS_SIZE[k], c=CLASS_COLORS[k],
                       alpha=CLASS_ALPHA[k], marker="o" if k > 0 else ".",
                       linewidths=0, rasterized=True,
                       label=f"{CLASS_NAMES[k]} ({m.sum()}, {m.mean()*100:.1f}%)")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]"); ax.set_zlabel("Z [Mpc]")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        leg = ax.legend(loc="upper left", fontsize=9, markerscale=3, framealpha=0.9)
        for lh in leg.legend_handles:
            try:
                lh.set_alpha(1.0)
            except Exception:
                pass
    fig.suptitle(f"T2 3-D U-Net T-web environment (lambda_th={LAMBDA_TH})  |  "
                 f"4-class accuracy={accuracy:.3f}  |  view elev={elev} azim={azim}",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p_4class = f"{OUTDIR}/t2_wedge_4class_pred_vs_truth.png"
    fig.savefig(p_4class, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] wrote {p_4class}")

    # ---- Per-class highlight figures (target class colored, rest faint grey) ----
    per_class_paths = {}
    highlight = {"FILAMENT": ("t2_wedge_filament_pred_vs_truth.png", 2, "#ff7f0e"),
                 "WALL": ("t2_wedge_wall_pred_vs_truth.png", 1, "#2ca02c"),
                 "VOID": ("t2_wedge_void_pred_vs_truth.png", 0, "#1f77b4")}
    for cname, (fname, kcode, col) in highlight.items():
        fig = plt.figure(figsize=(18, 8.5))
        for i, (cls, title) in enumerate([(pred_cls, f"PREDICTED {cname}"),
                                          (truth_cls, f"TRUTH {cname}")]):
            m = cls == kcode
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.scatter(x[~m], y[~m], z[~m], s=0.5, c="0.72", alpha=0.10,
                       marker=".", linewidths=0, rasterized=True)
            ax.scatter(x[m], y[m], z[m], s=5, c=col, alpha=0.80,
                       marker="o", linewidths=0, rasterized=True)
            ax.set_title(f"{title}\n({m.sum()} gal, {m.mean()*100:.1f}%)", fontsize=13)
            ax.set_xlabel("X [Mpc]"); ax.set_ylabel("Y [Mpc]"); ax.set_zlabel("Z [Mpc]")
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        pr, rc, f1c = prf[cname]
        fig.suptitle(f"T2 3-D U-Net {cname} class: predicted vs truth  |  "
                     f"P={pr:.3f} R={rc:.3f} F1={f1c:.3f}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = f"{OUTDIR}/{fname}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        per_class_paths[cname] = p
        print(f"[png] wrote {p}")

    print("\nDONE. Outputs:")
    for p in png_paths:
        print(" ", p)
    print(" ", p_cont)
    print(" ", p_4class)
    for cname in highlight:
        print(" ", per_class_paths[cname])
    if html_path:
        print(" ", html_path)


if __name__ == "__main__":
    main()
