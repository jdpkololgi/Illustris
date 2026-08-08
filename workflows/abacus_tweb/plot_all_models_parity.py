#!/usr/bin/env python3
"""True-vs-predicted parity for EVERY completed P8 model on the identical validation rows.

Models: G-PATCH base/extension, U-PATCH base/extension, U-CIC-residual v2, MT4 multitracer, CIC.
All score the same 999,683 rotation-0 Bright authoritative cores (verified by parent_node_id).

  fig18_all_models_parity_lambda1.png  lambda1 parity, one panel per model, ranked by primary score
  fig19_all_models_lambda23.png        lambda2 and lambda3 parity for the same models
  fig20_all_models_summary.png         per-shell R2, amplitude ratio, and the score ladder
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("p8ev", REPO / "workflows/abacus_tweb/plot_p8_smoke_eval.py")
p8ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(p8ev)
A = Path("/pscratch/sd/d/dkololgi/abacus")
OUT = p8ev.OUT
SH = ["0.15–0.25", "0.25–0.35", "0.35–0.45", "0.45–0.55"]
SHK = ["0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]

# (label, path under $A, two-fold mean primary score, colour)  -- MT4 is rot-0 only
MODELS = [
    ("G-PATCH base",      "p8_recovery_v1/recovery_v1/graph",                                  0.4695, "#9ecae1"),
    ("G-PATCH extension", "p8_recovery_v1/convergence_extension_v1/graph",                     0.4867, "#3182bd"),
    ("U-PATCH base",      "p8_recovery_v1/recovery_v1/unet",                                   0.5035, "#bcbddc"),
    ("U-PATCH extension", "p8_recovery_v1/convergence_extension_v1/unet",                      0.5134, "#756bb1"),
    ("U-CIC-residual v2", "p8_recovery_v1/u_cic_resid_v2/unet_cic_residual",                   0.5214, "#1a9850"),
    ("MT4 multitracer*",  "p8_multitracer_v1/models/recovery/mt4_proxy_v1/unet_multitracer",   0.6074, "#d94801"),
]
NOTE = ("*MT4 = Bright+Faint context; its gain is NOT yet attributable to Faint structural "
        "information — the scrambled-Faint neural null (U-BF-NULL-v1) has not been run.")


def load(rot=0):
    ids, truth, short, meta = p8ev.load_rotation(rot)
    P = {}
    for lab, sub, score, col in MODELS:
        d = A / sub / f"rotation_{rot}/seed_42"
        rid = np.load(d / "best_validation_parent_node_id.npy")
        rp = np.load(d / "best_validation_eigenvalues.npy").astype(np.float64)
        o = np.argsort(rid); pos = np.searchsorted(rid[o], ids)
        assert (rid[o][pos] == ids).all(), lab
        P[lab] = rp[o[pos]]
    P["CIC (train-affine)"] = short["CIC (train-affine)"]
    return ids, truth, P, meta


def panel(ax, y, p, lo, hi, title, col):
    ax.hexbin(y, p, gridsize=90, bins="log", cmap="viridis", extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
    ax.axhline(0.2, color="gray", ls=":", lw=0.5); ax.axvline(0.2, color="gray", ls=":", lw=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(f"{title}\nR²={r2_score(y,p):.3f}  slope={np.polyfit(y,p,1)[0]:.2f}  "
                 f"σp/σt={p.std()/y.std():.2f}", fontsize=9.5, color=col, fontweight="bold")


def main():
    ids, truth, P, meta = load(0)
    labs = [m[0] for m in MODELS] + ["CIC (train-affine)"]
    cols = {m[0]: m[3] for m in MODELS}; cols["CIC (train-affine)"] = "#d95f02"

    # fig18 — lambda1
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    lo, hi = -0.62, 1.05
    for ax, lab in zip(axes.ravel(), labs):
        panel(ax, truth[:, 0], P[lab][:, 0], lo, hi, lab, cols[lab])
        ax.set_xlabel("true λ1"); ax.set_ylabel("predicted λ1")
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.02, 0.5, NOTE, fontsize=9, wrap=True, va="center",
                          bbox=dict(fc="#fff3cd", ec="0.6"))
    fig.suptitle("λ1 true vs predicted — every completed P8 model, identical rotation-0 validation fold "
                 "(999,683 authoritative cores)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "fig18_all_models_parity_lambda1.png", dpi=120); plt.close(fig)
    print("fig18 done")

    # fig19 — lambda2 / lambda3
    fig, axes = plt.subplots(2, 7, figsize=(26, 8))
    for r, k in enumerate((1, 2)):
        lo2, hi2 = (-0.3, 1.5) if k == 1 else (-0.15, 2.0)
        for c, lab in enumerate(labs):
            panel(axes[r, c], truth[:, k], P[lab][:, k], lo2, hi2, f"{lab}\nλ{k+1}", cols[lab])
            if r == 1: axes[r, c].set_xlabel(f"true λ{k+1}")
            if c == 0: axes[r, c].set_ylabel(f"predicted λ{k+1}")
    fig.suptitle("λ2 (top) and λ3 (bottom) — all models, same validation rows.  " + NOTE, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig19_all_models_lambda23.png", dpi=115); plt.close(fig)
    print("fig19 done")

    # fig20 — summary
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    x = np.arange(4)
    for lab in labs:
        per = [r2_score(truth[meta["shell"] == s, 0], P[lab][meta["shell"] == s, 0]) for s in range(4)]
        axes[0].plot(x, per, "o-", color=cols[lab], label=lab, ms=5,
                     lw=2.5 if "MT4" in lab else 1.5)
        rat = [P[lab][meta["shell"] == s, 0].std() / truth[meta["shell"] == s, 0].std() for s in range(4)]
        axes[1].plot(x, rat, "o-", color=cols[lab], ms=5, lw=2.5 if "MT4" in lab else 1.5)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xticks(x, SH); axes[0].set_ylim(-0.9, 0.8)
    axes[0].set_ylabel("R²(λ1)"); axes[0].set_title("Per-shell λ1 R² (rotation-0 validation)")
    axes[0].legend(fontsize=7.5, loc="lower left"); axes[0].grid(alpha=0.3)
    axes[1].axhline(1.0, color="k", ls="--", lw=1); axes[1].set_xticks(x, SH)
    axes[1].set_ylabel("σ_pred / σ_true"); axes[1].set_title("Amplitude ratio (1 = no shrinkage; >1 = noise amplification)")
    axes[1].grid(alpha=0.3)
    ax = axes[2]
    sc = [m[2] for m in MODELS] + [0.185]
    bars = ax.barh(range(len(labs)), sc, color=[cols[l] for l in labs])
    ax.set_yticks(range(len(labs)), labs, fontsize=9); ax.invert_yaxis()
    ax.axvline(0.470, color="g", ls=":", lw=1.5); ax.text(0.474, 6.3, "promotion gate", fontsize=7.5, color="g", rotation=90)
    ax.axvline(0.440, color="k", ls=":", lw=1.2); ax.text(0.415, 6.3, "frozen R0", fontsize=7.5, rotation=90)
    for i, v in enumerate(sc): ax.text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 0.72); ax.set_xlabel("primary score (two-fold mean macro R²(λ1); MT4 = rot-0 only)")
    ax.set_title("Score ladder")
    fig.suptitle("All completed P8 models — per-shell behaviour, amplitude calibration, and score ladder",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig20_all_models_summary.png", dpi=125); plt.close(fig)
    print("fig20 done")


if __name__ == "__main__":
    main()
