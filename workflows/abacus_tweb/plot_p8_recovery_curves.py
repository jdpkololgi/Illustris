#!/usr/bin/env python3
"""Learning curves for the P8 exposure-aware recovery runs (reads live or finished runs).

Panels: windowed training loss | validation macro R2(lambda1) per epoch with frozen comparators |
per-shell evolution | first-three-shell diagnostic vs the classical adoption bar.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1")
OUT = Path("/pscratch/sd/d/dkololgi/abacus/figures/p8_smoke_eval")
MODELS = {"graph": ("G-PATCH", "#2166ac"), "unet": ("U-PATCH", "#7b3294")}
SHELLS = ["0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
SHELL_C = ["#c6dbef", "#6baed6", "#2171b5", "#d62728"]
# frozen comparators (P0 evidence freeze / P8 screen summary)
R0_MACRO, CIC_MACRO, CIC_FIRST3 = 0.440, 0.185, 0.520


def load(m, rot=0):
    d = R / m / f"rotation_{rot}/seed_42"
    hist = [json.loads(l) for l in open(d / "epoch_history.jsonl")]
    trace = [json.loads(l) for l in open(d / "loss_trace.jsonl")]
    done = (d / "RECOVERY_COMPLETE").exists()
    return hist, trace, done


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
    status = []
    for m, (nm, col) in MODELS.items():
        hist, trace, done = load(m)
        status.append(f"{nm}: {len(hist)} epochs{' (COMPLETE)' if done else ' (RUNNING)'}")
        eps = [h["epoch"] for h in hist]
        macro = [h["primary_macro_r2_lambda1"] for h in hist]
        axes[0].plot([t["global_step"] for t in trace],
                     [t["training_weighted_mse_window"] for t in trace], color=col, lw=0.8, label=nm)
        axes[1].plot(eps, macro, "o-", color=col, label=nm, ms=5)
        best = max(range(len(macro)), key=lambda i: macro[i])
        axes[1].scatter([eps[best]], [macro[best]], s=150, facecolors="none", edgecolors=col, lw=2, zorder=5)
        ls = "-" if m == "graph" else "--"
        for si, s in enumerate(SHELLS):
            axes[2].plot(eps, [h["per_shell_lambda1_r2"][s] for h in hist], ls, color=SHELL_C[si],
                         marker="o" if m == "graph" else "s", ms=3.5,
                         label=s.replace("p", ".").replace("_", "–") if m == "graph" else None)
        f3 = [sum(h["per_shell_lambda1_r2"][s] for s in SHELLS[:3]) / 3 for h in hist]
        axes[3].plot(eps, f3, "o-", color=col, label=nm, ms=5)

    axes[1].axhline(R0_MACRO, color="k", ls=":", lw=1.2)
    axes[1].text(0.5, R0_MACRO + 0.006, "frozen R0 spatial-holdout macro 0.440", fontsize=7.5)
    axes[1].axhline(CIC_MACRO, color="#d95f02", ls=":", lw=1.2)
    axes[1].text(0.5, CIC_MACRO + 0.006, "CIC macro 0.185 (collapses in sparse shell)", fontsize=7.5, color="#d95f02")
    axes[3].axhline(CIC_FIRST3, color="#d95f02", ls="--", lw=1.5)
    axes[3].text(0.5, CIC_FIRST3 + 0.006, "CIC first-three 0.520  ← the classical bar", fontsize=8, color="#d95f02")

    axes[0].set_xlabel("patch update"); axes[0].set_ylabel("windowed weighted MSE"); axes[0].set_yscale("log")
    axes[0].set_title("Training loss (25-patch window)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch (= all 10,351 cores once)"); axes[1].set_ylabel("validation macro R²(λ1)")
    axes[1].set_title("Primary metric — circles mark best epoch"); axes[1].legend(fontsize=8, loc="lower right")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("per-shell R²(λ1)")
    axes[2].set_title("Per-shell (solid=G, dashed=U)"); axes[2].legend(fontsize=7.5, title="shell", loc="lower right")
    axes[3].set_xlabel("epoch"); axes[3].set_ylabel("mean R²(λ1), shells 1–3")
    axes[3].set_title("Tracer-supported shells vs classical bar"); axes[3].legend(fontsize=8, loc="lower right")
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle("P8 exposure-aware recovery, rotation 0 — " + " | ".join(status), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig9_recovery_curves.png", dpi=130)
    print("saved fig9 |", " | ".join(status))


if __name__ == "__main__":
    main()
