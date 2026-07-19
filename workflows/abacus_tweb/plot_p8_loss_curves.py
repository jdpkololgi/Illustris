#!/usr/bin/env python3
"""Plot P8 training/validation curves from loss_trace.jsonl + screen_summary history.

Works for any run that has them. For the frozen 2026-07-19 short screens the history has a
SINGLE point and no loss_trace, so this script reports that explicitly rather than drawing a
misleading line through one sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P8 = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
OUT = Path("/pscratch/sd/d/dkololgi/abacus/figures/p8_smoke_eval")
RUNS = [("G-PATCH", "g_patch", "#2166ac"), ("U-PATCH", "u_patch", "#7b3294")]


def load(model_dir: str, rot: int):
    d = P8 / f"{model_dir}/rotation_{rot}/seed_42"
    hist = json.load(open(d / "screen_summary.json"))["history"]
    trace = []
    f = d / "loss_trace.jsonl"
    if f.exists():
        trace = [json.loads(l) for l in open(f) if l.strip()]
    return hist, trace


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have_curve = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for nm, md, col in RUNS:
        for rot, ls in ((0, "-"), (2, "--")):
            hist, trace = load(md, rot)
            if trace:
                have_curve = True
                axes[0].plot([t["step"] for t in trace],
                             [t["training_loss_window_mean"] for t in trace],
                             ls, color=col, label=f"{nm} rot{rot}")
            else:
                axes[0].scatter([h["step"] for h in hist],
                                [h["training_loss"] for h in hist],
                                color=col, marker="o" if rot == 0 else "s", s=70,
                                label=f"{nm} rot{rot} (single sample)")
            axes[1].plot([h["step"] for h in hist],
                         [h["primary_macro_r2_lambda1"] for h in hist],
                         ls + "o", color=col, label=f"{nm} rot{rot}")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("training loss (weighted MSE, scaled)")
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("validation macro R²(λ1)")
    axes[0].set_title("Training loss"); axes[1].set_title("Validation primary metric")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if not have_curve:
        msg = ("NO CURVE EXISTS for the frozen short screens: training loss was recorded only at the\n"
               "single evaluation step (eval_every == steps == 2000) and as an INSTANTANEOUS one-patch\n"
               "value, so the rot0/rot2 offset is patch-draw noise, not an optimization difference.\n"
               "Trainers are now instrumented (--loss-log-every, loss_trace.jsonl); recovery reruns\n"
               "will produce real curves.")
        fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10.5,
                 bbox=dict(fc="#fff3cd", ec="0.5"))
    fig.suptitle("P8 optimization curves", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig7_loss_curves.png", dpi=130)
    print("wrote", OUT / "fig7_loss_curves.png", "| curves present:", have_curve)


if __name__ == "__main__":
    main()
