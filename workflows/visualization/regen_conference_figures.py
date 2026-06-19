"""
Regenerate the two Cambridge-talk headline figures from a trained wedge-NPE run,
in the GraphWeb plot-style-guide theme (true-black, cosmic-web class colors):

  1. TARP coverage         -> tarp_coverage.png
  2. T-Web class fractions -> class_fractions.png  (predicted posterior vs CACTUS truth)

Reuses the (tested) model/data loaders and batched sampler from
plot_flowjax_posteriors. Auto-detects increment_mode from the saved model.

Usage:
    python workflows/visualization/regen_conference_figures.py \
        --model_path .../flowjax_sbi_model_seed_42_*.pkl --output_dir .../conference
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
for _p in (
    "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages",
    "/global/u2/d/dkololgi/.local/lib/python3.10/site-packages",
):
    while _p in sys.path:
        sys.path.remove(_p)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax

from shared.plot_style import apply_style, COSMIC_WEB_COLORS, CLASS_ORDER, ACCENT_COLORS
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues, posterior_to_classprobs
from workflows.sbi.plot_flowjax_posteriors import (
    load_flowjax_model, load_data, create_gnn_and_flow, batched_sample_posterior,
)

try:
    import tarp
    TARP_AVAILABLE = True
except ImportError:
    TARP_AVAILABLE = False


def main(args):
    apply_style()
    os.makedirs(args.output_dir, exist_ok=True)

    gnn_params, config, target_scaler, flow_filename, increment_mode = load_flowjax_model(args.model_path)
    graph, targets, train_mask, val_mask, test_mask, eigenvalues_raw = load_data(increment_mode=increment_mode)
    print(f"[conference figures] increment_mode={increment_mode}")

    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
    all_emb = np.array(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
    test_emb = all_emb[np.asarray(test_mask)]
    test_targets_scaled = np.asarray(targets)[np.asarray(test_mask)]
    test_eig_raw = np.asarray(eigenvalues_raw)[np.asarray(test_mask)]
    n_test = test_emb.shape[0]
    print(f"[conference figures] test nodes: {n_test}")

    key = jax.random.key(123)

    # ---- 1. TARP coverage -------------------------------------------------
    if TARP_AVAILABLE:
        n_tarp = min(args.num_tarp, n_test)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_test, n_tarp, replace=False)
        key, k = jax.random.split(key)
        samp = batched_sample_posterior(flow, test_emb[idx], args.num_samples, k)  # [n,K,3] scaled
        ecp, alpha = tarp.get_tarp_coverage(
            np.transpose(samp, (1, 0, 2)), test_targets_scaled[idx], norm=True, bootstrap=False
        )
        np.savez(os.path.join(args.output_dir, "tarp_coverage.npz"), ecp=ecp, alpha=alpha)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], ls="--", lw=2, color="#F2F2F2", alpha=0.6, label="Ideal")
        ax.plot(alpha, ecp, lw=2.5, color=ACCENT_COLORS["magenta"], label="NPE coverage")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(r"Credibility level $\alpha$")
        ax.set_ylabel("Expected coverage probability")
        ax.set_title(f"TARP coverage — linear-increment NPE ({n_tarp} galaxies)")
        ax.legend(loc="lower right")
        p = os.path.join(args.output_dir, "tarp_coverage.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {p}")
    else:
        print("TARP unavailable; skipping coverage figure.")

    # ---- 2. T-Web class fractions: predicted posterior vs CACTUS truth ----
    n_cls = min(args.num_classprob, n_test)
    rng = np.random.default_rng(7)
    cidx = rng.choice(n_test, n_cls, replace=False)
    key, k2 = jax.random.split(key)
    samp_all = batched_sample_posterior(flow, test_emb[cidx], args.num_samples, k2)
    samp_raw = samples_to_raw_eigenvalues(samp_all, target_scaler, increment_mode)
    cp = posterior_to_classprobs(samp_raw, lambda_th=args.lambda_th)
    pred = {c: float(np.mean(cp[c])) for c in CLASS_ORDER}
    tcp = posterior_to_classprobs(test_eig_raw[cidx][:, None, :], lambda_th=args.lambda_th)
    true = {c: float(np.mean(tcp[c])) for c in CLASS_ORDER}
    print(f"[conference figures] class-prob consistency: {cp['consistency_max_abs_diff']:.2e}")
    print("  predicted:", {c: round(pred[c], 3) for c in CLASS_ORDER})

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(CLASS_ORDER)); w = 0.38
    colors = [COSMIC_WEB_COLORS[c] for c in CLASS_ORDER]
    ax.bar(x - w/2, [true[c] for c in CLASS_ORDER], w, color=colors, alpha=0.55,
           edgecolor="#F2F2F2", linewidth=0.8, label="CACTUS truth")
    ax.bar(x + w/2, [pred[c] for c in CLASS_ORDER], w, color=colors, alpha=1.0,
           edgecolor="#F2F2F2", linewidth=0.8, label="NPE posterior")
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER])
    ax.set_ylabel("Class fraction")
    ax.set_title(rf"T-Web class fractions from posteriors ($\lambda_{{th}}$={args.lambda_th})")
    ax.legend()
    p = os.path.join(args.output_dir, "class_fractions.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")
    print("  truth    :", {c: round(true[c], 3) for c in CLASS_ORDER})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate Cambridge-talk NPE figures (themed)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_tarp", type=int, default=500)
    ap.add_argument("--num_classprob", type=int, default=3000)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--lambda_th", type=float, default=0.2)
    main(ap.parse_args())
