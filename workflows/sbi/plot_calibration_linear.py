#!/usr/bin/env python3
"""SBC + TARP calibration for the production linear-increment NPE (slide 7).

Two themed figures for the calibration headline:
  * tarp_linear.png — TARP coverage with a **bootstrap 1σ error band** (the error
    profile): expected-coverage-probability vs credibility level; on-diagonal =
    calibrated, above = over-confident.
  * sbc_linear.png  — simulation-based-calibration rank histograms for λ₁,λ₂,λ₃ with
    the expected-uniform line and a ±2σ binomial scatter band; bars inside the band =
    calibrated (flat), ∪-shape = over-confident, ∩-shape = under-confident.

Reuses the eval machinery in plot_flowjax_posteriors.py (model load, cache load, GNN
encode, posterior sampling). GPU job. Defaults to the production SI linear model.
Writes both PNGs into the DESI SI run dir by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import jax  # noqa: E402

_ILL = Path(os.environ.get("ILLUSTRIS_ROOT", "/global/homes/d/dkololgi/TNG/Illustris")).resolve()
for _p in (str(_ILL), str(_ILL / "workflows" / "sbi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shared.plot_style import apply_style, finalize_axes, ACCENT_COLORS  # noqa: E402
from plot_flowjax_posteriors import (  # noqa: E402
    load_flowjax_model, load_data, create_gnn_and_flow, batched_sample_posterior,
)
import tarp  # noqa: E402

DEF_MODEL = ("/pscratch/sd/d/dkololgi/abacus/sbi_runs/path1_wedge_flowjax_3d_Bcorrected_linear_si/"
             "flowjax_sbi_model_seed_42_20260620_160956.pkl")
DEF_CACHE = ("/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si/"
             "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl")
DEF_OUT = "/pscratch/sd/d/dkololgi/graphweb_desi/flowjax_inference_outputs/desi_wedge_flowjax_linear_si"
LNAME = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]


def get_test_set(model_path, cache_path, seed):
    gnn_params, config, _scaler, flow_filename, increment_mode = load_flowjax_model(model_path)
    graph, targets, _tr, _va, test_mask, _eig = load_data(data_path=cache_path,
                                                          increment_mode=increment_mode)
    key = jax.random.key(seed)
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, key)
    emb_key, samp_key = jax.random.split(key)
    emb = np.asarray(gnn.apply(gnn_params, emb_key, graph, is_training=False))[test_mask]
    tgt = np.asarray(targets[test_mask])
    return emb, tgt, flow, samp_key, increment_mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--cache", default=DEF_CACHE)
    ap.add_argument("--outdir", type=Path, default=Path(DEF_OUT))
    ap.add_argument("--num-test", type=int, default=3000, help="TARP/SBC test points (more => tighter band)")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-bootstrap", type=int, default=200)
    ap.add_argument("--sbc-bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    apply_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    emb, tgt, flow, key, mode = get_test_set(args.model, args.cache, args.seed)
    print(f"[cal] model increment_mode={mode}; test set {len(emb)}", flush=True)
    n = min(args.num_test, len(emb))
    idx = np.random.default_rng(args.seed).choice(len(emb), n, replace=False)
    samples = np.asarray(batched_sample_posterior(flow, emb[idx], args.num_samples, key))  # [n, S, 3]
    true = tgt[idx]                                                                          # [n, 3]

    # ---------------- TARP with bootstrap error band ----------------
    samples_tarp = np.transpose(samples, (1, 0, 2))  # [S, n, 3]
    ecp, alpha = tarp.get_tarp_coverage(samples_tarp, true, norm=True,
                                        bootstrap=True, num_bootstrap=args.num_bootstrap)
    ecp = np.atleast_2d(ecp)
    mean, std = ecp.mean(0), ecp.std(0)
    dev = float(np.max(np.abs(mean - alpha)))
    col = ACCENT_COLORS["magenta"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=2, color="#9a9a93", label="ideal (calibrated)")
    ax.fill_between(alpha, mean - std, mean + std, color=col, alpha=0.30,
                    label="bootstrap 1$\\sigma$")
    ax.plot(alpha, mean, color=col, lw=2.4, label="linear-increment NPE")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.15)
    ax.text(0.035, 0.965, f"max |ECP$-\\alpha$| = {dev:.3f}\n(N={n} test points)",
            transform=ax.transAxes, va="top", fontsize=11, color="#C9C9C9")
    finalize_axes(ax, "TARP coverage — production NPE",
                  r"credibility level $\alpha$", "expected coverage probability",
                  legend=True, legend_loc="lower right")
    p_tarp = args.outdir / "tarp_linear.png"
    fig.savefig(p_tarp, bbox_inches="tight"); plt.close(fig)
    print(f"[cal] wrote {p_tarp}  (max dev {dev:.3f})", flush=True)

    # ---------------- SBC rank histograms with expected band ----------------
    ranks = (samples < true[:, None, :]).mean(axis=1)   # [n, 3] in [0,1]
    nb = args.sbc_bins
    p = 1.0 / nb
    mu = n * p
    sig = np.sqrt(n * p * (1 - p))                       # binomial scatter of per-bin count
    edges = np.linspace(0, 1, nb + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for i in range(3):
        ax = axes[i]
        ax.axhspan(mu - 2 * sig, mu + 2 * sig, color="#9a9a93", alpha=0.22, label=r"expected $\pm2\sigma$")
        ax.axhline(mu, color="#9a9a93", ls="--", lw=1.5)
        ax.hist(ranks[:, i], bins=edges, color=ACCENT_COLORS["blue"], alpha=0.85, edgecolor="#000000")
        finalize_axes(ax, LNAME[i], f"posterior rank of true {LNAME[i]}",
                      "count" if i == 0 else "", legend=(i == 0))
        ax.set_xlim(0, 1)
    fig.suptitle("Simulation-based calibration (SBC) — production linear-increment NPE", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_sbc = args.outdir / "sbc_linear.png"
    fig.savefig(p_sbc, bbox_inches="tight"); plt.close(fig)
    print(f"[cal] wrote {p_sbc}", flush=True)


if __name__ == "__main__":
    main()
