#!/usr/bin/env python3
"""Three-way TARP overlay: softplus vs linear vs raw eigenvalue parameterisations.

Slide-6 methods point — "lower NLL != a better posterior, which is why we
calibration-test": raw eigenvalues win test NLL but are over-confident (TARP curve
bows ABOVE the diagonal), while linear increments hug the diagonal. Recomputes TARP
coverage for each of the three trained Abacus models on its own test split and
overlays them on one themed axes (with bootstrap 1-sigma bands).

Reuses the eval machinery from plot_flowjax_posteriors.py (model load, cache load,
GNN encode, posterior sampling). GPU job (samples the flows).

Writes three_way_tarp.png into the DESI SI run dir by default (deck consolidation).
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

_CACHE = "/pscratch/sd/d/dkololgi/abacus/sbi_caches"
_RUNS = "/pscratch/sd/d/dkololgi/abacus/sbi_runs"
# (label, model_pkl, cache_pkl, colour) — the 2026-06-19 parameterisation-study runs.
MODELS = [
    ("Softplus increments",
     f"{_RUNS}/path1_wedge_flowjax_3d_testA_reg/flowjax_sbi_model_seed_42_20260618_110810.pkl",
     f"{_CACHE}/path1_flowjax_3d/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl",
     ACCENT_COLORS["blue"]),
    ("Linear increments",
     f"{_RUNS}/path1_wedge_flowjax_3d_Bcorrected_linear/flowjax_sbi_model_seed_42_20260618_164955.pkl",
     f"{_CACHE}/path1_flowjax_3d_lineareig/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl",
     ACCENT_COLORS["magenta"]),
    ("Raw eigenvalues",
     f"{_RUNS}/path1_wedge_flowjax_3d_testB_raweig/flowjax_sbi_model_seed_42_20260618_110809.pkl",
     f"{_CACHE}/path1_flowjax_3d_raweig/processed_jraph_data_mc1e+09_v2_scaled_3_raw_eig.pkl",
     ACCENT_COLORS["red"]),
]


def tarp_curve(model_path, cache_path, num_test, num_samples, seed):
    """Return (alpha, ecp[num_bootstrap, n_bins]) TARP coverage for one model."""
    gnn_params, config, _scaler, flow_filename, increment_mode = load_flowjax_model(model_path)
    graph, targets, _tr, _va, test_mask, _eig = load_data(data_path=cache_path,
                                                          increment_mode=increment_mode)
    key = jax.random.key(seed)
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, key)
    emb_key, samp_key = jax.random.split(key)
    all_emb = gnn.apply(gnn_params, emb_key, graph, is_training=False)
    test_emb = np.asarray(all_emb[test_mask])
    test_tgt = np.asarray(targets[test_mask])

    n = min(num_test, len(test_emb))
    idx = np.random.default_rng(seed).choice(len(test_emb), n, replace=False)
    samples = batched_sample_posterior(flow, test_emb[idx], num_samples, samp_key)  # [n, S, 3]
    samples_tarp = np.transpose(np.asarray(samples), (1, 0, 2))                      # [S, n, 3]
    ecp, alpha = tarp.get_tarp_coverage(samples_tarp, test_tgt[idx], norm=True,
                                        bootstrap=True, num_bootstrap=100)
    return alpha, np.atleast_2d(ecp), increment_mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/graphweb_desi/flowjax_inference_outputs/"
        "desi_wedge_flowjax_linear_si/three_way_tarp.png"))
    ap.add_argument("--num-test", type=int, default=500)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], ls="--", lw=2, color="#9a9a93", label="Ideal (calibrated)")
    for label, mp, cp, color in MODELS:
        alpha, ecp, mode = tarp_curve(mp, cp, args.num_test, args.num_samples, args.seed)
        mean, std = ecp.mean(0), ecp.std(0)
        dev = float(np.max(np.abs(mean - alpha)))
        ax.fill_between(alpha, mean - std, mean + std, color=color, alpha=0.20)
        ax.plot(alpha, mean, color=color, lw=2.3, label=f"{label}  (max dev {dev:.02f})")
        print(f"[tarp] {label} ({mode}): max|ECP-alpha| = {dev:.3f}", flush=True)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.text(0.035, 0.965, "above diagonal = over-confident",
            transform=ax.transAxes, va="top", fontsize=10, color="#C9C9C9")
    finalize_axes(ax, "TARP coverage by target parameterisation",
                  r"credibility level $\alpha$", "expected coverage probability",
                  legend=True, legend_loc="lower right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)
    print(f"[tarp] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
