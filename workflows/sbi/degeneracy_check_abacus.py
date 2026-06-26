#!/usr/bin/env python3
"""Quick posterior-degeneracy diagnostic on the Abacus SI self-eval (in-distribution
reference for the DESI degeneracy check). Reuses the loaders/samplers from
plot_flowjax_posteriors.py: load model -> recreate GNN+flow -> embed test split ->
batched-sample posteriors -> run the same 4 checks as the DESI side.
"""
from __future__ import annotations
import argparse
import numpy as np
import jax

from plot_flowjax_posteriors import (
    load_flowjax_model, load_data, create_gnn_and_flow, batched_sample_posterior,
)
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n-nodes", type=int, default=20000, help="test nodes to sample")
    ap.add_argument("--num-samples", type=int, default=128)
    args = ap.parse_args()

    gnn_params, config, target_scaler, flow_filename, increment_mode = load_flowjax_model(args.model_path)
    graph, targets, train_mask, val_mask, test_mask, eigenvalues_raw = load_data(
        data_path=args.data_path, increment_mode=increment_mode)
    print(f"increment_mode={increment_mode}  test={int(np.sum(test_mask))}")

    master_key = jax.random.key(42)
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, master_key)
    all_emb = gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False)
    test_emb = np.array(all_emb[test_mask])

    n = min(args.n_nodes, test_emb.shape[0])
    np.random.seed(42)
    idx = np.random.choice(test_emb.shape[0], n, replace=False)
    emb = test_emb[idx]
    S_scaled = batched_sample_posterior(flow, emb, args.num_samples, jax.random.key(123))  # [n,K,3]
    # raw eigenvalues, UNSORTED (mirror the DESI lambda_samples_subset convention)
    S = np.stack([samples_to_raw_eigenvalues(S_scaled[i], target_scaler, increment_mode)
                  for i in range(n)], axis=0).astype(np.float64)
    names = ["lambda1", "lambda2", "lambda3"]

    print("\n[1] collapse-to-marginal: between-galaxy SD(means) / within-galaxy SD")
    sub_mean = S.mean(axis=1); between = sub_mean.std(axis=0); within = S.std(axis=1).mean(0)
    for i, nm in enumerate(names):
        print(f"  {nm:8s} between={between[i]:.4f} within={within[i]:.4f} ratio={between[i]/within[i]:.3f}")

    print("\n[2] width collapse: CV of per-galaxy width; spike fraction")
    w = S.std(axis=1)
    for i, nm in enumerate(names):
        q = np.percentile(w[:, i], [1, 50, 99])
        print(f"  {nm:8s} q01/50/99={q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f} CV={w[:,i].std()/w[:,i].mean():.3f}")
    print(f"  frac width<1e-3: {np.mean(w.min(axis=1) < 1e-3):.4%}")

    print("\n[3] inter-eigenvalue collapse: mean within-posterior corr")
    Sc = S - S.mean(axis=1, keepdims=True)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        r = (Sc[:, :, a] * Sc[:, :, b]).mean(1) / (Sc[:, :, a].std(1) * Sc[:, :, b].std(1) + 1e-12)
        print(f"  corr({names[a]},{names[b]}) mean={np.nanmean(r):+.3f} median={np.nanmedian(r):+.3f}")

    print("\n[4] embedding effective rank (participation ratio)")
    C = np.cov(emb.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 0][::-1]
    pr = (ev.sum() ** 2) / (ev ** 2).sum()
    print(f"  dim={emb.shape[1]} eff_rank={pr:.1f} top5_frac={ev[:5].sum()/ev.sum():.3f} top1_frac={ev[0]/ev.sum():.3f}")


if __name__ == "__main__":
    main()
