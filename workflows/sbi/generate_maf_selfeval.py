#!/usr/bin/env python3
"""Generate the MAF self-eval npz (node_index + raw-lambda posterior means) that
gate_g6_fmpe_frozen_head.py consumes as the MAF baseline. CPU-only.

Loads a trained FlowJAX MAF model, extracts frozen GNN embeddings, samples the
MAF posterior (128/node, batched vmap), converts scaled linear increments -> raw
(lambda1,lambda2,lambda3) via the target scaler + cumsum (matching gate_g6's
to_raw_lambda), and saves per-node posterior means.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import jax
    from plot_flowjax_posteriors import (load_flowjax_model, load_data,
                                         create_gnn_and_flow, batched_sample_posterior)

    gnn_params, config, target_scaler, flow_filename, increment_mode = load_flowjax_model(
        str(args.model_path))
    print(f"increment_mode={increment_mode}, latent_size={config.get('latent_size')}")
    graph, targets, train_mask, val_mask, test_mask, eig_raw = load_data(
        data_path=str(args.cache), increment_mode=increment_mode)
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(args.seed))
    emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
    print(f"embeddings: {emb.shape}")

    S = batched_sample_posterior(flow, emb, args.num_samples, jax.random.key(args.seed + 1))
    S = np.asarray(S, np.float64)                                   # [N, num_samples, 3] scaled increments
    print(f"posterior samples: {S.shape}")

    # scaled linear increments -> raw (lambda1,lambda2,lambda3): inverse-scale then cumsum
    shp = S.shape
    inc = target_scaler.inverse_transform(S.reshape(-1, 3)).reshape(shp)
    if increment_mode == "linear":
        lam = np.cumsum(inc, axis=-1)
    else:
        # softplus increments: lambda1 anchor + cumulative softplus of increments
        lam = np.empty_like(inc)
        lam[..., 0] = inc[..., 0]
        lam[..., 1] = inc[..., 0] + np.logaddexp(0.0, inc[..., 1])
        lam[..., 2] = lam[..., 1] + np.logaddexp(0.0, inc[..., 2])

    lambda_mean = lam.mean(axis=1)                                  # [N, 3] raw
    node_index = np.arange(len(emb), dtype=np.int64)

    # sanity: posterior-mean R2 on the test split should match the model's report (~0.817 lambda1)
    from sklearn.metrics import r2_score
    te = np.asarray(test_mask).astype(bool)
    print("posterior-mean R2 (test):",
          [f"{r2_score(eig_raw[te, k], lambda_mean[te, k]):.4f}" for k in range(3)])

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, node_index=node_index, lambda_mean=lambda_mean)
    print(f"wrote {args.out_npz}  (node_index {node_index.shape}, lambda_mean {lambda_mean.shape})")


if __name__ == "__main__":
    main()
