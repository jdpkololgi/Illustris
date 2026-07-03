#!/usr/bin/env python3
"""Gate G6 — FMPE vs MAF posterior head on FROZEN GNN embeddings (roadmap v2).

Head-only comparison: the trained SI GraphNet encoder is frozen; its 80-d node
embeddings condition (a) the existing MAF (baseline numbers from the self-eval
npz) and (b) a new sbi-package FMPE (flow matching) trained on the same train
split and scaled targets. Same conditioning => any difference is the density
estimator. GO if FMPE matches/beats MAF on R^2 AND calibration (SBC/coverage).

Runs fully on CPU (JAX_PLATFORMS=cpu for the frozen forward pass; torch CPU for
FMPE). No production edits; embeddings cached to npz for reuse (G4 etc.).
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.stats import kstest, spearmanr
from sklearn.metrics import r2_score


def get_embeddings(model_path, cache_path, out_npz):
    if Path(out_npz).exists():
        d = np.load(out_npz)
        return d["emb"], None
    import jax
    from plot_flowjax_posteriors import load_flowjax_model, load_data, create_gnn_and_flow
    gnn_params, config, target_scaler, flow_filename, increment_mode = load_flowjax_model(str(model_path))
    graph, targets, train_mask, val_mask, test_mask, eig_raw = load_data(
        data_path=str(cache_path), increment_mode=increment_mode)
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
    emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
    np.savez_compressed(out_npz, emb=emb)
    print(f"embeddings extracted: {emb.shape} -> {out_npz}")
    return emb, None


def to_raw_lambda(samples_scaled, target_scaler):
    """scaled linear increments -> raw (lambda1, lambda2, lambda3) via cumsum."""
    shp = samples_scaled.shape
    inc = target_scaler.inverse_transform(samples_scaled.reshape(-1, 3)).reshape(shp)
    return np.cumsum(inc, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--self-eval-npz", type=Path, required=True)
    ap.add_argument("--emb-npz", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    emb, _ = get_embeddings(args.model_path, args.cache, args.emb_npz)
    cache = pickle.load(open(args.cache, "rb"))
    theta = np.asarray(cache["regression_targets"], np.float64)      # scaled targets
    eig_raw = np.asarray(cache["eigenvalues_raw"], np.float64)
    scaler = cache["target_scaler"]
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])

    import torch
    from sbi.inference import FMPE
    from sbi.utils import BoxUniform
    torch.manual_seed(args.seed)
    lo = theta.min(0) - 0.1 * np.ptp(theta, 0)
    hi = theta.max(0) + 0.1 * np.ptp(theta, 0)
    prior = BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                       high=torch.tensor(hi, dtype=torch.float32))
    tr = train | val   # MAF trained with train(+val monitoring); give FMPE the same pool
    trainer = FMPE(prior=prior)
    trainer.append_simulations(torch.tensor(theta[tr], dtype=torch.float32),
                               torch.tensor(emb[tr], dtype=torch.float32))
    est = trainer.train(training_batch_size=512)
    posterior = trainer.build_posterior(est)

    # eval subset of the test split
    ti = np.where(test)[0]
    sub = rng.permutation(len(ti))[: args.n_eval]
    idx = ti[sub]
    X = torch.tensor(emb[idx], dtype=torch.float32)
    try:
        S = posterior.sample_batched((args.n_samples,), x=X, show_progress_bars=False)
        S = np.asarray(S.detach().cpu(), np.float64)                  # [n_samples, n_eval, 3]
        S = np.transpose(S, (1, 0, 2))                                # [n_eval, n_samples, 3]
    except Exception as e:
        print(f"(sample_batched unavailable: {e}); falling back to loop")
        S = np.stack([np.asarray(posterior.sample((args.n_samples,), x=X[i:i+1],
                                                  show_progress_bars=False).detach().cpu())
                      for i in range(len(idx))], axis=0)

    lam = to_raw_lambda(S, scaler)                                    # [n_eval, S, 3]
    mean_f = lam.mean(axis=1)
    truth = eig_raw[idx]

    # MAF baseline on the SAME rows (self-eval npz)
    d = np.load(args.self_eval_npz)
    order = {int(n): i for i, n in enumerate(d["node_index"])}
    rows = np.array([order[int(i)] for i in idx])
    maf_mean = d["lambda_mean"][rows].astype(np.float64)

    print(f"\n{'':10s}  {'MAF R2':>8s}  {'FMPE R2':>8s}   (n_eval={len(idx)}, frozen embeddings)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        print(f"{nm:10s}  {r2_score(truth[:,k], maf_mean[:,k]):8.3f}  {r2_score(truth[:,k], mean_f[:,k]):8.3f}")
    clu = truth[:, 0] > 0.2
    if clu.sum() > 30:
        print(f"cluster-slice lambda1 Spearman: MAF {spearmanr(truth[clu,0], maf_mean[clu,0]).statistic:+.2f}"
              f"  FMPE {spearmanr(truth[clu,0], mean_f[clu,0]).statistic:+.2f}  (n={clu.sum()})")

    # calibration: SBC ranks (in scaled-target space) + central coverage on raw lambda1
    ranks = (S < theta[idx][:, None, :]).mean(axis=1)                 # [n_eval,3]
    print("\ncalibration (FMPE): KS-uniform p per dim:",
          [f"{kstest(ranks[:,k], 'uniform').pvalue:.3f}" for k in range(3)])
    for q in (0.68, 0.90):
        lo_q = np.quantile(lam[:, :, 0], (1-q)/2, axis=1)
        hi_q = np.quantile(lam[:, :, 0], 1-(1-q)/2, axis=1)
        cov = np.mean((truth[:, 0] >= lo_q) & (truth[:, 0] <= hi_q))
        print(f"  lambda1 central {int(q*100)}% coverage: {cov:.3f}")

    print("\nGATE G6: GO if FMPE R2 >= MAF R2 - 0.01 AND calibration comparable/better.")


if __name__ == "__main__":
    main()
