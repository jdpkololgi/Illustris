#!/usr/bin/env python3
"""Evaluate the tiled ñ-conditioned model on the HELD-OUT test region (RA>=150), per shell.

The spatial holdout means test nodes were NEVER in training (no leakage guard needed). For
each tile: GNN embeddings -> flow posterior samples at test-mask nodes -> raw eigenvalues ->
per-shell R^2(λ1/λ2/λ3), cluster-slice Spearman, 68/90% central coverage on λ1. This is the
real readout: does ñ-conditioning hold accuracy+coverage across z 0.05-0.55 (vs S1(b) zero-shot
collapse, shell-0 R^2=-1.09)?
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
_bad = ("/global/homes/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/homes/d/dkololgi/.local/lib/python3.11/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.11/site-packages")
for _p in _bad:
    while _p in sys.path:
        sys.path.remove(_p)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import numpy as np
import jax
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from plot_flowjax_posteriors import load_flowjax_model, create_gnn_and_flow, batched_sample_posterior
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues


def metrics(truth, mean, lam):
    r2 = [r2_score(truth[:, k], mean[:, k]) for k in range(3)]
    clu = truth[:, 0] > 0.2
    sp = spearmanr(truth[clu, 0], mean[clu, 0]).statistic if clu.sum() > 20 else np.nan
    covs = []
    for q in (0.68, 0.90):
        lo = np.quantile(lam[:, :, 0], (1 - q) / 2, axis=1)
        hi = np.quantile(lam[:, :, 0], 1 - (1 - q) / 2, axis=1)
        covs.append(float(np.mean((truth[:, 0] >= lo) & (truth[:, 0] <= hi))))
    return r2, sp, covs, int(clu.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    gnn_params, config, target_scaler, flow_filename, inc = load_flowjax_model(str(args.model_path))
    print(f"model {args.model_path.name}; increment_mode={inc}")
    manifest = json.loads((args.tiles_dir / "manifest.json").read_text())

    # accumulate test predictions per shell
    per_shell = {}   # shell -> dict(truth=[], mean=[], lam=[])
    for t in manifest["tiles"]:
        if t["test"] == 0:
            continue
        p = pickle.load(open(args.tiles_dir / t["file"], "rb"))
        graph = p["graph"]; eig_raw = np.asarray(p["eigenvalues_raw"])
        test_mask = np.asarray(p["masks"][2]).astype(bool)
        idx = np.where(test_mask)[0]
        gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
        emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
        S = batched_sample_posterior(flow, emb[idx], args.n_samples, jax.random.key(7))
        lam = np.stack([samples_to_raw_eigenvalues(S[i], target_scaler, inc) for i in range(len(idx))], 0)
        sh = t["shell"]
        d = per_shell.setdefault(sh, dict(truth=[], mean=[], lam=[]))
        d["truth"].append(eig_raw[idx]); d["mean"].append(lam.mean(1)); d["lam"].append(lam)

    print(f"\n{'shell':12s} {'n_test':>7s} {'R2_l1':>7s} {'R2_l2':>7s} {'R2_l3':>7s} "
          f"{'cluSp':>6s} {'cov68':>6s} {'cov90':>6s}")
    result = {}
    allt, allm, alll = [], [], []
    for sh in ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]:
        if sh not in per_shell:
            continue
        d = per_shell[sh]
        truth = np.concatenate(d["truth"]); mean = np.concatenate(d["mean"]); lam = np.concatenate(d["lam"])
        allt.append(truth); allm.append(mean); alll.append(lam)
        r2, sp, covs, nclu = metrics(truth, mean, lam)
        print(f"{sh:12s} {len(truth):7d} {r2[0]:7.3f} {r2[1]:7.3f} {r2[2]:7.3f} "
              f"{sp:6.2f} {covs[0]:6.3f} {covs[1]:6.3f}")
        result[sh] = dict(n=len(truth), r2=r2, cluster_spearman=sp, cov68=covs[0], cov90=covs[1], n_clu=nclu)
    truth = np.concatenate(allt); mean = np.concatenate(allm); lam = np.concatenate(alll)
    r2, sp, covs, nclu = metrics(truth, mean, lam)
    print(f"{'ALL':12s} {len(truth):7d} {r2[0]:7.3f} {r2[1]:7.3f} {r2[2]:7.3f} "
          f"{sp:6.2f} {covs[0]:6.3f} {covs[1]:6.3f}")
    result["ALL"] = dict(n=len(truth), r2=r2, cluster_spearman=sp, cov68=covs[0], cov90=covs[1], n_clu=nclu)
    print("\nBaseline (S1b zero-shot, OLD wedge model on full-density shells): "
          "shell0 R2λ1=-1.09; grid CNN best-case 0.002@z0.05 / 0.429@z0.45.")

    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=1))
        print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
