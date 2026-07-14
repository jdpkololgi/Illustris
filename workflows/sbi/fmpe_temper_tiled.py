#!/usr/bin/env python3
"""FMPE head + posterior tempering on the frozen TILED ñ-conditioned encoder (G6 recipe).

Stage 1 (jax, GPU): freeze the trained tiled GNN encoder; extract 80-d node embeddings per
tile for all ACTIVE nodes (train/val/test regions), with scaled targets, raw eigenvalues, shell.
Stage 2 (torch, CPU): train an sbi FMPE (flow-matching) head on TRAIN-region (embedding->theta);
tune a tempering factor tau on the VAL region to hit nominal 68% λ1 coverage; EVALUATE on the
disjoint TEST region per shell. FMPE vs the MAF baseline (R²) and calibration (tempered coverage).

Tempering inflates samples about their per-node mean by tau -> posterior MEAN (hence R²) is
UNCHANGED; only the width (coverage) is recalibrated. This is a calibration step, not an accuracy
lever. Splits are the spatial holdout (train RA<145 / val 145-150 / test RA>=150).
"""
from __future__ import annotations
import sys, os, json, pickle
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
from scipy.stats import kstest
from sklearn.metrics import r2_score

SHELLS = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]


def extract_embeddings(model_path, tiles_dir, out_npz):
    if Path(out_npz).exists():
        d = np.load(out_npz, allow_pickle=True)
        return d["emb"], d["theta"], d["eig_raw"], d["region"], d["shell"]
    import jax
    from plot_flowjax_posteriors import load_flowjax_model, create_gnn_and_flow
    gnn_params, config, target_scaler, flow_filename, inc = load_flowjax_model(str(model_path))
    manifest = json.loads((Path(tiles_dir) / "manifest.json").read_text())
    E, TH, EI, RG, SH = [], [], [], [], []
    for t in manifest["tiles"]:
        p = pickle.load(open(Path(tiles_dir) / t["file"], "rb"))
        graph = p["graph"]
        tr, va, te = (np.asarray(m).astype(bool) for m in p["masks"])
        region = np.full(len(tr), -1, np.int8); region[tr] = 0; region[va] = 1; region[te] = 2
        act = region >= 0
        if not act.any():
            continue
        gnn, _ = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
        emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
        E.append(emb[act]); TH.append(np.asarray(p["regression_targets"])[act])
        EI.append(np.asarray(p["eigenvalues_raw"])[act]); RG.append(region[act])
        SH.append(np.array([t["shell"]] * int(act.sum())))
    emb = np.concatenate(E).astype(np.float32); theta = np.concatenate(TH).astype(np.float32)
    eig_raw = np.concatenate(EI); region = np.concatenate(RG); shell = np.concatenate(SH)
    np.savez_compressed(out_npz, emb=emb, theta=theta, eig_raw=eig_raw, region=region, shell=shell,
                        scaler_mean=target_scaler.mean_, scaler_scale=target_scaler.scale_)
    print(f"embeddings: {emb.shape} train/val/test={np.bincount(region+0)}")
    return emb, theta, eig_raw, region, shell


def to_raw(S_scaled, mean, scale):
    inc = S_scaled * scale + mean          # inverse StandardScaler
    return np.cumsum(inc, axis=-1)          # linear increments -> raw eigenvalues


def temper(S, tau):
    mu = S.mean(1, keepdims=True)
    return mu + tau * (S - mu)


def cov(lam, truth, q):
    lo = np.quantile(lam[:, :, 0], (1 - q) / 2, axis=1); hi = np.quantile(lam[:, :, 0], 1 - (1 - q) / 2, axis=1)
    return float(np.mean((truth[:, 0] >= lo) & (truth[:, 0] <= hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--emb-npz", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=120000)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    emb, theta, eig_raw, region, shell = extract_embeddings(args.model_path, args.tiles_dir, args.emb_npz)
    d = np.load(args.emb_npz); s_mean, s_scale = d["scaler_mean"], d["scaler_scale"]

    import torch
    from sbi.inference import FMPE
    from sbi.utils import BoxUniform
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FMPE device: {dev}")
    tr = region == 0; va = region == 1; te = region == 2
    tri = np.where(tr)[0]
    if len(tri) > args.max_train:
        tri = rng.permutation(tri)[: args.max_train]
    lo = theta.min(0) - 0.2 * np.ptp(theta, 0); hi = theta.max(0) + 0.2 * np.ptp(theta, 0)
    prior = BoxUniform(low=torch.tensor(lo).to(dev), high=torch.tensor(hi).to(dev))
    trainer = FMPE(prior=prior, device=dev)
    trainer.append_simulations(torch.tensor(theta[tri]), torch.tensor(emb[tri]))
    print(f"training FMPE on {len(tri)} train-region nodes...")
    est = trainer.train(training_batch_size=1024)
    post = trainer.build_posterior(est)

    def sample(idx):
        X = torch.tensor(emb[idx]).to(dev)
        S = post.sample_batched((args.n_samples,), x=X, show_progress_bars=False)
        return np.transpose(np.asarray(S.detach().cpu()), (1, 0, 2))   # [n, n_samples, 3]

    # tune tau on VAL to hit 68% λ1 coverage (bisection; coverage monotone in tau)
    vi = np.where(va)[0]; vi = rng.permutation(vi)[:8000]
    Sv = sample(vi); tv = eig_raw[vi]
    def valcov(tau): return cov(to_raw(temper(Sv, tau), s_mean, s_scale), tv, 0.68)
    tlo, thi = 0.5, 8.0
    for _ in range(40):
        tm = 0.5 * (tlo + thi)
        if valcov(tm) < 0.68: tlo = tm
        else: thi = tm
    tau = 0.5 * (tlo + thi)
    print(f"tempering tau={tau:.3f} (val λ1 cov68 {valcov(tau):.3f})")

    # EVAL on TEST per shell
    print(f"\n{'shell':12s} {'n':>7s} {'R2_l1(FMPE)':>11s} {'cov68':>6s} {'cov90':>6s} {'SBC_p_l1':>8s}")
    res = {"tau": float(tau)}
    order = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
    all_idx = []
    for sh in order:
        idx = np.where(te & (shell == sh))[0]
        if len(idx) < 50: continue
        if len(idx) > 8000: idx = rng.permutation(idx)[:8000]
        all_idx.append(idx)
        S = temper(sample(idx), tau); lam = to_raw(S, s_mean, s_scale); truth = eig_raw[idx]
        r2 = r2_score(truth[:, 0], lam[:, :, 0].mean(1))
        ranks = (S[:, :, 0] < ((truth[:, 0][:, None] - s_mean[0]) / s_scale[0] - 0)).mean(1)  # approx SBC on l1 incr
        # proper SBC rank of truth-increment among samples (increment space)
        tr_inc0 = (truth[:, 0] - s_mean[0]) / s_scale[0]
        rk = (S[:, :, 0] < tr_inc0[:, None]).mean(1)
        sbc = kstest(rk, "uniform").pvalue
        c68, c90 = cov(lam, truth, 0.68), cov(lam, truth, 0.90)
        print(f"{sh:12s} {len(idx):7d} {r2:11.3f} {c68:6.3f} {c90:6.3f} {sbc:8.3f}")
        res[sh] = dict(n=len(idx), r2_l1=r2, cov68=c68, cov90=c90, sbc_p=sbc)
    ai = np.concatenate(all_idx)
    S = temper(sample(ai), tau); lam = to_raw(S, s_mean, s_scale); truth = eig_raw[ai]
    r2 = r2_score(truth[:, 0], lam[:, :, 0].mean(1))
    print(f"{'ALL':12s} {len(ai):7d} {r2:11.3f} {cov(lam,truth,0.68):6.3f} {cov(lam,truth,0.90):6.3f}")
    res["ALL"] = dict(n=len(ai), r2_l1=r2, cov68=cov(lam, truth, 0.68), cov90=cov(lam, truth, 0.90))
    print("\nMAF baseline (best-val, eval_tiled_heldout): ALL R2_l1=0.340, cov68=0.653, cov90=0.876.")
    print("FMPE changes ACCURACY (R2) via a better head; tempering fixes WIDTH (coverage) only.")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1)); print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
