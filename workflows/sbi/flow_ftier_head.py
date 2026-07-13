#!/usr/bin/env python3
"""Normalizing-flow posterior head on the F-tier physics point estimate.

The F-tier gives a strong deterministic eigenvalue prediction (~0.84). This trains
an amortized conditional flow (FMPE and NPE-MAF) by MAXIMUM LIKELIHOOD with that
prediction as the conditioning summary — MLE calibrates the per-eigenvalue MARGINALS
(the thing SBC tests), which the energy-score F3 did not. CPU-only.

Reads the conditioning npz written by `gate_ftier_v2.py --save-cond`.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from scipy.stats import kstest, spearmanr
from sklearn.metrics import r2_score
import torch


def calib(samp, truth, tag):   # samp [Ne,K], truth [Ne]
    rank = (samp < truth[:, None]).mean(1)
    ks = kstest(rank, "uniform").pvalue
    lo, hi = np.quantile(samp, (0.16, 0.84), axis=1); c68 = np.mean((truth >= lo) & (truth <= hi))
    lo, hi = np.quantile(samp, (0.05, 0.95), axis=1); c90 = np.mean((truth >= lo) & (truth <= hi))
    print(f"  {tag:16s} SBC KS-p={ks:.4f}  cov68={c68:.3f}  cov90={c90:.3f}")
    return ks, c68, c90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-npz", type=Path, required=True)
    ap.add_argument("--k-eval", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=1500, help="subsample test galaxies for calibration eval (FMPE ODE sampling is slow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--samples-npz", type=Path, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    d = np.load(args.cond_npz)
    cond = d["cond"].astype(np.float64)                       # [N,3] F-tier predicted eigenvalues
    eig = d["eig"].astype(np.float64)                         # [N,3] true raw eigenvalues
    mu, sd = d["mu"].astype(np.float64), d["sd"].astype(np.float64)
    tr, va, te = d["train"].astype(bool), d["val"].astype(bool), d["test"].astype(bool)
    theta = (eig - mu) / sd                                   # scaled target
    x = (cond - cond[tr].mean(0)) / cond[tr].std(0)           # standardized conditioning
    print(f"N={len(eig)}  cond dim={x.shape[1]}  train={tr.sum()} test={te.sum()}")

    from sbi.inference import FMPE, NPE, NPSE
    from sbi.utils import BoxUniform
    lo = theta.min(0) - 0.2 * np.ptp(theta, 0); hi = theta.max(0) + 0.2 * np.ptp(theta, 0)
    prior = BoxUniform(low=torch.tensor(lo, dtype=torch.float32), high=torch.tensor(hi, dtype=torch.float32))
    pool = tr | va
    th = torch.tensor(theta[pool], dtype=torch.float32); xx = torch.tensor(x[pool], dtype=torch.float32)

    ti = np.where(te)[0]
    if args.n_eval and args.n_eval < len(ti):
        ti = np.sort(rng.choice(ti, args.n_eval, replace=False))
    Xte = torch.tensor(x[ti], dtype=torch.float32)
    truth = eig[ti]
    results = {}
    for name, Trainer in [("NPSE", NPSE), ("FMPE", FMPE), ("MAF", NPE)]:
        print(f"\n=== {name} (MLE) ===")
        tr_ = Trainer(prior=prior)
        tr_.append_simulations(th, xx)
        est = tr_.train(training_batch_size=512)
        post = tr_.build_posterior(est)
        try:
            S = post.sample_batched((args.k_eval,), x=Xte, show_progress_bars=False)
            S = np.transpose(np.asarray(S.detach()), (1, 0, 2))    # [Ne,K,3] scaled
        except Exception as e:
            print(f"  batched sample failed ({e}); looping");
            S = np.stack([np.asarray(post.sample((args.k_eval,), x=Xte[i:i+1],
                          show_progress_bars=False).detach()) for i in range(len(ti))])
        lam = S * sd + mu                                         # [Ne,K,3] raw
        pm = lam.mean(1)
        print(f"  posterior-mean R2: " + " ".join(f"{r2_score(truth[:,k], pm[:,k]):.3f}" for k in range(3)))
        clu = truth[:, 0] > 0.2
        print(f"  cluster-slice l1 Spearman: {spearmanr(truth[clu,0], pm[clu,0]).statistic:+.2f}")
        for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
            calib(lam[:, :, k], truth[:, k], nm)
        calib(lam.sum(2), truth.sum(1), "trace(=delta)")
        results[name] = lam
        if name in ("FMPE", "NPSE") and args.samples_npz:
            args.samples_npz.parent.mkdir(parents=True, exist_ok=True)
            out = args.samples_npz if name == "FMPE" else args.samples_npz.with_name(
                args.samples_npz.stem + "_npse" + args.samples_npz.suffix)
            np.savez_compressed(out, samples_test=np.transpose(lam, (1, 0, 2)),
                                truth_test=truth, test_index=ti)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        f.write("flow_ftier_head (MLE flow on F-tier physics point estimate)\n")
        for name in results:
            lam = results[name]; pm = lam.mean(1)
            f.write(f"[{name}] pmean_R2 " + " ".join(f"{r2_score(truth[:,k],pm[:,k]):.4f}" for k in range(3)) + "\n")
            for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
                ks, c68, c90 = calib(lam[:, :, k], truth[:, k], nm)
                f.write(f"  {name} {nm}: SBC_KS_p={ks:.4f} cov68={c68:.3f} cov90={c90:.3f}\n")
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
