#!/usr/bin/env python3
"""Pb — Posterior tempering / variance recalibration of the G3+FMPE head.

P3 (gate_g6_fmpe_frozen_head) found FMPE beats MAF on accuracy but BOTH flows
under-cover (posteriors too narrow): FMPE lambda1 central 68% cov = 0.594
(nominal 0.68), SBC KS-uniform p ~ 0. This script applies post-hoc posterior
tempering: inflate posterior samples about their per-node mean by a scalar (or
per-dim) factor tau, calibrate tau on a held-out CALIB half of the test split to
hit nominal 68% coverage, then EVALUATE on the disjoint EVAL half.

Discipline: FMPE is trained on train|val (as gate_g6), so val is IN-sample for
the flow. The only truly held-out rows are the TEST split -> we halve TEST into
disjoint CALIB (tune tau) and EVAL (report) sets. Never tune and report on the
same rows.

Tempering acts about the per-node posterior mean, so the posterior MEAN (hence
R^2) is exactly unchanged. The scaled-increment -> raw-lambda map is affine
(StandardScaler.inverse_transform + cumsum), so tempering scaled samples about
their mean by tau equals tempering raw lambda about its mean by the same tau.

CPU-only (JAX_PLATFORMS=cpu for the frozen GNN forward pass reuse via cache;
torch CPU for FMPE). No production edits.
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.stats import kstest
from sklearn.metrics import r2_score


def to_raw_lambda(samples_scaled, target_scaler):
    """scaled linear increments -> raw (lambda1,lambda2,lambda3) via inverse-scale + cumsum."""
    shp = samples_scaled.shape
    inc = target_scaler.inverse_transform(samples_scaled.reshape(-1, 3)).reshape(shp)
    return np.cumsum(inc, axis=-1)


def central_coverage(lam_raw, truth, dim, q):
    """Fraction of truth[:,dim] inside the central-q credible interval of lam_raw[:,:,dim]."""
    lo = np.quantile(lam_raw[:, :, dim], (1 - q) / 2, axis=1)
    hi = np.quantile(lam_raw[:, :, dim], 1 - (1 - q) / 2, axis=1)
    return np.mean((truth[:, dim] >= lo) & (truth[:, dim] <= hi))


def temper(S_scaled, tau):
    """Inflate scaled samples about their per-node mean by tau (scalar or per-dim (3,))."""
    mu = S_scaled.mean(axis=1, keepdims=True)
    return mu + np.asarray(tau) * (S_scaled - mu)


def calibrate_scalar(S_scaled, scaler, truth, dim, q, target, tau_lo=0.5, tau_hi=8.0, iters=40):
    """Bisection for scalar tau so central-q coverage of raw lambda[dim] == target.
    Coverage is monotone increasing in tau. Returns tau."""
    def cov(tau):
        lam = to_raw_lambda(temper(S_scaled, tau), scaler)
        return central_coverage(lam, truth, dim, q)
    lo, hi = tau_lo, tau_hi
    # ensure bracket
    if cov(hi) < target:
        return hi
    if cov(lo) > target:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cov(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_perdim_scaled(S_scaled, dim, q, target, tau_lo=0.5, tau_hi=8.0, iters=40):
    """Bisection for tau_k so the SCALED-space central-q coverage of scaled-dim k
    hits target (recalibrates each SBC marginal directly). truth in scaled space
    is theta[idx][:,dim]."""
    pass  # replaced below (needs theta) — see main


def sbc_ks(S_scaled, theta_rows):
    """SBC KS-uniform p per dim: rank = frac posterior samples below truth (scaled)."""
    ranks = (S_scaled < theta_rows[:, None, :]).mean(axis=1)  # [n,3]
    return [kstest(ranks[:, k], "uniform").pvalue for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--emb-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=6000,
                    help="test nodes evaluated (split in half: calib/eval)")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tarp", action="store_true", help="run TARP before/after (needs tarp pkg)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    emb = np.load(args.emb_npz)["emb"]
    cache = pickle.load(open(args.cache, "rb"))
    theta = np.asarray(cache["regression_targets"], np.float64)   # scaled increments
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
    tr = train | val   # same pool as gate_g6 / P3
    print(f"training FMPE on {tr.sum()} nodes (train|val), seed {args.seed} ...", flush=True)
    trainer = FMPE(prior=prior)
    trainer.append_simulations(torch.tensor(theta[tr], dtype=torch.float32),
                               torch.tensor(emb[tr], dtype=torch.float32))
    est = trainer.train(training_batch_size=512)
    posterior = trainer.build_posterior(est)

    # evaluate on a subset of the TEST split (truly held out from the flow)
    ti = np.where(test)[0]
    sub = rng.permutation(len(ti))[: args.n_eval]
    idx = ti[sub]
    print(f"sampling {args.n_samples} posterior draws for {len(idx)} test nodes ...", flush=True)
    X = torch.tensor(emb[idx], dtype=torch.float32)
    try:
        S = posterior.sample_batched((args.n_samples,), x=X, show_progress_bars=False)
        S = np.asarray(S.detach().cpu(), np.float64)      # [n_samples, n_eval, 3]
        S = np.transpose(S, (1, 0, 2))                    # [n_eval, n_samples, 3]
    except Exception as e:
        print(f"(sample_batched unavailable: {e}); looping")
        S = np.stack([np.asarray(posterior.sample((args.n_samples,), x=X[i:i+1],
                                                   show_progress_bars=False).detach().cpu())
                      for i in range(len(idx))], axis=0)

    theta_idx = theta[idx]
    truth = eig_raw[idx]

    # disjoint calib / eval halves of the evaluated test rows
    perm = rng.permutation(len(idx))
    half = len(idx) // 2
    A = perm[:half]   # CALIB
    B = perm[half:]   # EVAL

    def slice_rows(rows):
        return S[rows], theta_idx[rows], truth[rows]

    S_A, th_A, tr_A = slice_rows(A)
    S_B, th_B, tr_B = slice_rows(B)

    # ---- baseline (tau=1) ----
    lam_A0 = to_raw_lambda(S_A, scaler)
    lam_B0 = to_raw_lambda(S_B, scaler)
    r2_B = [r2_score(tr_B[:, k], lam_B0.mean(axis=1)[:, k]) for k in range(3)]

    # ---- calibrate scalar tau on CALIB half (A) ----
    tau68 = calibrate_scalar(S_A, scaler, tr_A, dim=0, q=0.68, target=0.68)
    tau90 = calibrate_scalar(S_A, scaler, tr_A, dim=0, q=0.90, target=0.90)

    # ---- calibrate per-dim tau_k on CALIB half: hit each raw lambda_k 68% coverage ----
    # (bisection per dim; coverage of lambda_k depends on scaled dims 0..k via cumsum,
    #  but is monotone in a per-dim inflation of the dims it uses; we inflate only dim k
    #  in scaled space to keep it a clean 1-D solve, applied cumulatively.)
    taus_perdim = np.ones(3)
    for k in range(3):
        def cov_k(tk):
            tvec = taus_perdim.copy(); tvec[k] = tk
            lam = to_raw_lambda(temper(S_A, tvec), scaler)
            return central_coverage(lam, tr_A, k, 0.68)
        lo_t, hi_t = 0.5, 8.0
        if cov_k(hi_t) < 0.68:
            taus_perdim[k] = hi_t
        elif cov_k(lo_t) > 0.68:
            taus_perdim[k] = lo_t
        else:
            for _ in range(40):
                mid = 0.5 * (lo_t + hi_t)
                if cov_k(mid) < 0.68:
                    lo_t = mid
                else:
                    hi_t = mid
            taus_perdim[k] = 0.5 * (lo_t + hi_t)

    print(f"\ntau68(scalar,lambda1)={tau68:.3f}  tau90(scalar,lambda1)={tau90:.3f}")
    print(f"per-dim tau (lambda1/2/3 68%): {np.round(taus_perdim,3)}")

    # ---- EVALUATE on EVAL half (B) for each tau setting ----
    def report(tag, tau, fh):
        Stemp = temper(S_B, tau)
        lam = to_raw_lambda(Stemp, scaler)
        ks = sbc_ks(Stemp, th_B)
        cov68 = [central_coverage(lam, tr_B, k, 0.68) for k in range(3)]
        cov90 = [central_coverage(lam, tr_B, k, 0.90) for k in range(3)]
        r2 = [r2_score(tr_B[:, k], lam.mean(axis=1)[:, k]) for k in range(3)]
        line = (f"[{tag}] tau={np.round(np.asarray(tau),3)}\n"
                f"    cov68 (l1,l2,l3) = {[f'{c:.3f}' for c in cov68]}  (nominal 0.68)\n"
                f"    cov90 (l1,l2,l3) = {[f'{c:.3f}' for c in cov90]}  (nominal 0.90)\n"
                f"    SBC KS-uniform p = {[f'{p:.3f}' for p in ks]}\n"
                f"    post-mean R2     = {[f'{r:.3f}' for r in r2]}\n")
        print(line, flush=True)
        fh.write(line + "\n")
        return dict(cov68=cov68, cov90=cov90, ks=ks, r2=r2)

    out_txt = args.out_dir / "pb_tempering_result.txt"
    with open(out_txt, "w") as fh:
        fh.write("Pb — Posterior tempering of G3 GraphNet + FMPE head\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"FMPE trained on train|val ({tr.sum()} nodes); evaluated on TEST subset.\n")
        fh.write(f"n_eval(test)={len(idx)}  n_samples={args.n_samples}  "
                 f"CALIB={len(A)} EVAL={len(B)} (disjoint halves of test)\n")
        fh.write(f"Calibrated tau on CALIB, reported below on held-out EVAL.\n\n")
        fh.write(f"chosen scalar tau68(lambda1) = {tau68:.4f}\n")
        fh.write(f"scalar tau90(lambda1)        = {tau90:.4f}   "
                 f"(tau68 vs tau90 close => pure scale miscal; differ => shape)\n")
        fh.write(f"per-dim tau (l1,l2,l3 @68%)  = {np.round(taus_perdim,4).tolist()}\n\n")
        fh.write("--- EVAL-half results (held out; NOT used to tune tau) ---\n\n")
        base = report("baseline tau=1", 1.0, fh)
        sc68 = report("scalar tau68", tau68, fh)
        sc90 = report("scalar tau90", tau90, fh)
        pdim = report("per-dim tau68", taus_perdim, fh)

        # verdict
        c68 = sc68["cov68"][0]; c90 = sc68["cov90"][0]
        ks_ok = all(p > 0.05 for p in sc68["ks"])
        cov_ok = abs(c68 - 0.68) < 0.03 and abs(c90 - 0.90) < 0.03
        fh.write("=" * 60 + "\n")
        fh.write(f"scale-vs-shape: tau68={tau68:.3f} tau90={tau90:.3f} "
                 f"(ratio {tau90/tau68:.3f}) -> "
                 f"{'consistent scale miscal' if abs(tau90-tau68)<0.15 else 'SHAPE miscal (single tau cannot fix both)'}\n")
        fh.write(f"scalar-tau68 EVAL lambda1: cov68={c68:.3f} cov90={c90:.3f}; "
                 f"SBC KS p={[f'{p:.3f}' for p in sc68['ks']]}\n")
        verdict = ("SHIPPABLE: tempering achieves ~nominal coverage on held-out test "
                   "AND SBC uniform" if (cov_ok and ks_ok) else
                   "NOT fully calibrated: " +
                   ("coverage near nominal but SBC still non-uniform => residual SHAPE miscal"
                    if cov_ok and not ks_ok else
                    "coverage still off nominal after tempering"))
        fh.write(f"VERDICT: {verdict}\n")
        print("VERDICT:", verdict)

    # ---- optional TARP before/after on EVAL half (scaled space) ----
    if args.tarp:
        try:
            import tarp
            for tag, tau in [("baseline", 1.0), ("scalar_tau68", tau68)]:
                St = temper(S_B, tau)
                samples_tarp = np.transpose(St, (1, 0, 2))  # [n_samples, n_evals, 3]
                ecp, alpha = tarp.get_tarp_coverage(samples_tarp, th_B, norm=True, bootstrap=False)
                np.savez(args.out_dir / f"tarp_{tag}.npz", ecp=ecp, alpha=alpha)
                # summary: max |ECP - alpha|
                dev = float(np.max(np.abs(ecp - alpha)))
                with open(out_txt, "a") as fh:
                    fh.write(f"TARP[{tag}]: max|ECP-alpha| = {dev:.3f}\n")
                print(f"TARP[{tag}]: max|ECP-alpha| = {dev:.3f}")
        except Exception as e:
            print(f"TARP skipped: {e}")

    np.savez_compressed(args.out_dir / "pb_samples.npz",
                        idx=idx, A=A, B=B, S_scaled=S.astype(np.float32),
                        theta_idx=theta_idx, truth=truth,
                        tau68=tau68, tau90=tau90, taus_perdim=taus_perdim)
    print(f"\nwrote {out_txt} and pb_samples.npz")


if __name__ == "__main__":
    main()
