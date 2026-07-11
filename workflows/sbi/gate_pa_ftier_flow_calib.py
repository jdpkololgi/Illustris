#!/usr/bin/env python3
"""Gate Pa — is an F-tier-encoder-conditioned flow a CALIBRATED posterior at DESI density?

Branch (a) of the production-VAC calibration work (F1-calibration gate,
docs/plan_field_level_multimodal.md §3.3 heads/losses + §10). Analogous to
gate_g6_fmpe_frozen_head.py, but the conditioning is the FROZEN F-tier encoder's
per-node invariant latent h_i (EGNNEncoder output, width 64), extracted on the
nzharm (DESI-density) cache by gate_t4 --save-embeddings.

Trains BOTH an FMPE (flow matching) and an NPE-MAF head (sbi package) on the SAME
train+val split and the SAME scaled eigenvalue-increment targets, conditioned on
the F-tier embeddings. Reports posterior-mean R^2, SBC KS-uniform p per eigenvalue,
and central coverage of lambda1 at nominal 68% / 90% — byte-identical calibration
machinery to gate_g6, so the number is comparable to the P3 G3+flow reference
(raw wedge under-covered ~0.594@68, KS p<0.01).

Runs fully on CPU (JAX not needed; sbi/torch CPU). No production edits; embeddings
read from the npz written by gate_t4.
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


def to_raw_lambda(samples_scaled, target_scaler):
    """scaled linear increments -> raw (lambda1, lambda2, lambda3) via cumsum."""
    shp = samples_scaled.shape
    inc = target_scaler.inverse_transform(samples_scaled.reshape(-1, 3)).reshape(shp)
    return np.cumsum(inc, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--emb-npz", type=Path, required=True,
                    help="npz from gate_t4 --save-embeddings (key 'emb', [N,width]).")
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = np.load(args.emb_npz)
    emb = d["emb"].astype(np.float32)
    cache = pickle.load(open(args.cache, "rb"))
    theta = np.asarray(cache["regression_targets"], np.float64)      # scaled increments
    eig_raw = np.asarray(cache["eigenvalues_raw"], np.float64)
    scaler = cache["target_scaler"]
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    assert emb.shape[0] == theta.shape[0] == eig_raw.shape[0], (
        f"row mismatch: emb {emb.shape} theta {theta.shape} eig {eig_raw.shape}")
    print(f"emb {emb.shape}  theta {theta.shape}  "
          f"train/val/test {train.sum()}/{val.sum()}/{test.sum()}")

    import torch
    from sbi.inference import FMPE, NPE
    from sbi.utils import BoxUniform
    torch.manual_seed(args.seed)

    lo = theta.min(0) - 0.1 * np.ptp(theta, 0)
    hi = theta.max(0) + 0.1 * np.ptp(theta, 0)
    prior = BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                       high=torch.tensor(hi, dtype=torch.float32))
    tr = train | val
    th_tr = torch.tensor(theta[tr], dtype=torch.float32)
    x_tr = torch.tensor(emb[tr], dtype=torch.float32)

    posteriors = {}
    # FMPE (flow matching) and NPE-MAF, same conditioning + train pool.
    print("\n=== training FMPE ===")
    tf = FMPE(prior=prior)
    tf.append_simulations(th_tr, x_tr)
    posteriors["FMPE"] = tf.build_posterior(tf.train(training_batch_size=512))
    print("\n=== training NPE-MAF ===")
    tm = NPE(prior=prior, density_estimator="maf")
    tm.append_simulations(th_tr, x_tr)
    posteriors["MAF"] = tm.build_posterior(tm.train(training_batch_size=512))

    # eval subset of the test split
    ti = np.where(test)[0]
    sub = rng.permutation(len(ti))[: args.n_eval]
    idx = ti[sub]
    X = torch.tensor(emb[idx], dtype=torch.float32)
    truth = eig_raw[idx]
    theta_eval = theta[idx]

    def sample_posterior(post):
        try:
            S = post.sample_batched((args.n_samples,), x=X, show_progress_bars=False)
            S = np.asarray(S.detach().cpu(), np.float64)                 # [n_s, n_eval, 3]
            return np.transpose(S, (1, 0, 2))                            # [n_eval, n_s, 3]
        except Exception as e:
            print(f"(sample_batched unavailable: {e}); falling back to loop")
            return np.stack([np.asarray(post.sample((args.n_samples,), x=X[i:i + 1],
                             show_progress_bars=False).detach().cpu())
                             for i in range(len(idx))], axis=0)

    out = []
    def emit(s):
        print(s); out.append(s)

    emit(f"gate_pa_ftier_flow_calib  emb={args.emb_npz.name}  n_eval={len(idx)}  "
         f"n_samples={args.n_samples}  seed={args.seed}")
    emit(f"F-tier encoder (frozen h_i, width={emb.shape[1]}) -> flow head @ nzharm (DESI density)")
    emit("")

    results = {}
    for tag in ("FMPE", "MAF"):
        S = sample_posterior(posteriors[tag])          # [n_eval, n_s, 3] scaled
        lam = to_raw_lambda(S, scaler)                 # [n_eval, n_s, 3] raw
        mean_f = lam.mean(axis=1)
        results[tag] = dict(S=S, lam=lam, mean=mean_f)

    emit(f"{'':10s}  " + "  ".join(f"{t+' R2':>9s}" for t in ("FMPE", "MAF"))
         + "    (F-tier MSE-head nzharm: 0.838; GraphNet 0.775/0.811/0.891; G3 0.804)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2s = [r2_score(truth[:, k], results[t]["mean"][:, k]) for t in ("FMPE", "MAF")]
        emit(f"{nm:10s}  " + "  ".join(f"{v:9.3f}" for v in r2s))
    clu = truth[:, 0] > 0.2
    if clu.sum() > 30:
        sps = [spearmanr(truth[clu, 0], results[t]["mean"][clu, 0]).statistic
               for t in ("FMPE", "MAF")]
        emit(f"cluster-slice lambda1 Spearman: "
             + "  ".join(f"{t} {v:+.2f}" for t, v in zip(("FMPE", "MAF"), sps))
             + f"  (n={int(clu.sum())})")

    # calibration: SBC ranks (scaled-target space) + central lambda1 coverage (raw).
    def calib(tag):
        S = results[tag]["S"]; lam = results[tag]["lam"]
        ranks = (S < theta_eval[:, None, :]).mean(axis=1)               # [n_eval,3]
        ks = [kstest(ranks[:, k], "uniform").pvalue for k in range(3)]
        emit("")
        emit(f"calibration ({tag}): SBC KS-uniform p per dim: "
             + f"[{', '.join(f'{p:.3f}' for p in ks)}]")
        for q in (0.68, 0.90):
            lo_q = np.quantile(lam[:, :, 0], (1 - q) / 2, axis=1)
            hi_q = np.quantile(lam[:, :, 0], 1 - (1 - q) / 2, axis=1)
            cov = np.mean((truth[:, 0] >= lo_q) & (truth[:, 0] <= hi_q))
            emit(f"  {tag} lambda1 central {int(q*100)}% coverage: {cov:.3f} (nominal {q:.2f})")

    calib("FMPE")
    calib("MAF")

    emit("")
    emit("REFERENCE (P3, G3 GraphNet + flow, RAW over-dense wedge):")
    emit("  FMPE SBC KS p 0.000/0.003/0.001; lambda1 cov 0.594@68 0.829@90")
    emit("  MAF  SBC KS p 0.009/0.006/0.017; lambda1 cov 0.610@68 0.837@90")
    emit("VERDICT test: is F-tier+flow @ nzharm better/similar/worse on coverage & SBC?")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(out) + "\n")
    print(f"\nsummary written: {args.out_file}")


if __name__ == "__main__":
    main()
