#!/usr/bin/env python3
"""Per-shell + macro R2 for the full-range DTFE run, matched to the GraphNet's scoring.

classical_tidal_baseline reports POOLED test R2; the production GraphNet's headline is MACRO-shell
(mean over shells). This reuses the SAME predictions the baseline saved (pred_eigs_dtfe.npy, raw
eigenvalues), applies the SAME global train-fit affine calibration, and reports pooled + per-shell +
macro on the s3c spatial-holdout test mask -- the exact set the GraphNet was scored on.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

S = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_cache.pkl")
PREDS = Path("/pscratch/sd/d/dkololgi/abacus/classical_baseline/fullrange_holdout")

GNET_VAL_MACRO = 0.456        # A1_sqrt (production 8-d GraphNet, spatial holdout, VAL)
GNET_TEST_POOLED = 0.514      # R0 held-out pooled lambda1
DTFE_DENSE = (0.534, 0.604, 0.634)   # dense wedge (2.4x denser) -- the sparsity comparison


def cal_r2(pred, eig, tr, te, shell):
    """Global affine cal (fit on train) then pooled + per-shell + macro test R2, per eigenvalue."""
    rows = {}
    for k, nm in enumerate(("l1", "l2", "l3")):
        p, y = pred[:, k], eig[:, k]
        A = np.stack([p[tr], np.ones(tr.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pc = coef[0] * p + coef[1]
        pooled = float(r2_score(y[te], pc[te]))
        per = {s: float(r2_score(y[te & (shell == s)], pc[te & (shell == s)]))
               for s in sorted(set(shell[te].tolist()))}
        macro = float(np.mean(list(per.values())))
        rows[nm] = {"pooled": pooled, "macro": macro, "per_shell": per}
    return rows


def main():
    c = pickle.load(open(S, "rb"))
    eig = np.asarray(c["eigenvalues_raw"], np.float64)
    tr, va, te = (np.asarray(m).astype(bool) for m in c["masks"])
    shell = np.asarray(c["shell"])
    for est in ("dtfe", "cic"):
        f = PREDS / f"pred_eigs_{est}.npy"
        if not f.exists():
            print(f"[skip] {f} missing"); continue
        pred = np.load(f).astype(np.float64)
        r = cal_r2(pred, eig, tr, te, shell)
        print(f"\n=== {est.upper()} full-range spatial holdout (test RA>=150, cal) ===")
        for nm in ("l1", "l2", "l3"):
            d = r[nm]
            ps = "  ".join(f"{s.split('_')[0]}:{v:.3f}" for s, v in d["per_shell"].items())
            print(f"  {nm}: pooled {d['pooled']:.3f}  macro {d['macro']:.3f}   [{ps}]")
        if est == "dtfe":
            print(f"\n  COMPARATORS (lambda1):")
            print(f"    DTFE full-range macro {r['l1']['macro']:.3f} / pooled {r['l1']['pooled']:.3f}")
            print(f"    GraphNet A1_sqrt VAL macro {GNET_VAL_MACRO:.3f} | R0 test pooled {GNET_TEST_POOLED:.3f}")
            print(f"    DTFE on DENSE wedge (2.4x denser): {DTFE_DENSE[0]:.3f}")
            print(f"    => sparsity tax on DTFE lambda1 = {DTFE_DENSE[0] - r['l1']['pooled']:+.3f} (dense - full-range pooled)")


if __name__ == "__main__":
    main()
