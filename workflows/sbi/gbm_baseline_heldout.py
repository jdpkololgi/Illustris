#!/usr/bin/env python3
"""GBM baseline vs the tiled GNN, on the SAME held-out test region (RA>=150), per shell.

Ablation question: how much does the GraphNet's message-passing + edge features (unit vectors,
density contrast) add OVER a plain gradient-boosted tree on the SAME 8 per-node features
(7 geometric incl. degree/density/neigh-density/I_eig + ñ)? Trees are invariant to the
per-node monotonic box-cox, so we feed the tile node features directly. Point R^2 for
λ1/λ2/λ3 (accuracy) + quantile-GBM 68/90% intervals for λ1 (calibration), grouped by shell,
printed in the SAME format as eval_tiled_heldout.py for a direct read-across.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

SHELLS = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]


def gbm(**kw):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=7,
                                         l2_regularization=1.0, **kw)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    manifest = json.loads((args.tiles_dir / "manifest.json").read_text())

    Xtr, ytr, Xte, yte, shte = [], [], [], [], []
    for t in manifest["tiles"]:
        p = pickle.load(open(args.tiles_dir / t["file"], "rb"))
        x = np.asarray(p["graph"].nodes, np.float64)      # [N,8] node feats incl ñ (col 7)
        eig = np.asarray(p["eigenvalues_raw"], np.float64)
        tr = np.asarray(p["masks"][0]).astype(bool); te = np.asarray(p["masks"][2]).astype(bool)
        if tr.any():
            Xtr.append(x[tr]); ytr.append(eig[tr])
        if te.any():
            Xte.append(x[te]); yte.append(eig[te]); shte.append(np.array([t["shell"]] * int(te.sum())))
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte); shte = np.concatenate(shte)
    print(f"GBM train nodes={len(Xtr)} test nodes={len(Xte)} feats={Xtr.shape[1]}")

    # mean models (R^2) for the 3 eigenvalues
    pred = np.zeros_like(yte)
    for k in range(3):
        m = gbm().fit(Xtr, ytr[:, k]); pred[:, k] = m.predict(Xte)
    # quantile models for λ1 coverage
    qmods = {q: gbm(loss="quantile", quantile=q).fit(Xtr, ytr[:, 0])
             for q in (0.16, 0.84, 0.05, 0.95)}
    q16, q84, q05, q95 = (qmods[q].predict(Xte) for q in (0.16, 0.84, 0.05, 0.95))

    def block(mask, name):
        y = yte[mask]; pr = pred[mask]
        r2 = [r2_score(y[:, k], pr[:, k]) for k in range(3)]
        clu = y[:, 0] > 0.2
        sp = spearmanr(y[clu, 0], pr[clu, 0]).statistic if clu.sum() > 20 else np.nan
        c68 = float(np.mean((y[:, 0] >= q16[mask]) & (y[:, 0] <= q84[mask])))
        c90 = float(np.mean((y[:, 0] >= q05[mask]) & (y[:, 0] <= q95[mask])))
        print(f"{name:12s} {mask.sum():7d} {r2[0]:7.3f} {r2[1]:7.3f} {r2[2]:7.3f} {sp:6.2f} {c68:6.3f} {c90:6.3f}")
        return dict(n=int(mask.sum()), r2=r2, cluster_spearman=sp, cov68=c68, cov90=c90)

    print(f"\n{'shell':12s} {'n_test':>7s} {'R2_l1':>7s} {'R2_l2':>7s} {'R2_l3':>7s} {'cluSp':>6s} {'cov68':>6s} {'cov90':>6s}")
    res = {}
    for sh in SHELLS:
        m = shte == sh
        if m.any():
            res[sh] = block(m, sh)
    res["ALL"] = block(np.ones(len(yte), bool), "ALL")
    print("\ncompare vs eval_tiled_heldout.py (GNN). GBM uses node feats only — no message "
          "passing, no edge unit-vectors/contrast. GNN >> GBM => graph structure earns its cost.")
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=1)); print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
