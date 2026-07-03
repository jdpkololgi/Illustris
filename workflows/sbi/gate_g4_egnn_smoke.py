#!/usr/bin/env python3
"""Gate G4 SMOKE — EGNN-lite (scalarized geometric messages) vs the Attentional
GraphNetwork baseline (roadmap v2, symmetry-axis quick test).

This is the CHEAP first look, not the full steerable e3nn build: an invariant-
feature EGNN (Satorras-style) whose messages consume rotation-about-observer
invariant scalars of the raw geometry — |r_ij| and the LOS-projected parallel /
transverse split — alongside the same curated node features (cache graph.nodes,
byte-identical to the baseline's inputs) on the same Delaunay edge set and the
same seed-42 splits. Point-estimate MSE head -> test R^2 per eigenvalue.

Honest caveats printed with the result: (a) point estimate vs the baseline's
posterior mean (MSE directly optimises R^2 - slight advantage to the smoke);
(b) invariant scalarization, not full type-2 steerable output. GO criterion for
proceeding to the full e3nn build: smoke within ~0.02 of, or above, baseline
lambda1 R^2 = 0.774. Torch; GPU if available, CPU fallback.
"""
from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class EGNNLite(nn.Module):
    def __init__(self, nfeat, negeo, width=96, layers=5):
        super().__init__()
        self.embed = nn.Linear(nfeat, width)
        self.msg = nn.ModuleList()
        self.upd = nn.ModuleList()
        for _ in range(layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                          nn.Linear(width, width)))
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def _layer(self, h, src, dst, egeo, msg, upd):
        n = h.shape[0]
        m = msg(torch.cat([h[src], h[dst], egeo], dim=1))
        agg = torch.zeros(n, m.shape[1], device=h.device, dtype=m.dtype)
        cnt = torch.zeros(n, 1, device=h.device, dtype=m.dtype)
        agg.index_add_(0, dst, m)
        cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device, dtype=m.dtype))
        return h + upd(torch.cat([h, agg / cnt.clamp(min=1)], dim=1))

    def forward(self, h, src, dst, egeo):
        h = self.embed(h)
        for msg, upd in zip(self.msg, self.upd):
            # gradient checkpointing: recompute each layer's edge tensors in backward
            # instead of holding ~1.5M-edge activations for all layers (OOM otherwise)
            h = checkpoint(self._layer, h, src, dst, egeo, msg, upd, use_reentrant=False)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--gnn-arrays", type=Path, required=True, help="wedge gnn_arrays npz (edge pairs)")
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    cache = pickle.load(open(args.cache, "rb"))
    X = np.asarray(cache["graph"].nodes, np.float64)         # identical inputs to baseline
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])

    pos = np.load(args.points_xyz).astype(np.float64)
    ei = np.load(args.gnn_arrays)["edge_index"].astype(np.int64)
    src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])  # bidirectional
    r = pos[dst] - pos[src]
    d = np.linalg.norm(r, axis=1)
    los = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rpar_s = np.einsum("ij,ij->i", r, los[src]) / np.maximum(d, 1e-12)
    rpar_d = np.einsum("ij,ij->i", r, los[dst]) / np.maximum(d, 1e-12)
    egeo = np.column_stack([np.log(d / np.median(d)), rpar_s, rpar_d,
                            np.sqrt(np.clip(1 - rpar_s**2, 0, 1))])
    print(f"nodes={len(X)}, directed edges={len(src)}, egeo dims={egeo.shape[1]}")

    mu, sd = eig[train].mean(0), eig[train].std(0)
    Y = (eig - mu) / sd
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    h, yt = t(X), t(Y)
    srct, dstt = t(src, torch.long), t(dst, torch.long)
    eg = t(egeo)
    trm, vam, tem = t(train, torch.bool), t(val, torch.bool), t(test, torch.bool)

    model = EGNNLite(X.shape[1], egeo.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        out = model(h, srct, dstt, eg)
        loss = ((out[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vout = model(h, srct, dstt, eg)
                vloss = float(((vout[vam] - yt[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 250 == 0:
                print(f"step {step:5d}  train {float(loss):.4f}  val {vloss:.4f}  "
                      f"({time.time()-t0:.0f}s)")
            if patience >= 12:
                print(f"early stop at step {step}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(h, srct, dstt, eg).cpu().numpy() * sd + mu
    ti = np.where(test)[0]
    print(f"\n{'':10s}  {'EGNN-lite R2':>12s}   (baseline GraphNet posterior-mean: l1 0.774, l2 0.810, l3 0.891)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        print(f"{nm:10s}  {r2_score(eig[ti,k], pred[ti,k]):12.3f}")
    clu = eig[ti, 0] > 0.2
    print(f"cluster-slice lambda1 Spearman: {spearmanr(eig[ti,0][clu], pred[ti,0][clu]).statistic:+.2f} "
          f"(baseline 0.54; n={clu.sum()})")
    print("\nCaveats: point-estimate MSE head (R2-favoured) vs posterior mean; invariant "
          "scalarization (not steerable). GATE: proceed to full e3nn build if lambda1 R2 "
          ">= ~0.75 here.")


if __name__ == "__main__":
    main()
