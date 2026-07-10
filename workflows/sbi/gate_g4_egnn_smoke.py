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
    """Invariant-message GN block. aggregation='mean' (part 1) or 'attention'
    (part 2): multi-head softmax over incoming edges with INVARIANT logits from
    [h_src, h_dst, egeo] — the single-variable change vs part 1."""

    def __init__(self, nfeat, negeo, width=96, layers=5, aggregation="mean", heads=4):
        super().__init__()
        assert width % heads == 0
        self.aggregation, self.heads = aggregation, heads
        self.embed = nn.Linear(nfeat, width)
        self.msg = nn.ModuleList()
        self.upd = nn.ModuleList()
        self.att = nn.ModuleList()
        for _ in range(layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                          nn.Linear(width, width)))
            self.att.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, heads)))
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def _segment_softmax(self, logits, dst, n):
        mx = torch.full((n, logits.shape[1]), -1e30, device=logits.device, dtype=logits.dtype)
        mx.scatter_reduce_(0, dst[:, None].expand(-1, logits.shape[1]), logits,
                           reduce="amax", include_self=True)
        w = torch.exp(logits - mx[dst])
        den = torch.zeros_like(mx).index_add_(0, dst, w)
        return w / den[dst].clamp(min=1e-12)

    def _layer(self, h, src, dst, egeo, msg, upd, att):
        n = h.shape[0]
        pair = torch.cat([h[src], h[dst], egeo], dim=1)
        m = msg(pair)
        if self.aggregation == "attention":
            alpha = self._segment_softmax(att(pair), dst, n)          # [E, H]
            E, W = m.shape
            mh = m.view(E, self.heads, W // self.heads) * alpha[:, :, None]
            agg = torch.zeros(n, self.heads, W // self.heads, device=h.device, dtype=m.dtype)
            agg.index_add_(0, dst, mh)
            agg = agg.view(n, W)
        else:
            agg = torch.zeros(n, m.shape[1], device=h.device, dtype=m.dtype)
            cnt = torch.zeros(n, 1, device=h.device, dtype=m.dtype)
            agg.index_add_(0, dst, m)
            cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device, dtype=m.dtype))
            agg = agg / cnt.clamp(min=1)
        return h + upd(torch.cat([h, agg], dim=1))

    def forward(self, h, src, dst, egeo):
        h = self.embed(h)
        for msg, upd, att in zip(self.msg, self.upd, self.att):
            # gradient checkpointing: recompute each layer's edge tensors in backward
            # instead of holding ~1.5M-edge activations for all layers (OOM otherwise)
            h = checkpoint(self._layer, h, src, dst, egeo, msg, upd, att, use_reentrant=False)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--gnn-arrays", type=Path, default=None,
                    help="wedge gnn_arrays npz (edge pairs); omit with --build-radius-mpc")
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggregation", choices=["mean", "attention"], default="mean")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--positions-only", action="store_true",
                    help="G4-PROPER P1a-ii (point-cloud control): DROP the curated "
                         "cache node features; nodes get [1, |pos|/median] only "
                         "(observational scalars); all other information enters "
                         "through the geometric edge scalars. Cache is used for "
                         "eigenvalue targets + splits ONLY.")
    ap.add_argument("--build-radius-mpc", type=float, default=None,
                    help="build the neighbourhood at LOAD TIME from positions "
                         "(cKDTree radius pairs) instead of reading a prebuilt "
                         "edge_index — the model consumes the catalogue purely as "
                         "a spatial point distribution.")
    ap.add_argument("--out-file", type=Path, default=None,
                    help="also write the R2 summary to this file (used by the "
                         "unattended g4 chain for completion detection).")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    # Fail-fast on GPU: an intermittent srun gres-binding miss can let a guard
    # subprocess see cuda=True while THIS process's own check returns False,
    # silently CPU-crawling for the full run. Force cuda and assert.
    dev = "cuda"
    assert torch.cuda.is_available(), (
        "gate_g4_egnn_smoke requires CUDA but torch.cuda.is_available() is "
        "False in this process — GPU not bound; abort instead of CPU-crawling.")
    print(f"device: {dev}  (torch.cuda.is_available()={torch.cuda.is_available()})")

    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])

    pos = np.load(args.points_xyz).astype(np.float64)
    if args.positions_only:
        rad = np.linalg.norm(pos, axis=1)
        X = np.column_stack([np.ones(len(pos)), rad / np.median(rad)])
        print("positions-only mode: node features = [1, |pos|/median] "
              "(curated features EXCLUDED)")
    else:
        X = np.asarray(cache["graph"].nodes, np.float64)     # identical inputs to baseline

    if args.build_radius_mpc is not None:
        from scipy.spatial import cKDTree
        pairs = cKDTree(pos).query_pairs(args.build_radius_mpc,
                                         output_type="ndarray")
        ei = pairs.T.astype(np.int64)
        print(f"load-time radius({args.build_radius_mpc:.2f} Mpc) graph: "
              f"{ei.shape[1]} undirected pairs (no prebuilt edge_index)")
    else:
        assert args.gnn_arrays is not None, "--gnn-arrays or --build-radius-mpc required"
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

    print(f"aggregation: {args.aggregation} (heads={args.heads})")
    model = EGNNLite(X.shape[1], egeo.shape[1], aggregation=args.aggregation,
                     heads=args.heads).to(dev)
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
    lines = []
    print(f"\n{'':10s}  {'EGNN-lite R2':>12s}   (baseline GraphNet posterior-mean: l1 0.774, l2 0.810, l3 0.891)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = r2_score(eig[ti, k], pred[ti, k])
        print(f"{nm:10s}  {r2:12.3f}")
        lines.append(f"{nm}: R2={r2:.4f}")
    clu = eig[ti, 0] > 0.2
    sp = spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic
    print(f"cluster-slice lambda1 Spearman: {sp:+.2f} "
          f"(baseline 0.54; n={clu.sum()})")
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        hdr = (f"gate_g4_egnn_smoke aggregation={args.aggregation} "
               f"positions_only={args.positions_only} "
               f"build_radius_mpc={args.build_radius_mpc} seed={args.seed}")
        args.out_file.write_text(hdr + "\n" + "\n".join(lines) + "\n")
        print(f"summary written: {args.out_file}")
    print("\nCaveats: point-estimate MSE head (R2-favoured) vs posterior mean; invariant "
          "scalarization (not steerable). GATE: proceed to full e3nn build if lambda1 R2 "
          ">= ~0.75 here.")


if __name__ == "__main__":
    main()
