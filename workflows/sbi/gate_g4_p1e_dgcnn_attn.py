#!/usr/bin/env python3
"""Gate G4-PROPER run E — ATTENTIONAL DGCNN: dynamic feature-space kNN + attention.

The one candidate in the §5A gamut whose graph is genuinely NOT fixed: every layer
recomputes a kNN graph in LEARNED FEATURE SPACE (Wang et al. 2019 DGCNN), so the
receptive field is selected by the network during training rather than by any
fixed rule of the coordinates. Per JDPK (2026-07-04): EdgeConv's max-pool
aggregation is replaced by GAT/GAPNet-style ATTENTION (invariant logits), so the
aggregation axis matches every other wave-1 run and E isolates the
candidate-selection axis alone:

    E - D  =  learned dynamic candidates  vs  fixed physical (radius) candidates,
              at matched inputs (positions+LOS only) and matched aggregation
              (multi-head attention).

Design notes:
- Layer 0 kNN is in COORDINATE space (canonical DGCNN); layers >0 in feature
  space (torch.no_grad selection — gradients flow through selected edges only).
- Node inputs [1, |pos|/median]; per-edge geometry = the same invariant scalars
  as run D (log|r|, LOS-parallel splits, transverse). All learned features are
  functions of observer-rotation invariants, so feature-space kNN is itself
  rotation-invariant — avoiding DGCNN's usual symmetry breakage from raw
  coordinates in feature space.
- Supervision: 3 sorted eigenvalues, per-component standardised (non-equivariant
  net -> NO fixed-frame tensor head; logged policy).
- Honest caveat (plan §5A): attention fixes the WEIGHTING of candidates, not the
  CANDIDATE SET — with fixed k, feature-similar distant nodes can evict physical
  neighbours entirely, and attention cannot attend to an absent edge. Layer-0
  coordinate kNN guarantees physical locality once; whether learned selection
  helps or hurts after that is exactly what E measures.
"""
from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


@torch.no_grad()
def knn_chunked(x: torch.Tensor, k: int, chunk: int = 4096,
                pos: torch.Tensor = None, radius_cap: float = None) -> torch.Tensor:
    """Row-chunked brute-force kNN (excl. self) — [N,F] -> [N,k] indices.

    If radius_cap is set, the feature-space kNN is restricted to candidates
    within `radius_cap` in PHYSICAL space (pos): 'learned selection within a
    physical envelope'. This is the graph-construction knob that directly
    addresses why run E (uncapped) lost — it keeps the adaptive candidate
    selection but forbids the non-local roaming. (A node with < k physical
    neighbours falls back to its nearest ones, so voids degrade gracefully.)
    """
    n = x.shape[0]
    out = torch.empty(n, k, dtype=torch.long, device=x.device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        rows = torch.arange(hi - lo, device=x.device)
        cols = torch.arange(lo, hi, device=x.device)
        d = torch.cdist(x[lo:hi], x)
        d[rows, cols] = torch.inf                        # exclude self
        if radius_cap is not None and pos is not None:
            ds = torch.cdist(pos[lo:hi], pos)
            far = ds > radius_cap
            far[rows, cols] = True
            # keep row usable if it would otherwise be all-inf (deep void node):
            # fall back to physical-kNN by using the spatial distance there
            allfar = far.all(dim=1)
            d = d.masked_fill(far, torch.inf)
            if allfar.any():
                d[allfar] = ds[allfar]                    # physical fallback
        out[lo:hi] = d.topk(k, largest=False).indices
    return out


def edge_geo(pos, los, i, j):
    """Invariant edge scalars for arbitrary (i<-j) pairs — mirrors run D's egeo."""
    r = pos[j] - pos[i]
    d = r.norm(dim=1)
    dn = d.clamp(min=1e-12)
    rpar_i = (r * los[i]).sum(1) / dn
    rpar_j = (r * los[j]).sum(1) / dn
    med = d.median().clamp(min=1e-12)
    return torch.stack([torch.log(dn / med), rpar_i, rpar_j,
                        (1 - rpar_i ** 2).clamp(0, 1).sqrt()], dim=1)


class DynAttnLayer(nn.Module):
    def __init__(self, dim: int, geo_dim: int = 4, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.msg = nn.Sequential(nn.Linear(2 * dim + geo_dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim), nn.SiLU())
        self.att = nn.Sequential(nn.Linear(2 * dim + geo_dim, 32), nn.SiLU(),
                                 nn.Linear(32, heads))
        self.upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))

    def forward(self, h, pos, los, idx):
        n, k = idx.shape
        i = torch.arange(n, device=h.device).repeat_interleave(k)
        j = idx.reshape(-1)
        geo = edge_geo(pos, los, i, j)
        hi, hj = h[i], h[j]
        m = self.msg(torch.cat([hi, hj - hi, geo], dim=1))         # [N*k, dim]
        logits = self.att(torch.cat([hi, hj, geo], dim=1))         # [N*k, H]
        a = torch.softmax(logits.view(n, k, self.heads), dim=1)    # over k nbrs
        m = m.view(n, k, self.heads, -1)                           # [N,k,H,dim/H]
        agg = (a.unsqueeze(-1) * m).sum(dim=1).reshape(n, -1)      # [N, dim]
        return h + self.upd(torch.cat([h, agg], dim=1))


class AttnDGCNN(nn.Module):
    def __init__(self, nfeat, dim=128, layers=4, heads=4, k=20, radius_cap=None):
        super().__init__()
        self.k = k
        self.radius_cap = radius_cap
        self.embed = nn.Linear(nfeat, dim)
        self.layers = nn.ModuleList(DynAttnLayer(dim, heads=heads)
                                    for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, x, pos, los, idx0):
        h = self.embed(x)
        for li, layer in enumerate(self.layers):
            idx = idx0 if li == 0 else knn_chunked(
                h.detach(), self.k, pos=pos, radius_cap=self.radius_cap)
            h = layer(h, pos, los, idx)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="SI cache pkl — eigenvalue targets + splits ONLY")
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=200.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--curated-features", action="store_true",
                    help="use the curated (Delaunay/cuGraph-derived) node features "
                         "from the cache instead of positions-only [1,|pos|/median]. "
                         "Run F: dynamic feature-space graph WITH curated features.")
    ap.add_argument("--knn-radius-cap", type=float, default=None,
                    help="restrict the dynamic feature-space kNN to candidates "
                         "within this physical radius (Mpc). None = uncapped "
                         "(canonical DGCNN, = run E).")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    print(f"device: {dev}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    pos_np = np.load(args.points_xyz).astype(np.float64)
    if args.curated_features:
        X = np.asarray(cache["graph"].nodes, np.float64)   # Delaunay/cuGraph feats
        feat_desc = f"curated Delaunay features ({X.shape[1]} cols)"
    else:
        rad = np.linalg.norm(pos_np, axis=1)
        X = np.column_stack([np.ones(len(pos_np)), rad / np.median(rad)])
        feat_desc = "positions-only [1, |pos|/median]"
    cap_desc = ("uncapped (canonical DGCNN)" if args.knn_radius_cap is None
                else f"physical-cap {args.knn_radius_cap:.2f} Mpc")
    print(f"inputs: {feat_desc}; nodes={len(X)}; dynamic kNN k={args.k}, "
          f"{cap_desc} (layer 0 = coordinate kNN, layers >0 = feature space)")

    mu, sd = eig[train].mean(0), eig[train].std(0)
    Y = (eig - mu) / sd
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    x, yt = t(X), t(Y)
    pos = t(pos_np)
    los = pos / pos.norm(dim=1, keepdim=True)
    trm, vam = t(train, torch.bool), t(val, torch.bool)

    idx0 = knn_chunked(pos, args.k)          # coordinate kNN, fixed -> once
    model = AttnDGCNN(X.shape[1], dim=args.dim, layers=args.layers,
                      heads=args.heads, k=args.k,
                      radius_cap=args.knn_radius_cap).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: dim={args.dim}, layers={args.layers}, heads={args.heads}, "
          f"k={args.k}, params={n_par:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for step in range(args.steps):
        if time.time() - t0 > args.minutes * 60:
            print(f"wall-clock budget reached at step {step}")
            break
        model.train()
        opt.zero_grad()
        out = model(x, pos, los, idx0)
        loss = ((out[trm] - yt[trm]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if step % 25 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vout = model(x, pos, los, idx0)
                vloss = float(((vout[vam] - yt[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {kk: v.detach().clone()
                              for kk, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 250 == 0:
                print(f"step {step:5d}  train {float(loss):.4f}  val {vloss:.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            if patience >= 12:
                print(f"early stop at step {step}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, pos, los, idx0).cpu().numpy() * sd + mu

    ti = np.where(test)[0]
    lines = [f"G4-PROPER attentional DGCNN (dynamic feature-space kNN, k={args.k})",
             f"inputs={feat_desc}; kNN={cap_desc}",
             f"params={n_par:,}  best_val={best_val:.4f}",
             "anchors: Delaunay baseline l1 0.774 | union@3749 0.804 | E(pos,uncapped) 0.507", ""]
    print(f"\n{'':10s}  {'E R2':>10s}   (anchors: baseline 0.774; union@3749 0.804)")
    for kk, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = r2_score(eig[ti, kk], pred[ti, kk])
        print(f"{nm:10s}  {r2:10.3f}")
        lines.append(f"{nm}: R2={r2:.4f}")
    clu = eig[ti, 0] > 0.2
    sp = spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic
    print(f"cluster-slice lambda1 Spearman: {sp:+.2f} (baseline 0.54; n={clu.sum()})")
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    lines.append("caveat: point-estimate MSE head (R2-favoured) vs posterior mean; "
                 "E-D isolates dynamic candidates vs fixed radius candidates")
    (args.out_dir / "p1e_dgcnn_attn_results.txt").write_text("\n".join(lines) + "\n")
    torch.save({"state_dict": best_state, "args": vars(args),
                "mu": mu, "sd": sd, "params": n_par},
               args.out_dir / "p1e_dgcnn_attn_model.pt")
    np.savez(args.out_dir / "p1e_dgcnn_attn_pred.npz", pred=pred, test_idx=ti)
    print(f"saved results/model/preds to {args.out_dir}")


if __name__ == "__main__":
    main()
