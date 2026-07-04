#!/usr/bin/env python3
"""Gate G4-PROPER P1b — SEGNN-style steerable MPNN with invariant-logit attention.

The FIRST equivariant candidate of the pre-registered bake-off
(docs/plan_g4_proper_equivariant_tensor.md §5A): steerable messages (e3nn tensor
products with spherical harmonics of the edge direction), REQUIRED attention with
invariant logits (segment softmax; SE(3)-Transformer construction, so equivariance
is exact), and a 1x0e+1x2e tensor head whose differentiable diagonalisation
(eigvalsh) is supervised on the EXISTING rotation-invariant eigenvalues (Tier A —
no tweb changes, no frame rotation).

Scope discipline (§0): inputs are positions + LOS ONLY. No curated node features,
no curated edge_attr (density_contrast excluded). The prebuilt npz supplies ONLY
edge_index (the neighbourhood definition — union or radius-only, per the 2x2
factorial); all geometry is recomputed from points_xyz.

Design invariants:
- Targets scaled by a SINGLE global affine (train-split mean/std pooled over all
  three eigenvalues). Per-component scaling would be tensor-INCONSISTENT: no one
  symmetric tensor has its sorted eigenvalues rescaled per sorted position.
- eigvalsh loss is eigenvalue-only: its backward is V diag(dL/dlam) V^T, which has
  no 1/(lam_i - lam_j) gap terms -> safe at near-degeneracy.
- P0 equivariance self-test runs FIRST (float64, random rotation; tensor must map
  as R T R^T, eigenvalues invariant to <1e-8). Training aborts if it fails.

Caveats printed with the result (as in gate_g4_egnn_smoke.py): point-estimate MSE
head is R^2-favoured vs the baseline's posterior mean. Comparison anchors: Delaunay
baseline 0.774 (l1), and the matched-budget graph-construction controls
(union@3749 = 0.804; P1a radius-only control from the same wave).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.nn import Gate

SH_IRREPS = o3.Irreps("1x0e+1x1o+1x2e")   # edge/LOS harmonics, dim 9
HEAD_OUT = o3.Irreps("1x0e+1x2e")         # trace + symmetric-traceless = 6 comps


def soft_one_hot(d: torch.Tensor, n_basis: int, r_max: float) -> torch.Tensor:
    """Gaussian radial basis on [0, r_max]; |r| beyond r_max clipped (long
    Delaunay void edges keep their true unit vector, only the basis saturates)."""
    centers = torch.linspace(0.0, r_max, n_basis, device=d.device, dtype=d.dtype)
    width = r_max / (n_basis - 1)
    return torch.exp(-0.5 * ((d.clamp(max=r_max)[:, None] - centers) / width) ** 2)


def segment_softmax(logits: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Numerically stable softmax over edges grouped by destination node."""
    m = torch.full((n_seg,) + logits.shape[1:], -torch.inf,
                   device=logits.device, dtype=logits.dtype)
    m = m.index_reduce(0, seg, logits, "amax", include_self=True)
    ex = torch.exp(logits - m[seg])
    den = torch.zeros_like(m).index_add_(0, seg, ex)
    return ex / den[seg].clamp(min=1e-30)


class SteerableAttnLayer(nn.Module):
    """One steerable message-passing layer with multi-head invariant-logit
    attention: per head, values are a tensor product h_src (x) Y(r_hat) with
    radial-MLP weights; logits are an MLP of INVARIANT scalars only
    (0e channels of h_src, h_dst, radial basis) -> exact equivariance."""

    def __init__(self, hidden: o3.Irreps, heads: int, n_basis: int,
                 att_dropout: float):
        super().__init__()
        self.heads = heads
        self.n_scalar = hidden[0].mul                      # leading 0e block dims
        assert hidden[0].ir == o3.Irrep("0e")
        # per-head value irreps: hidden multiplicities / heads
        head_irreps = o3.Irreps([(mul // heads, ir) for mul, ir in hidden])
        self.tps = nn.ModuleList()
        self.radials = nn.ModuleList()
        self.att_mlps = nn.ModuleList()
        for _ in range(heads):
            tp = o3.FullyConnectedTensorProduct(
                hidden, SH_IRREPS, head_irreps,
                shared_weights=False, internal_weights=False)
            self.tps.append(tp)
            self.radials.append(nn.Sequential(
                nn.Linear(n_basis, 64), nn.SiLU(), nn.Linear(64, tp.weight_numel)))
            self.att_mlps.append(nn.Sequential(
                nn.Linear(2 * self.n_scalar + n_basis, 64), nn.SiLU(),
                nn.Linear(64, 1)))
        cat_irreps = (head_irreps * heads).simplify()
        # gate: scalars(silu) + gates(sigmoid) gating the l>0 channels
        gated = o3.Irreps([(mul, ir) for mul, ir in hidden if ir.l > 0])
        n_gates = sum(mul for mul, _ in gated)
        self.gate = Gate(f"{hidden[0].mul}x0e", [torch.nn.functional.silu],
                         f"{n_gates}x0e", [torch.sigmoid], gated)
        self.lin_up = o3.Linear(cat_irreps, self.gate.irreps_in)
        self.att_dropout = att_dropout

    def _chunk_values(self, h, src_c, sh_c, rb_c, alphas_c):
        """Weighted steerable values for one edge chunk (all heads, concat).
        The E-chunk x weight_numel radial intermediates live only inside this
        function -> checkpointing it bounds peak memory."""
        outs = []
        for k in range(self.heads):
            v = self.tps[k](h[src_c], sh_c, self.radials[k](rb_c))
            outs.append(alphas_c[:, k:k + 1] * v)
        return torch.cat(outs, dim=1)                        # [Ec, sum head_dim]

    def forward(self, h, src, dst, sh, rb, training: bool,
                edge_chunk: int = 500_000):
        n = h.shape[0]
        s = h[:, :self.n_scalar]                             # invariant channels
        inv = torch.cat([s[src], s[dst], rb], dim=1)         # cheap: ~70 f/edge
        alphas = []
        for k in range(self.heads):
            a = segment_softmax(self.att_mlps[k](inv).squeeze(-1), dst, n)
            if training and self.att_dropout > 0:
                keep = (torch.rand_like(a) > self.att_dropout).to(a.dtype)
                a = a * keep / (1.0 - self.att_dropout)
            alphas.append(a)
        alphas = torch.stack(alphas, dim=1)                  # [E, heads]

        e = src.shape[0]
        dim = sum(self.tps[k].irreps_out.dim for k in range(self.heads))
        agg = torch.zeros(n, dim, device=h.device, dtype=h.dtype)
        for lo in range(0, e, edge_chunk):
            hi = min(lo + edge_chunk, e)
            if training:
                vw = torch.utils.checkpoint.checkpoint(
                    self._chunk_values, h, src[lo:hi], sh[lo:hi], rb[lo:hi],
                    alphas[lo:hi], use_reentrant=False)
            else:
                vw = self._chunk_values(h, src[lo:hi], sh[lo:hi], rb[lo:hi],
                                        alphas[lo:hi])
            agg = agg.index_add(0, dst[lo:hi], vw)
        return h + self.gate(self.lin_up(agg))


class SEGNNTensorNet(nn.Module):
    """positions+LOS -> steerable encoder -> 1x0e+1x2e -> symmetric 3x3 tensor."""

    def __init__(self, hidden="32x0e+16x1o+8x2e", layers=4, heads=4,
                 n_basis=8, r_max=30.0, att_dropout=0.1):
        super().__init__()
        self.hidden = o3.Irreps(hidden)
        self.r_max = r_max
        self.n_basis = n_basis
        self.embed = o3.Linear(SH_IRREPS, self.hidden)      # LOS harmonics in
        self.layers = nn.ModuleList(
            SteerableAttnLayer(self.hidden, heads, n_basis, att_dropout)
            for _ in range(layers))
        self.head = o3.Linear(self.hidden, HEAD_OUT)
        self.ct = CartesianTensor("ij=ji")                  # (0e+2e) <-> sym 3x3
        # cached change-of-basis (registered so .to(device)/.double() move it)
        self.register_buffer("rtp", self.ct.reduced_tensor_products().change_of_basis
                             if hasattr(self.ct, "reduced_tensor_products") else
                             torch.empty(0), persistent=False)

    def geometry(self, pos, src, dst):
        r = pos[dst] - pos[src]
        d = r.norm(dim=1)
        sh = o3.spherical_harmonics(SH_IRREPS, r / d.clamp(min=1e-12)[:, None],
                                    normalize=False, normalization="component")
        rb = soft_one_hot(d, self.n_basis, self.r_max)
        return sh, rb

    def forward(self, pos, los, src, dst, use_checkpoint=True):
        sh, rb = self.geometry(pos, src, dst)
        h0 = o3.spherical_harmonics(SH_IRREPS, los, normalize=False,
                                    normalization="component")
        h = self.embed(h0)
        for layer in self.layers:
            if use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, src, dst, sh, rb, self.training,
                    use_reentrant=False)
            else:
                h = layer(h, src, dst, sh, rb, self.training)
        irr = self.head(h)                                   # [N, 6] as 1x0e+1x2e
        T = self.ct.to_cartesian(irr)                        # [N, 3, 3] symmetric
        return T

    def eigenvalues(self, pos, los, src, dst, use_checkpoint=True):
        T = self.forward(pos, los, src, dst, use_checkpoint)
        return torch.linalg.eigvalsh(T)                      # ascending = l1<=l2<=l3


def p0_selftest(device):
    """Equivariance gate: rotate inputs+LOS -> tensor maps as R T R^T,
    eigenvalues invariant. float64, tiny random graph. Abort on failure."""
    torch.manual_seed(0)
    n = 256
    pos = torch.randn(n, 3, dtype=torch.float64, device=device) * 20 + \
        torch.tensor([80.0, 0, 0], dtype=torch.float64, device=device)
    los = pos / pos.norm(dim=1, keepdim=True)
    d2 = torch.cdist(pos, pos)
    src, dst = torch.where((d2 > 0) & (d2 < 15.0))
    model = SEGNNTensorNet(hidden="8x0e+4x1o+2x2e", layers=2, heads=2,
                           att_dropout=0.0).to(device).double()
    model.eval()
    with torch.no_grad():
        R = o3.rand_matrix(dtype=torch.float64, device=device)
        T1 = model(pos, los, src, dst, use_checkpoint=False)
        T2 = model(pos @ R.T, los @ R.T, src, dst, use_checkpoint=False)
        dev_t = (T2 - R @ T1 @ R.T).abs().max().item()
        e1 = torch.linalg.eigvalsh(T1)
        e2 = torch.linalg.eigvalsh(T2)
        dev_e = (e1 - e2).abs().max().item()
    print(f"[P0] equivariance: max|T(Rx) - R T(x) R^T| = {dev_t:.3e}, "
          f"max|dlam| = {dev_e:.3e}")
    assert dev_t < 1e-8 and dev_e < 1e-8, "P0 equivariance FAILED — aborting"
    print("[P0] PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="baseline SI cache pkl (eigenvalues_raw + masks ONLY)")
    ap.add_argument("--gnn-arrays", type=Path, required=True,
                    help="wedge gnn_arrays npz — edge_index ONLY (union or radius)")
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=195.0,
                    help="wall-clock training budget")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.08,
                    help="parity with baseline")
    ap.add_argument("--att-dropout", type=float, default=0.1)
    ap.add_argument("--hidden", default="32x0e+16x1o+8x2e")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    p0_selftest(dev)
    if args.selftest_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: eigenvalue targets + shared splits; geometry from positions ----
    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])

    pos_np = np.load(args.points_xyz).astype(np.float64)
    ei = np.load(args.gnn_arrays)["edge_index"].astype(np.int64)
    src_np = np.concatenate([ei[0], ei[1]])
    dst_np = np.concatenate([ei[1], ei[0]])                  # bidirectional
    print(f"nodes={len(pos_np)}, directed edges={len(src_np)} "
          f"(undirected pairs={ei.shape[1]}) from {args.gnn_arrays.name}")

    # SINGLE global affine over train-split eigenvalues (tensor-consistent)
    mu = float(eig[train].mean())
    sd = float(eig[train].std())
    Y = (eig - mu) / sd
    print(f"global target affine: mu={mu:.5f}, sd={sd:.5f}")

    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    pos = t(pos_np)
    los = pos / pos.norm(dim=1, keepdim=True)
    src, dst = t(src_np, torch.long), t(dst_np, torch.long)
    yt = t(Y)
    trm, vam = t(train, torch.bool), t(val, torch.bool)

    model = SEGNNTensorNet(hidden=args.hidden, layers=args.layers,
                           heads=args.heads, att_dropout=args.att_dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: hidden={args.hidden}, layers={args.layers}, heads={args.heads}, "
          f"params={n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    budget_s = args.minutes * 60.0
    step = 0
    sched_calibrated = False
    while step < args.max_steps and (time.time() - t0) < budget_s:
        model.train()
        opt.zero_grad()
        lam = model.eigenvalues(pos, los, src, dst)
        loss = ((lam[trm] - yt[trm]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if not sched_calibrated and step == 10:
            # re-fit the cosine horizon to the measured step time so the LR
            # actually anneals within the wall-clock budget
            step_s = (time.time() - t0) / 11.0
            est_total = max(200, int(budget_s / step_s))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=est_total, last_epoch=step)
            sched_calibrated = True
            print(f"[sched] ~{step_s:.2f}s/step -> cosine T_max={est_total}")
        if step % 25 == 0 or step == args.max_steps - 1:
            model.eval()
            with torch.no_grad():
                vlam = model.eigenvalues(pos, los, src, dst, use_checkpoint=False)
                vloss = float(((vlam[vam] - yt[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 100 == 0:
                print(f"step {step:5d}  train {float(loss):.4f}  val {vloss:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)",
                      flush=True)
            if patience >= 40:
                print(f"early stop at step {step}")
                break
        step += 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model.eigenvalues(pos, los, src, dst,
                                 use_checkpoint=False).cpu().numpy() * sd + mu

    ti = np.where(test)[0]
    tag = args.gnn_arrays.stem
    lines = [f"G4-PROPER P1b (SEGNN steerable + attention) — graph: {tag}",
             f"params={n_par:,}  steps={step}  best_val={best_val:.4f}",
             "anchors: Delaunay baseline l1 0.774 | union control@3749 0.804",
             ""]
    print(f"\n{'':10s}  {'P1b R2':>10s}   (anchors: baseline 0.774; "
          f"union control@3749 0.804)")
    r2s = {}
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2s[nm] = r2_score(eig[ti, k], pred[ti, k])
        print(f"{nm:10s}  {r2s[nm]:10.3f}")
        lines.append(f"{nm}: R2={r2s[nm]:.4f}")
    clu = eig[ti, 0] > 0.2
    sp = spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic
    print(f"cluster-slice lambda1 Spearman: {sp:+.2f} (baseline 0.54; n={clu.sum()})")
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    lines.append("caveat: point-estimate MSE head (R2-favoured) vs posterior mean")

    (args.out_dir / f"p1b_segnn_results_{tag}.txt").write_text("\n".join(lines) + "\n")
    torch.save({"state_dict": best_state, "args": vars(args) | {"mu": mu, "sd": sd},
                "r2": r2s, "params": n_par, "steps": step},
               args.out_dir / f"p1b_segnn_model_{tag}.pt")
    with open(args.out_dir / f"p1b_segnn_pred_{tag}.npz", "wb") as f:
        np.savez(f, pred=pred, test_idx=ti)
    print(f"saved results/model/preds to {args.out_dir}")
    print("\nGATE P1b: lambda1 R2 >= ~0.75 AND beats the P1a radius control "
          "beyond seed noise -> proceed to Equiformer-class second test.")


if __name__ == "__main__":
    main()
