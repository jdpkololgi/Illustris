#!/usr/bin/env python3
"""F-tier v2 (graph->field->Poisson) — upgraded components over gate_t4 v1.

v2 upgrades (docs/plan_field_level_multimodal.md §12): union graph, attention
aggregation encoder, union edge_attr (point-cloud geometry) features, TSC scatter,
optional apodized survey-mask channel, and a U-Net OR FNO field decoder. The FFT
physics layer + analytic Cardano eigensolver are REUSED unchanged from v1
(validated at voxel R^2>=0.992). Ordering + eigenvectors are free (symmetric tensor).

Runs on the RAW wedge (union arrays + edge_attr precomputed there; F-tier is
density-robust per P2, nzharm 0.838 ~ raw 0.840). Baseline to beat: v1 raw 0.840.

  A: --scatter tsc --decoder unet
  B: --scatter tsc --decoder fno --survey-mask
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# reuse validated physics + eigensolver + U-Net from v1
from gate_t4_graph_field_poisson import PhysicsLayer, eigvalsh3x3, UNet3D, RSMOOTH_MPC

RA_MIN, RA_MAX, DEC_MIN, DEC_MAX = 120.0, 160.0, 14.5, 30.6


# ---------------------------------------------------------------- attention encoder
class EGNNAttnEncoder(nn.Module):
    """Invariant-message GN with multi-head ATTENTION aggregation (softmax over
    incoming edges, invariant logits). Returns per-node latents (width)."""

    def __init__(self, nfeat, negeo, width=64, layers=5, heads=4):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.embed = nn.Linear(nfeat, width)
        self.msg, self.upd, self.att = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                          nn.Linear(width, width)))
            self.att.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, heads)))

    def _seg_softmax(self, logits, dst, n):
        mx = torch.full((n, logits.shape[1]), -1e30, device=logits.device, dtype=logits.dtype)
        mx.scatter_reduce_(0, dst[:, None].expand(-1, logits.shape[1]), logits, reduce="amax",
                           include_self=True)
        w = torch.exp(logits - mx[dst])
        den = torch.zeros_like(mx).index_add_(0, dst, w)
        return w / den[dst].clamp(min=1e-12)

    def _layer(self, h, src, dst, egeo, msg, upd, att):
        n = h.shape[0]
        pair = torch.cat([h[src], h[dst], egeo], dim=1)
        m = msg(pair)
        alpha = self._seg_softmax(att(pair), dst, n)          # [E,H]
        E, Wd = m.shape
        mh = m.view(E, self.heads, Wd // self.heads) * alpha[:, :, None]
        agg = torch.zeros(n, self.heads, Wd // self.heads, device=h.device, dtype=m.dtype)
        agg.index_add_(0, dst, mh)
        return h + upd(torch.cat([h, agg.view(n, Wd)], dim=1))

    def forward(self, h, src, dst, egeo):
        h = self.embed(h)
        for msg, upd, att in zip(self.msg, self.upd, self.att):
            h = checkpoint(self._layer, h, src, dst, egeo, msg, upd, att, use_reentrant=False)
        return h


# ---------------------------------------------------------------- geometry (CIC/TSC + mask)
def _stencil_1d(w, scheme):
    """w:(N,) position in cell units. Return list of (int_index(N,), weight(N,))."""
    if scheme == "cic":
        base = np.floor(w - 0.5).astype(np.int64)
        out = []
        for off in (0, 1):
            i = base + off
            out.append((i, 1.0 - np.abs(w - (i + 0.5))))
        return out
    near = np.round(w - 0.5).astype(np.int64)               # nearest cell centre index
    out = []
    for off in (-1, 0, 1):
        i = near + off
        s = np.abs(w - (i + 0.5))
        wt = np.where(s <= 0.5, 0.75 - s * s, np.where(s <= 1.5, 0.5 * (1.5 - s) ** 2, 0.0))
        out.append((i, wt))
    return out


class Geometry:
    def __init__(self, pos, cell, pad, device, scheme="tsc"):
        self.cell = cell
        self.lo = pos.min(0) - pad
        hi = pos.max(0) + pad
        self.shape = tuple(int(np.ceil((hi - self.lo)[i] / cell)) for i in range(3))
        D, H, W = self.shape
        wc = (pos - self.lo) / cell                          # position in cell units
        sx, sy, sz = (_stencil_1d(wc[:, d], scheme) for d in range(3))
        idx_list, wgt_list = [], []
        for ix, wx in sx:
            for iy, wy in sy:
                for iz, wz in sz:
                    ii = np.clip(ix, 0, D - 1); jj = np.clip(iy, 0, H - 1); kk = np.clip(iz, 0, W - 1)
                    idx_list.append((ii * H + jj) * W + kk)
                    wgt_list.append(wx * wy * wz)
        self.flat = torch.tensor(np.stack(idx_list, 1), dtype=torch.long, device=device)   # (N,K)
        self.wgt = torch.tensor(np.stack(wgt_list, 1), dtype=torch.float32, device=device)  # (N,K)
        self.n = len(pos); self.numel = D * H * W; self.K = self.flat.shape[1]

    def scatter(self, latents):
        C = latents.shape[1]
        grid = torch.zeros(C, self.numel, device=latents.device, dtype=latents.dtype)
        for c in range(self.K):
            grid.index_add_(1, self.flat[:, c], (latents * self.wgt[:, c:c + 1]).T)
        return grid.reshape((C,) + self.shape)

    def counts(self, device, dtype):
        cnt = torch.zeros(self.numel, device=device, dtype=dtype)
        for c in range(self.K):
            cnt.index_add_(0, self.flat[:, c], self.wgt[:, c])
        return cnt.reshape(self.shape)

    def gather(self, field):
        flat = field.reshape(-1)
        vals = torch.zeros(self.n, device=field.device, dtype=field.dtype)
        for c in range(self.K):
            vals = vals + self.wgt[:, c] * flat[self.flat[:, c]]
        return vals

    def survey_mask(self, r_gal, apod_mpc=6.0):
        """Apodized indicator of the wedge footprint on the grid (see plan §12)."""
        D, H, W = self.shape
        ax = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * self.cell for i in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        rr = np.sqrt(gx**2 + gy**2 + gz**2)
        ra = np.degrees(np.arctan2(gy, gx)) % 360.0
        dec = np.degrees(np.arcsin(np.clip(gz / np.maximum(rr, 1e-9), -1, 1)))
        hard = ((ra >= RA_MIN) & (ra <= RA_MAX) & (dec >= DEC_MIN) & (dec <= DEC_MAX)
                & (rr >= np.quantile(r_gal, 0.001)) & (rr <= np.quantile(r_gal, 0.999)))
        return gaussian_filter(hard.astype(np.float32), sigma=apod_mpc / self.cell)


# ---------------------------------------------------------------- FNO decoder
class SpectralConv3d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.cout = cout; self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w = nn.ParameterList([
            nn.Parameter(scale * torch.rand(cin, cout, *modes, dtype=torch.cfloat)) for _ in range(4)])

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x):
        B, C, D, H, Wd = x.shape
        m1, m2, m3 = (min(self.modes[0], D), min(self.modes[1], H), min(self.modes[2], Wd // 2 + 1))
        xft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out = torch.zeros(B, self.cout, D, H, Wd // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2, :m3] = self._mul(xft[:, :, :m1, :m2, :m3], self.w[0][:, :, :m1, :m2, :m3])
        out[:, :, -m1:, :m2, :m3] = self._mul(xft[:, :, -m1:, :m2, :m3], self.w[1][:, :, :m1, :m2, :m3])
        out[:, :, :m1, -m2:, :m3] = self._mul(xft[:, :, :m1, -m2:, :m3], self.w[2][:, :, :m1, :m2, :m3])
        out[:, :, -m1:, -m2:, :m3] = self._mul(xft[:, :, -m1:, -m2:, :m3], self.w[3][:, :, :m1, :m2, :m3])
        return torch.fft.irfftn(out, s=(D, H, Wd), dim=[-3, -2, -1])


class FNO3d(nn.Module):
    def __init__(self, c_in, width=24, modes=(12, 12, 12), layers=4):
        super().__init__()
        self.lift = nn.Conv3d(c_in, width, 1)
        self.specs = nn.ModuleList([SpectralConv3d(width, width, modes) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv3d(width, width, 1), nn.SiLU(), nn.Conv3d(width, 1, 1))

    def forward(self, x):                                    # x [C,D,H,W]
        x = x.unsqueeze(0)
        x = self.lift(x)
        for s, w in zip(self.specs, self.ws):
            x = F.silu(s(x) + w(x))
        return self.proj(x)[0, 0]


# ---------------------------------------------------------------- model
class FTierV2(nn.Module):
    def __init__(self, nfeat, negeo, geom, phys, mask_ch, width=64, decoder="unet",
                 unet_base=16, fno_width=24, fno_modes=(12, 12, 12)):
        super().__init__()
        self.enc = EGNNAttnEncoder(nfeat, negeo, width=width)
        c_in = width + 1 + (1 if mask_ch is not None else 0)   # latents + counts (+ mask)
        self.decoder = decoder
        if decoder == "fno":
            self.dec = FNO3d(c_in, width=fno_width, modes=fno_modes)
        else:
            self.dec = UNet3D(c_in=c_in, base=unet_base)
        self.geom, self.phys, self.mask_ch = geom, phys, mask_ch
        self.log_amp = nn.Parameter(torch.zeros(()))

    def forward(self, h, src, dst, egeo, counts_ch):
        lat = self.enc(h, src, dst, egeo)
        grid = self.geom.scatter(lat)                          # (width,D,H,W)
        chans = [grid, counts_ch[None]]
        if self.mask_ch is not None:
            chans.append(self.mask_ch[None])
        x = torch.cat(chans, 0)                                # (C,D,H,W)
        # UNet3D (imported from v1) expects a batch dim (5D); FNO3d adds its own.
        delta = self.dec(x if self.decoder == "fno" else x[None]) * torch.exp(self.log_amp)
        comps = self.phys.components(delta)
        idx = {"x": 0, "y": 1, "z": 2}
        T = torch.zeros(self.geom.n, 3, 3, device=delta.device, dtype=delta.dtype)
        for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
            v = self.geom.gather(comps[a + b])
            T[:, idx[a], idx[b]] = v; T[:, idx[b], idx[a]] = v
        return eigvalsh3x3(T), delta


def zscore(a, eps=1e-6):
    m, s = a.mean(0), a.std(0)
    return (a - m) / np.maximum(s, eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--gnn-arrays", type=Path, required=True, help="union arrays npz (edge_index, edge_attr)")
    ap.add_argument("--scatter", choices=["cic", "tsc"], default="tsc")
    ap.add_argument("--decoder", choices=["unet", "fno"], default="unet")
    ap.add_argument("--survey-mask", action="store_true")
    ap.add_argument("--cell-mpc", type=float, default=6.0)
    ap.add_argument("--pad-mpc", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-file", type=Path, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    if not args.smoke:
        assert torch.cuda.is_available(), "F-tier v2 requires CUDA (fail-fast, no CPU-crawl)."
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  scatter={args.scatter} decoder={args.decoder} mask={args.survey_mask}")

    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    pos = np.load(args.points_xyz).astype(np.float64)
    arr = np.load(args.gnn_arrays)
    X = zscore(np.asarray(arr["x"], np.float64)) if "x" in arr else zscore(np.asarray(cache["graph"].nodes, np.float64))
    ei = arr["edge_index"].astype(np.int64)
    eattr = np.asarray(arr["edge_attr"], np.float64) if "edge_attr" in arr else None
    print(f"nodes={len(pos)} feats={X.shape[1]} edges={ei.shape[1]} eattr={None if eattr is None else eattr.shape}")

    src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])
    r = pos[dst] - pos[src]; d = np.linalg.norm(r, axis=1)
    los = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rps = np.einsum("ij,ij->i", r, los[src]) / np.maximum(d, 1e-12)
    rpd = np.einsum("ij,ij->i", r, los[dst]) / np.maximum(d, 1e-12)
    egeo = [np.log(d / np.median(d)), rps, rpd, np.sqrt(np.clip(1 - rps**2, 0, 1))]
    if eattr is not None:
        ea = np.concatenate([eattr, eattr], axis=0)            # both directions
        ea[:, 0] = np.log1p(ea[:, 0]); ea[:, -1] = np.log1p(np.abs(ea[:, -1]))  # tame big-range cols
        egeo.append(zscore(ea))
    egeo = np.column_stack([e if e.ndim > 1 else e[:, None] for e in egeo])
    print(f"edge features negeo={egeo.shape[1]}")

    r_gal = np.linalg.norm(pos, axis=1)
    geom = Geometry(pos, args.cell_mpc, args.pad_mpc, dev, scheme=args.scatter)
    print(f"grid {geom.shape} = {geom.numel/1e6:.1f}M cells, K={geom.K}")
    phys = PhysicsLayer(geom.shape, args.cell_mpc, RSMOOTH_MPC, dev)
    mask_ch = None
    if args.survey_mask:
        mk = geom.survey_mask(r_gal)
        mask_ch = torch.tensor(mk, dtype=torch.float32, device=dev)
        print(f"survey mask channel: mean={float(mk.mean()):.3f}")

    mu, sd = eig[train].mean(0), eig[train].std(0)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    h, eg = t(X), t(egeo)
    srct, dstt = t(src, torch.long), t(dst, torch.long)
    yt = t((eig - mu) / sd); trm, vam = t(train, torch.bool), t(val, torch.bool)
    counts_ch = geom.counts(dev, torch.float32); counts_ch = counts_ch / counts_ch.mean().clamp(min=1e-6)
    mu_t, sd_t = t(mu), t(sd)

    model = FTierV2(X.shape[1], egeo.shape[1], geom, phys, mask_ch, width=args.width,
                    decoder=args.decoder).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k")

    if args.smoke:
        lam, delta = model(h, srct, dstt, eg, counts_ch)
        loss = ((((lam - mu_t) / sd_t)[trm] - yt[trm]) ** 2).mean(); loss.backward()
        gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        print(f"SMOKE ok: lam {tuple(lam.shape)} delta {tuple(delta.shape)} loss {float(loss):.3f} "
              f"grad {gn:.1f} asc {float((lam[:,0]<=lam[:,1]).float().mean()):.3f}")
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best, best_state, pat, t0 = np.inf, None, 0, time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        lam, _ = model(h, srct, dstt, eg, counts_ch)
        loss = ((((lam - mu_t) / sd_t)[trm] - yt[trm]) ** 2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                lam, _ = model(h, srct, dstt, eg, counts_ch)
                vl = float(((((lam - mu_t) / sd_t)[vam] - yt[vam]) ** 2).mean())
            if vl < best - 1e-4:
                best, pat = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if step % 250 == 0:
                print(f"step {step:5d} train {float(loss):.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if pat >= 15:
                print(f"early stop {step}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred, _ = model(h, srct, dstt, eg, counts_ch); pred = pred.cpu().numpy()
    ti = np.where(test)[0]; lines = []
    print(f"\n{'':10s} {'v2 R2':>8s}  (v1 raw 0.840/0.897/0.930; CNN 0.876; G3 0.804)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = r2_score(eig[ti, k], pred[ti, k]); lines.append(f"{nm}: R2={r2:.4f}")
        print(f"{nm:10s} {r2:8.3f}")
    clu = eig[ti, 0] > 0.2
    sp = float(spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic)
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    print(f"cluster-slice lambda1 Spearman: {sp:+.2f} (n={int(clu.sum())})")
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(f"gate_ftier_v2 scatter={args.scatter} decoder={args.decoder} "
                                 f"mask={args.survey_mask} seed={args.seed}\n" + "\n".join(lines) + "\n")
        print(f"summary: {args.out_file}")


if __name__ == "__main__":
    main()
