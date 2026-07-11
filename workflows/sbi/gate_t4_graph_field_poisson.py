#!/usr/bin/env python3
"""Gate T4 / F1 — graph -> field -> Poisson (field-as-OUTPUT).

The centerpiece of `docs/plan_field_level_multimodal.md` (§3). Instead of
regressing eigenvalues pointwise, the graph model DECODES a scalar density
field delta_hat, and the tidal tensor is obtained by a FIXED, parameter-free,
differentiable FFT physics layer:

  galaxy graph --(EGNNLite encoder)--> per-node latents h_i
    --(differentiable CIC scatter)--> coarse padded grid (C+1 channels)
    --(small 3-D U-Net)--> delta_hat (1 channel, padded, apodized)
    --(torch physics layer)--> T_ij(k) = (k_i k_j/k^2) W_R(k) delta_hat_k
    --(differentiable trilinear gather at galaxies)--> T_ij per galaxy
    --(eigvalsh, float64)--> lambda_1<=2<=3

The ONLY learnable job is producing delta_hat; the density->tidal map is exact
mathematics (validated in classical_tidal_baseline.py --mode validate-solver at
voxel R^2 >= 0.992). Eigenvalues fall out of the physics; symmetry / trace=delta
/ rotational consistency are guaranteed by construction. Eigenvectors come free
(F4 science) with no e3nn irreps and no box->observer tensor rotation.

Eval convention is byte-identical to gate_g4_egnn_smoke.py: cache
`eigenvalues_raw` + `masks`, per-eigenvalue test R^2 + cluster-slice Spearman.
GATE (F1): lambda1 R^2 >= G3 (0.804) AND calibration >= current flow -> GO F2.

Torch; GPU if available, CPU fallback (smoke).
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

PI = np.pi
LITTLE_H = 0.6736
RSMOOTH_MPC = 7.0 / LITTLE_H  # target smoothing in the wedge's Mpc coordinates


# ------------------------------------------------------------------ encoder
class EGNNEncoder(nn.Module):
    """Invariant-message GN block (from gate_g4_egnn_smoke), returning per-node
    LATENTS (width) instead of a 3-eigenvalue head."""

    def __init__(self, nfeat, negeo, width=96, layers=5):
        super().__init__()
        self.embed = nn.Linear(nfeat, width)
        self.msg, self.upd = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                          nn.Linear(width, width)))

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
            h = checkpoint(self._layer, h, src, dst, egeo, msg, upd, use_reentrant=False)
        return h


# ------------------------------------------------------------------ field decoder
class UNet3D(nn.Module):
    """Small 3-D U-Net: (C_in) channels on the padded grid -> 1 channel delta_hat."""

    def __init__(self, c_in, base=16):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(co, co, 3, padding=1), nn.SiLU())
        self.e0 = blk(c_in, base)
        self.e1 = blk(base, base * 2)
        self.pool = nn.AvgPool3d(2)
        self.mid = blk(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.d1 = blk(base * 4, base * 2)
        self.d0 = blk(base * 3, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(self.pool(e0))
        m = self.mid(self.pool(e1))
        d1 = self.d1(torch.cat([self._match(self.up(m), e1), e1], 1))
        d0 = self.d0(torch.cat([self._match(self.up(d1), e0), e0], 1))
        return self.out(d0)[0, 0]  # (D,H,W)

    @staticmethod
    def _match(a, ref):
        # resize upsampled tensor to EXACTLY ref spatial size (odd dims after
        # pooling can make the upsample off by one in either direction)
        if a.shape[2:] == ref.shape[2:]:
            return a
        return nn.functional.interpolate(a, size=ref.shape[2:], mode="trilinear",
                                         align_corners=False)


# ------------------------------------------------------------------ grid + interp geometry
class Geometry:
    """Padded Cartesian grid; precomputed CIC corner indices/weights for the
    galaxy positions (used for BOTH the scatter and the gather, differentiably)."""

    def __init__(self, pos, cell, pad, device):
        self.cell = cell
        self.lo = pos.min(0) - pad
        hi = pos.max(0) + pad
        self.shape = tuple(int(np.ceil((hi - self.lo)[i] / cell)) for i in range(3))
        D, H, W = self.shape
        u = (pos - self.lo) / cell - 0.5
        i0 = np.floor(u).astype(np.int64)
        f = (u - i0).astype(np.float32)
        idx8, w8 = [], []
        for dx in (0, 1):
            wx = (1 - f[:, 0]) if dx == 0 else f[:, 0]
            for dy in (0, 1):
                wy = (1 - f[:, 1]) if dy == 0 else f[:, 1]
                for dz in (0, 1):
                    wz = (1 - f[:, 2]) if dz == 0 else f[:, 2]
                    ii = np.clip(i0[:, 0] + dx, 0, D - 1)
                    jj = np.clip(i0[:, 1] + dy, 0, H - 1)
                    kk = np.clip(i0[:, 2] + dz, 0, W - 1)
                    idx8.append((ii * H + jj) * W + kk)   # flat index into (D,H,W)
                    w8.append(wx * wy * wz)
        self.flat = torch.tensor(np.stack(idx8, 1), dtype=torch.long, device=device)   # (N,8)
        self.wgt = torch.tensor(np.stack(w8, 1), dtype=torch.float32, device=device)   # (N,8)
        self.n = len(pos)
        self.numel = D * H * W

    def scatter(self, latents):
        """(N,C) node latents -> (C, D,H,W) grid via CIC deposit (differentiable)."""
        C = latents.shape[1]
        grid = torch.zeros(C, self.numel, device=latents.device, dtype=latents.dtype)
        for c in range(8):
            grid.index_add_(1, self.flat[:, c], (latents * self.wgt[:, c:c + 1]).T)
        return grid.reshape((C,) + self.shape)

    def counts(self, device, dtype):
        cnt = torch.zeros(self.numel, device=device, dtype=dtype)
        for c in range(8):
            cnt.index_add_(0, self.flat[:, c], self.wgt[:, c])
        return cnt.reshape(self.shape)

    def gather(self, field):
        """(D,H,W) field -> (N,) values at galaxy positions (differentiable)."""
        flat = field.reshape(-1)
        vals = torch.zeros(self.n, device=field.device, dtype=field.dtype)
        for c in range(8):
            vals = vals + self.wgt[:, c] * flat[self.flat[:, c]]
        return vals


def eigvalsh3x3(T, eps=1e-9):
    """Analytic ascending eigenvalues of a batch of symmetric 3x3 matrices.

    Trigonometric (Cardano) method — fully differentiable, and avoids the
    cuSOLVER batched-eigvalsh workspace blowup on GPU (documented gotcha). A
    small eps keeps gradients finite at near-degeneracy.
    """
    a00, a11, a22 = T[:, 0, 0], T[:, 1, 1], T[:, 2, 2]
    a01, a02, a12 = T[:, 0, 1], T[:, 0, 2], T[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01**2 + a02**2 + a12**2
    p2 = (a00 - q)**2 + (a11 - q)**2 + (a22 - q)**2 + 2.0 * p1
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=eps))
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = torch.clamp(detB / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    phi = torch.acos(r) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)                       # largest
    e3 = q + 2.0 * p * torch.cos(phi + 2.0 * PI / 3.0)      # smallest
    e2 = 3.0 * q - e1 - e3
    return torch.stack([e3, e2, e1], dim=1)                 # ascending


# ------------------------------------------------------------------ physics layer (torch, exact)
class PhysicsLayer:
    """T_ij(k) = (k_i k_j / k^2) exp(-0.5 (kR)^2) delta_hat_k. Fixed, no params."""

    def __init__(self, shape, cell, rsmooth, device):
        kx = torch.fft.fftfreq(shape[0], d=cell, device=device) * 2 * PI
        ky = torch.fft.fftfreq(shape[1], d=cell, device=device) * 2 * PI
        kz = torch.fft.rfftfreq(shape[2], d=cell, device=device) * 2 * PI
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX**2 + KY**2 + KZ**2
        k2[0, 0, 0] = 1.0
        sm = torch.exp(-0.5 * k2 * rsmooth**2)
        sm[0, 0, 0] = 0.0
        self.shape = shape
        self.kern = sm / k2
        self.K = {"x": KX, "y": KY, "z": KZ}

    def components(self, delta):
        dk = torch.fft.rfftn(delta) * self.kern
        out = {}
        for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
            out[a + b] = torch.fft.irfftn(self.K[a] * self.K[b] * dk, s=self.shape)
        return out


# ------------------------------------------------------------------ full model
class GraphFieldPoisson(nn.Module):
    def __init__(self, nfeat, negeo, geom: Geometry, phys: PhysicsLayer, width=96, unet_base=16):
        super().__init__()
        self.enc = EGNNEncoder(nfeat, negeo, width=width)
        self.unet = UNet3D(c_in=width + 1, base=unet_base)  # +1 counts channel
        self.geom, self.phys = geom, phys
        self.log_amp = nn.Parameter(torch.zeros(()))  # global amplitude freedom

    def forward(self, h, src, dst, egeo, counts_ch):
        lat = self.enc(h, src, dst, egeo)                     # (N, width)
        grid = self.geom.scatter(lat)                          # (width, D,H,W)
        x = torch.cat([grid, counts_ch[None]], 0)[None]        # (1, width+1, D,H,W)
        delta = self.unet(x) * torch.exp(self.log_amp)         # (D,H,W)
        comps = self.phys.components(delta)
        idx = {"x": 0, "y": 1, "z": 2}
        T = torch.zeros(self.geom.n, 3, 3, device=delta.device, dtype=delta.dtype)
        for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
            v = self.geom.gather(comps[a + b])
            T[:, idx[a], idx[b]] = v
            T[:, idx[b], idx[a]] = v
        lam = eigvalsh3x3(T)                                   # ascending, analytic (no cuSOLVER)
        return lam, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--gnn-arrays", type=Path, default=None)
    ap.add_argument("--build-radius-mpc", type=float, default=10.0)
    ap.add_argument("--cell-mpc", type=float, default=4.0)
    ap.add_argument("--pad-mpc", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--unet-base", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="few steps, assert shapes+backward")
    ap.add_argument("--out-file", type=Path, default=None)
    ap.add_argument("--save-embeddings", type=Path, default=None,
                    help="opt-in: after training, dump the encoder's per-node invariant "
                         "latent h_i (shape [N,width]) + masks/eig to an npz for a flow head.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Fail-fast on GPU: an intermittent srun gres-binding miss can let the guard
    # subprocess see cuda=True while THIS process's own check returns False,
    # silently CPU-crawling for the full run (bit T3 and T4-seed43). Force cuda
    # and assert rather than fall back.
    dev = "cuda"
    assert torch.cuda.is_available(), (
        "gate_t4 requires CUDA but torch.cuda.is_available() is False in this "
        "process — GPU not bound to this step; abort instead of CPU-crawling.")
    print(f"device: {dev}  (torch.cuda.is_available()={torch.cuda.is_available()})")

    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    X = np.asarray(cache["graph"].nodes, np.float64)
    pos = np.load(args.points_xyz).astype(np.float64)
    print(f"nodes={len(pos)}, node-feats={X.shape[1]}")

    if args.gnn_arrays is not None:
        ei = np.load(args.gnn_arrays)["edge_index"].astype(np.int64)
    else:
        from scipy.spatial import cKDTree
        ei = cKDTree(pos).query_pairs(args.build_radius_mpc, output_type="ndarray").T.astype(np.int64)
    src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])
    r = pos[dst] - pos[src]; d = np.linalg.norm(r, axis=1)
    los = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rpar_s = np.einsum("ij,ij->i", r, los[src]) / np.maximum(d, 1e-12)
    rpar_d = np.einsum("ij,ij->i", r, los[dst]) / np.maximum(d, 1e-12)
    egeo = np.column_stack([np.log(d / np.median(d)), rpar_s, rpar_d,
                            np.sqrt(np.clip(1 - rpar_s**2, 0, 1))])
    print(f"radius({args.build_radius_mpc}) graph: {ei.shape[1]} pairs -> {len(src)} directed edges")

    geom = Geometry(pos, args.cell_mpc, args.pad_mpc, dev)
    print(f"grid {geom.shape} = {geom.numel/1e6:.1f}M cells @ {args.cell_mpc} Mpc")
    phys = PhysicsLayer(geom.shape, args.cell_mpc, RSMOOTH_MPC, dev)

    mu, sd = eig[train].mean(0), eig[train].std(0)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    h, eg = t(X), t(egeo)
    srct, dstt = t(src, torch.long), t(dst, torch.long)
    yt = t((eig - mu) / sd)
    trm, vam = t(train, torch.bool), t(val, torch.bool)
    counts_ch = geom.counts(dev, torch.float32)
    counts_ch = counts_ch / counts_ch.mean().clamp(min=1e-6)
    sd_t, mu_t = t(sd), t(mu)

    model = GraphFieldPoisson(X.shape[1], egeo.shape[1], geom, phys,
                              width=args.width, unet_base=args.unet_base).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e3:.0f}k")

    if args.smoke:
        lam, delta = model(h, srct, dstt, eg, counts_ch)
        loss = ((((lam - mu_t) / sd_t)[trm] - yt[trm]) ** 2).mean()
        loss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        print(f"SMOKE ok: lam {tuple(lam.shape)} delta {tuple(delta.shape)} "
              f"loss {float(loss):.3f} gradnorm {gnorm:.2f} "
              f"asc-frac {float((lam[:,0]<=lam[:,1]).float().mean()):.3f}")
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        lam, _ = model(h, srct, dstt, eg, counts_ch)
        pred_std = (lam - mu_t) / sd_t
        loss = ((pred_std[trm] - yt[trm]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                lam, _ = model(h, srct, dstt, eg, counts_ch)
                vloss = float(((((lam - mu_t) / sd_t)[vam] - yt[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 250 == 0:
                print(f"step {step:5d} train {float(loss):.4f} val {vloss:.4f} ({time.time()-t0:.0f}s)")
            if patience >= 15:
                print(f"early stop at {step}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred, _ = model(h, srct, dstt, eg, counts_ch)
        pred = pred.cpu().numpy()
    if args.save_embeddings is not None:
        with torch.no_grad():
            emb = model.enc(h, srct, dstt, eg).detach().cpu().numpy()  # (N, width) invariant h_i
        args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_embeddings, emb=emb.astype(np.float32),
                            train=train, val=val, test=test,
                            eig_raw=eig.astype(np.float64), pred_eigs=pred.astype(np.float64),
                            seed=args.seed, width=args.width)
        print(f"embeddings saved: {emb.shape} -> {args.save_embeddings}")

    ti = np.where(test)[0]
    lines, scores = [], {}
    print(f"\n{'':10s} {'F1 R2':>8s}  (GraphNet 0.775/0.811/0.891; G3 0.804; classical DTFE 0.552/0.641/0.663)")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = r2_score(eig[ti, k], pred[ti, k])
        scores[nm] = float(r2)
        print(f"{nm:10s} {r2:8.3f}")
        lines.append(f"{nm}: R2={r2:.4f}")
    clu = eig[ti, 0] > 0.2
    sp = float(spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic)
    scores["cluster_lambda1_spearman"] = sp
    print(f"cluster-slice lambda1 Spearman: {sp:+.2f} (baseline 0.54; n={int(clu.sum())})")
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(f"gate_t4_graph_field_poisson seed={args.seed} cell={args.cell_mpc} "
                                 f"radius={args.build_radius_mpc}\n" + "\n".join(lines) + "\n")
        json.dump({"scores": scores, "args": vars(args) | {"cache": str(args.cache),
                   "points_xyz": str(args.points_xyz), "out_file": str(args.out_file)}},
                  open(args.out_file.with_suffix(".json"), "w"), indent=2, default=str)
        print(f"summary written: {args.out_file}")
    print("\nGATE F1: lambda1 R2 >= G3 (0.804) AND calibration >= flow -> GO F2. "
          "Physics layer is fixed/validated; only delta_hat is learned.")


if __name__ == "__main__":
    main()
