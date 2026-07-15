#!/usr/bin/env python3
"""Workstream C — pooled, selection-aware full-range 3-D U-Net challenger.

QUESTION: S1(b) measured a CNN trained WITHIN each shell at 0.902/0.847/0.722/0.429 over z0.15-0.55
(macro ~0.725) vs the incumbent ñ-conditioned GraphNet's 0.456 macro. That run's "grid is dead" verdict
was driven substantially by its 0.002 at z0.05-0.15 -- now known to be the CORRUPT-LABEL shell
(permutation-null targets, BOX_INDEX==-1), which is excluded from the VAC. Within the actual VAC range
the grid lost to the GraphNet in ZERO shells. But those were per-shell models (best case, and per-shell
models reintroduce the seams that ñ-conditioning exists to remove). C asks the untested question:

    does ONE pooled, selection-aware U-Net hold that skill across the full range?

DIFFERENCES vs gate_t2_cnn_counts.py (which this is adapted from):
  1. POOLED full range (z0.15-0.55 in a single grid) instead of one model per shell.
  2. SELECTION-AWARE channels: gate_t2 already had delta=counts/mu and the apodized mask; this adds an
     explicit expected-count channel log1p(mu) so the model can see n(z) directly rather than only
     through the contrast.
  3. LOS channels for RSD: 3 radial unit-vector components r_hat at each voxel. A Cartesian CNN
     otherwise cannot know which direction is the line of sight -- and it varies across the wedge --
     so it cannot learn the anisotropic (RSD-elongated) structure without them.
  4. GATE METRIC = VAL macro-shell lambda1 R^2, matching how the GraphNet is scored. gate_t2 scored
     POOLED R2 on TEST; pooled hides the sparse high-z shell, and the RA>=150 test region must stay
     sealed until a finalist is frozen (per the memo). Test is refused unless --unseal-test.

The cache is built by s3c_build_cnn_fullrange_cache.py, whose match gate proves the split is
byte-identical to the tiled cache the GraphNet trained on (129,113/18,629/52,848) -- so this is a true
head-to-head, not two models on two datasets.

Run under the cosmic_env absolute python with PYTHONNOUSERSITE=1 (repo CLAUDE.md). GPU; prefer hbm80g.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------- constants
RA_MIN, RA_MAX = 120.0, 160.0        # verified against the s2 shells: data spans exactly this
DEC_MIN, DEC_MAX = 14.5, 30.6

# The comparator that matters: the incumbent GraphNet on the SAME split (A1_sqrt, v2 8-d, tau=0.5).
GRAPHNET_A1SQRT = {"best_macro": 0.456, "best_pooled": 0.5162, "best_val_nll": 2.7150,
                   "per_shell": {"0p15_0p25": 0.53, "0p25_0p35": 0.49,
                                 "0p35_0p45": 0.50, "0p45_0p55": 0.29}}
# S1(b) per-shell CNN (best case, trained WITHIN each shell) -- the skill C is trying to hold pooled.
S1B_CNN_INSHELL = {"0p15_0p25": 0.902, "0p25_0p35": 0.847, "0p35_0p45": 0.722, "0p45_0p55": 0.429}


def radial_nbar(r: np.ndarray, omega_sr: float, bin_mpc: float = 10.0):
    """Smoothed n(r) [gal/Mpc^3] within the wedge solid angle."""
    edges = np.arange(r.min() - bin_mpc, r.max() + 2 * bin_mpc, bin_mpc)
    counts, edges = np.histogram(r, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = omega_sr * centers**2 * np.diff(edges)
    nbar = gaussian_filter1d(counts.astype(np.float64), sigma=2.0) / shell_vol
    return centers, nbar


class WedgeGrid:
    """Padded Cartesian grid around the wedge with an apodized survey mask.

    Axis convention: grid tensor indexed (ix, iy, iz); cell centers at lo + (i+0.5)*cell.
    """

    def __init__(self, xyz: np.ndarray, cell: float, pad: float):
        self.cell = cell
        self.lo = xyz.min(axis=0) - pad
        hi = xyz.max(axis=0) + pad
        self.shape = tuple(int(np.ceil((hi - self.lo)[i] / cell)) for i in range(3))
        print(f"grid shape {self.shape} = {np.prod(self.shape)/1e6:.2f}M cells, cell={cell} Mpc")
        ax = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * cell for i in range(3)]
        self.gx, self.gy, self.gz = np.meshgrid(*ax, indexing="ij", sparse=True)

    def frac_index(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.lo) / self.cell - 0.5

    def radius(self) -> np.ndarray:
        return np.sqrt(self.gx**2 + self.gy**2 + self.gz**2)

    def survey_mask(self, r_gal: np.ndarray, apod_mpc: float = 6.0) -> np.ndarray:
        rr = self.radius()
        ra = np.degrees(np.arctan2(self.gy, self.gx)) % 360.0
        dec = np.degrees(np.arcsin(np.clip(self.gz / np.maximum(rr, 1e-9), -1, 1)))
        hard = (
            (ra >= RA_MIN) & (ra <= RA_MAX)
            & (dec >= DEC_MIN) & (dec <= DEC_MAX)
            & (rr >= np.quantile(r_gal, 0.001)) & (rr <= np.quantile(r_gal, 0.999))
        )
        return gaussian_filter(hard.astype(np.float32), sigma=apod_mpc / self.cell)

    def expected_counts(self, r_centers, nbar, mask_apod) -> np.ndarray:
        rr = self.radius()
        nbar_grid = np.interp(rr, r_centers, nbar, left=0.0, right=0.0).astype(np.float32)
        return nbar_grid * mask_apod * self.cell**3

    def los_hat(self) -> np.ndarray:
        """(3,nx,ny,nz) radial unit vector = the line-of-sight direction at each voxel (RSD axis)."""
        rr = np.maximum(self.radius(), 1e-9)
        ones = np.ones(self.shape, np.float32)
        return np.stack([(self.gx / rr) * ones, (self.gy / rr) * ones, (self.gz / rr) * ones], 0)

    def cic_deposit(self, xyz: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.shape, dtype=np.float32)
        u = (xyz - self.lo) / self.cell - 0.5
        i0 = np.floor(u).astype(np.int64)
        f = u - i0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = i0 + np.array([dx, dy, dz])
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0])
                         * np.where(dy, f[:, 1], 1 - f[:, 1])
                         * np.where(dz, f[:, 2], 1 - f[:, 2]))
                    ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
                    np.add.at(counts, (idx[ok, 0], idx[ok, 1], idx[ok, 2]), w[ok].astype(np.float32))
        return counts


def delta_from_counts(counts: np.ndarray, mu: np.ndarray, mu_floor: float = 0.05) -> np.ndarray:
    delta = np.zeros_like(counts)
    ref = float(mu[mu > 0].mean() if np.any(mu > 0) else 1.0)
    ok = mu > mu_floor * ref
    delta[ok] = counts[ok] / mu[ok] - 1.0
    return delta


# ----------------------------------------------------------------------------- 3-D U-Net
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class UNet3D(nn.Module):
    def __init__(self, in_ch, lat_ch=32, base=24):
        super().__init__()
        self.enc0 = conv_block(in_ch, base)
        self.enc1 = conv_block(base, base * 2)
        self.enc2 = conv_block(base * 2, base * 4)
        self.bott = conv_block(base * 4, base * 4)
        self.dec2 = conv_block(base * 4 + base * 4, base * 2)
        self.dec1 = conv_block(base * 2 + base * 2, base)
        self.dec0 = conv_block(base + base, base)
        self.out = nn.Conv3d(base, lat_ch, 1)
        self.pool = nn.MaxPool3d(2, ceil_mode=True)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self._up(b, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], 1))
        d0 = self.dec0(torch.cat([self._up(d1, e0), e0], 1))
        return self.out(d0)


class CNNCountsModel(nn.Module):
    def __init__(self, in_ch, lat_ch=32, base=24, head_width=128):
        super().__init__()
        self.unet = UNet3D(in_ch, lat_ch=lat_ch, base=base)
        self.head = nn.Sequential(
            nn.Linear(lat_ch, head_width), nn.SiLU(),
            nn.Linear(head_width, head_width), nn.SiLU(),
            nn.Linear(head_width, 3),
        )

    def forward(self, vox, grid_pts):
        lat = self.unet(vox)
        sampled = F.grid_sample(lat, grid_pts, mode="bilinear",
                                align_corners=True, padding_mode="border")
        feat = sampled[0, :, 0, 0, :].transpose(0, 1)
        return self.head(feat)


def make_grid_coords(frac_idx: np.ndarray, shape) -> torch.Tensor:
    nx, ny, nz = shape
    norm = np.empty_like(frac_idx)
    norm[:, 0] = 2.0 * frac_idx[:, 0] / (nx - 1) - 1.0
    norm[:, 1] = 2.0 * frac_idx[:, 1] / (ny - 1) - 1.0
    norm[:, 2] = 2.0 * frac_idx[:, 2] / (nz - 1) - 1.0
    g = np.stack([norm[:, 2], norm[:, 1], norm[:, 0]], axis=1)   # grid_sample order (x=nz,y=ny,z=nx)
    return torch.tensor(g, dtype=torch.float32).view(1, 1, 1, -1, 3)


def shell_r2(eig, pred, mask, shell, k=0):
    """Per-shell + pooled + macro R^2 for eigenvalue k -- same convention as the tiled GraphNet."""
    rows = {}
    for tag in sorted(set(shell[mask].tolist())):
        s = mask & (shell == tag)
        if s.sum() > 2:
            rows[tag] = float(r2_score(eig[s, k], pred[s, k]))
    pooled = float(r2_score(eig[mask, k], pred[mask, k]))
    macro = float(np.nanmean(list(rows.values()))) if rows else np.nan
    return pooled, macro, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_cache.pkl"))
    ap.add_argument("--points-xyz", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_points.npy"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/C_unet_fullrange/scores.json"))
    ap.add_argument("--cell-mpc", type=float, default=5.0)
    ap.add_argument("--pad-mpc", type=float, default=40.0)
    ap.add_argument("--lat-ch", type=int, default=32)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--no-los", action="store_true", help="ablate the LOS/RSD channels")
    ap.add_argument("--no-mu", action="store_true", help="ablate the explicit expected-count channel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--unseal-test", action="store_true",
                    help="score the RA>=150 TEST region. Per the memo this is allowed ONLY for a "
                         "frozen finalist -- never during selection.")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every = 40, 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  torch {torch.__version__}")
    if dev == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    # ---- data
    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    shell = np.asarray(cache["shell"])
    pos = np.load(args.points_xyz).astype(np.float64)
    assert len(pos) == len(eig) == len(shell), "cache/points row mismatch"
    n = len(pos)
    print(f"nodes(field)={n:,}  active train/val/test = {train.sum():,}/{val.sum():,}/{test.sum():,}")
    print(f"provenance: {cache.get('provenance', {}).get('matched_to', '?')}")

    # ---- voxelize
    grid = WedgeGrid(pos, cell=args.cell_mpc, pad=args.pad_mpc)
    r_gal = np.linalg.norm(pos, axis=1)
    omega = np.radians(RA_MAX - RA_MIN) * (np.sin(np.radians(DEC_MAX)) - np.sin(np.radians(DEC_MIN)))
    r_centers, nbar = radial_nbar(r_gal, omega)
    mask = grid.survey_mask(r_gal)
    mu = grid.expected_counts(r_centers, nbar, mask)
    counts = grid.cic_deposit(pos)
    delta = delta_from_counts(counts, mu)
    print(f"counts sum={counts.sum():.0f} (N={n}); mu sum={mu.sum():.0f}; "
          f"delta range [{delta.min():.2f},{delta.max():.2f}]")

    inmask = mask > 0.05
    ch_counts = np.log1p(counts).astype(np.float32)
    ch_counts = (ch_counts - ch_counts[inmask].mean()) / (ch_counts[inmask].std() + 1e-6)
    ch_delta = np.clip(delta, -1.0, 20.0).astype(np.float32)
    ch_mask = mask.astype(np.float32)
    chans = [ch_counts, ch_delta, ch_mask]
    names = ["log1p_counts", "delta", "mask"]
    if not args.no_mu:
        ch_mu = np.log1p(mu).astype(np.float32)
        ch_mu = (ch_mu - ch_mu[inmask].mean()) / (ch_mu[inmask].std() + 1e-6)
        chans.append(ch_mu); names.append("log1p_mu")           # explicit selection / n(z)
    if not args.no_los:
        los = grid.los_hat().astype(np.float32)                 # (3,nx,ny,nz) RSD axis
        chans += [los[0], los[1], los[2]]; names += ["los_x", "los_y", "los_z"]
    vox_np = np.stack(chans, axis=0)[None]
    print(f"channels ({len(names)}): {names}")
    print(f"vox tensor {vox_np.shape}  ~{vox_np.nbytes/1e9:.2f}GB")

    frac = grid.frac_index(pos)
    grid_pts = make_grid_coords(frac, grid.shape).to(dev)

    # axis-order guard (identical to T2): grid_sample must agree with map_coordinates
    with torch.no_grad():
        vtest = torch.tensor(counts[None, None], dtype=torch.float32, device=dev)
        s = F.grid_sample(vtest, grid_pts, mode="bilinear", align_corners=True,
                          padding_mode="border")[0, 0, 0, 0, :].cpu().numpy()
        from scipy.ndimage import map_coordinates
        ref = map_coordinates(counts, frac.T, order=1, mode="nearest")
        rr = np.corrcoef(s, ref)[0, 1]
        print(f"[axis-order check] grid_sample vs map_coordinates counts corr = {rr:.4f}")
        if rr < 0.99:
            raise RuntimeError(f"grid_sample axis order looks wrong (corr {rr:.3f})")
        del vtest

    # ---- standardize targets on TRAIN only
    tmu, tsd = eig[train].mean(0), eig[train].std(0)
    Y = (eig - tmu) / tsd
    vox = torch.tensor(vox_np, dtype=torch.float32, device=dev)
    del vox_np
    yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    trm = torch.tensor(train, dtype=torch.bool, device=dev)
    vam = torch.tensor(val, dtype=torch.bool, device=dev)

    model = CNNCountsModel(in_ch=vox.shape[1], lat_ch=args.lat_ch, base=args.base).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M  (lat_ch={args.lat_ch}, base={args.base})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best_val, best_state, patience = np.inf, None, 0
    best_macro_seen, best_macro_step = -np.inf, -1
    t0 = time.time()
    step = 0
    for step in range(args.steps):
        model.train()
        opt.zero_grad()
        out = model(vox, grid_pts)
        loss = ((out[trm] - yt[trm]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vout = model(vox, grid_pts)
                vloss = float(((vout[vam] - yt[vam]) ** 2).mean())
                pv = vout.cpu().numpy() * tsd + tmu
            _, macro_now, _ = shell_r2(eig, pv, val, shell, k=0)
            if macro_now > best_macro_seen:
                best_macro_seen, best_macro_step = macro_now, step
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
                print(f"step {step:5d}  train {float(loss.detach()):.4f}  val {vloss:.4f}  "
                      f"best {best_val:.4f}  val macro λ1 {macro_now:.3f}  "
                      f"({time.time()-t0:.0f}s, {mem:.1f}GB)")
            if patience >= args.patience:
                print(f"early stop at step {step}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)          # restore best-val-loss (honest early stopping)
    model.eval()
    with torch.no_grad():
        pred = model(vox, grid_pts).cpu().numpy() * tsd + tmu

    # ---- GATE: VAL macro-shell lambda1 R^2 (test stays sealed)
    pooled_v, macro_v, rows_v = shell_r2(eig, pred, val, shell, k=0)
    print(f"\n=== WORKSTREAM C — VAL (gate; RA>=150 test SEALED) ===")
    print(f"{'shell':12s} {'C U-Net':>9s} {'GraphNet A1_sqrt':>17s} {'S1b CNN in-shell':>17s}")
    for tag in sorted(rows_v):
        print(f"{tag:12s} {rows_v[tag]:9.3f} {GRAPHNET_A1SQRT['per_shell'].get(tag, float('nan')):17.3f} "
              f"{S1B_CNN_INSHELL.get(tag, float('nan')):17.3f}")
    print(f"{'MACRO':12s} {macro_v:9.3f} {GRAPHNET_A1SQRT['best_macro']:17.3f} "
          f"{np.mean(list(S1B_CNN_INSHELL.values())):17.3f}")
    print(f"{'pooled':12s} {pooled_v:9.3f} {GRAPHNET_A1SQRT['best_pooled']:17.3f}")
    print(f"\nbest val macro λ1 seen during training: {best_macro_seen:.3f} (step {best_macro_step}); "
          f"macro at best-val-loss checkpoint: {macro_v:.3f}")
    print(f"GATE: beat GraphNet macro {GRAPHNET_A1SQRT['best_macro']:.3f} by >= +0.02 to justify a pivot.")
    lam23 = {}
    for k, nm in ((1, "lambda2"), (2, "lambda3")):
        p, m, _ = shell_r2(eig, pred, val, shell, k=k)
        lam23[nm] = {"pooled": p, "macro": m}
        print(f"  {nm}: val pooled {p:.3f}  macro {m:.3f}")
    clu = eig[val, 0] > 0.2
    sp = float(spearmanr(eig[val, 0][clu], pred[val, 0][clu]).statistic) if clu.sum() > 2 else float("nan")
    print(f"  cluster-slice λ1 Spearman (val): {sp:+.3f} (n={int(clu.sum())})")

    scores = {"val_lambda1": {"pooled": pooled_v, "macro": macro_v, "per_shell": rows_v},
              "val_best_macro_seen": best_macro_seen, "val_best_macro_step": best_macro_step,
              "val_lambda23": lam23, "val_cluster_slice_lambda1_spearman": sp}

    if args.unseal_test:
        print("\n*** TEST UNSEALED (finalist only) ***")
        pooled_t, macro_t, rows_t = shell_r2(eig, pred, test, shell, k=0)
        print(f"  test λ1 pooled {pooled_t:.3f}  macro {macro_t:.3f}  per-shell {rows_t}")
        scores["test_lambda1"] = {"pooled": pooled_t, "macro": macro_t, "per_shell": rows_t}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gate": "C_unet_fullrange_pooled", "cache": str(args.cache),
               "cell_mpc": args.cell_mpc, "pad_mpc": args.pad_mpc, "grid_shape": list(grid.shape),
               "channels": names, "lat_ch": args.lat_ch, "base": args.base, "seed": args.seed,
               "steps_run": step + 1, "n_params": int(nparam), "best_val_mse": best_val,
               "graphnet_a1sqrt": GRAPHNET_A1SQRT, "s1b_cnn_inshell": S1B_CNN_INSHELL,
               "scores": scores, "runtime_s": time.time() - t0,
               "test_sealed": not args.unseal_test}
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    np.save(args.out_json.with_suffix(".pred_eigs.npy"), pred.astype(np.float32))
    print(f"\nscores json: {args.out_json}")


if __name__ == "__main__":
    main()
