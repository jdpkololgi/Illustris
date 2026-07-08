#!/usr/bin/env python3
"""Gate T2 — CNN-on-counts control (roadmap G5 concretized).

Question: does a 3-D CNN on a *dumb voxelized view* of the wedge galaxy counts
recover the GraphNet's eigenvalue-regression skill, or does the graph's
sparse-tracer geometry add real signal beyond a Cartesian counts field?

Pipeline (no graph anywhere):
  galaxies (observer-frame comoving Mpc)
    -> voxelize onto a padded Cartesian wedge grid: channels =
       [log1p(CIC counts), overdensity delta=counts/mu-1, apodized survey mask]
    -> small 3-D U-Net  grid(C_in) -> latent volume grid(C_lat)
    -> differentiable trilinear sample of the latent volume at each galaxy's
       observer-frame position
    -> MLP head -> 3 eigenvalues (standardized)
    -> MSE on the standardized targets, train mask only; early-stop on val loss;
       restore best; un-standardize predictions.

Eval convention is byte-identical to gate_g4_egnn_smoke.py / the classical
baseline: r2_score(eig[test,k], pred[test,k]) per lambda1/2/3 (ascending), plus
the cluster-slice lambda1 Spearman (clu = eig[test,0] > 0.2). Reference numbers
printed alongside: GraphNet+NPE 0.775/0.811/0.891 and the classical (DTFE) floor
0.552/0.641/0.663.

Reuses the WedgeGrid geometry/CIC/mask/nbar helpers from
workflows/abacus_tweb/classical_tidal_baseline.py (no FFT tidal solve — the CNN
learns the map). cell-mpc is a CLI arg; start coarse (5 Mpc -> ~150^3).

Torch; GPU if available, CPU fallback for the smoke. Env hygiene: run under the
cosmic_env absolute python with PYTHONNOUSERSITE=1 (see repo CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- constants
# (mirror classical_tidal_baseline.py so the voxel view matches T1 exactly)
LITTLE_H = 0.6736
RA_MIN, RA_MAX = 120.0, 160.0
DEC_MIN, DEC_MAX = 14.5, 30.6

GRAPHNET_BASELINE = {"lambda1": 0.775, "lambda2": 0.811, "lambda3": 0.891}
CLASSICAL_FLOOR = {"lambda1": 0.552, "lambda2": 0.641, "lambda3": 0.663}  # DTFE (T1)


# ----------------------------------------------------------------------------- wedge grid
# NOTE: geometry helpers copied from classical_tidal_baseline.py::WedgeGrid so this
# script is standalone (the task requires ONE new file, no edits elsewhere).
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

    Axis convention: the grid tensor is indexed (ix, iy, iz) over the three
    Cartesian axes; cell centers at lo + (i+0.5)*cell.
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
        """Fractional cell-center index of each point: u in [0, n-1] per axis. (N,3)"""
        return (xyz - self.lo) / self.cell - 0.5

    def survey_mask(self, r_gal: np.ndarray, apod_mpc: float = 6.0) -> np.ndarray:
        rr = np.sqrt(self.gx**2 + self.gy**2 + self.gz**2)
        ra = np.degrees(np.arctan2(self.gy, self.gx)) % 360.0
        dec = np.degrees(np.arcsin(np.clip(self.gz / np.maximum(rr, 1e-9), -1, 1)))
        hard = (
            (ra >= RA_MIN) & (ra <= RA_MAX)
            & (dec >= DEC_MIN) & (dec <= DEC_MAX)
            & (rr >= np.quantile(r_gal, 0.001)) & (rr <= np.quantile(r_gal, 0.999))
        )
        return gaussian_filter(hard.astype(np.float32), sigma=apod_mpc / self.cell)

    def expected_counts(self, r_centers, nbar, mask_apod) -> np.ndarray:
        rr = np.sqrt(self.gx**2 + self.gy**2 + self.gz**2)
        nbar_grid = np.interp(rr, r_centers, nbar, left=0.0, right=0.0).astype(np.float32)
        return nbar_grid * mask_apod * self.cell**3

    def cic_deposit(self, xyz: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.shape, dtype=np.float32)
        u = (xyz - self.lo) / self.cell - 0.5
        i0 = np.floor(u).astype(np.int64)
        f = (u - i0).astype(np.float32)
        for dx in (0, 1):
            wx = (1 - f[:, 0]) if dx == 0 else f[:, 0]
            for dy in (0, 1):
                wy = (1 - f[:, 1]) if dy == 0 else f[:, 1]
                for dz in (0, 1):
                    wz = (1 - f[:, 2]) if dz == 0 else f[:, 2]
                    np.add.at(
                        counts,
                        (
                            np.clip(i0[:, 0] + dx, 0, self.shape[0] - 1),
                            np.clip(i0[:, 1] + dy, 0, self.shape[1] - 1),
                            np.clip(i0[:, 2] + dz, 0, self.shape[2] - 1),
                        ),
                        wx * wy * wz,
                    )
        return counts


def delta_from_counts(counts: np.ndarray, mu: np.ndarray, mu_floor: float = 0.05) -> np.ndarray:
    """Overdensity where the expected count is meaningful, 0 outside the survey."""
    delta = np.zeros_like(counts, dtype=np.float32)
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
    """Small 3-D U-Net: in_ch grid -> lat_ch latent volume (same spatial size).

    Two down/up levels with trilinear up-sampling back to the exact encoder skip
    shapes (handles non-power-of-2 dims without conv-transpose size juggling).
    """

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
        e0 = self.enc0(x)            # full res
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self._up(b, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], 1))
        d0 = self.dec0(torch.cat([self._up(d1, e0), e0], 1))
        return self.out(d0)          # (1, lat_ch, nx, ny, nz)


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
        """vox: (1,C,nx,ny,nz); grid_pts: (1,1,1,N,3) normalized [-1,1] sample coords."""
        lat = self.unet(vox)                                   # (1, L, nx, ny, nz)
        sampled = F.grid_sample(lat, grid_pts, mode="bilinear",
                                align_corners=True, padding_mode="border")
        feat = sampled[0, :, 0, 0, :].transpose(0, 1)          # (N, L)
        return self.head(feat)


def make_grid_coords(frac_idx: np.ndarray, shape) -> torch.Tensor:
    """Map fractional cell-center indices (N,3) over axes (ix,iy,iz) to a
    grid_sample grid tensor (1,1,1,N,3).

    grid_sample (5-D) last-dim order is (x,y,z) where x indexes the LAST spatial
    dim (nz), y the middle (ny), z the first (nx). align_corners=True maps
    fractional index u on an axis of length n to 2*u/(n-1) - 1.
    """
    nx, ny, nz = shape
    norm = np.empty_like(frac_idx)
    norm[:, 0] = 2.0 * frac_idx[:, 0] / (nx - 1) - 1.0  # ix
    norm[:, 1] = 2.0 * frac_idx[:, 1] / (ny - 1) - 1.0  # iy
    norm[:, 2] = 2.0 * frac_idx[:, 2] / (nz - 1) - 1.0  # iz
    # reorder to grid_sample's (x=nz, y=ny, z=nx)
    g = np.stack([norm[:, 2], norm[:, 1], norm[:, 0]], axis=1)
    return torch.tensor(g, dtype=torch.float32).view(1, 1, 1, -1, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--out-file", type=Path, default=None,
                    help="write the R2 summary (used for completion detection).")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="write full scores + predicted eigenvalues alongside.")
    ap.add_argument("--cell-mpc", type=float, default=5.0, help="grid cell size (Mpc)")
    ap.add_argument("--pad-mpc", type=float, default=40.0, help="zero padding per side (Mpc)")
    ap.add_argument("--lat-ch", type=int, default=32)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 40 steps, verifies data-load + fwd/bwd + eval plumbing.")
    args = ap.parse_args()

    if args.smoke:
        args.steps = 40
        args.eval_every = 5
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  torch {torch.__version__}")

    # ---- data
    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)          # (N,3) ascending
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    pos = np.load(args.points_xyz).astype(np.float64)              # (N,3) observer-frame Mpc
    assert len(pos) == len(eig), f"pos {len(pos)} != eig {len(eig)}"
    n = len(pos)
    print(f"nodes={n}  train/val/test = {train.sum()}/{val.sum()}/{test.sum()}")

    # ---- voxelize (fixed input; matches T1 exactly)
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

    # channels: log1p(counts), overdensity, apodized mask
    ch_counts = np.log1p(counts).astype(np.float32)
    ch_counts = (ch_counts - ch_counts[mask > 0.05].mean()) / (ch_counts[mask > 0.05].std() + 1e-6)
    ch_delta = np.clip(delta, -1.0, 20.0).astype(np.float32)
    ch_mask = mask.astype(np.float32)
    vox_np = np.stack([ch_counts, ch_delta, ch_mask], axis=0)[None]  # (1,3,nx,ny,nz)
    print(f"vox tensor {vox_np.shape}  ~{vox_np.nbytes/1e6:.0f}MB")

    frac = grid.frac_index(pos)
    grid_pts = make_grid_coords(frac, grid.shape).to(dev)

    # sanity: sample the counts channel at galaxies via grid_sample and check it
    # correlates with the analytic CIC value (axis-order / normalization guard).
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

    # ---- standardize targets on train
    tmu, tsd = eig[train].mean(0), eig[train].std(0)
    Y = (eig - tmu) / tsd
    vox = torch.tensor(vox_np, dtype=torch.float32, device=dev)
    yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    trm = torch.tensor(train, dtype=torch.bool, device=dev)
    vam = torch.tensor(val, dtype=torch.bool, device=dev)

    model = CNNCountsModel(in_ch=vox_np.shape[1], lat_ch=args.lat_ch, base=args.base).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M  (lat_ch={args.lat_ch}, base={args.base})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        out = model(vox, grid_pts)
        loss = ((out[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vout = model(vox, grid_pts)
                vloss = float(((vout[vam] - yt[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                print(f"step {step:5d}  train {float(loss.detach()):.4f}  val {vloss:.4f}  "
                      f"best {best_val:.4f}  ({time.time()-t0:.0f}s)")
            if patience >= args.patience:
                print(f"early stop at step {step}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(vox, grid_pts).cpu().numpy() * tsd + tmu

    # ---- eval (byte-identical convention to gate_g4_egnn_smoke / classical baseline)
    ti = np.where(test)[0]
    print(f"\n{'':10s}  {'T2 CNN R2':>10s}   {'GraphNet':>9s}   {'DTFE floor':>10s}")
    lines, scores = [], {}
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = float(r2_score(eig[ti, k], pred[ti, k]))
        scores[nm] = r2
        print(f"{nm:10s}  {r2:10.3f}   {GRAPHNET_BASELINE[nm]:9.3f}   {CLASSICAL_FLOOR[nm]:10.3f}")
        lines.append(f"{nm}: R2={r2:.4f}")
    clu = eig[ti, 0] > 0.2
    sp = float(spearmanr(eig[ti, 0][clu], pred[ti, 0][clu]).statistic)
    print(f"cluster-slice lambda1 Spearman: {sp:+.3f} (baseline 0.54; n={int(clu.sum())})")
    lines.append(f"cluster_slice_lambda1_spearman: {sp:.4f} (n={int(clu.sum())})")
    scores["cluster_slice_lambda1_spearman"] = sp
    scores["n_cluster"] = int(clu.sum())

    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        hdr = (f"gate_t2_cnn_counts cell_mpc={args.cell_mpc} lat_ch={args.lat_ch} "
               f"base={args.base} seed={args.seed} grid={grid.shape} params={nparam}")
        args.out_file.write_text(hdr + "\n" + "\n".join(lines) + "\n")
        print(f"summary written: {args.out_file}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gate": "T2_cnn_counts", "cache": str(args.cache),
            "cell_mpc": args.cell_mpc, "pad_mpc": args.pad_mpc,
            "grid_shape": list(grid.shape), "lat_ch": args.lat_ch, "base": args.base,
            "seed": args.seed, "steps_run": step + 1, "n_params": int(nparam),
            "graphnet_baseline": GRAPHNET_BASELINE, "classical_floor": CLASSICAL_FLOOR,
            "scores": scores, "runtime_s": time.time() - t0,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        np.save(args.out_json.with_suffix(".pred_eigs.npy"), pred.astype(np.float32))
        print(f"scores json: {args.out_json}")

    print("\nGATE read: CNN-on-counts vs GraphNet (0.775/0.811/0.891) vs classical "
          "DTFE floor (0.552/0.641/0.663). ~= GraphNet -> graph story needs "
          "rewriting; ~= floor -> voxel view adds nothing over linear recon; "
          "in-between -> sparse-tracer geometry carries real signal.")


if __name__ == "__main__":
    main()
