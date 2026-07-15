#!/usr/bin/env python3
"""ML-baseline transfer test: the T2 3-D U-Net (leaky-split champion, λ1 0.876) on the disjoint wedge.

Companion to the GraphNet transfer test (RA120-160 -> RA200-240): the GraphNet fell 0.804 -> 0.421 at
deployment, below the classical floor (DTFE 0.534 on the same wedge/test mask). This asks whether the
OTHER architecture family collapses the same way, i.e. whether the failure is model-specific or a
property of transductive random-split training on this volume.

Phase 1 replicates the T2 run exactly (same cache/masks/points/constants/hyperparams, seed 42) because
gate_t2 never saved weights -- and gates on reproducing its λ1 (3-seed band 0.871-0.880) before any
transfer number is produced. Phase 2 voxelizes the NEW wedge with its OWN nbar/mask/mu and own in-mask
channel standardisation (the voxel analogue of own-graph SI medians -- what deployment has), predicts
with the frozen weights, and scores on the transfer cache's test mask: the SAME 95,220 galaxies the
GraphNet and DTFE were scored on.

Model/geometry classes are imported from gate_t2_cnn_counts.py (single source, no drift).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gate_t2", REPO / "workflows/sbi/gate_t2_cnn_counts.py")
t2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t2)

W = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign")
TRAIN_CACHE = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/"
                   "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_sbi_cache_3d_lineareig_si.pkl")
TRAIN_POINTS = W / "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_points_xyz.npy"
NEW_CACHE = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_TRANSFER_ra200_240_uniongraph/"
                 "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl")
NEW_TARGETS = W / "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra200_240_dec14p5_30p6_z0p2_0p3_wedge_targets.fits"


def build_channels(pos, ra_min, ra_max, cell, pad):
    """T2's voxel view, with the wedge RA bounds as parameters (T2 hardcodes 120-160)."""
    t2.RA_MIN, t2.RA_MAX = ra_min, ra_max          # module constants used by survey_mask/omega
    grid = t2.WedgeGrid(pos, cell=cell, pad=pad)
    r_gal = np.linalg.norm(pos, axis=1)
    omega = np.radians(ra_max - ra_min) * (np.sin(np.radians(t2.DEC_MAX)) - np.sin(np.radians(t2.DEC_MIN)))
    r_centers, nbar = t2.radial_nbar(r_gal, omega)
    mask = grid.survey_mask(r_gal)
    mu = grid.expected_counts(r_centers, nbar, mask)
    counts = grid.cic_deposit(pos)
    delta = t2.delta_from_counts(counts, mu)
    ch_counts = np.log1p(counts).astype(np.float32)
    inm = mask > 0.05
    ch_counts = (ch_counts - ch_counts[inm].mean()) / (ch_counts[inm].std() + 1e-6)   # own-wedge (SI analogue)
    vox = np.stack([ch_counts, np.clip(delta, -1.0, 20.0).astype(np.float32), mask.astype(np.float32)], 0)[None]
    return grid, vox, counts, r_gal


def predict(model, vox_np, grid, pos, dev, counts):
    grid_pts = t2.make_grid_coords(grid.frac_index(pos), grid.shape).to(dev)
    with torch.no_grad():                              # axis-order guard, same as T2
        vtest = torch.tensor(counts[None, None], dtype=torch.float32, device=dev)
        s = F.grid_sample(vtest, grid_pts, mode="bilinear", align_corners=True,
                          padding_mode="border")[0, 0, 0, 0, :].cpu().numpy()
        from scipy.ndimage import map_coordinates
        ref = map_coordinates(counts, grid.frac_index(pos).T, order=1, mode="nearest")
        rr = np.corrcoef(s, ref)[0, 1]
        if rr < 0.99:
            raise RuntimeError(f"grid_sample axis order wrong (corr {rr:.3f})")
        vox = torch.tensor(vox_np, dtype=torch.float32, device=dev)
        out = model(vox, grid_pts).cpu().numpy()
    return out


def r2_cols(y, p, m):
    from sklearn.metrics import r2_score
    return [float(r2_score(y[m, k], p[m, k])) for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/pscratch/sd/d/dkololgi/abacus/field_level_tests/T2_transfer"))
    ap.add_argument("--cell-mpc", type=float, default=5.0)
    ap.add_argument("--pad-mpc", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}")

    # ---------------- Phase 1: replicate T2 on the TRAINING wedge (weights were never saved)
    cache = pickle.load(open(TRAIN_CACHE, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    pos = np.load(TRAIN_POINTS).astype(np.float64)
    assert len(pos) == len(eig), f"train wedge pos {len(pos)} != eig {len(eig)}"
    print(f"TRAIN wedge: {len(pos):,} nodes  masks {train.sum():,}/{val.sum():,}/{test.sum():,} (RANDOM split, as T2)")

    grid, vox_np, counts, _ = build_channels(pos, 120.0, 160.0, args.cell_mpc, args.pad_mpc)
    grid_pts = t2.make_grid_coords(grid.frac_index(pos), grid.shape).to(dev)
    tmu, tsd = eig[train].mean(0), eig[train].std(0)
    Y = torch.tensor((eig - tmu) / tsd, dtype=torch.float32, device=dev)
    vox = torch.tensor(vox_np, dtype=torch.float32, device=dev)
    trm = torch.tensor(train, dtype=torch.bool, device=dev)
    vam = torch.tensor(val, dtype=torch.bool, device=dev)

    model = t2.CNNCountsModel(in_ch=3, lat_ch=32, base=24).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best_val, best_state, patience, t0 = np.inf, None, 0, time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        out = model(vox, grid_pts)
        loss = ((out[trm] - Y[trm]) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vloss = float(((model(vox, grid_pts)[vam] - Y[vam]) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 250 == 0:
                print(f"step {step:5d} train {float(loss):.4f} val {vloss:.4f} best {best_val:.4f} ({time.time()-t0:.0f}s)")
            if patience >= args.patience:
                print(f"early stop at {step}"); break
    model.load_state_dict(best_state); model.eval()

    pred_tr = predict(model, vox_np, grid, pos, dev, counts) * tsd + tmu
    r2_home = r2_cols(eig, pred_tr, test)
    print(f"\n[SANITY] home-wedge test R2 (random split): λ1={r2_home[0]:.3f} λ2={r2_home[1]:.3f} λ3={r2_home[2]:.3f}")
    print(f"         T2 reference (3 seeds): 0.871-0.880 / ~0.905 / ~0.933")
    if r2_home[0] < 0.85:
        raise RuntimeError(f"T2 replication failed (λ1 {r2_home[0]:.3f} < 0.85) -- transfer number would be untrustworthy")
    torch.save({"state_dict": best_state, "tmu": tmu, "tsd": tsd, "cell": args.cell_mpc,
                "pad": args.pad_mpc, "seed": args.seed, "home_test_r2": r2_home},
               args.out_dir / "t2_model_seed42.pt")

    # ---------------- Phase 2: TRANSFER to the disjoint wedge (own-wedge channels, frozen weights)
    ncache = pickle.load(open(NEW_CACHE, "rb"))
    neig = np.asarray(ncache["eigenvalues_raw"], np.float64)
    _, _, ntest = (np.asarray(m).astype(bool) for m in ncache["masks"])
    npoints_path = W / "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra200_240_dec14p5_30p6_z0p2_0p3_points_xyz.npy"
    npos = np.load(npoints_path).astype(np.float64)
    assert len(npos) == len(neig), f"new wedge pos {len(npos)} != eig {len(neig)}"
    print(f"\nNEW wedge: {len(npos):,} nodes, scoring on transfer test mask = {ntest.sum():,} "
          f"(same galaxies as GraphNet transfer + DTFE)")

    ngrid, nvox_np, ncounts, nr_gal = build_channels(npos, 200.0, 240.0, args.cell_mpc, args.pad_mpc)
    pred_new = predict(model, nvox_np, ngrid, npos, dev, ncounts) * tsd + tmu

    # interior mask (>=25 Mpc from wedge edges), for parity with the DTFE report
    import fitsio
    tt = fitsio.read(NEW_TARGETS, columns=["RA", "DEC"])
    ang = np.degrees(25.0 / nr_gal)
    interior = ((tt["RA"] > 200 + ang) & (tt["RA"] < 240 - ang)
                & (tt["DEC"] > t2.DEC_MIN + ang) & (tt["DEC"] < t2.DEC_MAX - ang)
                & (nr_gal > np.quantile(nr_gal, 0.001) + 25) & (nr_gal < np.quantile(nr_gal, 0.999) - 25))

    r2_tr = r2_cols(neig, pred_new, ntest)
    r2_int = r2_cols(neig, pred_new, ntest & interior)
    print(f"\n=== U-NET TRANSFER (RA200-240) ===")
    print(f"  test R2      : λ1={r2_tr[0]:.3f} λ2={r2_tr[1]:.3f} λ3={r2_tr[2]:.3f}   (n={int(ntest.sum()):,})")
    print(f"  interior R2  : λ1={r2_int[0]:.3f} λ2={r2_int[1]:.3f} λ3={r2_int[2]:.3f}   (n={int((ntest & interior).sum()):,})")
    print(f"  comparators  : GraphNet transfer 0.421/0.498/0.524 | DTFE(cal) 0.534/0.604/0.634 | home leaky 0.876/0.905/0.933")

    np.savez_compressed(args.out_dir / "t2_transfer_pred_ra200_240.npz",
                        pred=pred_new.astype(np.float32), true=neig.astype(np.float32),
                        test_mask=ntest, interior=interior)
    with open(args.out_dir / "t2_transfer_scores.json", "w") as f:
        json.dump({"home_test_r2_random_split": r2_home, "transfer_test_r2": r2_tr,
                   "transfer_interior_r2": r2_int, "n_test": int(ntest.sum()),
                   "seed": args.seed, "cell_mpc": args.cell_mpc,
                   "channel_standardisation": "own-wedge in-mask (SI analogue)"}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
