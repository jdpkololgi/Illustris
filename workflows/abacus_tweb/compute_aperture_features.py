#!/usr/bin/env python3
"""Workstream B1/B3 — fixed-aperture density + NN-scale node channels for the valid shells.

Per galaxy, on the shell's observer-frame positions (1:1 with the shell cache node order):
  * N_R      : neighbour counts inside fixed physical apertures R = 7, 10, 14 Mpc/h
               (7 Mpc/h matches the T-Web label smoothing) -> stored as log1p(N)
  * contrast : log[(N_R+eps) / (ntilde(z)*V_R + eps)]  — the SELECTION-AWARE contrast, i.e.
               observed vs EXPECTED counts. Keeps physical density DISTINCT from the selection
               covariate ntilde; we keep BOTH N and the contrast (memo: do not divide away the
               cosmological density signal — supply expected counts as a separate covariate).
  * d_NN     : distance to the nearest neighbour (local sampling scale) -> log(d+eps)

Output: one npz per shell, [N_nodes, 7], aligned with the shell node order, consumed by the
enriched-cache build. NOTE: apertures are truncated for galaxies within ~R of a wedge/z boundary
(same edge caveat as the existing graph features) — the gutter/holdout already de-weights edges.
CPU only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import fitsio
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree

SHELLS = ["0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]   # valid range only
S2 = Path("/pscratch/sd/d/dkololgi/abacus/s2_shells")
FEAT_NAMES = ["logN_ap7", "logN_ap10", "logN_ap14",
              "contrast_ap7", "contrast_ap10", "contrast_ap14", "log_dNN"]
EPS = 1e-3


def ntilde_of_z(z, sp):
    zg = np.asarray(sp["grid_z"]); nt = np.asarray(sp["ntilde"])
    n = np.interp(np.clip(z, zg.min(), zg.max()), zg, nt)
    return np.maximum(n, sp["ntilde_floor"])          # per Mpc^3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntilde-spline", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/conditioning/ntilde_spline_v1_frozen.json"))
    ap.add_argument("--apertures-hmpc", type=float, nargs="+", default=[7.0, 10.0, 14.0])
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/aperture_features_v1"))
    args = ap.parse_args()
    sp = json.loads(Path(args.ntilde_spline).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    h = cosmo.h
    print(f"apertures {args.apertures_hmpc} Mpc/h  (h={h:.4f} -> "
          f"{[round(R/h,2) for R in args.apertures_hmpc]} Mpc)")

    for tag in SHELLS:
        pos = np.load(S2 / f"shell_{tag}_final_points_xyz.npy").astype(np.float64)
        t = fitsio.read(S2 / f"shell_{tag}_final_wedge_targets.fits", columns=["Z", "LAMBDA1"])
        z = t["Z"].astype(np.float64)
        assert len(pos) == len(z), f"{tag}: pos {len(pos)} != targets {len(z)}"
        tree = cKDTree(pos)
        nt = ntilde_of_z(z, sp)
        cols = []
        counts = {}
        for R in args.apertures_hmpc:
            Rm = R / h
            # query_ball_point counts include the galaxy itself -> subtract for neighbours
            n_self = tree.query_ball_point(pos, Rm, return_length=True)
            N = np.maximum(n_self.astype(np.float64) - 1.0, 0.0)
            counts[R] = N
            cols.append(np.log1p(N))
        for R in args.apertures_hmpc:
            Rm = R / h
            V = (4.0 / 3.0) * np.pi * Rm ** 3
            mu = nt * V                                   # expected counts from the selection
            cols.append(np.log((counts[R] + EPS) / (mu + EPS)))
        # nearest-neighbour distance (k=2: first is self)
        dnn, _ = tree.query(pos, k=2)
        cols.append(np.log(dnn[:, 1] + EPS))
        X = np.column_stack(cols).astype(np.float32)
        assert X.shape[1] == len(FEAT_NAMES)
        np.savez_compressed(args.out_dir / f"aperture_{tag}.npz", X=X, names=np.array(FEAT_NAMES))
        # quick informativeness readout vs the truth (diagnostic only, NOT a gate)
        from scipy.stats import spearmanr
        l1 = t["LAMBDA1"].astype(float)
        sp7 = spearmanr(X[:, 0], l1).statistic
        spc7 = spearmanr(X[:, 3], l1).statistic
        spnn = spearmanr(X[:, 6], l1).statistic
        print(f"[{tag}] n={len(z):6d}  medN7={np.median(counts[7.0]):6.1f}  "
              f"Spearman(logN_ap7,λ1)={sp7:+.3f}  (contrast_ap7,λ1)={spc7:+.3f}  (log_dNN,λ1)={spnn:+.3f}")
    print(f"\nSaved aperture features -> {args.out_dir}  cols={FEAT_NAMES}")


if __name__ == "__main__":
    main()
