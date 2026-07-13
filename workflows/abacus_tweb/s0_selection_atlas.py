#!/usr/bin/env python3
"""S0 — full-z-range selection atlas (roadmap §3b, S-track).

Produces the three things every downstream S-gate needs:
  1. n(z) 0.03-0.62 for DESI DR2 and the sentinelfix mock (wedge box = known solid
     angle, both catalogs complete there) + the mock/DESI ratio per shell.
  2. Smooth ñ(z) CONDITIONING FUNCTIONS for BOTH datasets (spline of log n(z),
     evaluated on a fine grid; each dataset conditions on its OWN ñ — the design
     decision that dissolves the high-z mock deficit).
  3. Per-broad-shell STRUCTURAL stats quantifying architecture OOD across the range:
     mean spacing, NN distance, degree within the 10 Mpc/h union radius (graph side),
     and voxel occupancy at 5/6 Mpc cells (grid side — the CNN/F-tier stress preview
     the P2 caveat says is untested).

CPU only. Outputs: atlas JSON + npz + themed figure in a new scratch dir.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline

_ILL = Path(os.environ.get("ILLUSTRIS_ROOT", "/global/homes/d/dkololgi/TNG/Illustris")).resolve()
if str(_ILL) not in sys.path:
    sys.path.insert(0, str(_ILL))
from shared.plot_style import apply_style, ACCENT_COLORS  # noqa: E402

DESI = "/pscratch/sd/d/dkololgi/graphweb_desi/catalogs/bgs_maglim_bright_galaxy_zwarn0_dchi2ge25.fits"
MOCK = "/pscratch/sd/d/dkololgi/abacus/mock_bgs_maglim_sentinelfix.fits"
UNION_R_MPC = 14.78          # 10 Mpc/h comoving
BROAD = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.62)]


def load(path, ra_box, dec_box):
    names = [c.upper() for c in fitsio.FITS(path)[1].get_colnames()]
    ra_c = "RA" if "RA" in names else "TARGET_RA"
    dec_c = "DEC" if "DEC" in names else "TARGET_DEC"
    t = fitsio.read(path, columns=[ra_c, dec_c, "Z"])
    sel = ((t[ra_c] >= ra_box[0]) & (t[ra_c] < ra_box[1]) &
           (t[dec_c] >= dec_box[0]) & (t[dec_c] < dec_box[1]))
    return (np.asarray(t[ra_c][sel], np.float64), np.asarray(t[dec_c][sel], np.float64),
            np.asarray(t["Z"][sel], np.float64))


def to_xyz(ra, dec, z):
    zt = np.linspace(0, 0.75, 6000)
    dt = cosmo.comoving_distance(zt).value
    d = np.interp(z, zt, dt)
    r, dd = np.deg2rad(ra), np.deg2rad(dec)
    return np.vstack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)]).T


def nz_and_spline(z, edges, omega):
    cnt = np.histogram(z, bins=edges)[0].astype(float)
    dcom = cosmo.comoving_distance(edges).value
    vshell = (omega / 3.0) * (dcom[1:] ** 3 - dcom[:-1] ** 3)
    n = cnt / vshell
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 20
    sp = UnivariateSpline(mid[ok], np.log(n[ok]), w=np.sqrt(cnt[ok]),
                          s=len(mid[ok]) * 2.0, k=3)
    zg = np.arange(edges[0], edges[-1] + 1e-9, 0.002)
    ntilde = np.exp(sp(zg))
    return cnt, n, mid, zg, ntilde


def shell_stats(pos, z, lo, hi, n_shell):
    m = (z >= lo) & (z < hi)
    if m.sum() < 200:
        return None
    p = pos[m]
    tree = cKDTree(p)
    sub = p[np.random.default_rng(0).permutation(len(p))[: min(20000, len(p))]]
    dnn = tree.query(sub, k=2)[0][:, 1]
    deg = tree.query_ball_point(sub, UNION_R_MPC, return_length=True) - 1
    # voxel occupancy in the shell's bounding box (grid-OOD preview)
    occ = {}
    for cell in (5.0, 6.0):
        ix = np.floor((p - p.min(0)) / cell).astype(np.int64)
        dims = ix.max(0) + 1
        # occupancy within the wedge shell volume approximated by occupied bbox voxels
        keys = (ix[:, 0] * dims[1] + ix[:, 1]) * dims[2] + ix[:, 2]
        nocc = len(np.unique(keys))
        ntot = int(np.prod(dims))
        occ[f"occ_frac_cell{int(cell)}"] = nocc / ntot
        occ[f"mean_count_occ_cell{int(cell)}"] = len(p) / nocc
    return dict(n_gal=int(m.sum()), nbar=float(n_shell),
                spacing=float(n_shell ** (-1 / 3)) if n_shell > 0 else None,
                med_nn=float(np.median(dnn)), med_deg_unionR=float(np.median(deg)),
                frac_deg0=float((deg == 0).mean()), **occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", nargs=2, type=float, default=[120, 160])
    ap.add_argument("--dec", nargs=2, type=float, default=[14.5, 30.6])
    ap.add_argument("--zmin", type=float, default=0.03)
    ap.add_argument("--zmax", type=float, default=0.62)
    ap.add_argument("--dz", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/s0_selection_atlas"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    dra = np.deg2rad(args.ra[1] - args.ra[0])
    sind = np.sin(np.deg2rad(args.dec[1])) - np.sin(np.deg2rad(args.dec[0]))
    omega = dra * sind
    edges = np.arange(args.zmin, args.zmax + args.dz / 2, args.dz)

    out = {"omega_sr": omega, "union_radius_mpc": UNION_R_MPC}
    figdata = {}
    for tag, path in (("desi", DESI), ("mock", MOCK)):
        ra, dec, z = load(path, args.ra, args.dec)
        if tag == "mock":   # sentinel guard (should be ~0 post-fix; assert visibility)
            sw = ((z >= 0.585) & (z < 0.595)).mean()
            print(f"[mock] sentinel-window frac = {sw:.5f} (expect ~2e-5)")
        cnt, n, mid, zg, ntilde = nz_and_spline(z, edges, omega)
        out[f"{tag}_count"] = cnt.tolist()
        out[f"{tag}_n"] = n.tolist()
        out["z_mid"] = mid.tolist()
        out[f"{tag}_ntilde_grid_z"] = zg.tolist()
        out[f"{tag}_ntilde"] = ntilde.tolist()
        pos = to_xyz(ra, dec, z)
        shells = {}
        print(f"\n[{tag}] per-shell structural stats (union R = {UNION_R_MPC} Mpc):")
        print("  shell        N        nbar       spacing  medNN   medDeg  deg0%   occ5   cnt|occ5")
        for lo, hi in BROAD:
            msk = (mid >= lo) & (mid < hi)
            nbar = float(np.mean(n[msk])) if msk.any() else 0.0
            s = shell_stats(pos, z, lo, hi, nbar)
            if s is None:
                print(f"  {lo:.2f}-{hi:.2f}  (too few)"); continue
            shells[f"{lo:.2f}-{hi:.2f}"] = s
            print(f"  {lo:.2f}-{hi:.2f} {s['n_gal']:8d}  {s['nbar']:.3e}  {s['spacing']:6.1f}  "
                  f"{s['med_nn']:5.1f}  {s['med_deg_unionR']:5.1f}  {100*s['frac_deg0']:5.2f}  "
                  f"{s['occ_frac_cell5']:.3f}  {s['mean_count_occ_cell5']:.2f}")
        out[f"{tag}_shell_stats"] = shells
        figdata[tag] = (mid, n, zg, ntilde)

    ratio = np.array(out["mock_count"]) / np.maximum(np.array(out["desi_count"]), 1)
    out["mock_over_desi_count_ratio"] = ratio.tolist()
    (args.out_dir / "s0_atlas.json").write_text(json.dumps(out, indent=1) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    for tag, c in (("desi", ACCENT_COLORS["magenta"]), ("mock", ACCENT_COLORS["blue"])):
        mid, n, zg, ntilde = figdata[tag]
        ax1.semilogy(mid, n, ".", ms=4, color=c, alpha=0.6)
        ax1.semilogy(zg, ntilde, "-", lw=2, color=c, label=f"{tag} ñ(z) spline")
    ax1.axvspan(0.2, 0.3, alpha=0.10, color="#9a9a93", label="training wedge (so far)")
    ax1.set_xlabel("z"); ax1.set_ylabel(r"$n(z)$ [Mpc$^{-3}$]"); ax1.legend(fontsize=9)
    ax1.set_title("S0: selection functions — the ñ(z) conditioning inputs")
    ax2.plot(np.array(out["z_mid"]), ratio, "-", lw=2, color=ACCENT_COLORS["red"])
    ax2.axhline(1, ls=":", color="#9a9a93"); ax2.set_ylim(0, 1.6)
    ax2.set_xlabel("z"); ax2.set_ylabel("mock / DESI counts")
    ax2.set_title("mock deficit grows with z (condition on ñ, not z)")
    fig.savefig(args.out_dir / "s0_selection_atlas.png", bbox_inches="tight", dpi=180)
    print(f"\nSaved: {args.out_dir}/s0_atlas.json + s0_selection_atlas.png")


if __name__ == "__main__":
    main()
