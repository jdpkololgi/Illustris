#!/usr/bin/env python3
"""Classical tidal-reconstruction baseline for the path1 wedge (no ML).

Measures how much of the GraphNet's eigenvalue-regression performance is
recoverable by textbook density reconstruction + an exact FFT tidal solve:

  galaxies (observer-frame, redshift-space, Mpc)
    -> density estimate on a grid          [CIC / CIC+Gauss / DTFE / Wiener]
    -> T_ij(k) = (k_i k_j / k^2) W_R(k) delta_k   (same convention as cactus)
    -> eigenvalues at galaxy positions (ascending, = LAMBDA1/2/3 convention)
    -> per-eigenvalue R2 / Spearman on the SAME test split as the GraphNet
       baseline (path1 lineareig_si cache masks).

This bounds the headroom available to learned models (GraphNet, equivariant,
field-level): whatever a Wiener-filtered linear reconstruction already gets is
not evidence of representation learning. See docs/plan_field_level_multimodal.md.

Modes
-----
wedge            main baseline (default)
validate-solver  voxel-level check of the FFT tidal solve against the stored
                 cactus eig_vals slabs on a subbox of the 10% particle grid.

Run on a CPU compute node (cosmic_env). Example:
  srun -n 1 -c 64 --cpu-bind=cores bash -lc '... classical_tidal_baseline.py --mode wedge'
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------- paths
WEDGE_DIR = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign")
WEDGE_PREFIX = "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3"
CACHE_PATH = Path(
    "/pscratch/sd/d/dkololgi/abacus/sbi_caches/"
    "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_sbi_cache_3d_lineareig_si.pkl"
)
DENSITY_GRID_PATH = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/"
    "AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy"
)
TWEB_SLAB_DIR = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v3/"
    "dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/"
    "backend_optimized_ngrid_2048_rsmooth_7"
)
DEFAULT_OUT = Path("/pscratch/sd/d/dkololgi/abacus/classical_baseline")

# Abacus c000 cosmology; coordinates were built with astropy Planck18 (h=0.6766)
# but labels are index-matched, so only the smoothing-scale conversion uses h and
# the 0.4% difference is negligible.
LITTLE_H = 0.6736
RSMOOTH_MPC_H = 7.0                      # truth smoothing (cactus Rsmooth)
RSMOOTH_MPC = RSMOOTH_MPC_H / LITTLE_H   # in the wedge's Mpc coordinates

# Wedge footprint (from the wedge name; used for the analytic survey mask)
RA_MIN, RA_MAX = 120.0, 160.0
DEC_MIN, DEC_MAX = 14.5, 30.6

GRAPHNET_BASELINE = {"lambda1": 0.775, "lambda2": 0.811, "lambda3": 0.891}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("wedge", "validate-solver"), default="wedge")
    p.add_argument("--cell-mpc", type=float, default=3.0, help="grid cell size (Mpc)")
    p.add_argument("--pad-mpc", type=float, default=100.0, help="zero padding per side (Mpc)")
    p.add_argument(
        "--estimators",
        type=str,
        default="cic,cic_g3,cic_g5,dtfe,wiener",
        help="comma list from {cic,cic_g3,cic_g5,dtfe,wiener}",
    )
    p.add_argument("--interior-margin-mpc", type=float, default=25.0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    # wedge overrides (defaults = the RA120-160 training wedge; added for the RA200-240 transfer wedge)
    p.add_argument("--wedge-prefix", type=str, default=None,
                   help="Override WEDGE_PREFIX (points_xyz/wedge_targets under WEDGE_DIR).")
    p.add_argument("--cache-path", type=Path, default=None,
                   help="Override CACHE_PATH (supplies the scoring masks; eigenvalue row-alignment is verified).")
    p.add_argument("--ra-min", type=float, default=None, help="Override wedge RA_MIN (deg).")
    p.add_argument("--ra-max", type=float, default=None, help="Override wedge RA_MAX (deg).")
    # validate-solver options (grid units: cells of the 2048^3 box, 0.9766 Mpc/h)
    p.add_argument("--val-origin", type=int, nargs=3, default=(512, 512, 512))
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--val-interior-cells", type=int, default=64,
                   help="cells trimmed from each subbox face before comparing")
    return p.parse_args()


# ----------------------------------------------------------------------------- tidal solve
def tidal_components(delta: np.ndarray, cell: float, rsmooth: float) -> dict[str, np.ndarray]:
    """T_ij(k) = (k_i k_j / k^2) exp(-0.5 (k R)^2) delta_k -> 6 unique T_ij grids.

    Matches cactus run_tweb (periodic FFT branch): phi_k = -delta_k/k^2 and
    T_ij = d_i d_j phi  ==>  T_ij(k) = (k_i k_j / k^2) delta_k.
    """
    shape = delta.shape
    dk = np.fft.rfftn(delta)
    ks = [np.fft.fftfreq(n, d=cell) * 2.0 * np.pi for n in shape[:-1]]
    ks.append(np.fft.rfftfreq(shape[-1], d=cell) * 2.0 * np.pi)
    kx, ky, kz = np.meshgrid(*ks, indexing="ij", sparse=True)
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    smooth = np.exp(-0.5 * k2 * rsmooth**2)
    smooth[0, 0, 0] = 0.0
    base = dk * smooth / k2
    del dk, smooth
    kvec = {"x": kx, "y": ky, "z": kz}
    out = {}
    for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
        out[a + b] = np.fft.irfftn(kvec[a] * kvec[b] * base, s=shape, axes=(0, 1, 2)).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- wedge helpers
def load_wedge():
    xyz = np.load(WEDGE_DIR / f"{WEDGE_PREFIX}_points_xyz.npy")  # (N,3) Mpc, observer frame
    from astropy.io import fits

    with fits.open(WEDGE_DIR / f"{WEDGE_PREFIX}_wedge_targets.fits") as h:
        t = h[1].data
        lam = np.stack(
            [np.asarray(t["LAMBDA1"]), np.asarray(t["LAMBDA2"]), np.asarray(t["LAMBDA3"])], axis=1
        ).astype(np.float64)
        ra = np.asarray(t["RA"], dtype=np.float64)
        dec = np.asarray(t["DEC"], dtype=np.float64)
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    if not np.allclose(np.asarray(cache["eigenvalues_raw"]), lam, atol=1e-6):
        raise RuntimeError("cache eigenvalues_raw != wedge FITS LAMBDA columns; row alignment broken")
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    return xyz, lam, ra, dec, train, val, test


def radial_nbar(r: np.ndarray, omega_sr: float, bin_mpc: float = 10.0):
    """Smoothed n(r) [gal/Mpc^3] within the wedge solid angle."""
    edges = np.arange(r.min() - bin_mpc, r.max() + 2 * bin_mpc, bin_mpc)
    counts, edges = np.histogram(r, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = omega_sr * centers**2 * np.diff(edges)
    nbar = gaussian_filter1d(counts.astype(np.float64), sigma=2.0) / shell_vol
    return centers, nbar


class WedgeGrid:
    """Padded Cartesian grid around the wedge with an apodized survey mask."""

    def __init__(self, xyz: np.ndarray, cell: float, pad: float):
        self.cell = cell
        self.lo = xyz.min(axis=0) - pad
        hi = xyz.max(axis=0) + pad
        self.shape = tuple(int(np.ceil((hi - self.lo)[i] / cell)) for i in range(3))
        print(f"grid shape {self.shape} = {np.prod(self.shape)/1e6:.1f}M cells, cell={cell} Mpc")
        ax = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * cell for i in range(3)]
        self.gx, self.gy, self.gz = np.meshgrid(*ax, indexing="ij", sparse=True)

    def point_coords(self, xyz: np.ndarray) -> np.ndarray:
        return ((xyz - self.lo) / self.cell - 0.5).T  # for map_coordinates

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
    ok = mu > mu_floor * float(mu[mu > 0].mean() if np.any(mu > 0) else 1.0)
    delta[ok] = counts[ok] / mu[ok] - 1.0
    return delta


def dtfe_point_density(xyz: np.ndarray) -> np.ndarray:
    """Classic DTFE: rho_i = (D+1) / V_star_i with V_star = sum of incident tet volumes."""
    tri = Delaunay(xyz)
    simp = tri.simplices
    a, b, c, d = (xyz[simp[:, i]] for i in range(4))
    vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
    vstar = np.zeros(len(xyz))
    for col in range(4):
        np.add.at(vstar, simp[:, col], vol)
    rho = np.zeros(len(xyz))
    ok = vstar > 0
    rho[ok] = 4.0 / vstar[ok]
    return rho, tri


def wiener_filter_delta(delta: np.ndarray, cell: float) -> np.ndarray:
    """Data-driven Wiener filter: W(k) = (P(k) - P_noise)/P(k), noise from the high-k plateau."""
    dk = np.fft.rfftn(delta)
    ks = [np.fft.fftfreq(n, d=cell) * 2.0 * np.pi for n in delta.shape[:-1]]
    ks.append(np.fft.rfftfreq(delta.shape[-1], d=cell) * 2.0 * np.pi)
    kx, ky, kz = np.meshgrid(*ks, indexing="ij", sparse=True)
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    p3d = np.abs(dk) ** 2
    kmax = kmag.max()
    edges = np.linspace(0, kmax, 81)
    which = np.digitize(kmag.ravel(), edges)
    psum = np.bincount(which, weights=p3d.ravel(), minlength=82)
    pcnt = np.bincount(which, minlength=82)
    pk = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), 0.0)
    # noise plateau: mean of the top-quartile-k bins
    hi = pk[int(0.75 * 80):81]
    pnoise = float(np.mean(hi[hi > 0])) if np.any(hi > 0) else 0.0
    wk = np.clip((pk - pnoise) / np.maximum(pk, 1e-30), 0.0, 1.0)
    w3d = wk[which].reshape(kmag.shape)
    return np.fft.irfftn(dk * w3d, s=delta.shape, axes=(0, 1, 2)).astype(np.float32)


# ----------------------------------------------------------------------------- scoring
def score_predictions(pred, truth, train, test, label, results, interior=None):
    """Per-eigenvalue raw R2, train-calibrated R2, Spearman on the test split."""
    entry = {}
    for k, nm in enumerate(("lambda1", "lambda2", "lambda3")):
        p, y = pred[:, k], truth[:, k]
        finite = np.isfinite(p)
        te = test & finite
        tr = train & finite
        A = np.stack([p[tr], np.ones(tr.sum())], axis=1)
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p_cal = coef[0] * p + coef[1]
        entry[nm] = {
            "r2_raw": float(r2_score(y[te], p[te])),
            "r2_cal": float(r2_score(y[te], p_cal[te])),
            "spearman": float(spearmanr(y[te], p[te]).statistic),
            "n_test": int(te.sum()),
            "frac_finite": float(finite.mean()),
        }
        if interior is not None:
            ti = te & interior
            entry[nm]["r2_cal_interior"] = float(r2_score(y[ti], p_cal[ti]))
            entry[nm]["n_test_interior"] = int(ti.sum())
    results[label] = entry
    print(f"  {label}: " + "  ".join(
        f"{nm} R2cal={entry[nm]['r2_cal']:.3f} (raw {entry[nm]['r2_raw']:.3f})"
        for nm in ("lambda1", "lambda2", "lambda3")))


def print_table(results, interior_col: bool):
    hdr = "| Estimator | λ1 R² (cal) | λ2 R² (cal) | λ3 R² (cal) | λ1 raw | λ1 ρ_s |"
    if interior_col:
        hdr += " λ1 R² int |"
    print("\n" + hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    rows = []
    for label, e in results.items():
        row = (
            f"| {label} | {e['lambda1']['r2_cal']:.3f} | {e['lambda2']['r2_cal']:.3f} "
            f"| {e['lambda3']['r2_cal']:.3f} | {e['lambda1']['r2_raw']:.3f} "
            f"| {e['lambda1']['spearman']:.3f} |"
        )
        if interior_col:
            row += f" {e['lambda1'].get('r2_cal_interior', float('nan')):.3f} |"
        rows.append(row)
        print(row)
    gb = GRAPHNET_BASELINE
    print(f"| GraphNet+NPE (Delaunay, curated) | {gb['lambda1']:.3f} | {gb['lambda2']:.3f} "
          f"| {gb['lambda3']:.3f} | — | — |" + (" — |" if interior_col else ""))
    return [hdr] + rows


# ----------------------------------------------------------------------------- modes
def run_wedge(args) -> None:
    t0 = time.time()
    xyz, lam, ra_gal, dec_gal, train, val, test = load_wedge()
    n = len(xyz)
    r_gal = np.linalg.norm(xyz, axis=1)
    print(f"wedge: {n:,} galaxies, r in [{r_gal.min():.0f},{r_gal.max():.0f}] Mpc; "
          f"Rsmooth = {RSMOOTH_MPC:.2f} Mpc (= {RSMOOTH_MPC_H} Mpc/h)")

    grid = WedgeGrid(xyz, cell=args.cell_mpc, pad=args.pad_mpc)
    omega = np.radians(RA_MAX - RA_MIN) * (np.sin(np.radians(DEC_MAX)) - np.sin(np.radians(DEC_MIN)))
    r_centers, nbar = radial_nbar(r_gal, omega)
    mask = grid.survey_mask(r_gal)
    mu = grid.expected_counts(r_centers, nbar, mask)
    counts = grid.cic_deposit(xyz)
    print(f"deposited counts sum={counts.sum():.0f} (N={n}); expected-counts sum={mu.sum():.0f}")

    # interior mask (away from wedge edges) for the edge-effect diagnostic
    m = args.interior_margin_mpc
    ang_margin = np.degrees(m / r_gal)
    interior = (
        (ra_gal > RA_MIN + ang_margin) & (ra_gal < RA_MAX - ang_margin)
        & (dec_gal > DEC_MIN + ang_margin) & (dec_gal < DEC_MAX - ang_margin)
        & (r_gal > np.quantile(r_gal, 0.001) + m) & (r_gal < np.quantile(r_gal, 0.999) - m)
    )
    print(f"interior galaxies: {interior.sum():,}/{n:,}")

    coords = grid.point_coords(xyz)
    results: dict = {}
    wanted = [e.strip() for e in args.estimators.split(",") if e.strip()]

    def eval_delta(delta, label):
        comps = tidal_components(delta, grid.cell, RSMOOTH_MPC)
        # trace check: sum of diagonal = smoothed delta (Poisson consistency)
        tr_grid = comps["xx"] + comps["yy"] + comps["zz"]
        dk = np.fft.rfftn(delta)
        ks = [np.fft.fftfreq(nn, d=grid.cell) * 2 * np.pi for nn in delta.shape[:-1]]
        ks.append(np.fft.rfftfreq(delta.shape[-1], d=grid.cell) * 2 * np.pi)
        kx, ky, kz = np.meshgrid(*ks, indexing="ij", sparse=True)
        sm = np.exp(-0.5 * (kx**2 + ky**2 + kz**2) * RSMOOTH_MPC**2)
        sm[0, 0, 0] = 0.0
        ds = np.fft.irfftn(dk * sm, s=delta.shape, axes=(0, 1, 2))
        tr_err = float(np.max(np.abs(tr_grid - ds)) / max(np.max(np.abs(ds)), 1e-30))
        print(f"  [{label}] trace-vs-smoothed-delta max rel err: {tr_err:.2e}")
        tij = np.empty((n, 3, 3), dtype=np.float64)
        idx = {"x": 0, "y": 1, "z": 2}
        for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
            v = map_coordinates(comps[a + b], coords, order=1, mode="nearest")
            tij[:, idx[a], idx[b]] = v
            tij[:, idx[b], idx[a]] = v
        pred = np.linalg.eigvalsh(tij)  # ascending, matches LAMBDA1<=2<=3
        score_predictions(pred, lam, train, test, label, results, interior=interior)
        np.save(args.out_dir / f"pred_eigs_{label}.npy", pred.astype(np.float32))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "cic" in wanted:
        eval_delta(delta_from_counts(counts, mu), "cic")
    for est, rext in (("cic_g3", 3.0), ("cic_g5", 5.0)):
        if est in wanted:
            rext_mpc = rext / LITTLE_H
            sm_counts = gaussian_filter(counts, sigma=rext_mpc / grid.cell)
            sm_mu = gaussian_filter(mu, sigma=rext_mpc / grid.cell)
            eval_delta(delta_from_counts(sm_counts, sm_mu), f"{est}")
    if "wiener" in wanted:
        eval_delta(wiener_filter_delta(delta_from_counts(counts, mu), grid.cell), "wiener")
    if "dtfe" in wanted:
        print("  DTFE: triangulating + star volumes...")
        rho, tri = dtfe_point_density(xyz)
        interp = LinearNDInterpolator(tri, rho, fill_value=0.0)
        # evaluate on grid points inside the survey mask only (hull is bigger than wedge)
        gxyz = np.stack(np.meshgrid(
            grid.lo[0] + (np.arange(grid.shape[0]) + 0.5) * grid.cell,
            grid.lo[1] + (np.arange(grid.shape[1]) + 0.5) * grid.cell,
            grid.lo[2] + (np.arange(grid.shape[2]) + 0.5) * grid.cell,
            indexing="ij"), axis=-1).reshape(-1, 3)
        sel = (mask > 0.05).ravel()
        rho_grid = np.zeros(gxyz.shape[0], dtype=np.float32)
        chunk = 2_000_000
        idxs = np.nonzero(sel)[0]
        for s in range(0, len(idxs), chunk):
            ii = idxs[s:s + chunk]
            rho_grid[ii] = interp(gxyz[ii])
        rho_grid = rho_grid.reshape(grid.shape)
        nbar_grid = np.zeros_like(rho_grid)
        pos = mu > 0
        nbar_grid[pos] = mu[pos] / (grid.cell**3)
        delta_dtfe = np.zeros_like(rho_grid)
        ok = nbar_grid > 0
        delta_dtfe[ok] = np.clip(rho_grid[ok] / nbar_grid[ok] - 1.0, -1.0, 200.0) * mask[ok]
        eval_delta(delta_dtfe, "dtfe")

    table_lines = print_table(results, interior_col=True)
    payload = {
        "mode": "wedge",
        "cache": str(CACHE_PATH),
        "cell_mpc": args.cell_mpc,
        "pad_mpc": args.pad_mpc,
        "rsmooth_mpc": RSMOOTH_MPC,
        "graphnet_baseline": GRAPHNET_BASELINE,
        "results": results,
        "table": table_lines,
        "runtime_s": time.time() - t0,
    }
    out = args.out_dir / "classical_baseline_scores.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}  ({payload['runtime_s']:.0f}s)")


def run_validate_solver(args) -> None:
    """Recompute eigenvalues on a subbox of the 10% particle grid; compare to cactus."""
    t0 = time.time()
    ngrid, box = 2048, 2000.0
    cell = box / ngrid
    i0, j0, k0 = args.val_origin
    s = args.val_size
    dens = np.load(DENSITY_GRID_PATH, mmap_mode="r")
    # global mean via x-slab chunks (cactus normalized by the FULL-box mean)
    gsum, cnt = 0.0, 0
    for a in range(0, ngrid, 128):
        blk = np.asarray(dens[a:a + 128], dtype=np.float64)
        gsum += blk.sum()
        cnt += blk.size
    gmean = gsum / cnt
    print(f"global mean density: {gmean:.6f}  ({time.time()-t0:.0f}s)")
    sub = np.asarray(dens[i0:i0 + s, j0:j0 + s, k0:k0 + s], dtype=np.float32)
    delta = sub / gmean - 1.0
    del sub
    print(f"subbox {s}^3 at ({i0},{j0},{k0}), delta range [{delta.min():.2f},{delta.max():.2f}]")

    comps = tidal_components(delta, cell=cell, rsmooth=RSMOOTH_MPC_H)
    trim = args.val_interior_cells
    sl = slice(trim, s - trim)
    m = s - 2 * trim
    tens = np.empty((m, m, m, 3, 3), dtype=np.float32)
    idx = {"x": 0, "y": 1, "z": 2}
    for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
        v = comps[a + b][sl, sl, sl]
        tens[..., idx[a], idx[b]] = v
        tens[..., idx[b], idx[a]] = v
    del comps
    mine = np.linalg.eigvalsh(tens).astype(np.float32)  # ascending
    del tens

    # stored cactus eigenvalues for the interior x-planes
    truth = np.empty((m, m, m, 3), dtype=np.float32)
    x_lo, x_hi = i0 + trim, i0 + s - trim
    for rank_file in sorted(TWEB_SLAB_DIR.glob("*.npz")):
        with np.load(rank_file) as d:
            xs, xe = int(d["x_start"]), int(d["x_end"])
            if xe <= x_lo or xs >= x_hi:
                continue
            ev = d["eig_vals"]  # (3, nx_local, ngrid, ngrid), ascending assumed
            lo, hi = max(xs, x_lo), min(xe, x_hi)
            truth[lo - x_lo:hi - x_lo] = np.moveaxis(
                ev[:, lo - xs:hi - xs, j0 + trim:j0 + s - trim, k0 + trim:k0 + s - trim], 0, -1
            )
    truth = np.sort(truth, axis=-1)  # guard against ordering-convention drift
    report = {}
    for k, nm in enumerate(("lambda1", "lambda2", "lambda3")):
        a, b = truth[..., k].ravel(), mine[..., k].ravel()
        report[nm] = {
            "r2": float(r2_score(a, b)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "truth_std": float(a.std()),
        }
        print(f"  {nm}: R2={report[nm]['r2']:.5f}  rms={report[nm]['rms_diff']:.4f} "
              f"(truth std {report[nm]['truth_std']:.4f})  max|d|={report[nm]['max_abs_diff']:.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "solver_validation.json"
    with open(out, "w") as f:
        json.dump({"mode": "validate-solver", "origin": [i0, j0, k0], "size": s,
                   "trim": trim, "global_mean": gmean, "report": report,
                   "runtime_s": time.time() - t0}, f, indent=2)
    print(f"wrote {out}  ({time.time()-t0:.0f}s)")


def main() -> None:
    args = parse_args()
    global WEDGE_PREFIX, CACHE_PATH, RA_MIN, RA_MAX
    if args.wedge_prefix is not None:
        WEDGE_PREFIX = args.wedge_prefix
    if args.cache_path is not None:
        CACHE_PATH = args.cache_path
    if args.ra_min is not None:
        RA_MIN = args.ra_min
    if args.ra_max is not None:
        RA_MAX = args.ra_max
    if args.mode == "wedge":
        run_wedge(args)
    else:
        run_validate_solver(args)


if __name__ == "__main__":
    main()
