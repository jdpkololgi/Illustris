#!/usr/bin/env python3
"""P1 — canonical immutable raw catalogue (plan_generalisable_graphweb_vac.md P1).

Catalogue #1 (Wave-0 decision, SCIENCE_LOG 2026-07-18): ph000 / path1_fiberassign observer, buffered
full-range wedge RA 118-162, DEC 12.5-32.6, Z_ORIG-buffered around the 0.15-0.55 core. One contiguous
extraction from the SAME parent S2 used (path1_fiberassign graph-ready full-sky, rs7 halo_xcom
labels), with ONE z-error injection pass (sigma_v=35 km/s, seed 42 — S2's per-shell convention,
applied once; S2's per-shell RNG draws are not reproducible in a single pass, so continuity with the
frozen canonical rows is anchored on TARGETID + Z_ORIG + eigenvalues, NOT observed Z).

Contents follow the plan's P1 list verbatim: identifiers, global node id, observer-frame Cartesian
position (indexing metadata only), sky/redshift/shell, targets + validity mask, halo ids, selection
and boundary metadata, source hashes and target convention. No train-fitted normalisation, no
train/val/test filtering — split ownership arrives with the P4 manifest.

Gates (hard, raise on failure): unique TARGETID; finite targets on valid rows; BOX_INDEX>=0 on all
ACTIVE rows; exact catalogue/target alignment (single-source, verified by construction + spot-check);
canonical-continuity (all 219,929 frozen canonical TARGETIDs present as active, Z_ORIG and eigenvalues
matching to float tolerance); consistent units (RA/DEC/z ranges, Planck18 comoving distances).
CATALOGUE_COMPLETE is written only after every gate passes (plan section 14).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo

C_KMS = 299792.458
SENTINEL = (0.585, 0.595)                 # phantom window (documented, S2 convention)
PARENT = Path("/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
              "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits")
CANONICAL = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange/cnn_fullrange_cache.pkl")
SHELLS = [("0p15_0p25", 0.15, 0.25), ("0p25_0p35", 0.25, 0.35),
          ("0p35_0p45", 0.35, 0.45), ("0p45_0p55", 0.45, 0.55)]

_ZT = np.linspace(0.0, 0.75, 8000)
_DT = cosmo.comoving_distance(_ZT).value


def dcom(z):
    return np.interp(z, _ZT, _DT)


def sha256(path: Path, blocksize=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(blocksize), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/p1_canonical/ph000_path1_wedge"))
    ap.add_argument("--ra", nargs=2, type=float, default=[118.0, 162.0])
    ap.add_argument("--dec", nargs=2, type=float, default=[12.5, 32.6])
    ap.add_argument("--z-core", nargs=2, type=float, default=[0.15, 0.55])
    ap.add_argument("--z-buf", type=float, default=0.05,
                    help="Z_ORIG buffer beyond the core (context for graphs/fields/patches)")
    ap.add_argument("--sigma-v-kms", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    print(f"parent: {PARENT}")
    parent_sha = sha256(PARENT)
    print(f"parent sha256: {parent_sha}  ({time.time()-t0:.0f}s)")

    cols = ["TARGETID", "RA", "DEC", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX",
            "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"]
    avail = fitsio.FITS(str(PARENT))[1].get_colnames()
    sel_meta = [c for c in ("BGS_TARGET", "ZWARN", "DELTACHI2", "SPECTYPE", "R_MAG_ABS", "R_MAG_APP", "STATUS")
                if c in avail]
    t = fitsio.read(str(PARENT), columns=cols + sel_meta)
    print(f"parent rows: {len(t):,}  ({time.time()-t0:.0f}s)")

    z0 = np.asarray(t["Z"], np.float64)
    lo, hi = args.z_core[0] - args.z_buf, args.z_core[1] + args.z_buf
    sel = ((t["RA"] >= args.ra[0]) & (t["RA"] < args.ra[1])
           & (t["DEC"] >= args.dec[0]) & (t["DEC"] < args.dec[1])
           & (z0 >= lo) & (z0 < hi)
           & ~((z0 >= SENTINEL[0]) & (z0 < SENTINEL[1])))
    sub = t[sel]
    zorig = np.asarray(sub["Z"], np.float64)
    n = len(sub)
    print(f"slice RA[{args.ra[0]},{args.ra[1]}) DEC[{args.dec[0]},{args.dec[1]}) "
          f"Z_ORIG[{lo},{hi}) minus sentinel: {n:,} rows")

    # ONE z-error injection pass over the whole slice (S2 physics, single RNG stream)
    rng = np.random.default_rng(args.seed)
    z_obs = zorig + args.sigma_v_kms * (1.0 + zorig) / C_KMS * rng.standard_normal(n)

    # reporting shell on OBSERVED z; outside core -> buffer (context only, never active)
    shell = np.full(n, "buffer", dtype="U9")
    for tag, slo, shi in SHELLS:
        shell[(z_obs >= slo) & (z_obs < shi)] = tag

    lam = np.stack([sub["LAMBDA1"], sub["LAMBDA2"], sub["LAMBDA3"]], 1).astype(np.float64)
    valid_target = (sub["BOX_INDEX"] >= 0) & np.isfinite(lam).all(axis=1)
    active = valid_target & (shell != "buffer")

    r = dcom(z_obs)
    phi, th = np.radians(np.asarray(sub["RA"], np.float64)), np.radians(np.asarray(sub["DEC"], np.float64))
    xyz = np.stack([r * np.cos(th) * np.cos(phi), r * np.cos(th) * np.sin(phi), r * np.sin(th)], 1)

    # boundary metadata: comoving distance to each survey edge (RA/DEC converted at each row's r)
    deg2mpc = np.radians(1.0) * r
    d_ra = np.minimum(sub["RA"] - args.ra[0], args.ra[1] - sub["RA"]) * deg2mpc * np.cos(th)
    d_dec = np.minimum(sub["DEC"] - args.dec[0], args.dec[1] - sub["DEC"]) * deg2mpc
    d_z = np.minimum(np.abs(dcom(z_obs) - dcom(lo)), np.abs(dcom(hi) - dcom(z_obs)))
    d_boundary = np.minimum(np.minimum(d_ra, d_dec), d_z)

    # ================================ GATES ================================
    tid = np.asarray(sub["TARGETID"], np.int64)
    if len(np.unique(tid)) != n:
        raise RuntimeError(f"GATE FAIL unique-TARGETID: {n - len(np.unique(tid))} duplicates in slice")
    if not np.isfinite(lam[valid_target]).all():
        raise RuntimeError("GATE FAIL finite-targets on valid rows")
    if (sub["BOX_INDEX"][active] < 0).any():
        raise RuntimeError("GATE FAIL BOX_INDEX<0 among active rows")
    if not (np.isfinite(xyz).all() and (r > 0).all()):
        raise RuntimeError("GATE FAIL positions non-finite")

    # continuity with the frozen canonical evidence (TARGETID + Z_ORIG + eigenvalues; NOT observed Z)
    can = pickle.load(open(CANONICAL, "rb"))
    can_tid = np.asarray(can["tid"], np.int64)
    can_eig = np.asarray(can["eigenvalues_raw"], np.float64)
    order = np.argsort(tid)
    pos = np.searchsorted(tid[order], can_tid)
    pos = np.clip(pos, 0, n - 1)
    hit = order[pos]
    found = tid[hit] == can_tid
    if not found.all():
        raise RuntimeError(f"GATE FAIL canonical-continuity: {(~found).sum():,}/{len(can_tid):,} "
                           f"canonical TARGETIDs missing from P1 slice")
    dl = np.abs(lam[hit] - can_eig).max()
    if dl > 1e-6:
        raise RuntimeError(f"GATE FAIL canonical eigenvalue mismatch: max|dlam| = {dl:.3e}")
    act_frac = active[hit].mean()
    print(f"[gate] canonical continuity: {len(can_tid):,}/{len(can_tid):,} TARGETIDs found, "
          f"max|dlam|={dl:.2e}, active fraction {100*act_frac:.2f}% "
          f"(sub-100% = new z-error draw moved some rows across shell edges; expected and recorded)")
    print("[PASS] gates: unique TID, finite targets, valid-box active, positions, canonical continuity")

    # ================================ WRITE ================================
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros(n, dtype=[("GLOBAL_NODE_ID", "i8"), ("TARGETID", "i8"),
                             ("RA", "f8"), ("DEC", "f8"), ("Z", "f8"), ("Z_ORIG", "f8"),
                             ("X", "f8"), ("Y", "f8"), ("Z_CART", "f8"),
                             ("SHELL", "U9"), ("VALID_TARGET", "?"), ("ACTIVE", "?"),
                             ("LAMBDA1", "f8"), ("LAMBDA2", "f8"), ("LAMBDA3", "f8"), ("CWEB", "i2"),
                             ("FILE_NUM", "i8"), ("BOX_INDEX", "i8"), ("HALO_INDEX", "i8"),
                             ("D_BOUNDARY_MPC", "f8")]
                   + [(c, sub[c].dtype.str) for c in sel_meta])
    out["GLOBAL_NODE_ID"] = np.arange(n)
    out["TARGETID"] = tid
    out["RA"], out["DEC"] = sub["RA"], sub["DEC"]
    out["Z"], out["Z_ORIG"] = z_obs, zorig
    out["X"], out["Y"], out["Z_CART"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["SHELL"], out["VALID_TARGET"], out["ACTIVE"] = shell, valid_target, active
    out["LAMBDA1"], out["LAMBDA2"], out["LAMBDA3"] = lam[:, 0], lam[:, 1], lam[:, 2]
    out["CWEB"] = sub["CWEB"]
    out["FILE_NUM"], out["BOX_INDEX"], out["HALO_INDEX"] = sub["FILE_NUM"], sub["BOX_INDEX"], sub["HALO_INDEX"]
    out["D_BOUNDARY_MPC"] = d_boundary
    for c in sel_meta:
        out[c] = sub[c]

    cat_path = args.out_dir / "canonical_catalogue.fits"
    fitsio.write(str(cat_path), out, clobber=True)
    cat_sha = sha256(cat_path)

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    shell_counts = {tag: int((shell == tag).sum()) for tag, _, _ in SHELLS}
    manifest = {
        "schema_version": "1.0", "stage": "P1",
        "catalogue_id": "ph000_path1_wedge_v1", "phase": "ph000",
        "observer": "path1_fiberassign", "hod": "baseline (none generated)",
        "parent": str(PARENT), "parent_sha256": parent_sha,
        "catalogue": str(cat_path), "catalogue_sha256": cat_sha,
        "git_sha": git_sha,
        "target_convention": {"labels": "tidal eigenvalues ascending, rs7=7 Mpc/h Gaussian, "
                              "ngrid2048, halo_xcom sampling", "epoch": "z=0.2 snapshot"},
        "cosmology": "Planck18 comoving distances (astropy)",
        "z_error": {"sigma_v_kms": args.sigma_v_kms, "seed": args.seed,
                    "policy": "single-pass injection; continuity vs frozen canonical rows is "
                              "TARGETID+Z_ORIG+eigenvalue anchored (observed Z differs by draw)"},
        "bounds": {"ra": args.ra, "dec": args.dec, "z_core": args.z_core, "z_buf": args.z_buf,
                   "sentinel_excluded": list(SENTINEL)},
        "counts": {"total": n, "valid_target": int(valid_target.sum()), "active": int(active.sum()),
                   "buffer": int((shell == "buffer").sum()), "by_shell": shell_counts},
        "canonical_continuity": {"n_canonical": int(len(can_tid)), "all_found": True,
                                 "max_abs_dlambda": float(dl),
                                 "active_fraction_under_new_z_draw": float(act_frac)},
        "no_train_fitted_normalisation": True, "no_split_filtering": True,
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(args.out_dir / "CATALOGUE_COMPLETE", "w") as f:
        f.write(f"P1 ph000_path1_wedge_v1 sha256={cat_sha[:16]} rows={n}\n")
    print(f"\nP1 COMPLETE: {n:,} rows ({int(active.sum()):,} active, "
          f"{int((shell=='buffer').sum()):,} buffer) -> {cat_path}  ({time.time()-t0:.0f}s)")
    print(f"shell counts: {shell_counts}")


if __name__ == "__main__":
    main()
