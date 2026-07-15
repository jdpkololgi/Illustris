#!/usr/bin/env python3
"""S3c — pooled FULL-RANGE cache for the Workstream-C 3-D U-Net challenger.

WHY: R1 showed the graph encoder's feature lever is dead (aperture channels bought capacity, not
information). S1(b) measured a within-shell CNN at 0.902/0.847/0.722/0.429 over z0.15-0.55 (macro
~0.725) vs the incumbent GraphNet's 0.456 -- but that CNN was trained PER SHELL (best case) and the
"grid is dead" verdict was driven substantially by its 0.002 at z0.05-0.15, which we now know is the
CORRUPT-LABEL shell (permutation-null targets; BOX_INDEX==-1) and is excluded from the VAC. Workstream
C asks the untested question: does ONE pooled, selection-aware U-Net hold that skill across the full
range without per-shell seams?

The comparison is only meaningful if the CNN sees EXACTLY the galaxies and the split the GraphNet saw.
So this mirrors s3b_build_tiled_caches.py bit-for-bit:
  - cross-shell dedup   : a TARGETID is active only in the shell whose core-z contains its Z
  - region assignment   : GLOBAL halo-centroid RA -> region (halo-disjoint, incl. cross-shell halos),
                          NOT per-node RA
  - gutter              : |RA - 145| or |RA - 150| < 15 Mpc (converted to deg at each node's z)
  - valid_box           : BOX_INDEX >= 0 (out-of-box == scrambled labels -> never active)
  - active              : ~in_gutter & is_core_shell & valid_box
                          (s3b also ANDs `lcore`, the tile's core-RA window; the tiles' cores tile the
                          full RA range of each shell, so pooling them is exactly this.)
A MANDATORY gate then checks the resulting train/val/test counts against the tiled cache manifest --
if they differ, the "matched" claim is false and the build raises rather than emit a misleading cache.

FIELD vs LABELS: the density field is built from ALL deduped galaxies (positions are observables, and
the GraphNet's tiles likewise message-pass over non-active buffer nodes); only `active` nodes carry
labels/loss. Positions are observer-frame comoving Mpc from (RA, DEC, Z) -- Z is the OBSERVED redshift
(RSD included), which is all DESI gives us at inference.

Emits (row-aligned):
  cnn_fullrange_cache.pkl : eigenvalues_raw (N,3), masks (train,val,test), z, ra, dec, shell, tid
  cnn_fullrange_points.npy: (N,3) observer-frame comoving Mpc

Run under the cosmic_env absolute python (see repo CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo

SHELLS = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
CORE_Z = {"0p05_0p15": (0.05, 0.15), "0p15_0p25": (0.15, 0.25), "0p25_0p35": (0.25, 0.35),
          "0p35_0p45": (0.35, 0.45), "0p45_0p55": (0.45, 0.55)}
CACHE_TMPL = "sbi_caches/s2_shell_{tag}_si_union/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
TARGETS_TMPL = "s2_shells/shell_{tag}_final_wedge_targets.fits"

_ZT = np.linspace(0.0, 0.75, 8000)
_DT = cosmo.comoving_distance(_ZT).value


def dcom(z):
    return np.interp(z, _ZT, _DT)


def deg_per_mpc(z):
    return np.degrees(1.0 / np.maximum(dcom(z), 1e-6))


def region_of(ra, ra_train_hi, ra_test_lo):
    r = np.ones(np.shape(ra), np.int8)           # 1=val
    r = np.where(ra < ra_train_hi, 0, r)         # 0=train
    r = np.where(ra >= ra_test_lo, 2, r)         # 2=test
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/pscratch/sd/d/dkololgi/abacus"))
    ap.add_argument("--ra-train-hi", type=float, default=145.0)
    ap.add_argument("--ra-test-lo", type=float, default=150.0)
    ap.add_argument("--gutter-mpc", type=float, default=15.0)
    ap.add_argument("--min-zlo", type=float, default=0.15,
                    help="drop shells whose core z_lo is below this (0p05_0p15 = corrupt labels)")
    ap.add_argument("--tiled-manifest", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3b_tiled_valid_v3_aper/manifest.json"),
                    help="tiled cache to match against (mandatory gate)")
    ap.add_argument("--tiled-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3b_tiled_valid_v3_aper"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3c_cnn_fullrange"))
    args = ap.parse_args()

    shells = [s for s in SHELLS if CORE_Z[s][0] >= args.min_zlo]
    dropped = [s for s in SHELLS if CORE_Z[s][0] < args.min_zlo]
    print(f"VALID shells (core z_lo >= {args.min_zlo}): {shells}   DROPPED (corrupt labels): {dropped}")

    # ---- load shells -----------------------------------------------------------------
    sd = {}
    for tag in shells:
        c = pickle.load(open(args.root / CACHE_TMPL.format(tag=tag), "rb"))
        tg = fitsio.read(args.root / TARGETS_TMPL.format(tag=tag),
                         columns=["TARGETID", "RA", "DEC", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
        eig = np.asarray(c["eigenvalues_raw"], np.float64)
        box = np.asarray(c["box_index"], np.int32)
        assert len(tg) == len(eig) == len(box), f"{tag}: row mismatch fits/eig/box"
        sd[tag] = dict(tid=tg["TARGETID"].astype(np.int64), ra=tg["RA"].astype(np.float64),
                       dec=tg["DEC"].astype(np.float64), z=tg["Z"].astype(np.float64),
                       halo=np.stack([tg["FILE_NUM"], tg["BOX_INDEX"], tg["HALO_INDEX"]], 1).astype(np.int64),
                       eig=eig, box=box)
        print(f"  {tag}: {len(eig):,} rows")

    # ---- cross-shell dedup (identical to s3b) -----------------------------------------
    core_shell = {}
    for tag in shells:
        d = sd[tag]
        lo, hi = CORE_Z[tag]
        inc = (d["z"] >= lo) & (d["z"] < hi)
        for t in d["tid"][inc]:
            core_shell.setdefault(int(t), tag)
    for tag in shells:                                   # buffer-only dups -> first shell seen
        for t in sd[tag]["tid"]:
            core_shell.setdefault(int(t), tag)

    # ---- global halo -> region by centroid RA (identical to s3b) ----------------------
    Hc = np.concatenate([sd[t]["halo"] for t in shells], 0)
    Rc = np.concatenate([sd[t]["ra"] for t in shells])
    hv = np.ascontiguousarray(Hc).view([('', Hc.dtype)] * 3).ravel()
    uk, inv = np.unique(hv, return_inverse=True)
    sum_ra = np.zeros(len(uk))
    cnt_h = np.zeros(len(uk))
    np.add.at(sum_ra, inv, Rc)
    np.add.at(cnt_h, inv, 1.0)
    halo_region = region_of(sum_ra / cnt_h, args.ra_train_hi, args.ra_test_lo)
    node_region_global = halo_region[inv]
    off = 0
    for tag in shells:
        n = len(sd[tag]["ra"])
        sd[tag]["reg"] = node_region_global[off:off + n]
        off += n

    # ---- per-shell active selection, keeping each TARGETID once -----------------------
    keep = {"tid": [], "ra": [], "dec": [], "z": [], "eig": [], "reg": [],
            "shell": [], "active": [], "halo": []}
    for tag in shells:
        d = sd[tag]
        is_core = np.array([core_shell[int(t)] == tag for t in d["tid"]])
        gut = args.gutter_mpc * deg_per_mpc(d["z"])
        in_gutter = (np.abs(d["ra"] - args.ra_train_hi) < gut) | (np.abs(d["ra"] - args.ra_test_lo) < gut)
        valid_box = d["box"] >= 0
        # dedup for the FIELD: each tid appears once, in its core shell
        sel = is_core
        active = (~in_gutter & valid_box)[sel]           # is_core already applied by `sel`
        keep["tid"].append(d["tid"][sel]); keep["ra"].append(d["ra"][sel])
        keep["dec"].append(d["dec"][sel]); keep["z"].append(d["z"][sel])
        keep["eig"].append(d["eig"][sel]); keep["reg"].append(d["reg"][sel])
        keep["halo"].append(d["halo"][sel])
        keep["shell"].append(np.full(int(sel.sum()), tag, dtype=object))
        keep["active"].append(active)
        print(f"  {tag}: kept {int(sel.sum()):,} unique  active {int(active.sum()):,}")

    tid = np.concatenate(keep["tid"]); ra = np.concatenate(keep["ra"])
    dec = np.concatenate(keep["dec"]); z = np.concatenate(keep["z"])
    eig = np.concatenate(keep["eig"], 0); reg = np.concatenate(keep["reg"])
    shell = np.concatenate(keep["shell"]); active = np.concatenate(keep["active"])
    halo = np.concatenate(keep["halo"], 0)

    train_m = active & (reg == 0)
    val_m = active & (reg == 1)
    test_m = active & (reg == 2)
    n = len(tid)
    print(f"\nN unique galaxies (field) = {n:,}")
    print(f"active train/val/test = {train_m.sum():,}/{val_m.sum():,}/{test_m.sum():,}")

    # ---- positions: observer-frame comoving Mpc from OBSERVED (RA, DEC, Z) -----------
    r = dcom(z)
    phi = np.radians(ra)
    th = np.radians(dec)
    xyz = np.stack([r * np.cos(th) * np.cos(phi), r * np.cos(th) * np.sin(phi), r * np.sin(th)], 1)

    # ================================ MANDATORY GATES ==================================
    # 1. matched split: counts must equal the tiled cache the GraphNet trained on, else the
    #    head-to-head is not a head-to-head and the number would be meaningless.
    tiled = {"train": 0, "val": 0, "test": 0}
    for p in sorted(Path(args.tiled_dir).glob("tile_*.pkl")):
        t = pickle.load(open(p, "rb"))
        tr, va, te = (np.asarray(m).astype(bool) for m in t["masks"])
        tiled["train"] += int(tr.sum()); tiled["val"] += int(va.sum()); tiled["test"] += int(te.sum())
    got = {"train": int(train_m.sum()), "val": int(val_m.sum()), "test": int(test_m.sum())}
    print(f"[gate] tiled cache counts  {tiled}")
    print(f"[gate] this cache counts   {got}")
    if got != tiled:
        raise RuntimeError(f"MATCH GATE FAILED: split differs from the tiled cache {tiled} vs {got} "
                           f"-- the CNN would not be comparable to the GraphNet.")
    print("[PASS] split matches the tiled cache EXACTLY (same galaxies, same holdout)")

    # 2. no TARGETID appears twice (dedup correctness -> no double-deposited galaxies)
    if len(np.unique(tid)) != n:
        raise RuntimeError(f"dedup failed: {n - len(np.unique(tid))} duplicate TARGETIDs")
    # 3. labels finite where active
    if not np.isfinite(eig[active]).all():
        raise RuntimeError("non-finite eigenvalues among active nodes")
    # 4. halo-disjoint train/test (the guarantee s3b enforces)
    hv2 = np.ascontiguousarray(halo).view([('', halo.dtype)] * 3).ravel()
    shared = np.intersect1d(np.unique(hv2[train_m]), np.unique(hv2[test_m]))
    if len(shared):
        raise RuntimeError(f"halo leak: {len(shared)} halos shared between train and test")
    # 5. every valid shell represented in train and val
    for tag in shells:
        s = shell == tag
        if not (train_m & s).any() or not (val_m & s).any():
            raise RuntimeError(f"shell {tag} missing from train or val")
    print("[PASS] mandatory gates: matched split, unique TID, finite labels, halo-disjoint, shell coverage")

    # ---- write ------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = {"eigenvalues_raw": eig, "masks": (train_m, val_m, test_m),
             "z": z, "ra": ra, "dec": dec, "shell": shell, "tid": tid, "active": active,
             "region": reg,
             "provenance": {"built_by": "s3c_build_cnn_fullrange_cache.py",
                            "matched_to": str(args.tiled_dir), "shells": shells,
                            "dropped_shells": dropped, "ra_train_hi": args.ra_train_hi,
                            "ra_test_lo": args.ra_test_lo, "gutter_mpc": args.gutter_mpc,
                            "z_is_observed_rsd": True}}
    with open(args.out_dir / "cnn_fullrange_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
    np.save(args.out_dir / "cnn_fullrange_points.npy", xyz.astype(np.float64))
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump({"n_nodes": n, "counts": got, "shells": shells, "dropped": dropped,
                   "bbox_extent_mpc": (xyz.max(0) - xyz.min(0)).tolist(),
                   "r_range_mpc": [float(r.min()), float(r.max())]}, f, indent=2)
    print(f"\nSaved -> {args.out_dir}  (N={n:,}, bbox {np.round(xyz.max(0) - xyz.min(0), 0)} Mpc)")


if __name__ == "__main__":
    main()
