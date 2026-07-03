#!/usr/bin/env python3
"""A3 step 4 — trim the buffered harmonized build to the exact wedge box.

Keeps nodes inside the final wedge (their features were computed with buffered
neighbours, so no boundary corruption), remaps edges to kept-node indexing, and
writes wedge-level artifacts row-aligned with a trimmed targets FITS + a cache
metadata json. Alignment guard: recompute xyz from the catalog and require a
match with the builder's points (KD-tree strict mapping if reordered).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import fitsio
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree


def radec_z_to_xyz(ra, dec, z):
    zt = np.linspace(0, 0.7, 4000)
    dt = cosmo.comoving_distance(zt).value
    d = np.interp(z, zt, dt)
    r, dd = np.deg2rad(np.asarray(ra, np.float64)), np.deg2rad(np.asarray(dec, np.float64))
    return np.vstack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffered-fits", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True, help="builder points_xyz.npy")
    ap.add_argument("--gnn-arrays", type=Path, required=True)
    ap.add_argument("--gnn-metadata", type=Path, required=True)
    ap.add_argument("--ra", nargs=2, type=float, default=[120.0, 160.0])
    ap.add_argument("--dec", nargs=2, type=float, default=[14.5, 30.6])
    ap.add_argument("--zr", nargs=2, type=float, default=[0.2, 0.3])
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    cat = fitsio.read(args.buffered_fits)
    pos_build = np.load(args.points_xyz).astype(np.float64)
    d = np.load(args.gnn_arrays)
    x, ei, ea = d["x"], d["edge_index"].astype(np.int64), d["edge_attr"]
    assert len(pos_build) == len(x), "points/features mismatch"

    # catalog-row -> builder-node mapping (guards against builder reordering)
    pos_cat = radec_z_to_xyz(cat["RA"], cat["DEC"], cat["Z"])
    if len(cat) == len(pos_build) and np.allclose(pos_cat, pos_build, atol=1e-3):
        cat2node = np.arange(len(cat))
        print("alignment: identity (no reordering)")
    else:
        dist, cat2node = cKDTree(pos_build).query(pos_cat, k=1)
        assert len(np.unique(cat2node)) == len(cat) and dist.max() < 1e-2, \
            f"ambiguous mapping (max dist {dist.max():.4f})"
        print(f"alignment: KD-tree mapping (max dist {dist.max():.2e})")

    inw = ((cat["RA"] >= args.ra[0]) & (cat["RA"] < args.ra[1]) &
           (cat["DEC"] >= args.dec[0]) & (cat["DEC"] < args.dec[1]) &
           (cat["Z"] >= args.zr[0]) & (cat["Z"] < args.zr[1]))
    rows = np.where(inw)[0]
    nodes = cat2node[rows]
    print(f"wedge nodes: {len(rows)} of {len(cat)} buffered")

    remap = np.full(len(x), -1, np.int64)
    remap[nodes] = np.arange(len(nodes))
    ekeep = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    ei_t = np.vstack([remap[ei[0][ekeep]], remap[ei[1][ekeep]]])
    print(f"edges: {ei.shape[1]} -> {ei_t.shape[1]} (both endpoints in wedge)")

    out = args.out_prefix
    np.savez_compressed(str(out) + "_gnn_arrays.npz", x=x[nodes], edge_index=ei_t,
                        edge_attr=ea[ekeep])
    np.save(str(out) + "_points_xyz.npy", pos_build[nodes])
    np.save(str(out) + "_buffered_row_ids.npy", rows)
    fitsio.write(str(out) + "_wedge_targets.fits", cat[rows], clobber=True)
    meta = json.loads(args.gnn_metadata.read_text())
    meta.update({"n_points": int(len(nodes)), "n_edges": int(ei_t.shape[1]),
                 "input_mode": "delaunay_buffered_trim_nzharm",
                 "outputs": {"gnn_arrays_npz": str(out) + "_gnn_arrays.npz"},
                 "buffered_parent_fits": str(args.buffered_fits)})
    Path(str(out) + "_gnn_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"Saved wedge artifacts at prefix: {out}")


if __name__ == "__main__":
    main()
