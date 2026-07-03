#!/usr/bin/env python3
"""G3 (rung b') — build a Delaunay ∪ radius(R) union graph for a wedge (roadmap v2).

Motivation: Delaunay's receptive field is fixed in NEIGHBOUR COUNT but variable in
PHYSICAL SCALE (narrowest in clusters), while the target lives at a fixed 7 Mpc/h.
Adding fixed-radius edges (~1.4x the target smoothing) gives every node a
physically matched neighbourhood; node features are UNCHANGED so this isolates the
connectivity axis alone.

New radius edges get edge_attr built with the ORIGINAL convention
(edge_length, unit x/y/z_dir, density_contrast = dst_density/src_density from the
node Density column). Existing Delaunay edges + attrs are reused byte-identical.
Outputs a new gnn_arrays npz + metadata json; originals untouched. CPU only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

DENSITY_COL = 2  # "Density" in node_feature_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-metadata-path", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True, help="wedge points_xyz.npy [N,3] Mpc")
    ap.add_argument("--radius-mpc", type=float, required=True, help="union radius in Mpc (comoving)")
    ap.add_argument("--out-prefix", type=Path, required=True, help="prefix for _gnn_arrays.npz + _gnn_metadata.json")
    args = ap.parse_args()

    meta = json.loads(args.gnn_metadata_path.read_text())
    d = np.load(meta["outputs"]["gnn_arrays_npz"])
    x = d["x"]; ei = d["edge_index"].astype(np.int64); ea = d["edge_attr"]
    pos = np.load(args.points_xyz).astype(np.float64)
    n = len(pos)
    assert x.shape[0] == n and int(ei.max()) < n
    print(f"nodes={n}; delaunay undirected pairs={ei.shape[1]}")

    # radius pairs (undirected, each once, i<j) — same storage convention as Delaunay
    tree = cKDTree(pos)
    pairs = tree.query_pairs(args.radius_mpc, output_type="ndarray")  # [P,2]
    print(f"radius({args.radius_mpc:.2f} Mpc) pairs={len(pairs)}")

    # drop radius pairs already present as Delaunay edges (either orientation)
    key_del = set(map(tuple, np.sort(ei.T, axis=1)))
    keep = np.array([tuple(p) not in key_del for p in np.sort(pairs, axis=1)], dtype=bool)
    new = pairs[keep]
    print(f"new (non-Delaunay) pairs={len(new)}  overlap dropped={len(pairs)-len(new)}")

    # edge_attr for new pairs, original convention (src->dst as stored)
    vec = pos[new[:, 1]] - pos[new[:, 0]]
    length = np.linalg.norm(vec, axis=1)
    unit = vec / np.maximum(length, 1e-12)[:, None]
    dens = x[:, DENSITY_COL].astype(np.float64)
    contrast = np.ones(len(new), dtype=np.float64)
    m = dens[new[:, 0]] > 0
    contrast[m] = dens[new[:, 1]][m] / dens[new[:, 0]][m]
    ea_new = np.column_stack([length, unit[:, 0], unit[:, 1], unit[:, 2], contrast]).astype(np.float32)

    ei_u = np.concatenate([ei, new.T.astype(np.int64)], axis=1)
    ea_u = np.concatenate([ea, ea_new], axis=0)
    med_len = np.median(ea_u[:, 0])
    print(f"union undirected pairs={ei_u.shape[1]} (x{ei_u.shape[1]/ei.shape[1]:.2f}); "
          f"median edge length {np.median(ea[:,0]):.2f} -> {med_len:.2f} Mpc")

    out_npz = Path(str(args.out_prefix) + "_gnn_arrays.npz")
    np.savez_compressed(out_npz, x=x, edge_index=ei_u, edge_attr=ea_u)
    meta_u = dict(meta)
    meta_u["input_mode"] = f"delaunay_union_radius_{args.radius_mpc:.2f}mpc"
    meta_u["n_edges"] = int(ei_u.shape[1])
    meta_u["union_radius_mpc"] = args.radius_mpc
    meta_u["union_parent_metadata"] = str(args.gnn_metadata_path)
    meta_u["outputs"] = {"gnn_arrays_npz": str(out_npz)}
    out_meta = Path(str(args.out_prefix) + "_gnn_metadata.json")
    out_meta.write_text(json.dumps(meta_u, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {out_npz}\nSaved: {out_meta}")


if __name__ == "__main__":
    main()
