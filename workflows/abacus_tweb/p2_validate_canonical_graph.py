#!/usr/bin/env python3
"""P2 validation — gates, edge-type provenance, and flags for the canonical graph.

Consumes the artifacts produced by the gold-validated chain (build_abacus_graph.py ->
abacus_graph_features_cugraph.py -> build_union_graph_arrays.py) on the P1 catalogue, verifies the
plan's P2 gates, derives edge-type provenance (delaunay / radius / both), attaches survey-boundary
and support flags from P1, and writes GRAPH_COMPLETE only if every gate passes. Only this script may
write GRAPH_COMPLETE (plan P2).

Node metrics follow the ESTABLISHED R0/A1 schema: computed on the Delaunay topology (Degree,
Clustering, tetrahedral Density, NeighDensity, inertia eigenvalues); the union supplies
message-passing edges with 5 edge features. Recorded in the manifest as a convention, not a change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import fitsio
import numpy as np

P1_DIR = Path("/pscratch/sd/d/dkololgi/abacus/p1_canonical/ph000_path1_wedge")


def sha_arr(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/p2_canonical/ph000_path1_wedge"))
    ap.add_argument("--prefix", default="p2_ph000_path1_wedge_v1")
    ap.add_argument("--union-prefix", default="p2_ph000_path1_wedge_v1_union_r14p78")
    args = ap.parse_args()
    G, PRE, UPRE = args.graph_dir, args.prefix, args.union_prefix

    cat = fitsio.read(P1_DIR / "canonical_catalogue.fits")
    p1_manifest = json.load(open(P1_DIR / "manifest.json"))
    n_p1 = len(cat)

    xyz = np.load(G / f"{PRE}_points_xyz.npy")
    del_edges = np.load(G / f"{PRE}_edges_combined_idx.npy")           # [E,2] delaunay pairs
    with np.load(G / f"{UPRE}_gnn_arrays.npz") as u:
        x = u["x"]; edge_index = u["edge_index"]; edge_attr = u["edge_attr"]
    n = xyz.shape[0]
    E_union = edge_index.shape[1]

    print(f"P1 rows {n_p1:,} | graph nodes {n:,} | delaunay pairs {len(del_edges):,} | "
          f"union directed edges {E_union:,} | node features {x.shape[1]}")

    # ---- gates -----------------------------------------------------------------------
    fails = []
    if n != n_p1:
        fails.append(f"node count {n} != P1 rows {n_p1}")
    # row alignment: graph points must be the P1 positions (same order)
    p1_xyz = np.stack([cat["X"], cat["Y"], cat["Z_CART"]], 1)
    dpos = float(np.max(np.abs(p1_xyz - xyz))) if n == n_p1 else np.inf
    if dpos > 0.05:                                       # builder recomputes from RA/DEC/Z; interp tol
        fails.append(f"position mismatch vs P1: max {dpos:.3f} Mpc")
    if edge_index.min() < 0 or edge_index.max() >= n:
        fails.append("senders/receivers out of bounds")
    if not np.isfinite(x).all() or not np.isfinite(edge_attr).all():
        fails.append("non-finite features")
    deg = np.bincount(edge_index[0], minlength=n) + np.bincount(edge_index[1], minlength=n)
    n_iso = int((deg == 0).sum())
    if n_iso > 0:
        fails.append(f"{n_iso} isolated nodes in the union graph")
    # no shell seams possible by construction (single contiguous build) — assert continuity of z
    # coverage instead: every shell present among active nodes
    for tag in ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"):
        if not ((cat["SHELL"] == tag) & cat["ACTIVE"]).any():
            fails.append(f"shell {tag} missing among active nodes")

    # ---- edge-type provenance --------------------------------------------------------
    und = np.sort(edge_index.T % n, axis=1)               # undirected pairs of the union
    upairs = np.unique(und, axis=0)
    dpairs = np.unique(np.sort(del_edges, axis=1), axis=0)
    dset = set(map(tuple, dpairs))
    is_del = np.fromiter((tuple(p) in dset for p in upairs), bool, len(upairs))
    lengths = np.linalg.norm(xyz[upairs[:, 0]] - xyz[upairs[:, 1]], axis=1)
    is_rad = lengths <= 14.78 + 1e-9
    etype = np.where(is_del & is_rad, 2, np.where(is_del, 0, 1)).astype(np.int8)
    if (~is_del & ~is_rad).any():
        fails.append(f"{int((~is_del & ~is_rad).sum())} union pairs neither delaunay nor radius")
    print(f"edge provenance: delaunay-only {(etype==0).sum():,}  radius-only {(etype==1).sum():,}  "
          f"both {(etype==2).sum():,}")

    if fails:
        raise RuntimeError("P2 GATES FAILED: " + "; ".join(fails))

    # ---- flags + manifest ------------------------------------------------------------
    np.savez_compressed(G / f"{UPRE}_edge_provenance.npz", pairs=upairs, edge_type=etype,
                        legend=np.array(["delaunay_only", "radius_only", "both"]))
    flags = {"survey_boundary_lt_15mpc": (cat["D_BOUNDARY_MPC"] < 15).astype(bool),
             "extreme_degree": (deg > np.quantile(deg, 0.999)).astype(bool),
             "buffer_node": (cat["SHELL"] == "buffer").astype(bool)}
    np.savez_compressed(G / f"{PRE}_node_flags.npz", **flags)

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    manifest = {
        "schema_version": "1.0", "stage": "P2", "catalogue_id": p1_manifest["catalogue_id"],
        "p1_catalogue_sha256": p1_manifest["catalogue_sha256"], "git_sha": git_sha,
        "topology": {"delaunay_pairs": int(len(dpairs)), "union_pairs": int(len(upairs)),
                     "union_directed_edges": int(E_union), "radius_mpc": 14.78,
                     "isolated_nodes": 0, "mean_degree": float(deg.mean())},
        "node_metric_convention": "R0/A1 established schema: metrics on Delaunay topology "
                                  "(Degree, Clustering, tetrahedral Density, NeighDensity, "
                                  "inertia eigenvalues); union supplies message-passing edges",
        "hashes": {"points": sha_arr(xyz), "union_arrays": sha_arr(x, edge_index, edge_attr),
                   "edge_provenance": sha_arr(upairs, etype)},
        "position_match_vs_p1_max_mpc": dpos,
    }
    with open(G / "p2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(G / "GRAPH_COMPLETE", "w") as f:
        f.write(f"P2 {p1_manifest['catalogue_id']} nodes={n} union_pairs={len(upairs)}\n")
    print(f"[PASS] all P2 gates; GRAPH_COMPLETE written -> {G}")


if __name__ == "__main__":
    main()
