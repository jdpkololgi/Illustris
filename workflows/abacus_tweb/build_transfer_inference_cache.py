#!/usr/bin/env python3
"""Build an INFERENCE cache for the pure-inductive transfer test (deployment rehearsal).

TEST: the union GraphNet (anchor: posterior-mean R2 0.8041/0.8461/0.8955, transductively trained on
the RA 120-160 wedge with a random split) is applied to a completely disjoint wedge (RA 200-240, same
DEC/z, same pipeline) where truth is known. This is exactly the DESI deployment path -- fresh graph,
training-fitted transforms, saved checkpoint -- run where it can be scored.

TRANSFORM POLICY (deployment-correct):
  - per-GRAPH normalisations use the NEW wedge's own statistics (that is what SI is FOR -- it is the
    transfer mechanism): node cols [0,2,3,4,5,6] /= own median; edge length /= own median.
  - FITTED transforms come from TRAINING (deployment has nothing else): node PowerTransformer and
    target StandardScaler from the production cache; the edge log+StandardScaler is refit on the
    TRAINING union npz by replaying the identical code path (the production cache does not store it),
    then applied to the new wedge. Same policy as the DESI inference script.

GOLD GATE (--gold-check): before building the transfer cache, replay this exact recipe on the
TRAINING union npz + training wedge_targets.fits and require the result to reproduce the production
cache's graph (nodes/edges/senders/receivers) and targets to float tolerance. If that fails, the
recipe is NOT the production recipe and the transfer number would be meaningless -- so we refuse.

Masks: train/val are 1000-node dummies (the eval-only trainer invocation never trains); test = ALL
valid nodes, so the final eval scores the whole new wedge.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import fitsio
import jax.numpy as jnp
import jraph
import numpy as np
from sklearn.preprocessing import StandardScaler

G = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions")
W = G / "wedges/path1_fiberassign"
TRAIN_UNION_NPZ = W / "path1_wedge_union_r10hmpc_gnn_arrays.npz"
TRAIN_TARGETS = W / "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_wedge_targets.fits"
PROD_CACHE = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph/"
                  "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl")


def build_edges(npz_path: Path):
    """Replicates build_abacus_sbi_cache._build_graph_from_npz edge path:
    bidirectional doubling -> SI (length/own median) -> log(length), log(contrast).
    Returns pre-scaler edge features + senders/receivers (scaler applied by caller)."""
    with np.load(npz_path) as d:
        x = d["x"].astype(np.float32)
        edge_index = d["edge_index"].astype(np.int64)
        edge_attr = d["edge_attr"].astype(np.float32)
    senders, receivers = edge_index[0], edge_index[1]

    rev = edge_attr.copy()
    rev[:, 1] *= -1.0
    rev[:, 2] *= -1.0
    rev[:, 3] *= -1.0
    rev[:, 4] = 1.0 / np.maximum(rev[:, 4], 1e-6)
    s2 = np.concatenate([senders, receivers])
    r2 = np.concatenate([receivers, senders])
    ea = np.concatenate([edge_attr, rev], axis=0)

    ea = ea.copy()
    med0 = np.median(ea[:, 0])
    ea[:, 0] = ea[:, 0] / max(float(med0), 1e-6)          # SI: own-graph median

    ea[:, 0] = np.log(np.maximum(ea[:, 0], 1e-6))
    ea[:, 4] = np.log(np.maximum(ea[:, 4], 1e-6))
    return x, ea, s2, r2


def si_nodes(x: np.ndarray) -> np.ndarray:
    x_si = np.asarray(x, np.float64).copy()
    for col in (0, 2, 3, 4, 5, 6):
        med = np.median(x_si[:, col])
        x_si[:, col] = x_si[:, col] / max(float(med), 1e-9)
    return x_si


def build_cache(npz_path: Path, targets_fits: Path, node_pt, target_scaler, edge_scaler):
    x_raw, ea, s2, r2 = build_edges(npz_path)
    ea[:, [0, 4]] = edge_scaler.transform(ea[:, [0, 4]])

    x_scaled = node_pt.transform(si_nodes(x_raw) + 1e-6).astype(np.float32)

    t = fitsio.read(targets_fits, columns=["LAMBDA1", "LAMBDA2", "LAMBDA3", "BOX_INDEX", "RA", "Z"])
    assert len(t) == x_raw.shape[0], f"targets rows {len(t)} != nodes {x_raw.shape[0]}"
    lam = np.stack([t["LAMBDA1"], t["LAMBDA2"], t["LAMBDA3"]], 1).astype(np.float64)
    linear = np.stack([lam[:, 0], lam[:, 1] - lam[:, 0], lam[:, 2] - lam[:, 1]], 1)
    reg = target_scaler.transform(linear).astype(np.float32)

    graph = jraph.GraphsTuple(
        nodes=jnp.array(x_scaled, jnp.float32), edges=jnp.array(ea, jnp.float32),
        senders=jnp.array(s2, jnp.int32), receivers=jnp.array(r2, jnp.int32),
        n_node=jnp.array([x_raw.shape[0]], jnp.int32), n_edge=jnp.array([len(s2)], jnp.int32),
        globals=None)
    return graph, reg, linear, lam, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-union-npz", type=Path,
                    default=W / "path1_wedge_ra200_240_union_r10hmpc_gnn_arrays.npz")
    ap.add_argument("--new-targets-fits", type=Path,
                    default=W / "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra200_240_dec14p5_30p6_z0p2_0p3_wedge_targets.fits")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_TRANSFER_ra200_240_uniongraph"))
    ap.add_argument("--gold-check", action="store_true", default=True,
                    help="verify the recipe reproduces the production cache first (default ON)")
    ap.add_argument("--no-gold-check", dest="gold_check", action="store_false")
    args = ap.parse_args()

    prod = pickle.load(open(PROD_CACHE, "rb"))
    node_pt = prod["node_feature_scaler"]
    target_scaler = prod["target_scaler"]

    # edge scaler: refit on the TRAINING union npz by the identical code path (not stored in cache)
    _, ea_train, _, _ = build_edges(TRAIN_UNION_NPZ)
    edge_scaler = StandardScaler().fit(ea_train[:, [0, 4]])

    if args.gold_check:
        print("=== GOLD GATE: replaying the recipe on the TRAINING wedge ===")
        g, reg, _, lam, _ = build_cache(TRAIN_UNION_NPZ, TRAIN_TARGETS, node_pt, target_scaler, edge_scaler)
        pg = prod["graph"]
        checks = {
            "nodes": (np.asarray(g.nodes), np.asarray(pg.nodes)),
            "edges": (np.asarray(g.edges), np.asarray(pg.edges)),
            "senders": (np.asarray(g.senders), np.asarray(pg.senders)),
            "receivers": (np.asarray(g.receivers), np.asarray(pg.receivers)),
            "regression_targets": (reg, np.asarray(prod["regression_targets"])),
            "eigenvalues_raw": (lam, np.asarray(prod["eigenvalues_raw"])),
        }
        for name, (a, b) in checks.items():
            if a.shape != b.shape:
                raise RuntimeError(f"GOLD GATE FAIL: {name} shape {a.shape} != {b.shape}")
            d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            print(f"  {name:20s} shape {a.shape}  max|diff| = {d:.3e}")
            if d > 1e-4:
                raise RuntimeError(f"GOLD GATE FAIL: {name} max|diff| {d:.3e} > 1e-4 -- recipe is NOT "
                                   f"the production recipe; a transfer number would be meaningless")
        print("[PASS] recipe reproduces the production cache -- transfer build is trustworthy\n")

    print("=== building TRANSFER cache (new wedge) ===")
    g, reg, linear, lam, t = build_cache(args.new_union_npz, args.new_targets_fits,
                                         node_pt, target_scaler, edge_scaler)
    n = int(np.asarray(g.n_node)[0])
    valid = t["BOX_INDEX"] >= 0
    if not np.isfinite(lam[valid]).all():
        raise RuntimeError("non-finite eigenvalues among valid nodes")
    print(f"nodes {n:,}  edges {int(np.asarray(g.n_edge)[0]):,}  valid_box {100*valid.mean():.1f}%")
    print(f"RA [{t['RA'].min():.2f},{t['RA'].max():.2f}]  z [{t['Z'].min():.3f},{t['Z'].max():.3f}]")
    if t["RA"].min() < 199.9 or t["RA"].max() > 240.1:
        raise RuntimeError("new wedge RA out of expected bounds")

    train_m = np.zeros(n, bool); train_m[np.where(valid)[0][:1000]] = True     # dummy (never trained)
    val_m = np.zeros(n, bool); val_m[np.where(valid)[0][1000:2000]] = True     # dummy
    test_m = valid & ~train_m & ~val_m                                          # score everything else
    print(f"masks: dummy-train {train_m.sum():,} / dummy-val {val_m.sum():,} / TEST {test_m.sum():,}")

    cls = (lam > 0.2).sum(axis=1).astype(np.int32)
    cache = {
        "graph": g, "regression_targets": reg,
        "regression_targets_raw": linear, "target_scaler": target_scaler,
        "eigenvalues_raw": lam, "masks": (train_m, val_m, test_m),
        "stats": {"transfer_test": True, "source_npz": str(args.new_union_npz),
                  "trained_on": "RA120-160 (transductive, random split)",
                  "applied_to": "RA200-240 (pure inductive)",
                  "transform_policy": "own-graph SI medians + TRAINING PowerTransformer/target_scaler/edge_scaler"},
        "node_feature_scaler": node_pt, "node_feature_power_method": prod.get("node_feature_power_method", "box-cox"),
        "classification_labels": cls, "box_index": t["BOX_INDEX"].astype(np.int32),
        "box_index_col": "BOX_INDEX", "excluded_box_index_minus_one": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
