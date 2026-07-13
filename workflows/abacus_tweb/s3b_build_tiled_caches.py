#!/usr/bin/env python3
"""S3b — TILED ñ-conditioned training set (roadmap: Phase-B OOM pivot, user-approved).

The pooled full-range graph (20.9M edges) OOMs one 80GB GPU. Fix: split the dense low-z
shells into buffered RA sub-volume tiles (aligned to the holdout boundaries {145,150}); keep
shells 2-4 whole. Each tile is a disjoint graph <= MAX_EDGES (fits one GPU); the trainer
mini-batches over tiles. Tiling by RA aligns with the spatial holdout (a tile's core RA puts
its nodes in one region), and matches the Phase-C deployment (tiled footprint).

Built by SLICING the existing per-shell union graphs (induced subgraph on core+buffer nodes) —
NO cuGraph/Delaunay rebuild. Per-shell ANGULAR buffer (union radius 14.78 Mpc ≈ 3.9° at z=0.05)
so core nodes near a tile edge keep their real neighbours (buffer nodes are passive).

Conditioning/normalisation identical to s3 (single source of truth): invert each shell's
per-shell box-cox → SI-only features → ONE pooled box-cox on ALL train-core nodes across tiles
→ append frozen-spline ñ as untransformed col 7. ONE pooled target scaler on pooled train.
Spatial holdout: train RA<145 / val 145-150 / test RA>=150 (per active node), halo-disjoint via
15 Mpc gutter (passive band) at the region boundaries + cross-shell/tile dedup by TARGETID.

Output: <out_dir>/tile_###.pkl (standard cache payloads) + shared_scalers.pkl + manifest.json.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jraph
import fitsio
from astropy.cosmology import Planck18 as cosmo
from sklearn.preprocessing import PowerTransformer, StandardScaler

SHELLS = ["0p05_0p15", "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55"]
CORE_Z = {"0p05_0p15": (0.05, 0.15), "0p15_0p25": (0.15, 0.25), "0p25_0p35": (0.25, 0.35),
          "0p35_0p45": (0.35, 0.45), "0p45_0p55": (0.45, 0.55)}
SHELL_ZLO = {t: CORE_Z[t][0] for t in SHELLS}
NODE_FEATURE_NAMES = ["Degree", "Clustering", "Density", "NeighDensity",
                      "I_eig1", "I_eig2", "I_eig3", "log_ntilde_std"]
UNION_R_MPC = 14.78
CACHE_TMPL = "sbi_caches/s2_shell_{tag}_si_union/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
TARGETS_TMPL = "s2_shells/shell_{tag}_final_wedge_targets.fits"

_ZT = np.linspace(0.0, 0.75, 8000); _DT = cosmo.comoving_distance(_ZT).value
def dcom(z): return np.interp(z, _ZT, _DT)
def deg_per_mpc(z): return np.degrees(1.0 / np.maximum(dcom(z), 1e-6))  # angular size of 1 Mpc


def ntilde_feature(z, sp):
    zg = np.asarray(sp["grid_z"]); nt = np.asarray(sp["ntilde"])
    n = np.interp(np.clip(z, zg.min(), zg.max()), zg, nt)
    return (np.log(np.maximum(n, sp["ntilde_floor"])) - sp["logn_mean"]) / sp["logn_std"]


def region_of(ra, ra_train_hi, ra_test_lo):
    r = np.ones(np.shape(ra), np.int8)          # 1=val
    r = np.where(ra < ra_train_hi, 0, r)         # 0=train
    r = np.where(ra >= ra_test_lo, 2, r)         # 2=test
    return r


def induced_subgraph(keep_idx, senders, receivers, edge_attr, n_full):
    """Induced subgraph on keep_idx. node_map sized to the FULL shell node count
    (senders/receivers index the full graph). Returns remapped se, re, ea, edge_mask."""
    node_map = -np.ones(int(n_full), np.int64)
    node_map[keep_idx] = np.arange(len(keep_idx))
    em = (node_map[senders] >= 0) & (node_map[receivers] >= 0)
    return node_map[senders[em]], node_map[receivers[em]], edge_attr[em], em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/pscratch/sd/d/dkololgi/abacus"))
    ap.add_argument("--ntilde-spline", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/conditioning/ntilde_spline_v1_frozen.json"))
    ap.add_argument("--ra-train-hi", type=float, default=145.0)
    ap.add_argument("--ra-test-lo", type=float, default=150.0)
    ap.add_argument("--gutter-mpc", type=float, default=15.0)
    ap.add_argument("--max-edges", type=int, default=4_000_000)
    ap.add_argument("--buffer-mpc", type=float, default=17.0)  # >= union radius, for connectivity
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/s3b_tiled_ntilde_uniongraph"))
    args = ap.parse_args()
    sp = json.loads(Path(args.ntilde_spline).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    BOUNDS = [args.ra_train_hi, args.ra_test_lo]

    # ---- load shells, invert box-cox, define tiles ----------------------------------
    shell_data = {}
    for tag in SHELLS:
        c = pickle.load(open(args.root / CACHE_TMPL.format(tag=tag), "rb"))
        g = c["graph"]
        x_si = c["node_feature_scaler"].inverse_transform(np.asarray(g.nodes, np.float64)) - 1e-6
        tg = fitsio.read(args.root / TARGETS_TMPL.format(tag=tag),
                         columns=["TARGETID", "RA", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
        shell_data[tag] = dict(
            x_si=x_si, se=np.asarray(g.senders, np.int64), re=np.asarray(g.receivers, np.int64),
            ea=np.asarray(g.edges, np.float64), n_edge=int(np.asarray(g.n_edge)[0]),
            tid=tg["TARGETID"].astype(np.int64), ra=tg["RA"].astype(np.float64),
            z=tg["Z"].astype(np.float64),
            halo=np.stack([tg["FILE_NUM"], tg["BOX_INDEX"], tg["HALO_INDEX"]], 1).astype(np.int64),
            tgt_raw=np.asarray(c["regression_targets_raw"], np.float64),
            eig_raw=np.asarray(c["eigenvalues_raw"], np.float64),
            box_index=np.asarray(c["box_index"], np.int32))

    # cross-shell dedup: a TARGETID active only in the shell whose core-z contains its Z
    core_shell = {}   # tid -> preferred shell tag
    for tag in SHELLS:
        d = shell_data[tag]; lo, hi = CORE_Z[tag]
        inc = (d["z"] >= lo) & (d["z"] < hi)
        for t in d["tid"][inc]:
            core_shell.setdefault(int(t), tag)
    # any tid never in-core (buffer-only dup) -> first shell it appears in
    for tag in SHELLS:
        for t in shell_data[tag]["tid"]:
            core_shell.setdefault(int(t), tag)

    # global halo -> region by centroid RA (halo-disjoint guarantee, incl. cross-shell halos)
    Hc = np.concatenate([shell_data[t]["halo"] for t in SHELLS], 0)
    Rc = np.concatenate([shell_data[t]["ra"] for t in SHELLS])
    hv = np.ascontiguousarray(Hc).view([('', Hc.dtype)] * 3).ravel()
    uk, inv = np.unique(hv, return_inverse=True)
    sum_ra = np.zeros(len(uk)); cnt_h = np.zeros(len(uk))
    np.add.at(sum_ra, inv, Rc); np.add.at(cnt_h, inv, 1.0)
    halo_region = region_of(sum_ra / cnt_h, args.ra_train_hi, args.ra_test_lo)  # per unique halo
    node_region_global = halo_region[inv]                                       # per global node
    off = 0
    for tag in SHELLS:
        n = len(shell_data[tag]["ra"])
        shell_data[tag]["node_region"] = node_region_global[off:off + n]; off += n

    # tile definitions: list of (shell, ra_core_lo, ra_core_hi)
    tiles = []
    for tag in SHELLS:
        d = shell_data[tag]; ra = d["ra"]
        if d["n_edge"] <= args.max_edges:
            tiles.append((tag, ra.min() - 1e-3, ra.max() + 1e-3)); continue
        # node cap per tile s.t. induced edges (core+buffer) <= max_edges. Buffer nodes add
        # edges on top of core (large angular buffer at low z), so size cores conservatively.
        cap = max(2000, int(0.32 * args.max_edges / d["n_edge"] * len(ra)))
        # split each holdout region separately so tiles never straddle a region boundary
        edges_all = np.array([ra.min() - 1e-3] + BOUNDS + [ra.max() + 1e-3])
        for a, b in zip(edges_all[:-1], edges_all[1:]):
            core = ra[(ra >= a) & (ra < b)]
            if len(core) == 0:
                continue
            ntile = max(1, int(np.ceil(len(core) / cap)))
            qs = np.quantile(core, np.linspace(0, 1, ntile + 1))
            qs[0] = a; qs[-1] = b
            for c_lo, c_hi in zip(qs[:-1], qs[1:]):
                if c_hi > c_lo:
                    tiles.append((tag, float(c_lo), float(c_hi)))
    print(f"defined {len(tiles)} tiles from {len(SHELLS)} shells")

    # ---- PASS 1: assemble tile node arrays, masks; collect pooled-train for scalers ---
    tile_blobs = []
    train_x_list, train_tgt_list = [], []
    for ti, (tag, c_lo, c_hi) in enumerate(tiles):
        d = shell_data[tag]; ra = d["ra"]; z = d["z"]
        buf_deg = args.buffer_mpc * deg_per_mpc(SHELL_ZLO[tag])  # angular buffer at shell's near edge
        core = (ra >= c_lo) & (ra < c_hi)
        keep = core | ((ra >= c_lo - buf_deg) & (ra < c_hi + buf_deg))
        keep_idx = np.where(keep)[0]
        se, re, ea, _ = induced_subgraph(keep_idx, d["se"], d["re"], d["ea"], len(d["ra"]))
        # local arrays
        lra, lz = ra[keep_idx], z[keep_idx]
        lcore = core[keep_idx]
        ltid = d["tid"][keep_idx]
        gut = args.gutter_mpc * deg_per_mpc(lz)
        in_gutter = (np.abs(lra - args.ra_train_hi) < gut) | (np.abs(lra - args.ra_test_lo) < gut)
        is_core_shell = np.array([core_shell[int(t)] == tag for t in ltid])
        active = lcore & ~in_gutter & is_core_shell
        reg = d["node_region"][keep_idx]        # global halo-disjoint region (not per-node RA)
        train_m = active & (reg == 0); val_m = active & (reg == 1); test_m = active & (reg == 2)
        blob = dict(tag=tag, c_lo=c_lo, c_hi=c_hi, keep_idx=keep_idx, se=se, re=re, ea=ea,
                    x_si=d["x_si"][keep_idx], z=lz, ra=lra, tid=ltid,
                    halo=d["halo"][keep_idx], tgt_raw=d["tgt_raw"][keep_idx],
                    eig_raw=d["eig_raw"][keep_idx], box_index=d["box_index"][keep_idx],
                    train_m=train_m, val_m=val_m, test_m=test_m, active=active)
        tile_blobs.append(blob)
        train_x_list.append(blob["x_si"][train_m]); train_tgt_list.append(blob["tgt_raw"][train_m])

    train_x = np.concatenate(train_x_list, 0); train_tgt = np.concatenate(train_tgt_list, 0)
    bc = PowerTransformer(method="box-cox").fit(train_x + 1e-6)
    ts = StandardScaler().fit(train_tgt)
    print(f"pooled scalers fit on {len(train_x)} train-core nodes across {len(tiles)} tiles")

    # ---- PASS 2: transform + write tiles ---------------------------------------------
    manifest = {"tiles": [], "n_tiles": len(tiles), "ntilde_spline_path": str(args.ntilde_spline),
                "node_feature_names": NODE_FEATURE_NAMES, "ntilde_feature_index": 7,
                "spatial_split": {"ra_train_hi": args.ra_train_hi, "ra_test_lo": args.ra_test_lo,
                                  "gutter_mpc": args.gutter_mpc, "buffer_mpc": args.buffer_mpc,
                                  "max_edges": args.max_edges, "halo_disjoint": True}}
    tot = {"train": 0, "val": 0, "test": 0, "nodes": 0, "edges": 0}
    all_train_halos, all_test_halos = set(), set()
    for ti, b in enumerate(tile_blobs):
        x_bc = bc.transform(b["x_si"] + 1e-6)
        ntf = ntilde_feature(b["z"], sp)[:, None]
        x_final = np.concatenate([x_bc, ntf], 1).astype(np.float32)
        tgt_scaled = ts.transform(b["tgt_raw"]).astype(np.float32)
        n = x_final.shape[0]
        graph = jraph.GraphsTuple(
            nodes=jnp.array(x_final), edges=jnp.array(b["ea"], jnp.float32),
            senders=jnp.array(b["se"], jnp.int32), receivers=jnp.array(b["re"], jnp.int32),
            n_node=jnp.array([n], jnp.int32), n_edge=jnp.array([b["se"].shape[0]], jnp.int32),
            globals=None)
        payload = dict(graph=graph, regression_targets=jnp.array(tgt_scaled),
                       regression_targets_raw=b["tgt_raw"], eigenvalues_raw=b["eig_raw"],
                       masks=(jnp.array(b["train_m"]), jnp.array(b["val_m"]), jnp.array(b["test_m"])),
                       box_index=b["box_index"], tile_index=ti, tile_shell=b["tag"],
                       tile_ra_core=[b["c_lo"], b["c_hi"]])
        pickle.dump(payload, open(args.out_dir / f"tile_{ti:03d}.pkl", "wb"))
        all_train_halos |= set(map(tuple, b["halo"][b["train_m"]]))
        all_test_halos |= set(map(tuple, b["halo"][b["test_m"]]))
        manifest["tiles"].append(dict(file=f"tile_{ti:03d}.pkl", shell=b["tag"],
                                      ra_core=[round(b["c_lo"],2), round(b["c_hi"],2)],
                                      n_node=int(n), n_edge=int(b["se"].shape[0]),
                                      train=int(b["train_m"].sum()), val=int(b["val_m"].sum()),
                                      test=int(b["test_m"].sum())))
        for k, m in (("train", b["train_m"]), ("val", b["val_m"]), ("test", b["test_m"])):
            tot[k] += int(m.sum())
        tot["nodes"] += n; tot["edges"] += int(b["se"].shape[0])

    # shared scalers (single source for train + inference)
    pickle.dump(dict(node_feature_scaler=bc, node_feature_power_method="box-cox",
                     target_scaler=ts, stats={"increment_mode": "linear",
                         "scaler_mean": ts.mean_.tolist(), "scaler_std": ts.scale_.tolist()},
                     node_feature_names=NODE_FEATURE_NAMES, ntilde_feature_index=7,
                     ntilde_spline_path=str(args.ntilde_spline)),
                open(args.out_dir / "shared_scalers.pkl", "wb"))
    manifest["totals"] = tot
    manifest["max_tile_edges"] = max(t["n_edge"] for t in manifest["tiles"])
    manifest["max_tile_nodes"] = max(t["n_node"] for t in manifest["tiles"])
    json.dump(manifest, open(args.out_dir / "manifest.json", "w"), indent=1)

    # ---- sanity gate -----------------------------------------------------------------
    print(f"\n=== TILED SANITY GATE ({len(tiles)} tiles) ===")
    print(f"totals: nodes={tot['nodes']} edges={tot['edges']} | active train/val/test="
          f"{tot['train']}/{tot['val']}/{tot['test']}")
    print(f"max tile: {manifest['max_tile_nodes']} nodes, {manifest['max_tile_edges']} edges "
          f"(budget {args.max_edges})")
    over = [t for t in manifest["tiles"] if t["n_edge"] > args.max_edges]
    print(f"[{'PASS' if not over else 'WARN'}] all tiles within edge budget ({len(over)} over)")
    leak = all_train_halos & all_test_halos
    print(f"[{'PASS' if not leak else 'FAIL'}] train/test halo-disjoint across ALL tiles: {len(leak)} shared")
    # every shell represented in all three regions (globally)
    for tag in SHELLS:
        rows = [t for t in manifest["tiles"] if t["shell"] == tag]
        tr, va, te = (sum(t[k] for t in rows) for k in ("train", "val", "test"))
        print(f"    shell {tag}: {len(rows)} tiles, train/val/test={tr}/{va}/{te}")
    print(f"\nSaved {len(tiles)} tiles + manifest + shared_scalers to {args.out_dir}")


if __name__ == "__main__":
    main()
