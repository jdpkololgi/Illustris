#!/usr/bin/env python3
"""S3 — pool the 5 S2 shell caches into ONE ñ-conditioned, spatially-split training
cache for Phase B (roadmap §3b/§4b).

Design (see SCIENCE_LOG 2026-07-13):
  * DISJOINT-UNION the 5 shells into one GraphsTuple (5 disconnected components) so the
    single-graph transductive trainer (jraph_sbi_flowjax.py) needs NO surgery.
  * SI stays PER-SHELL (already baked in each cache). We recover the SI-only node features
    by INVERTING each shell's stored per-shell box-cox, then fit ONE POOLED box-cox on the
    pooled TRAIN split — per-shell box-cox would Gaussianize each shell's contrast
    distribution separately and pre-remove the density-dependent structure ñ must explain.
  * ñ NODE FEATURE: append log_ntilde_std (frozen mock spline, fixed standardization) as the
    UNtransformed final node column (col 7). Excluded from SI (edge/node) and from box-cox by
    construction (appended AFTER both). Also skip-concatenated to the flow conditioning at the
    GNN readout (separate model edit) — so ñ reaches the flow undiluted.
  * SPATIAL HOLDOUT (halo-disjoint): train RA<145 · val/tempering 145-150 · test RA>=150,
    identical across all shells. Each (FILE_NUM,BOX_INDEX,HALO_INDEX) group assigned wholesale
    by centroid RA (kills the ~10 shell-0 straddlers). 15 Mpc transverse graph gutter around
    each boundary -> gutter nodes dropped from masks but kept as passive neighbours.
  * Cross-shell duplicates (204 galaxies in adjacent-shell buffers) deduped by TARGETID:
    keep the copy whose CORE z-range contains observed Z; the other stays passive (mask 0).

Inputs are the existing S2 per-shell caches + their final_wedge_targets FITS (aligned 1:1 with
cache node order, verified n_node==targets). Output is a single cache pickle consumable by the
trainer, plus a sidecar metadata JSON.
"""
from __future__ import annotations
import argparse
import json
import pickle
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
NODE_FEATURE_NAMES = ["Degree", "Clustering", "Density", "NeighDensity",
                      "I_eig1", "I_eig2", "I_eig3", "log_ntilde_std"]
SI_EXCLUDE = {"log_ntilde_std"}  # the covariate — never per-graph/median/box-cox normalised

CACHE_TMPL = "sbi_caches/s2_shell_{tag}_si_union/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
TARGETS_TMPL = "s2_shells/shell_{tag}_final_wedge_targets.fits"

# comoving-distance interpolant (Planck18, matches gate_s1 to_xyz)
_ZT = np.linspace(0.0, 0.75, 8000)
_DT = cosmo.comoving_distance(_ZT).value  # Mpc


def dcom(z):
    return np.interp(z, _ZT, _DT)


def ntilde_feature(z, spline):
    zg = np.asarray(spline["grid_z"]); nt = np.asarray(spline["ntilde"])
    floor = spline["ntilde_floor"]; mu = spline["logn_mean"]; sd = spline["logn_std"]
    n = np.interp(np.clip(z, zg.min(), zg.max()), zg, nt)
    return (np.log(np.maximum(n, floor)) - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/pscratch/sd/d/dkololgi/abacus"))
    ap.add_argument("--ntilde-spline", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/conditioning/ntilde_spline_v1_frozen.json"))
    ap.add_argument("--ra-train-hi", type=float, default=145.0)
    ap.add_argument("--ra-test-lo", type=float, default=150.0)
    ap.add_argument("--gutter-mpc", type=float, default=15.0)
    ap.add_argument("--out", type=Path,
                    default=Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches/"
                                 "s3_pooled_ntilde_uniongraph/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"))
    args = ap.parse_args()
    spline = json.loads(Path(args.ntilde_spline).read_text())
    print(f"frozen spline: {args.ntilde_spline}  logn_mean={spline['logn_mean']:.4f} "
          f"logn_std={spline['logn_std']:.4f}")

    # ---- load + invert box-cox per shell ---------------------------------------------
    xs, eas, sends, recvs = [], [], [], []
    ra_all, z_all, tid_all, halo_all, shell_all = [], [], [], [], []
    tgt_raw_all, eig_raw_all, boxidx_all = [], [], []
    offset = 0
    for si, tag in enumerate(SHELLS):
        cache = pickle.load(open(args.root / CACHE_TMPL.format(tag=tag), "rb"))
        g = cache["graph"]
        x_scaled = np.asarray(g.nodes, np.float64)               # [N,7] SI+per-shell box-cox
        scaler = cache["node_feature_scaler"]                    # per-shell box-cox
        x_si = scaler.inverse_transform(x_scaled) - 1e-6         # recover SI-only features
        n = x_si.shape[0]

        tg = fitsio.read(args.root / TARGETS_TMPL.format(tag=tag),
                         columns=["TARGETID", "RA", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
        assert len(tg) == n, f"{tag}: targets {len(tg)} != nodes {n}"

        se = np.asarray(g.senders, np.int64) + offset
        re = np.asarray(g.receivers, np.int64) + offset
        xs.append(x_si); eas.append(np.asarray(g.edges, np.float64))
        sends.append(se); recvs.append(re)
        ra_all.append(tg["RA"].astype(np.float64)); z_all.append(tg["Z"].astype(np.float64))
        tid_all.append(tg["TARGETID"].astype(np.int64))
        halo_all.append(np.stack([tg["FILE_NUM"], tg["BOX_INDEX"], tg["HALO_INDEX"]], axis=1).astype(np.int64))
        shell_all.append(np.full(n, si, np.int32))
        tgt_raw_all.append(np.asarray(cache["regression_targets_raw"], np.float64))   # linear increments
        eig_raw_all.append(np.asarray(cache["eigenvalues_raw"], np.float64))
        boxidx_all.append(np.asarray(cache["box_index"], np.int32))
        print(f"  [{tag}] nodes={n:6d} edges={se.shape[0]:8d}  box-cox inverted "
              f"(SI feat range [{x_si.min():.3g},{x_si.max():.3g}])")
        offset += n

    x_si = np.concatenate(xs, 0); edges = np.concatenate(eas, 0)
    senders = np.concatenate(sends, 0); receivers = np.concatenate(recvs, 0)
    ra = np.concatenate(ra_all); z = np.concatenate(z_all)
    tid = np.concatenate(tid_all); halo = np.concatenate(halo_all, 0)
    shell = np.concatenate(shell_all)
    tgt_raw = np.concatenate(tgt_raw_all, 0); eig_raw = np.concatenate(eig_raw_all, 0)
    box_index = np.concatenate(boxidx_all)
    N = x_si.shape[0]
    print(f"pooled: nodes={N} edges={senders.shape[0]}")

    # ---- active-node mask: dedup cross-shell + core-range assignment ------------------
    active = np.ones(N, bool)
    # a galaxy is "in core" if its observed Z lies in its own shell's core range
    lo = np.array([CORE_Z[SHELLS[s]][0] for s in shell])
    hi = np.array([CORE_Z[SHELLS[s]][1] for s in shell])
    in_core = (z >= lo) & (z < hi)
    # duplicate TARGETIDs: keep the in-core copy; if none in-core, keep first occurrence
    order = np.argsort(tid, kind="stable")
    tid_s = tid[order]
    dup_start = np.r_[True, tid_s[1:] != tid_s[:-1]]
    grp_id = np.cumsum(dup_start) - 1
    n_groups = grp_id[-1] + 1
    # per group, pick the winner index (prefer in_core)
    inv = np.empty(N, np.int64); inv[order] = grp_id       # per-node group id
    # vectorised winner per group: prefer in_core, then earliest position (unique -> no ties)
    pos = np.arange(N)
    score = in_core.astype(np.int64) * (N + 1) + (N - pos)  # prefer in_core, then earliest
    # reduce max per group
    winner_score = np.full(n_groups, -1, np.int64)
    np.maximum.at(winner_score, inv, score)
    keep = score == winner_score[inv]
    # guard against ties (shouldn't happen: pos unique) -> keep exactly one per group
    dup_removed = int((~keep).sum())
    active &= keep
    print(f"dedup: {n_groups} unique TARGETIDs from {N} rows; {dup_removed} buffer copies -> passive")

    # ---- halo-disjoint spatial regions by centroid RA --------------------------------
    # region codes: 0=train (RA<train_hi), 1=val (train_hi<=RA<test_lo), 2=test (RA>=test_lo)
    def region_of(ra_vals):
        r = np.full(ra_vals.shape, 1, np.int8)
        r[ra_vals < args.ra_train_hi] = 0
        r[ra_vals >= args.ra_test_lo] = 2
        return r
    # group by full halo key; assign whole group by its centroid RA
    hk = np.ascontiguousarray(halo).view([('', halo.dtype)] * 3).ravel()
    uk, inv_h = np.unique(hk, return_inverse=True)
    sum_ra = np.zeros(len(uk)); cnt_h = np.zeros(len(uk))
    np.add.at(sum_ra, inv_h, ra); np.add.at(cnt_h, inv_h, 1.0)
    centroid_ra = sum_ra / cnt_h
    grp_region = region_of(centroid_ra)
    region = grp_region[inv_h]                     # per-node region from its halo group
    n_straddle_fixed = int((region != region_of(ra)).sum())
    print(f"halo-disjoint reassignment moved {n_straddle_fixed} nodes to their group's region")

    # ---- 15 Mpc transverse graph gutter around each RA boundary -----------------------
    dtheta_deg = np.degrees(args.gutter_mpc / np.maximum(dcom(z), 1e-6))  # per-node ang. size
    near_b1 = np.abs(ra - args.ra_train_hi) < dtheta_deg
    near_b2 = np.abs(ra - args.ra_test_lo) < dtheta_deg
    gutter = near_b1 | near_b2
    active &= ~gutter
    print(f"gutter: {int(gutter.sum())} nodes within {args.gutter_mpc} Mpc of a boundary -> passive")

    train_mask = active & (region == 0)
    val_mask = active & (region == 1)
    test_mask = active & (region == 2)
    for nm, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"  {nm:5s}: {int(m.sum()):7d} active nodes")
    train_idx = np.where(train_mask)[0]

    # ---- ONE pooled box-cox on pooled TRAIN, then append ñ ---------------------------
    x_pos = x_si + 1e-6
    if np.any(x_pos.min(0) <= 0):
        print("WARN box-cox input min<=0 after +1e-6:", x_pos.min(0))
    bc = PowerTransformer(method="box-cox")
    bc.fit(x_pos[train_idx])
    x_bc = bc.transform(x_pos).astype(np.float64)                       # [N,7]
    ntf = ntilde_feature(z, spline).astype(np.float64)[:, None]         # [N,1] standardized
    x_final = np.concatenate([x_bc, ntf], axis=1).astype(np.float32)    # [N,8]
    print(f"node features: {x_final.shape}  (cols {NODE_FEATURE_NAMES})")
    print(f"  train-split box-cox mean={np.round(x_bc[train_idx].mean(0),3)}")
    print(f"  ñ feature (all/train) mean={ntf.mean():.3f}/{ntf[train_idx].mean():.3f} "
          f"std={ntf.std():.3f}")

    # ---- ONE pooled target scaler on pooled TRAIN ------------------------------------
    ts = StandardScaler().fit(tgt_raw[train_idx])
    tgt_scaled = ts.transform(tgt_raw).astype(np.float32)
    smin = tgt_scaled[train_idx].min(0); smax = tgt_scaled[train_idx].max(0)
    stats = {"increment_mode": "linear", "target_min": smin.tolist(), "target_max": smax.tolist(),
             "scaler_mean": ts.mean_.tolist(), "scaler_std": ts.scale_.tolist()}

    graph = jraph.GraphsTuple(
        nodes=jnp.array(x_final, jnp.float32), edges=jnp.array(edges, jnp.float32),
        senders=jnp.array(senders, jnp.int32), receivers=jnp.array(receivers, jnp.int32),
        n_node=jnp.array([N], jnp.int32), n_edge=jnp.array([senders.shape[0]], jnp.int32),
        globals=None)

    payload = {
        "graph": graph,
        "regression_targets": jnp.array(tgt_scaled),
        "regression_targets_raw": tgt_raw,
        "target_scaler": ts,
        "eigenvalues_raw": eig_raw,
        "masks": (jnp.array(train_mask), jnp.array(val_mask), jnp.array(test_mask)),
        "stats": stats,
        "node_feature_scaler": bc,
        "node_feature_power_method": "box-cox",
        "node_feature_names": NODE_FEATURE_NAMES,
        "si_exclude_features": sorted(SI_EXCLUDE),
        "ntilde_feature_index": 7,
        "ntilde_spline_path": str(args.ntilde_spline),
        "box_index": box_index,
        "spatial_split": {"ra_train_hi": args.ra_train_hi, "ra_test_lo": args.ra_test_lo,
                          "gutter_mpc": args.gutter_mpc, "halo_disjoint": True},
        "provenance": {"shells": SHELLS, "n_nodes": int(N),
                       "n_active": {"train": int(train_mask.sum()), "val": int(val_mask.sum()),
                                    "test": int(test_mask.sum())},
                       "dup_removed": dup_removed, "gutter_removed": int(gutter.sum())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    meta = {k: payload[k] for k in ("stats", "node_feature_names", "si_exclude_features",
                                    "ntilde_feature_index", "ntilde_spline_path",
                                    "spatial_split", "provenance")}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nSaved pooled cache: {args.out}\nSaved meta: {args.out.with_suffix('.meta.json')}")

    # ---- sanity gate ------------------------------------------------------------------
    print("\n=== SANITY GATE ===")
    # (1) ñ untouched by box-cox: last col == recomputed standardized feature
    ok_nt = np.allclose(np.asarray(graph.nodes)[:, 7], ntf[:, 0], atol=1e-5)
    print(f"[{'PASS' if ok_nt else 'FAIL'}] ñ column is the untransformed standardized feature")
    # (2) regions disjoint & non-empty per shell
    ok_reg = True
    for si, tag in enumerate(SHELLS):
        sm = shell == si
        cnts = [int((train_mask & sm).sum()), int((val_mask & sm).sum()), int((test_mask & sm).sum())]
        empty = min(cnts) == 0
        ok_reg &= not empty
        print(f"    shell {tag}: train/val/test active = {cnts}{'  <-- EMPTY' if empty else ''}")
    print(f"[{'PASS' if ok_reg else 'WARN'}] every shell populates all three regions")
    # (3) zero train<->test halo leakage
    train_halos = set(map(tuple, halo[train_mask]))
    test_halos = set(map(tuple, halo[test_mask]))
    leak = train_halos & test_halos
    print(f"[{'PASS' if not leak else 'FAIL'}] train/test halo-disjoint: {len(leak)} shared halos")
    # (4) no active node is a cross-shell duplicate
    act_tid = tid[active]
    _, c = np.unique(act_tid, return_counts=True)
    print(f"[{'PASS' if c.max()==1 else 'FAIL'}] active nodes are TARGETID-unique (max mult {c.max()})")


if __name__ == "__main__":
    main()
