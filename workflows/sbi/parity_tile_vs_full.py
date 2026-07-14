#!/usr/bin/env python3
"""Phase 3 — tile/full receptive-field parity test.

Does RA-tiling with a finite buffer alter the encoder's predictions vs the complete shell graph?
An 8-pass GNN has a large graph receptive field; a 17 Mpc buffer covers ~1 radius hop, not 8 hops.
On the z0.25-0.35 shell (fits as ONE whole-shell tile, so 'full' is available): compute full-graph
embeddings + posterior-mean λ1, then artificially split the SAME graph at its RA median into two
core bins and, for buffers {17,30,50} Mpc, rebuild the induced core+buffer subgraph, re-encode the
core nodes, and compare. Report |Δλ1| and embedding cosine-distance vs distance-to-cut (Mpc).

Gate: tiled≈full in the central region (far from the cut); the buffer at which the near-cut
disagreement becomes negligible sets the production tile buffer. Model-consistent (uses the cache
+ model the encoder was trained with); label validity is irrelevant to a receptive-field test.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
_bad = ("/global/homes/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/homes/d/dkololgi/.local/lib/python3.11/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.11/site-packages")
for _p in _bad:
    while _p in sys.path:
        sys.path.remove(_p)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import numpy as np
import fitsio
import jraph
from astropy.cosmology import Planck18 as cosmo

S2 = Path("/pscratch/sd/d/dkololgi/abacus/s2_shells")
_ZT = np.linspace(0, 0.75, 8000); _DT = cosmo.comoving_distance(_ZT).value
def dcom(z): return np.interp(z, _ZT, _DT)
def deg_per_mpc(z): return np.degrees(1.0 / np.maximum(dcom(z), 1e-6))


def induced(keep_idx, se, re, ea, n_full):
    nm = -np.ones(n_full, np.int64); nm[keep_idx] = np.arange(len(keep_idx))
    m = (nm[se] >= 0) & (nm[re] >= 0)
    return nm[se[m]], nm[re[m]], ea[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--shell", default="0p25_0p35")
    ap.add_argument("--buffers", type=float, nargs="+", default=[17, 30, 50])
    ap.add_argument("--n-samples", type=int, default=128)
    args = ap.parse_args()
    import jax
    from plot_flowjax_posteriors import load_flowjax_model, create_gnn_and_flow, batched_sample_posterior
    from shared.eigenvalue_transformations import samples_to_raw_eigenvalues

    gnn_params, config, tscaler, flow_file, inc = load_flowjax_model(str(args.model_path))
    manifest = json.loads((Path(args.tiles_dir) / "manifest.json").read_text())
    tinfo = [t for t in manifest["tiles"] if t["shell"] == args.shell][0]
    p = pickle.load(open(Path(args.tiles_dir) / tinfo["file"], "rb"))
    g = p["graph"]; eig = np.asarray(p["eigenvalues_raw"])
    x = np.asarray(g.nodes); se = np.asarray(g.senders); re = np.asarray(g.receivers); ea = np.asarray(g.edges)
    N = x.shape[0]
    # per-node RA/z via eigenvalue-triplet match to the shell FITS
    t = fitsio.read(S2 / f"shell_{args.shell}_final_wedge_targets.fits", columns=["RA", "Z", "LAMBDA1", "LAMBDA2", "LAMBDA3"])
    L = np.round(np.stack([t["LAMBDA1"], t["LAMBDA2"], t["LAMBDA3"]], 1).astype(np.float64), 5)
    keys = {tuple(L[i]): i for i in range(len(L))}
    et = np.round(eig, 5)
    gi = np.array([keys.get(tuple(et[i]), -1) for i in range(N)])
    ok = gi >= 0
    ra = np.full(N, np.nan); z = np.full(N, np.nan)
    ra[ok] = t["RA"][gi[ok]].astype(np.float64); z[ok] = t["Z"][gi[ok]].astype(np.float64)
    print(f"shell {args.shell}: N={N}, matched positions {ok.sum()}/{N}")

    def encode_l1(nodes, se_, re_, ea_, idx_out):
        gg = jraph.GraphsTuple(nodes=jax.numpy.asarray(nodes), edges=jax.numpy.asarray(ea_),
                               senders=jax.numpy.asarray(se_.astype(np.int32)),
                               receivers=jax.numpy.asarray(re_.astype(np.int32)),
                               n_node=jax.numpy.asarray([nodes.shape[0]]),
                               n_edge=jax.numpy.asarray([se_.shape[0]]), globals=None)
        gnn, flow = create_gnn_and_flow(config, flow_file, gg, jax.random.key(42))
        emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), gg, is_training=False))
        S = batched_sample_posterior(flow, emb[idx_out], args.n_samples, jax.random.key(7))
        lam = np.stack([samples_to_raw_eigenvalues(S[i], tscaler, inc) for i in range(len(idx_out))], 0)
        return emb[idx_out], lam.mean(1)[:, 0]

    # FULL graph
    emb_full, l1_full = encode_l1(x, se, re, ea, np.arange(N))

    # split at RA median -> 2 core bins; internal cut = RA_mid
    ra_mid = np.nanmedian(ra[ok])
    print(f"internal cut at RA={ra_mid:.2f}")
    bins = [(ra[ok].min() - 1e-3, ra_mid), (ra_mid, ra[ok].max() + 1e-3)]

    print(f"\n{'buffer_Mpc':>10s} {'n_core':>7s} {'mean|Δλ1|':>10s} {'|Δλ1|>0.05':>10s} "
          f"{'embcos':>7s} | disagreement by distance-to-cut (Mpc):")
    dist_bins = [(0, 15), (15, 30), (30, 60), (60, 1e9)]
    for B in args.buffers:
        dl1 = np.full(N, np.nan); ecos = np.full(N, np.nan)
        for (c_lo, c_hi) in bins:
            core = ok & (ra >= c_lo) & (ra < c_hi)
            buf_deg = B * deg_per_mpc(z)
            keep = core | (ok & (ra >= c_lo - buf_deg) & (ra < c_hi + buf_deg))
            keep_idx = np.where(keep)[0]
            se2, re2, ea2 = induced(keep_idx, se, re, ea, N)
            core_local = np.where(core[keep_idx])[0]
            emb_t, l1_t = encode_l1(x[keep_idx], se2, re2, ea2, core_local)
            gidx = keep_idx[core_local]
            dl1[gidx] = np.abs(l1_t - l1_full[gidx])
            a, b = emb_t, emb_full[gidx]
            ecos[gidx] = 1 - np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        m = np.isfinite(dl1)
        dcut = np.abs(ra - ra_mid) * (dcom(z) * np.pi / 180.0)   # transverse Mpc to the cut
        by = []
        for lo, hi in dist_bins:
            sel = m & (dcut >= lo) & (dcut < hi)
            by.append(f"[{lo:.0f}-{hi if hi<1e9 else '∞'}):{np.nanmean(dl1[sel]):.3f}" if sel.sum() else f"[{lo:.0f}):--")
        print(f"{B:10.0f} {int(m.sum()):7d} {np.nanmean(dl1[m]):10.4f} {np.mean(dl1[m]>0.05):10.3f} "
              f"{np.nanmean(ecos[m]):7.3f} | " + "  ".join(by))
    print("\nGate: central-region (>60 Mpc from cut) |Δλ1| must be ~0; the buffer at which near-cut "
          "|Δλ1| becomes negligible is the required production tile buffer. If even 50 Mpc leaves large "
          "near-cut disagreement, reduce message-passing depth or use exact k-hop halos.")


if __name__ == "__main__":
    main()
