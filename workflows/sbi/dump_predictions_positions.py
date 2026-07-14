#!/usr/bin/env python3
"""Dump per-galaxy predicted vs true eigenvalues + sky positions on the HELD-OUT test region,
for validation plots. Reconstructs each tile's galaxy positions from the manifest geometry
(deterministic keep_idx) since tiles don't store RA/z; cross-checks against stored eigenvalues.
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
from astropy.cosmology import Planck18 as cosmo

CORE_ZLO = {"0p05_0p15": 0.05, "0p15_0p25": 0.15, "0p25_0p35": 0.25, "0p35_0p45": 0.35, "0p45_0p55": 0.45}
S2 = Path("/pscratch/sd/d/dkololgi/abacus/s2_shells")
_ZT = np.linspace(0, 0.75, 8000); _DT = cosmo.comoving_distance(_ZT).value
def dcom(z): return np.interp(z, _ZT, _DT)
def deg_per_mpc(z): return np.degrees(1.0 / max(dcom(z), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()
    import jax
    from plot_flowjax_posteriors import load_flowjax_model, create_gnn_and_flow, batched_sample_posterior
    from shared.eigenvalue_transformations import samples_to_raw_eigenvalues

    gnn_params, config, target_scaler, flow_filename, inc = load_flowjax_model(str(args.model_path))
    tiles_dir = Path(args.tiles_dir); manifest = json.loads((tiles_dir / "manifest.json").read_text())
    buffer_mpc = manifest["spatial_split"]["buffer_mpc"]

    shell_cache = {}
    def load_shell(tag):
        if tag not in shell_cache:
            t = fitsio.read(S2 / f"shell_{tag}_final_wedge_targets.fits",
                            columns=["RA", "DEC", "Z", "LAMBDA1"])
            shell_cache[tag] = (t["RA"].astype(np.float64), t["DEC"].astype(np.float64),
                                t["Z"].astype(np.float64), t["LAMBDA1"].astype(np.float64))
        return shell_cache[tag]

    P, T, RA, DEC, Z, SH = [], [], [], [], [], []
    for tinfo in manifest["tiles"]:
        if tinfo["test"] == 0:
            continue
        tag = tinfo["shell"]; c_lo, c_hi = tinfo["ra_core"]
        p = pickle.load(open(tiles_dir / tinfo["file"], "rb"))
        graph = p["graph"]; eig_raw = np.asarray(p["eigenvalues_raw"])
        test_mask = np.asarray(p["masks"][2]).astype(bool)
        # reconstruct keep_idx (== tile node order) from manifest geometry
        ra_s, dec_s, z_s, l1_s = load_shell(tag)
        buf = buffer_mpc * deg_per_mpc(CORE_ZLO[tag])
        keep_idx = np.where((ra_s >= c_lo - buf) & (ra_s < c_hi + buf))[0]
        assert len(keep_idx) == eig_raw.shape[0], f"{tinfo['file']}: {len(keep_idx)} != {eig_raw.shape[0]}"
        assert np.allclose(l1_s[keep_idx], eig_raw[:, 0], atol=1e-4), f"{tinfo['file']}: eig misalign"

        gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
        emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
        ti = np.where(test_mask)[0]
        S = batched_sample_posterior(flow, emb[ti], args.n_samples, jax.random.key(7))
        lam = np.stack([samples_to_raw_eigenvalues(S[i], target_scaler, inc) for i in range(len(ti))], 0)
        gi = keep_idx[ti]                       # global shell rows for the test nodes
        P.append(lam.mean(1)); T.append(eig_raw[ti])
        RA.append(ra_s[gi]); DEC.append(dec_s[gi]); Z.append(z_s[gi]); SH.append(np.array([tag] * len(ti)))

    pred = np.concatenate(P); true = np.concatenate(T)
    np.savez_compressed(args.out_npz, pred=pred, true=true,
                        ra=np.concatenate(RA), dec=np.concatenate(DEC), z=np.concatenate(Z),
                        shell=np.concatenate(SH))
    print(f"saved {pred.shape[0]} test galaxies -> {args.out_npz}")
    for k, nm in enumerate(["l1", "l2", "l3"]):
        from sklearn.metrics import r2_score
        m = np.concatenate(Z) >= 0.15
        print(f"  {nm}: R2(all)={r2_score(true[:,k],pred[:,k]):.3f}  R2(z>=0.15)={r2_score(true[m,k],pred[m,k]):.3f}")


if __name__ == "__main__":
    main()
