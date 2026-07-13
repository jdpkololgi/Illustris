#!/usr/bin/env python3
"""S1(b) — winner zero-shot across the S2 shell caches: G3 union GraphNet + FMPE-era
flow (roadmap §3b amended row). The trained z0.2-0.3 model is evaluated UNTRAINED on
each shell: per-shell lambda1 R^2, cluster Spearman, and 68/90% coverage — the
production-fidelity encoder-at-sparsity readout (S2.5) for the graph branch.

Leakage guard: galaxies present in the ORIGINAL training wedge (z0.2-0.3, same box)
are excluded from evaluation via (FILE_NUM, BOX_INDEX) keys.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import fitsio
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from plot_flowjax_posteriors import (load_flowjax_model, load_data, create_gnn_and_flow,
                                     batched_sample_posterior)
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues

SHELL_DIR = Path("/pscratch/sd/d/dkololgi/abacus/sbi_caches")
SHELLS = ["s2_shell_0p05_0p15_si_union", "s2_shell_0p15_0p25_si_union",
          "s2_shell_0p25_0p35_si_union", "s2_shell_0p35_0p45_si_union",
          "s2_shell_0p45_0p55_si_union"]
TARGETS = {s: f"/pscratch/sd/d/dkololgi/abacus/s2_shells/{s.replace('s2_','').replace('_si_union','')}_final_wedge_targets.fits"
           for s in SHELLS}
TRAIN_WEDGE_TARGETS = ("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
                       "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_wedge_targets.fits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--max-eval", type=int, default=20000)
    args = ap.parse_args()
    import jax

    tw = fitsio.read(TRAIN_WEDGE_TARGETS, columns=["FILE_NUM", "BOX_INDEX"])
    train_keys = set(zip(tw["FILE_NUM"].tolist(), tw["BOX_INDEX"].tolist()))
    gnn_params, config, target_scaler, flow_filename, increment_mode = \
        load_flowjax_model(str(args.model_path))
    print(f"model: {args.model_path.name}; increment_mode={increment_mode}")
    print(f"\n{'shell':26s} {'n_eval':>7s} {'leak_excl':>9s} {'R2_l1':>7s} {'R2_l2':>7s} "
          f"{'R2_l3':>7s} {'cluSp':>6s} {'cov68':>6s} {'cov90':>6s}")

    for s in SHELLS:
        cache_path = SHELL_DIR / s / "processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
        graph, targets, trm, vam, tem, eig_raw = load_data(
            data_path=str(cache_path), increment_mode=increment_mode)
        tg = fitsio.read(TARGETS[s], columns=["FILE_NUM", "BOX_INDEX"])
        leak = np.fromiter(((f, b) in train_keys for f, b in
                            zip(tg["FILE_NUM"].tolist(), tg["BOX_INDEX"].tolist())),
                           dtype=bool, count=len(tg))
        ok = ~leak
        rng = np.random.default_rng(42)
        idx = np.where(ok)[0]
        if len(idx) > args.max_eval:
            idx = rng.permutation(idx)[: args.max_eval]

        gnn, flow = create_gnn_and_flow(config, flow_filename, graph, jax.random.key(42))
        emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
        S = batched_sample_posterior(flow, emb[idx], args.n_samples, jax.random.key(7))
        lam = np.stack([samples_to_raw_eigenvalues(S[i], target_scaler, increment_mode)
                        for i in range(len(idx))], axis=0)
        mean = lam.mean(axis=1)
        truth = np.asarray(eig_raw)[idx]

        r2 = [r2_score(truth[:, k], mean[:, k]) for k in range(3)]
        clu = truth[:, 0] > 0.2
        sp = spearmanr(truth[clu, 0], mean[clu, 0]).statistic if clu.sum() > 20 else np.nan
        covs = []
        for q in (0.68, 0.90):
            lo = np.quantile(lam[:, :, 0], (1-q)/2, axis=1)
            hi = np.quantile(lam[:, :, 0], 1-(1-q)/2, axis=1)
            covs.append(float(np.mean((truth[:, 0] >= lo) & (truth[:, 0] <= hi))))
        print(f"{s:26s} {len(idx):7d} {int(leak.sum()):9d} {r2[0]:7.3f} {r2[1]:7.3f} "
              f"{r2[2]:7.3f} {sp:6.2f} {covs[0]:6.3f} {covs[1]:6.3f}")

    print("\nReadout: R2 collapse and/or coverage drift vs shell = the unconditioned "
          "model's OOD failure profile (motivates Phase-B conditioning); shell-4 row = "
          "graph branch's encoder-at-sparsity verdict vs the CNN session.")


if __name__ == "__main__":
    main()
