#!/usr/bin/env python3
"""TILED ñ-conditioned GNN+flow trainer — CORRECTED optimizer semantics (2026-07-14).

Fixes the invalid Phase-B run (see RUNCARD_invalid.json):
 * PER-SCIENTIFIC-SAMPLE optimizer: each update samples one tile with probability
   p(t)=N_train_t / Σ N_train_u, so the expected gradient equals the per-galaxy empirical
   objective and is INDEPENDENT of tile count (was: one update per tile → dense-shell/tile
   count dominated, corrupt shell got 52% of updates).
 * GLOBAL-STEP learning-rate schedule over `--total-updates` UPDATE units (was: decay_steps in
   epoch units but advanced per-tile → LR floored ~20× too early).
 * Deterministic, resumable RNG (tile choice + dropout keyed on global step).
 * Separate best-VAL-NLL and best-VAL-λ1 checkpoints; test region NEVER used for selection.
 * Patience early-stopping in update units. Run lock to prevent concurrent writers.
 * Completion marker written only by the trainer, with total_updates + params hash.

Consumes <tiles-dir>/{manifest.json, shared_scalers.pkl, tile_###.pkl}. Save format matches
jraph_sbi_flowjax.py so downstream eval is unchanged.
"""
from __future__ import annotations
import argparse, json, os, sys, pickle, time, hashlib, uuid, atexit
from pathlib import Path
from collections import defaultdict

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

import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import optax
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal
from shared.graph_net_models import make_gnn_encoder
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues
from plot_flowjax_posteriors import batched_sample_posterior
from sklearn.metrics import r2_score


def load_tiles(tiles_dir):
    tiles_dir = Path(tiles_dir)
    manifest = json.loads((tiles_dir / "manifest.json").read_text())
    shared = pickle.load(open(tiles_dir / "shared_scalers.pkl", "rb"))
    tiles = []
    for t in manifest["tiles"]:
        p = pickle.load(open(tiles_dir / t["file"], "rb"))
        tr, va, te = (np.asarray(m).astype(bool) for m in p["masks"])
        tiles.append(dict(
            shell=t["shell"], graph=jax.device_put(p["graph"]),
            targets=jax.device_put(jnp.asarray(p["regression_targets"])),
            eig_raw=np.asarray(p["eigenvalues_raw"]),
            train=jax.device_put(jnp.asarray(tr)), val=jax.device_put(jnp.asarray(va)),
            val_np=va, n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum())))
    return manifest, shared, tiles


def main(args):
    print(f"jax devices: {jax.devices()}")
    os.makedirs(args.output_dir, exist_ok=True)
    # --- run lock (prevent concurrent writers to the same checkpoint) ---
    lock = os.path.join(args.output_dir, "train.lock")
    if os.path.exists(lock) and not args.force:
        raise SystemExit(f"[lock] {lock} exists — another trainer may be writing here. Use --force to override.")
    run_uuid = uuid.uuid4().hex
    open(lock, "w").write(f"pid={os.getpid()} uuid={run_uuid} t={time.time()}\n")
    atexit.register(lambda: os.path.exists(lock) and open(lock).read().count(run_uuid) and os.remove(lock))

    manifest, shared, tiles = load_tiles(args.tiles_dir)
    target_scaler = shared["target_scaler"]; inc = args.increment_mode
    train_tiles = np.array([i for i in range(len(tiles)) if tiles[i]["n_train"] > 0])
    p_train = np.array([tiles[i]["n_train"] for i in train_tiles], float)
    p_train /= p_train.sum()
    print(f"loaded {len(tiles)} tiles ({len(train_tiles)} with train nodes); totals {manifest['totals']}")
    print("  per-tile train-node sampling prob (node-proportional):")
    for i, pr in zip(train_tiles, p_train):
        print(f"    tile {i:2d} shell {tiles[i]['shell']} n_train={tiles[i]['n_train']:6d} p={pr:.4f}")

    gnn = hk.transform(make_gnn_encoder(num_passes=args.num_passes, latent_size=args.latent_size,
                                        num_heads=args.num_heads, dropout_rate=args.dropout))
    gnn_params = gnn.init(jax.random.PRNGKey(args.seed), tiles[0]["graph"], is_training=True)
    base = Normal(jnp.zeros(3), jnp.ones(3))
    flow = masked_autoregressive_flow(jax.random.PRNGKey(args.seed + 1), base_dist=base, cond_dim=args.latent_size,
                                      flow_layers=args.num_flow_layers, nn_width=args.flow_hidden_size,
                                      nn_depth=2, transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12))
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)

    # --- schedule in UPDATE units (one optim.update-pair per training step) ---
    sched_config = dict(total_updates=args.total_updates, warmup_updates=args.warmup_updates, lr=args.lr)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, args.warmup_updates,
                                               max(args.total_updates - args.warmup_updates, 1), 1e-5)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=args.weight_decay))
    gnn_opt = optim.init(gnn_params); flow_opt = optim.init(flow_arrays)

    def nll(gnn_p, flow_arr, graph, targets, mask, rng, is_training):
        emb = gnn.apply(gnn_p, rng, graph, is_training=is_training)
        fl = eqx.combine(flow_arr, flow_static)
        lp = jax.vmap(fl.log_prob)(targets, condition=emb)
        nm = jnp.sum(mask)
        return -jnp.sum(lp * mask) / jnp.maximum(nm, 1.0), (jnp.sum(lp * mask), nm)

    @jax.jit
    def update(gnn_p, gnn_o, flow_arr, flow_o, graph, targets, mask, rng):
        (loss, _), (gg, fg) = jax.value_and_grad(
            lambda g, f: nll(g, f, graph, targets, mask, rng, True), argnums=(0, 1), has_aux=True)(gnn_p, flow_arr)
        gu, gnn_o = optim.update(gg, gnn_o, gnn_p); gnn_p = optax.apply_updates(gnn_p, gu)
        fu, flow_o = optim.update(fg, flow_o, flow_arr); flow_arr = optax.apply_updates(flow_arr, fu)
        return gnn_p, gnn_o, flow_arr, flow_o, loss

    @jax.jit
    def val_nll_step(gnn_p, flow_arr, graph, targets, mask, rng):
        _, (slp, nm) = nll(gnn_p, flow_arr, graph, targets, mask, rng, False)
        return slp, nm

    def val_nll(gnn_p, flow_arr, step):
        slp, nm = 0.0, 0.0
        for t in tiles:
            if t["n_val"] == 0:
                continue
            s, n = val_nll_step(gnn_p, flow_arr, t["graph"], t["targets"], t["val"],
                                jax.random.fold_in(jax.random.PRNGKey(args.seed + 777), step))
            slp += float(s); nm += float(n)
        return -slp / max(nm, 1.0)

    def val_lambda1_r2(gnn_p, flow_arr):
        fl = eqx.combine(flow_arr, flow_static)
        per = defaultdict(lambda: ([], []))
        rng = np.random.default_rng(0)
        for t in tiles:
            if t["n_val"] == 0:
                continue
            emb = np.asarray(gnn.apply(gnn_p, jax.random.PRNGKey(0), t["graph"], is_training=False))
            vidx = np.where(t["val_np"])[0]
            if len(vidx) > args.l1_val_cap:
                vidx = rng.permutation(vidx)[: args.l1_val_cap]
            S = batched_sample_posterior(fl, emb[vidx], args.l1_val_samples, jax.random.key(3))
            lam = np.stack([samples_to_raw_eigenvalues(S[i], target_scaler, inc) for i in range(len(vidx))], 0)
            per[t["shell"]][0].append(lam.mean(1)[:, 0]); per[t["shell"]][1].append(t["eig_raw"][vidx][:, 0])
        pr, tr = [], []
        rows = {}
        for sh, (P, T) in per.items():
            P = np.concatenate(P); T = np.concatenate(T); pr.append(P); tr.append(T)
            rows[sh] = r2_score(T, P) if len(T) > 20 else float("nan")
        pooled = r2_score(np.concatenate(tr), np.concatenate(pr))
        macro = float(np.nanmean([v for v in rows.values()]))
        return pooled, macro, rows

    # --- resume (deterministic; refuse on schedule-config mismatch) ---
    ckpt = os.path.join(args.output_dir, f"tiled_v2_checkpoint_seed_{args.seed}.pkl")
    start_step = 0
    best_nll = np.inf; best_l1 = -np.inf
    best_nll_g, best_nll_f = jax.device_get(gnn_params), jax.device_get(flow_arrays)
    best_l1_g, best_l1_f = best_nll_g, best_nll_f
    since_improve = 0
    if args.resume and os.path.exists(ckpt):
        ck = pickle.load(open(ckpt, "rb"))
        if ck["sched_config"] != sched_config:
            raise SystemExit(f"[resume] schedule config mismatch: ckpt {ck['sched_config']} vs now {sched_config}. "
                             "Refusing (would corrupt the LR schedule).")
        gnn_params, gnn_opt = ck["gnn_params"], ck["gnn_opt"]
        flow_arrays, flow_opt = ck["flow_arrays"], ck["flow_opt"]
        start_step = ck["global_step"] + 1
        best_nll, best_l1 = ck["best_nll"], ck["best_l1"]
        best_nll_g, best_nll_f = ck["best_nll_g"], ck["best_nll_f"]
        best_l1_g, best_l1_f = ck["best_l1_g"], ck["best_l1_f"]
        since_improve = ck.get("since_improve", 0)
        print(f"[resume] from update {start_step} (best val NLL {best_nll:.4f}, best val λ1 R² {best_l1:.4f})")
    elif args.resume:
        print(f"[resume] no checkpoint at {ckpt}; starting fresh.")

    print(f"\n[train] {args.total_updates} updates, warmup {args.warmup_updates}, lr {args.lr}; "
          f"node-proportional tile sampling. val NLL every {args.val_every}, λ1 R² every {args.l1_every}.")
    t0 = time.time()
    for step in range(start_step, args.total_updates):
        ti = int(np.random.default_rng((args.seed << 20) + step).choice(train_tiles, p=p_train))
        t = tiles[ti]
        rng = jax.random.fold_in(jax.random.PRNGKey(args.seed + 12345), step)
        gnn_params, gnn_opt, flow_arrays, flow_opt, loss = update(
            gnn_params, gnn_opt, flow_arrays, flow_opt, t["graph"], t["targets"], t["train"], rng)

        if step % args.val_every == 0 or step == args.total_updates - 1:
            vnll = val_nll(gnn_params, flow_arrays, step)
            improved = vnll < best_nll - 1e-4
            if improved:
                best_nll = vnll; best_nll_g = jax.device_get(gnn_params); best_nll_f = jax.device_get(flow_arrays)
                since_improve = 0
            else:
                since_improve += args.val_every
            print(f"update {step:5d}  lr {float(sched(step)):.2e}  train NLL {float(loss):.4f}  "
                  f"val NLL {vnll:.4f}  best {best_nll:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        if step > 0 and (step % args.l1_every == 0 or step == args.total_updates - 1):
            pooled, macro, rows = val_lambda1_r2(gnn_params, flow_arrays)
            if pooled > best_l1:
                best_l1 = pooled; best_l1_g = jax.device_get(gnn_params); best_l1_f = jax.device_get(flow_arrays)
            print(f"    val λ1 R²: pooled {pooled:.3f}  macro {macro:.3f}  best {best_l1:.3f}  "
                  f"per-shell " + " ".join(f"{k}:{v:.2f}" for k, v in sorted(rows.items())), flush=True)

        if step > start_step and step % args.checkpoint_every == 0:
            tmp = ckpt + ".tmp"
            pickle.dump(dict(global_step=step, sched_config=sched_config,
                             gnn_params=jax.device_get(gnn_params), gnn_opt=jax.device_get(gnn_opt),
                             flow_arrays=jax.device_get(flow_arrays), flow_opt=jax.device_get(flow_opt),
                             best_nll=best_nll, best_l1=best_l1, best_nll_g=best_nll_g, best_nll_f=best_nll_f,
                             best_l1_g=best_l1_g, best_l1_f=best_l1_f, since_improve=since_improve), open(tmp, "wb"))
            os.replace(tmp, ckpt)

        if args.patience and since_improve >= args.patience:
            print(f"[early-stop] no val-NLL improvement in {since_improve} updates (patience {args.patience}).")
            break

    # --- save BOTH best-val-NLL and best-val-λ1 models (test never used for selection) ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    def save(tag, g, f):
        ff = os.path.join(args.output_dir, f"flowjax_sbi_flow_seed_{args.seed}_{tag}_{ts}.eqx")
        eqx.tree_serialise_leaves(ff, eqx.combine(f, flow_static))
        mf = os.path.join(args.output_dir, f"flowjax_sbi_model_seed_{args.seed}_{tag}_{ts}.pkl")
        pickle.dump(dict(gnn_params=g, config=vars(args), target_scaler=target_scaler,
                         use_transformed_eig=(inc != "raw"), increment_mode=inc, flow_filename=ff,
                         tiles_dir=str(args.tiles_dir), selection=tag), open(mf, "wb"))
        return mf
    mf_nll = save("bestNLL", best_nll_g, best_nll_f)
    mf_l1 = save("bestL1", best_l1_g, best_l1_f)
    h = hashlib.sha256(pickle.dumps(best_nll_g)).hexdigest()[:12]
    print(f"\nSaved best-NLL model -> {mf_nll}\nSaved best-λ1 model -> {mf_l1}")
    print(f"best val NLL {best_nll:.4f} | best val λ1 R² {best_l1:.4f}")
    with open(os.path.join(args.output_dir, "TRAINING_COMPLETE"), "w") as _f:
        _f.write(f"total_updates={args.total_updates} bestNLL_hash={h} model_nll={mf_nll} model_l1={mf_l1}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--increment_mode", type=str, default="linear")
    ap.add_argument("--total-updates", dest="total_updates", type=int, default=4000)
    ap.add_argument("--warmup-updates", dest="warmup_updates", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", dest="val_every", type=int, default=50)
    ap.add_argument("--l1-every", dest="l1_every", type=int, default=400)
    ap.add_argument("--l1-val-cap", dest="l1_val_cap", type=int, default=3000)
    ap.add_argument("--l1-val-samples", dest="l1_val_samples", type=int, default=64)
    ap.add_argument("--checkpoint_every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0, help="updates w/o val-NLL improvement before stop (0=off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="override run lock")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.08)
    ap.add_argument("--num_passes", type=int, default=8)
    ap.add_argument("--latent_size", type=int, default=80)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_flow_layers", type=int, default=5)
    ap.add_argument("--num_bins", type=int, default=8)
    ap.add_argument("--flow_hidden_size", type=int, default=128)
    main(ap.parse_args())
