#!/usr/bin/env python3
"""TILED ñ-conditioned GNN+flow trainer (Phase-B OOM pivot).

Mini-batches over the S3b tiles (each a disjoint graph <= ~4M edges that fits one GPU),
instead of one giant pooled graph that OOMs. Single GPU + jit (one compile cached per tile
shape). All tile INPUT arrays live on-device at once (~0.5GB total); only the current tile's
forward activation is transient (~50GB). Params carry across tiles (SGD over graphs).

Model/flow/optimizer/save-format IDENTICAL to jraph_sbi_flowjax.py so downstream eval
(plot_flowjax_posteriors.load_flowjax_model / create_gnn_and_flow) works unchanged.
Consumes <tiles-dir>/{manifest.json, shared_scalers.pkl, tile_###.pkl}.
"""
from __future__ import annotations
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import optax
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal
from shared.graph_net_models import make_gnn_encoder


def load_tiles(tiles_dir):
    tiles_dir = Path(tiles_dir)
    manifest = json.loads((tiles_dir / "manifest.json").read_text())
    shared = pickle.load(open(tiles_dir / "shared_scalers.pkl", "rb"))
    tiles = []
    for t in manifest["tiles"]:
        p = pickle.load(open(tiles_dir / t["file"], "rb"))
        tr, va, te = p["masks"]
        tiles.append(dict(
            graph=jax.device_put(p["graph"]),
            targets=jax.device_put(jnp.asarray(p["regression_targets"])),
            train=jax.device_put(jnp.asarray(tr)), val=jax.device_put(jnp.asarray(va)),
            test=jax.device_put(jnp.asarray(te)),
            n_train=int(np.asarray(tr).sum()), n_val=int(np.asarray(va).sum()),
            n_test=int(np.asarray(te).sum())))
    return manifest, shared, tiles


def main(args):
    key = jax.random.PRNGKey(args.seed)
    print(f"jax devices: {jax.devices()}")
    manifest, shared, tiles = load_tiles(args.tiles_dir)
    target_scaler = shared["target_scaler"]
    print(f"loaded {len(tiles)} tiles; totals {manifest['totals']}; "
          f"max tile {manifest['max_tile_nodes']} nodes / {manifest['max_tile_edges']} edges")

    gnn = hk.transform(make_gnn_encoder(num_passes=args.num_passes, latent_size=args.latent_size,
                                        num_heads=args.num_heads, dropout_rate=args.dropout))
    key, ik = jax.random.split(key)
    gnn_params = gnn.init(ik, tiles[0]["graph"], is_training=True)

    base = Normal(jnp.zeros(3), jnp.ones(3))
    key, fk = jax.random.split(key)
    flow = masked_autoregressive_flow(fk, base_dist=base, cond_dim=args.latent_size,
                                      flow_layers=args.num_flow_layers, nn_width=args.flow_hidden_size,
                                      nn_depth=2, transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12))
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)
    print(f"GNN params {sum(x.size for x in jax.tree_util.tree_leaves(gnn_params)):,}; "
          f"flow params {sum(x.size for x in jax.tree_util.tree_leaves(flow_arrays)):,}")

    warmup = min(500, args.epochs // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, max(args.epochs - warmup, warmup + 1), 1e-5)
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
        (loss, (slp, nm)), (gg, fg) = jax.value_and_grad(
            lambda g, f: nll(g, f, graph, targets, mask, rng, True), argnums=(0, 1), has_aux=True)(gnn_p, flow_arr)
        gu, gnn_o = optim.update(gg, gnn_o, gnn_p); gnn_p = optax.apply_updates(gnn_p, gu)
        fu, flow_o = optim.update(fg, flow_o, flow_arr); flow_arr = optax.apply_updates(flow_arr, fu)
        return gnn_p, gnn_o, flow_arr, flow_o, loss

    @jax.jit
    def val_step(gnn_p, flow_arr, graph, targets, mask, rng):
        _, (slp, nm) = nll(gnn_p, flow_arr, graph, targets, mask, rng, False)
        return slp, nm

    # resume
    start, best_val = 0, np.inf
    best_gnn, best_flow_arr = gnn_params, flow_arrays
    ckpt = os.path.join(args.output_dir, f"tiled_checkpoint_seed_{args.seed}.pkl")
    if args.resume and os.path.exists(ckpt):
        ck = pickle.load(open(ckpt, "rb"))
        gnn_params, gnn_opt = ck["gnn_params"], ck["gnn_opt"]
        flow_arrays, flow_opt = ck["flow_arrays"], ck["flow_opt"]
        start, best_val = ck["epoch"] + 1, ck["best_val"]
        best_gnn, best_flow_arr = ck["best_gnn"], ck["best_flow_arr"]
        key = jax.random.PRNGKey(ck["epoch"] + args.seed)
        print(f"[resume] from epoch {start} (best val NLL {best_val:.4f})")
    elif args.resume:
        print(f"[resume] no checkpoint at {ckpt}; starting fresh.")

    os.makedirs(args.output_dir, exist_ok=True)
    order = np.arange(len(tiles))
    print(f"\n[train] {args.epochs} epochs over {len(tiles)} tiles "
          f"(first epoch compiles ~{len(tiles)} tile shapes)")
    for epoch in range(start, args.epochs):
        key, sk = jax.random.split(key)
        np.random.RandomState(epoch).shuffle(order)
        ep_loss, ep_n = 0.0, 0
        for ti in order:
            t = tiles[ti]
            if t["n_train"] == 0:
                continue
            key, rng = jax.random.split(key)
            gnn_params, gnn_opt, flow_arrays, flow_opt, loss = update(
                gnn_params, gnn_opt, flow_arrays, flow_opt, t["graph"], t["targets"], t["train"], rng)
            ep_loss += float(loss) * t["n_train"]; ep_n += t["n_train"]
        train_nll = ep_loss / max(ep_n, 1)

        if epoch % args.report_every == 0 or epoch == args.epochs - 1:
            slp, nm = 0.0, 0
            for t in tiles:
                if t["n_val"] == 0:
                    continue
                key, rng = jax.random.split(key)
                s, n = val_step(gnn_params, flow_arrays, t["graph"], t["targets"], t["val"], rng)
                slp += float(s); nm += float(n)
            val_nll = -slp / max(nm, 1.0)
            dt = time.time()
            print(f"epoch {epoch:5d}  train NLL {train_nll:.4f}  val NLL {val_nll:.4f}", flush=True)
            if val_nll < best_val:
                best_val = val_nll; best_gnn = jax.device_get(gnn_params)
                best_flow_arr = jax.device_get(flow_arrays)

        if epoch % args.checkpoint_every == 0 and epoch > 0:
            tmp = ckpt + ".tmp"
            pickle.dump(dict(epoch=epoch, gnn_params=jax.device_get(gnn_params), gnn_opt=jax.device_get(gnn_opt),
                             flow_arrays=jax.device_get(flow_arrays), flow_opt=jax.device_get(flow_opt),
                             best_val=best_val, best_gnn=jax.device_get(best_gnn),
                             best_flow_arr=jax.device_get(best_flow_arr)), open(tmp, "wb"))
            os.replace(tmp, ckpt)

    # save in the ORIGINAL model format (downstream-compatible)
    ts = time.strftime("%Y%m%d_%H%M%S")
    flow_file = os.path.join(args.output_dir, f"flowjax_sbi_flow_seed_{args.seed}_{ts}.eqx")
    eqx.tree_serialise_leaves(flow_file, eqx.combine(best_flow_arr, flow_static))
    model_file = os.path.join(args.output_dir, f"flowjax_sbi_model_seed_{args.seed}_{ts}.pkl")
    pickle.dump(dict(gnn_params=jax.device_get(best_gnn), config=vars(args), target_scaler=target_scaler,
                     use_transformed_eig=(args.increment_mode != "raw"), increment_mode=args.increment_mode,
                     flow_filename=flow_file, tiles_dir=str(args.tiles_dir)), open(model_file, "wb"))
    print(f"\nSaved flow -> {flow_file}\nSaved model -> {model_file}\nbest val NLL {best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--increment_mode", type=str, default="linear")
    ap.add_argument("--epochs", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
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
