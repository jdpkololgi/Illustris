"""Partition-aware SBI trainer for Abacus caches.

This entrypoint keeps `jraph_sbi_flowjax.py` untouched and trains using
partition artifacts produced by `build_abacus_partition_batches.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from flowjax.distributions import Normal
from flowjax.flows import RationalQuadraticSpline, masked_autoregressive_flow

# Allow workflow script to resolve repo-root modules after reorganization.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import CANONICAL_OUTPUT_ROOT
from shared.graph_net_models import make_gnn_encoder
from shared.resource_requirements import require_gpu_slurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", required=True, help="Path to partition_manifest.json")
    parser.add_argument("--sbi-cache-path", required=True, help="Original SBI cache path (for scaler/raw eig metadata).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-passes", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-flow-layers", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--flow-hidden-size", type=int, default=128)
    parser.add_argument("--train-partition-limit", type=int, default=0, help="0 means all train partitions.")
    parser.add_argument("--val-partition-limit", type=int, default=8, help="Validation partitions per eval.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help=(
            "Enable activation rematerialization (jax.checkpoint) for the GNN forward pass "
            "during training to reduce peak memory usage."
        ),
    )
    parser.add_argument(
        "--train-progress-every",
        type=int,
        default=1,
        help="Print training progress every N partition steps within each epoch.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{CANONICAL_OUTPUT_ROOT}/sbi_partitioned",
        help="Directory for models/logs.",
    )
    return parser.parse_args()


def load_partition(path: Path) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    with np.load(path) as d:
        x = jnp.array(d["x"], dtype=jnp.float32)
        edge_index = d["edge_index"]
        edge_attr = jnp.array(d["edge_attr"], dtype=jnp.float32)
        targets = jnp.array(d["targets"])
        core_mask_local = jnp.array(d["core_mask_local"], dtype=bool)
        if "global_node_ids" in d:
            global_node_ids = jnp.array(d["global_node_ids"], dtype=jnp.int64)
        else:
            global_node_ids = jnp.arange(x.shape[0], dtype=jnp.int64)

    senders = jnp.array(edge_index[0], dtype=jnp.int32)
    receivers = jnp.array(edge_index[1], dtype=jnp.int32)
    graph = jraph.GraphsTuple(
        nodes=x,
        edges=edge_attr,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([x.shape[0]], dtype=jnp.int32),
        n_edge=jnp.array([senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    return graph, targets, core_mask_local, global_node_ids


def main(args: argparse.Namespace) -> None:
    require_gpu_slurm("jraph_sbi_flowjax_partitioned.py", min_gpus=1)
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70, flush=True)
    print("Partitioned SBI Trainer (FlowJAX + Haiku)", flush=True)
    print("=" * 70, flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Manifest: {args.partition_manifest}", flush=True)
    print(f"SBI cache: {args.sbi_cache_path}", flush=True)
    print(
        f"Config: epochs={args.epochs}, num_passes={args.num_passes}, "
        f"latent={args.latent_size}, flow_layers={args.num_flow_layers}, "
        f"activation_checkpointing={args.activation_checkpointing}",
        flush=True,
    )

    with open(args.partition_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = Path(args.partition_manifest).resolve().parent

    train_parts = [p for p in manifest["partitions"] if p["split"] == "train"]
    val_parts = [p for p in manifest["partitions"] if p["split"] == "val"]
    if args.train_partition_limit > 0:
        train_parts = train_parts[: args.train_partition_limit]
    if args.val_partition_limit > 0:
        val_parts = val_parts[: args.val_partition_limit]

    if not train_parts:
        raise ValueError("No train partitions found.")
    print(f"Train partitions: {len(train_parts)}, Val partitions: {len(val_parts)}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    # Read scaler/raw metadata from the monolithic cache once for output compatibility.
    with open(args.sbi_cache_path, "rb") as f:
        source_cache = pickle.load(f)
    target_scaler = source_cache.get("target_scaler")
    stats = source_cache.get("stats")

    # Initialize on first train partition
    rng = jax.random.key(args.seed)
    print("Loading first train partition for model init...", flush=True)
    first_graph, _, _, _ = load_partition(base_dir / train_parts[0]["file"])
    print(
        f"First partition stats: nodes={int(first_graph.n_node[0]):,}, "
        f"edges={int(first_graph.n_edge[0]):,}",
        flush=True,
    )
    gnn_fn = make_gnn_encoder(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    gnn = hk.transform(gnn_fn)
    rng, gnn_init_key = jax.random.split(rng)
    gnn_params = gnn.init(gnn_init_key, first_graph, is_training=True)

    rng, flow_key = jax.random.split(rng)
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(3), jnp.ones(3)),
        cond_dim=args.latent_size,
        flow_layers=args.num_flow_layers,
        nn_width=args.flow_hidden_size,
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),
    )
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)

    num_epochs = args.epochs
    warmup_steps = min(200, max(1, num_epochs // 10))
    decay_steps = max(num_epochs - warmup_steps, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay),
    )
    gnn_opt_state = optimizer.init(gnn_params)
    flow_opt_state = optimizer.init(flow_arrays)

    def _gnn_forward_train(gnn_p, step_key, graph):
        return gnn.apply(gnn_p, step_key, graph, is_training=True)

    def _gnn_forward_eval(gnn_p, step_key, graph):
        return gnn.apply(gnn_p, step_key, graph, is_training=False)

    if args.activation_checkpointing:
        # Rematerialize GNN forward activations in backward pass to lower peak VRAM.
        _gnn_forward_train = jax.checkpoint(_gnn_forward_train, prevent_cse=False)

    def _loss_with_embeddings(emb, flow_arr, targets, core_mask):
        flow_model = eqx.combine(flow_arr, flow_static)
        log_probs = jax.vmap(flow_model.log_prob)(targets, condition=emb)
        masked = log_probs * core_mask
        n_core = jnp.maximum(jnp.sum(core_mask), 1.0)
        nll = -jnp.sum(masked) / n_core
        return nll, jnp.sum(masked) / n_core

    def train_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        emb = _gnn_forward_train(gnn_p, step_key, graph)
        return _loss_with_embeddings(emb, flow_arr, targets, core_mask)

    def eval_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        emb = _gnn_forward_eval(gnn_p, step_key, graph)
        return _loss_with_embeddings(emb, flow_arr, targets, core_mask)

    @jax.jit
    def train_step(gnn_p, gnn_state, flow_arr, flow_state, graph, targets, core_mask, step_key):
        (nll, mean_logp), grads = jax.value_and_grad(
            lambda gp, fa: train_loss_fn(gp, fa, graph, targets, core_mask, step_key),
            argnums=(0, 1),
            has_aux=True,
        )(gnn_p, flow_arr)
        gnn_grads, flow_grads = grads
        gnn_updates, gnn_state_new = optimizer.update(gnn_grads, gnn_state, gnn_p)
        flow_updates, flow_state_new = optimizer.update(flow_grads, flow_state, flow_arr)
        gnn_p_new = optax.apply_updates(gnn_p, gnn_updates)
        flow_arr_new = optax.apply_updates(flow_arr, flow_updates)
        return gnn_p_new, gnn_state_new, flow_arr_new, flow_state_new, nll, mean_logp

    @jax.jit
    def eval_step(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        return eval_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key)

    best_val = float("inf")
    best = None
    history = {"train_nll": [], "val_nll": []}

    t0 = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch:04d} start", flush=True)
        rng, ep_key = jax.random.split(rng)
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_parts))
        train_nll_epoch = []
        n_train_steps = len(order)
        for step_idx, i in enumerate(order):
            p = train_parts[int(i)]
            if step_idx % max(1, args.train_progress_every) == 0:
                print(
                    f"  [train] epoch={epoch:04d} step={step_idx + 1}/{n_train_steps} "
                    f"partition={p['partition_id']}",
                    flush=True,
                )
            graph, targets, core_mask, _ = load_partition(base_dir / p["file"])
            ep_key, step_key = jax.random.split(ep_key)
            gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, nll, _ = train_step(
                gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, graph, targets, core_mask, step_key
            )
            train_nll_epoch.append(float(nll))

        mean_train_nll = float(np.mean(train_nll_epoch)) if train_nll_epoch else float("nan")
        history["train_nll"].append((epoch, mean_train_nll))

        if epoch % args.eval_every == 0:
            val_losses = []
            print(f"  [val] epoch={epoch:04d} evaluating {len(val_parts)} partitions", flush=True)
            for p in val_parts:
                graph, targets, core_mask, _ = load_partition(base_dir / p["file"])
                ep_key, step_key = jax.random.split(ep_key)
                (val_nll, _) = eval_step(gnn_params, flow_arrays, graph, targets, core_mask, step_key)
                val_losses.append(float(val_nll))
            mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
            history["val_nll"].append((epoch, mean_val))
            if mean_val < best_val:
                best_val = mean_val
                best = (
                    jax.device_get(gnn_params),
                    jax.device_get(flow_arrays),
                )
            print(f"Epoch {epoch:04d} | train_nll={mean_train_nll:.4f} | val_nll={mean_val:.4f}", flush=True)
        else:
            print(f"Epoch {epoch:04d} | train_nll={mean_train_nll:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s, best_val_nll={best_val:.4f}", flush=True)

    if best is None:
        best = (jax.device_get(gnn_params), jax.device_get(flow_arrays))
    best_gnn, best_flow_arrays = best
    best_flow = eqx.combine(best_flow_arrays, flow_static)
    ts = time.strftime("%Y%m%d_%H%M%S")
    flow_path = os.path.join(args.output_dir, f"partitioned_flow_seed_{args.seed}_{ts}.eqx")
    model_path = os.path.join(args.output_dir, f"partitioned_model_seed_{args.seed}_{ts}.pkl")
    logs_path = os.path.join(args.output_dir, f"partitioned_logs_seed_{args.seed}_{ts}.pkl")
    eqx.tree_serialise_leaves(flow_path, best_flow)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "gnn_params": best_gnn,
                "flow_path": flow_path,
                "config": vars(args),
                "target_scaler": target_scaler,
                "stats": stats,
                "partition_manifest": args.partition_manifest,
            },
            f,
        )
    with open(logs_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Saved flow: {flow_path}", flush=True)
    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved logs: {logs_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
