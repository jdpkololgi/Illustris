"""
Phase A training: integrated GNN + Flowjax conditional flow on local ego-graphs.

This script is intentionally independent of the existing transductive pipeline.
It consumes the cached global graph and extracts many small k-hop ego-graphs.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jraph
import equinox as eqx

from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal

# Make parent dir importable (so we can reuse shared modules without modifying them)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ILLUSTRIS_DIR = os.path.dirname(THIS_DIR)
if ILLUSTRIS_DIR not in sys.path:
    sys.path.insert(0, ILLUSTRIS_DIR)

from graph_net_models import make_gnn_encoder  # noqa: E402
from eigenvalue_transformations import samples_to_raw_eigenvalues  # noqa: E402
from subgraph_dataset import SubgraphBuilder, PaddedSubgraphBuilder, iter_center_batches  # noqa: E402
from tng_positions import load_tng_positions_via_groupcat, validate_edge_directions  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    k_hops: int
    max_nodes: int
    max_edges: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    use_transformed_eig: bool


def center_embeddings_from_batched_graph(batched: jraph.GraphsTuple) -> jnp.ndarray:
    """
    Extract per-graph center-node embeddings from a batched GraphsTuple.
    We enforce center node is local index 0 in each subgraph, so the center positions
    are the start indices of each graph's node block.
    """
    n_node = batched.n_node  # [B]
    offsets = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(n_node[:-1])])
    return offsets  # start indices of each graph in the batched node array


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    cfg = TrainConfig(
        k_hops=args.k_hops,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_transformed_eig=not args.no_transformed_eig,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase A: Local Subgraph SBI (Integrated GNN + Flowjax)")
    print("=" * 70)
    print(f"JAX Devices: {jax.devices()}")
    print(f"Config: k_hops={cfg.k_hops}, max_nodes={cfg.max_nodes}, max_edges={cfg.max_edges}, batch={cfg.batch_size}")

    # -------------------------------------------------------------------------
    # Load cached global graph + targets
    # -------------------------------------------------------------------------
    print(f"\nLoading cache: {args.cache_path}")
    with open(args.cache_path, "rb") as f:
        data = pickle.load(f)

    global_graph: jraph.GraphsTuple = data["graph"]
    targets = data["regression_targets"]  # [N, 3] (scaled transformed eig by default)
    train_mask, val_mask, test_mask = data["masks"]
    target_scaler = data.get("target_scaler")
    eigenvalues_raw = data.get("eigenvalues_raw")

    n_total = int(global_graph.nodes.shape[0])
    print(f"Global graph: N={n_total}, E={int(global_graph.edges.shape[0])}")

    if args.load_positions:
        print("\n[Optional] Loading TNG positions via groupcat (Utilities/Network_stats)...")
        pos = load_tng_positions_via_groupcat(masscut=args.masscut)
        print(f"Loaded pos shape: {pos.shape}")
        print("[Optional] Validating index alignment using cached edge direction vectors...")
        validate_edge_directions(pos, global_graph, n_checks=1024, seed=cfg.seed, atol=5e-3)
        print("Alignment check passed.")

    train_centers = np.asarray(jnp.where(train_mask)[0])
    val_centers = np.asarray(jnp.where(val_mask)[0])
    test_centers = np.asarray(jnp.where(test_mask)[0])
    print(f"Centers: train={len(train_centers)}, val={len(val_centers)}, test={len(test_centers)}")

    # -------------------------------------------------------------------------
    # Integrated model: GNN encoder + conditional flow
    # -------------------------------------------------------------------------
    gnn_fn = make_gnn_encoder(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    gnn = hk.transform(gnn_fn)

    # Initialize on ONE representative subgraph batch to set shapes.
    init_rng = jax.random.key(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    init_centers = train_centers[: min(cfg.batch_size, len(train_centers))]
    if args.static_shapes:
        builder = PaddedSubgraphBuilder.from_cache_arrays(global_graph, np.asarray(targets))
    else:
        builder = SubgraphBuilder.from_cache_arrays(global_graph, np.asarray(targets))
    init_batched_graph, init_theta = builder.batch(
        init_centers, k_hops=cfg.k_hops, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges
    )

    init_key, flow_key = jax.random.split(init_rng)
    gnn_params = gnn.init(init_key, init_batched_graph, is_training=True)

    base_dist = Normal(jnp.zeros(3), jnp.ones(3))
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=base_dist,
        cond_dim=args.latent_size,
        flow_layers=args.num_flow_layers,
        nn_width=args.flow_hidden_size,
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),
    )

    # Partition flow into arrays/static for JAX-friendly updates
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)

    # Unified optimizer step (applied to both parameter sets each update)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=max(1, cfg.epochs // 20),
        decay_steps=max(2, cfg.epochs - max(1, cfg.epochs // 20)),
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=cfg.weight_decay),
    )

    gnn_opt_state = optimizer.init(gnn_params)
    flow_opt_state = optimizer.init(flow_arrays)

    maybe_jit = (lambda f: f) if args.no_jit else jax.jit

    @maybe_jit
    def loss_and_aux(
        gnn_params_local,
        flow_arrays_local,
        batched_graph: jraph.GraphsTuple,
        theta: jnp.ndarray,  # [B,3]
        rng_key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        flow_model = eqx.combine(flow_arrays_local, flow_static)
        node_emb = gnn.apply(gnn_params_local, rng_key, batched_graph, is_training=True)  # [sumN, D]

        offsets = center_embeddings_from_batched_graph(batched_graph)
        # If using padded graphs, local index 0 is dummy, center is local index 1.
        center_offset = jnp.int32(1 if args.static_shapes else 0)
        cond = node_emb[offsets + center_offset]  # [B, D]

        # NLL = -mean log p(theta | cond)
        log_prob_fn = jax.vmap(flow_model.log_prob)
        logp = log_prob_fn(theta, condition=cond)  # [B]
        nll = -jnp.mean(logp)
        return nll, jnp.mean(logp)

    @maybe_jit
    def eval_loss(
        gnn_params_local,
        flow_arrays_local,
        batched_graph: jraph.GraphsTuple,
        theta: jnp.ndarray,  # [B,3]
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        """Validation loss with dropout OFF for apples-to-apples comparisons."""
        flow_model = eqx.combine(flow_arrays_local, flow_static)
        node_emb = gnn.apply(gnn_params_local, rng_key, batched_graph, is_training=False)  # [sumN, D]

        offsets = center_embeddings_from_batched_graph(batched_graph)
        center_offset = jnp.int32(1 if args.static_shapes else 0)
        cond = node_emb[offsets + center_offset]  # [B, D]

        log_prob_fn = jax.vmap(flow_model.log_prob)
        logp = log_prob_fn(theta, condition=cond)  # [B]
        return -jnp.mean(logp)

    @maybe_jit
    def update_step(
        gnn_params_local,
        gnn_opt_state_local,
        flow_arrays_local,
        flow_opt_state_local,
        batched_graph: jraph.GraphsTuple,
        theta: jnp.ndarray,
        rng_key: jax.Array,
    ):
        (loss, mean_logp), (gnn_grads, flow_grads) = jax.value_and_grad(
            loss_and_aux, argnums=(0, 1), has_aux=True
        )(gnn_params_local, flow_arrays_local, batched_graph, theta, rng_key)

        gnn_updates, new_gnn_opt_state = optimizer.update(gnn_grads, gnn_opt_state_local, gnn_params_local)
        new_gnn_params = optax.apply_updates(gnn_params_local, gnn_updates)

        flow_updates, new_flow_opt_state = optimizer.update(flow_grads, flow_opt_state_local, flow_arrays_local)
        new_flow_arrays = optax.apply_updates(flow_arrays_local, flow_updates)

        return new_gnn_params, new_gnn_opt_state, new_flow_arrays, new_flow_opt_state, loss, mean_logp

    # -------------------------------------------------------------------------
    # Training loop (Phase A)
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    t0 = time.time()
    train_losses = []
    val_losses = []

    step_rng = init_rng

    for epoch in range(cfg.epochs):
        epoch_losses = []

        for batch_idx, batch_centers in enumerate(iter_center_batches(train_centers, cfg.batch_size, rng=np_rng, shuffle=True)):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            batched_graph, theta = builder.batch(
                batch_centers, k_hops=cfg.k_hops, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges
            )

            step_rng, subkey = jax.random.split(step_rng)
            gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, loss, mean_logp = update_step(
                gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, batched_graph, theta, subkey
            )
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_losses.append(train_loss)

        if epoch % max(1, cfg.epochs // 50) == 0 or epoch == cfg.epochs - 1:
            # quick val estimate on a single batch
            val_batch = val_centers[: min(cfg.batch_size, len(val_centers))]
            val_graph, val_theta = builder.batch(
                val_batch, k_hops=cfg.k_hops, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges
            )
            step_rng, subkey = jax.random.split(step_rng)
            vloss = eval_loss(gnn_params, flow_arrays, val_graph, val_theta, subkey)
            val_losses.append((epoch, float(vloss)))
            print(
                f"Epoch {epoch:5d} | Train NLL {train_loss:.4f} | Val NLL {float(vloss):.4f} | Time {time.time()-t0:.1f}s"
            )

    # -------------------------------------------------------------------------
    # Save checkpoint (integrated model parts)
    # -------------------------------------------------------------------------
    if not args.no_save:
        print("\nSaving checkpoint...")
        # Flowjax/equinox models contain JAX functions in their static structure,
        # which are not reliably picklable. Use Equinox serialization for the flow
        # and a small pickle for metadata + Haiku parameters.
        flow_filename = os.path.join(args.output_dir, f"phaseA_local_subgraph_flow_{timestamp}.eqx")
        full_flow = eqx.combine(flow_arrays, flow_static)
        eqx.tree_serialise_leaves(flow_filename, full_flow)
        print(f"Saved flow: {flow_filename}")

        model_filename = os.path.join(args.output_dir, f"phaseA_local_subgraph_model_{timestamp}.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(
                {
                    "gnn_params": jax.device_get(gnn_params),
                    "flow_filename": flow_filename,
                    "config": vars(args),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "target_scaler": target_scaler,
                    "use_transformed_eig": cfg.use_transformed_eig,
                },
                f,
            )
        print(f"Saved model metadata: {model_filename}")

    # -------------------------------------------------------------------------
    # Minimal posterior check on a small test batch (point estimate)
    # -------------------------------------------------------------------------
    if eigenvalues_raw is not None:
        print("\nPosterior sanity check on test batch...")
        test_batch = test_centers[: min(cfg.batch_size, len(test_centers))]
        test_graph, test_theta = builder.batch(
            test_batch, k_hops=cfg.k_hops, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges
        )

        flow_model = eqx.combine(flow_arrays, flow_static)
        emb = gnn.apply(gnn_params, jax.random.key(cfg.seed + 999), test_graph, is_training=False)
        offsets = center_embeddings_from_batched_graph(test_graph)
        center_offset = 1 if args.static_shapes else 0
        cond = emb[offsets + center_offset]

        # One posterior sample per object as a crude point estimate
        sample_keys = jax.random.split(jax.random.key(cfg.seed + 1234), cond.shape[0])
        samples = jax.vmap(lambda k, c: flow_model.sample(k, condition=c))(sample_keys, cond)
        samples_np = np.array(samples)

        raw_preds = samples_to_raw_eigenvalues(samples_np, target_scaler, cfg.use_transformed_eig)
        raw_true = np.asarray(eigenvalues_raw)[np.asarray(test_batch)]

        mse = float(np.mean((raw_preds - raw_true) ** 2))
        print(f"Test batch raw-eigenvalue MSE (1-sample point estimate): {mse:.6e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache_path",
        type=str,
        default="/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl",
    )
    p.add_argument("--output_dir", type=str, default="/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/local_subgraph_phaseA/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.08)

    # Subgraph extraction
    p.add_argument("--k_hops", type=int, default=2)
    p.add_argument("--max_nodes", type=int, default=256)
    p.add_argument("--max_edges", type=int, default=4096)

    # GNN
    p.add_argument("--num_passes", type=int, default=6)
    p.add_argument("--latent_size", type=int, default=80)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)

    # Flow
    p.add_argument("--num_flow_layers", type=int, default=5)
    p.add_argument("--num_bins", type=int, default=8)
    p.add_argument("--flow_hidden_size", type=int, default=128)

    # Target space
    p.add_argument("--no_transformed_eig", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=5, help="Limit batches per epoch for smoke tests/debug.")
    p.add_argument("--no_save", action="store_true", help="Skip writing checkpoints (useful for quick smoke tests).")
    p.add_argument("--no_jit", action="store_true", help="Disable jax.jit to avoid recompiles with variable-shaped subgraphs (slower but robust).")
    p.add_argument("--static_shapes", action="store_true", help="Pad every ego-graph to fixed shapes (prevents XLA recompilation blowups).")
    p.add_argument("--load_positions", action="store_true", help="Load TNG positions via groupcat and validate node ordering.")
    p.add_argument("--masscut", type=float, default=1e9, help="Mass cut to use when loading positions via groupcat.")

    main(p.parse_args())


