"""
Evaluate a trained local-subgraph Flowjax SBI model on a subset of test nodes.

This is used in Phase A to compare against the existing transductive baseline
(e.g. `TNG_Illustris_outputs/sbi/flowjax_sbi_results_*.txt`).
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal

# Make parent dir importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ILLUSTRIS_DIR = os.path.dirname(THIS_DIR)
if ILLUSTRIS_DIR not in sys.path:
    sys.path.insert(0, ILLUSTRIS_DIR)

from graph_net_models import make_gnn_encoder  # noqa: E402
from eigenvalue_transformations import samples_to_raw_eigenvalues  # noqa: E402
from subgraph_dataset import PaddedSubgraphBuilder  # noqa: E402


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dimension R²."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main(args: argparse.Namespace) -> None:
    # CPU is safest on Perlmutter due to occasional CuDNN mismatch issues.
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    with open(args.model_pkl, "rb") as f:
        ckpt = pickle.load(f)

    config = ckpt["config"]
    gnn_params = ckpt["gnn_params"]
    flow_filename = ckpt["flow_filename"]
    target_scaler = ckpt["target_scaler"]
    use_transformed_eig = bool(ckpt.get("use_transformed_eig", True))

    # Load cache
    with open(args.cache_path, "rb") as f:
        data = pickle.load(f)
    graph = data["graph"]
    targets = np.asarray(data["regression_targets"])
    eigenvalues_raw = np.asarray(data["eigenvalues_raw"])
    train_mask, val_mask, test_mask = data["masks"]

    test_centers = np.asarray(jnp.where(test_mask)[0]).copy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(test_centers)
    test_centers = test_centers[: args.num_test]

    print(f"Evaluating {len(test_centers)} test nodes from cache {args.cache_path}")

    # Rebuild model
    gnn_fn = make_gnn_encoder(
        num_passes=int(config.get("num_passes", 2)),
        latent_size=int(config.get("latent_size", 80)),
        num_heads=int(config.get("num_heads", 8)),
        dropout_rate=float(config.get("dropout", 0.2)),
    )
    gnn = hk.transform(gnn_fn)

    # Flowjax needs a template ("like") to deserialize into.
    base_dist = Normal(jnp.zeros(3), jnp.ones(3))
    flow_key = jax.random.key(0)
    flow_template = masked_autoregressive_flow(
        flow_key,
        base_dist=base_dist,
        cond_dim=int(config.get("latent_size", 80)),
        flow_layers=int(config.get("num_flow_layers", 5)),
        nn_width=int(config.get("flow_hidden_size", 128)),
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=int(config.get("num_bins", 8)), interval=12),
    )
    flow = eqx.tree_deserialise_leaves(flow_filename, like=flow_template)

    builder = PaddedSubgraphBuilder.from_cache_arrays(graph, targets)

    @jax.jit
    def forward_batch(batched_graph, theta, key):
        emb = gnn.apply(gnn_params, key, batched_graph, is_training=False)
        # padded graphs => dummy at 0, center at 1
        n_node = batched_graph.n_node
        offsets = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(n_node[:-1])])
        cond = emb[offsets + 1]
        logp = jax.vmap(flow.log_prob)(theta, condition=cond)
        return logp, cond

    all_logp = []
    all_samples = []
    all_true_raw = []
    all_true_trans = []

    for i in range(0, len(test_centers), args.batch_size):
        batch_centers = test_centers[i : i + args.batch_size]
        batched_graph, theta = builder.batch(
            batch_centers, k_hops=args.k_hops, max_nodes=args.max_nodes, max_edges=args.max_edges
        )

        key = jax.random.key(args.seed + i)
        logp, cond = forward_batch(batched_graph, theta, key)
        all_logp.append(np.array(logp))

        # Posterior mean from multiple samples (more stable than 1 sample)
        base_key = jax.random.key(args.seed + 1000 + i)
        # JAX new-style keys: jax.random.split returns shape (N,) of scalar keys.
        keys = jax.random.split(base_key, cond.shape[0] * args.num_samples).reshape(cond.shape[0], args.num_samples)

        def sample_k(keys_k, c):
            return jax.vmap(lambda kk: flow.sample(kk, condition=c))(keys_k)

        samples = jax.vmap(sample_k)(keys, cond)  # [B, K, 3]
        all_samples.append(np.array(samples))

        all_true_trans.append(np.array(theta))
        all_true_raw.append(eigenvalues_raw[np.asarray(batch_centers)])

    logp = np.concatenate(all_logp, axis=0)
    samples = np.concatenate(all_samples, axis=0)  # [N, K, 3]
    true_trans = np.concatenate(all_true_trans, axis=0)
    true_raw = np.concatenate(all_true_raw, axis=0)

    # Convert all samples to raw eigenvalues and average over samples for point estimate
    pred_raw_all = samples_to_raw_eigenvalues(samples, target_scaler, use_transformed_eig)  # [N, K, 3]
    pred_raw = np.mean(pred_raw_all, axis=1)  # [N, 3]
    pred_trans = np.mean(samples, axis=1)  # [N, 3]

    mse_raw = float(np.mean((pred_raw - true_raw) ** 2))
    r2_raw = r2_score(true_raw, pred_raw)

    # Transformed-space R² (use posterior mean in target space)
    r2_trans = r2_score(true_trans, pred_trans)

    nll = float(-np.mean(logp))
    print("\n=== Local-subgraph eval (subset) ===")
    print(f"NLL: {nll:.4f}")
    print(f"Raw-eigen MSE: {mse_raw:.6e}")
    print(f"R² raw:  λ1={r2_raw[0]:.4f}, λ2={r2_raw[1]:.4f}, λ3={r2_raw[2]:.4f}, mean={float(np.mean(r2_raw)):.4f}")
    print(
        f"R² target-space: t0={r2_trans[0]:.4f}, t1={r2_trans[1]:.4f}, t2={r2_trans[2]:.4f}, mean={float(np.mean(r2_trans)):.4f}"
    )
    print(f"Posterior samples per node: {args.num_samples}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_pkl", type=str, required=True, help="Path to phaseA_local_subgraph_model_*.pkl")
    p.add_argument(
        "--cache_path",
        type=str,
        default="/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    # Subgraph shape (must match training static shapes)
    p.add_argument("--k_hops", type=int, default=1)
    p.add_argument("--max_nodes", type=int, default=64)
    p.add_argument("--max_edges", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)

    # Eval size
    p.add_argument("--num_test", type=int, default=1024)
    p.add_argument("--num_samples", type=int, default=32, help="Posterior samples per node for point estimate (mean).")
    main(p.parse_args())


