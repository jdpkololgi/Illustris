"""
SBI Pipeline for Cosmic Web Eigenvalue Posterior Estimation

This pipeline uses a Graph Neural Network encoder combined with a conditional
normalizing flow to learn the posterior distribution p(θ | graph) where θ 
are the eigenvalues (λ₁, λ₂, λ₃) for each node in the cosmic web graph.

Based on jraph_pipeline.py but adapted for Simulation-Based Inference.

Usage:
    python jraph_sbi_pipeline.py [--epochs 5000] [--seed 42]
"""
import os
import sys
from pathlib import Path

# Force priority for user installed packages to resolve numpy/scipy/astropy conflicts
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import pickle
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax
import distrax

from graph_net_models import make_graph_network_sbi
from shared.resource_requirements import require_gpu_slurm

# Import data loading utilities from existing pipeline
from jraph_pipeline import load_data, generate_data


#########################################################################
# Loss Functions
#########################################################################

def nll_loss_fn(params, graph, theta_true, mask, net_apply, rng):
    """
    Negative Log-Likelihood loss for SBI.
    
    Args:
        params: Model parameters
        graph: Input graph
        theta_true: True eigenvalues [N, 3]
        mask: Boolean mask indicating which nodes to include in loss
        net_apply: Haiku apply function
        rng: Random key
    
    Returns:
        loss: Mean NLL over masked nodes
        aux: (log_probs, num_masked) for metrics
    """
    # Forward pass: get log probabilities
    log_probs, embeddings = net_apply(params, rng, graph, theta=theta_true, is_training=True)
    
    # Mask the log probs
    masked_log_probs = log_probs * mask
    num_masked = jnp.sum(mask)
    
    # Negative log-likelihood (we want to minimize, so negate log_prob)
    nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
    
    return nll, (log_probs, num_masked)


#########################################################################
# Sampling Functions
#########################################################################

def create_sampling_fn(net_fn, params, graph, output_dim=3, num_flow_layers=5, 
                       num_bins=8, flow_hidden_size=128, range_min=-5.0, range_max=5.0):
    """
    Create a function that samples from the learned posterior.
    
    This recreates the flow architecture to enable sampling.
    """
    latent_size = 80  # Must match model
    params_per_dim = 3 * num_bins + 1
    total_flow_params = output_dim * params_per_dim
    
    def sample_fn(rng, embeddings, num_samples=100):
        """
        Sample from p(θ | embedding) for given embeddings.
        
        Args:
            rng: Random key
            embeddings: Node embeddings [N, latent_size] or [latent_size] for single node
            num_samples: Number of posterior samples per node
        
        Returns:
            samples: [N, num_samples, output_dim] or [num_samples, output_dim]
        """
        single_node = embeddings.ndim == 1
        if single_node:
            embeddings = embeddings[None, :]  # [1, latent_size]
        
        num_nodes = embeddings.shape[0]
        
        # We need to access the flow parameters from the trained model
        # This is a simplified version - in practice you'd want to properly
        # extract the conditioner weights
        
        def sample_single_node(embedding, node_rng):
            """Sample multiple times from one node's posterior."""
            sample_keys = jax.random.split(node_rng, num_samples)
            
            def sample_once(sample_rng):
                # Sample from base distribution
                z = jax.random.normal(sample_rng, shape=(output_dim,))
                
                # Apply forward transforms through all layers
                # Note: This requires the flow parameters which are in 'params'
                # For now, we return z as placeholder
                return z
            
            return jax.vmap(sample_once)(sample_keys)
        
        # Sample for each node
        node_keys = jax.random.split(rng, num_nodes)
        samples = jax.vmap(sample_single_node)(embeddings, node_keys)
        
        if single_node:
            samples = samples[0]  # Remove batch dim
        
        return samples
    
    return sample_fn


def sample_posterior_in_model(params, rng, graph, net_apply, num_samples=100, 
                               output_dim=3, num_flow_layers=5, num_bins=8,
                               flow_hidden_size=128, range_min=-5.0, range_max=5.0,
                               latent_size=80):
    """
    Sample from the learned posterior using the model's forward pass structure.
    
    This function needs to be JIT-compiled with the model for efficiency.
    """
    # Get embeddings
    _, embeddings = net_apply(params, rng, graph, theta=None, is_training=False)
    
    num_nodes = embeddings.shape[0]
    params_per_dim = 3 * num_bins + 1
    
    # We need to recreate the sampling logic here, accessing the flow parameters
    # This is complex because Haiku parameters are structured differently
    
    # For now, return embeddings and we'll implement proper sampling separately
    return embeddings


#########################################################################
# Training Pipeline
#########################################################################

def main(args):
    require_gpu_slurm("jraph_sbi_pipeline.py", min_gpus=1)
    print("=" * 70)
    print("SBI Pipeline: GNN + Conditional Normalizing Flow")
    print("=" * 70)
    print(f"JAX Devices: {jax.devices()}")
    num_devices = jax.local_device_count()
    print(f"Running on {num_devices} device(s).")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Job Timestamp: {timestamp}")
    
    # =====================================================================
    # 1. Load Data
    # =====================================================================
    print("\n[1/5] Loading data...")
    masscut = 1e9
    graph, targets, eigenvalue_scaler, masks = load_data(
        masscut=masscut, use_v2=True, prediction_mode='regression'
    )
    train_mask, val_mask, test_mask = masks
    
    print(f"Graph stats: Nodes={graph.n_node[0]}, Edges={graph.n_edge[0]}")
    print(f"Train size: {jnp.sum(train_mask)}, Val size: {jnp.sum(val_mask)}, Test size: {jnp.sum(test_mask)}")
    print(f"Targets shape: {targets.shape}")
    
    # Check for NaNs
    print(f"Checking for NaNs...")
    n_nans_nodes = jnp.sum(jnp.isnan(graph.nodes))
    n_nans_edges = jnp.sum(jnp.isnan(graph.edges))
    n_nans_targets = jnp.sum(jnp.isnan(targets))
    print(f"NaNs in graph nodes: {n_nans_nodes}")
    print(f"NaNs in graph edges: {n_nans_edges}")
    print(f"NaNs in targets: {n_nans_targets}")
    
    if n_nans_nodes > 0 or n_nans_edges > 0 or n_nans_targets > 0:
        print("CRITICAL: Data contains NaNs! Exiting.")
        return
    
    # =====================================================================
    # 2. Model Setup
    # =====================================================================
    print("\n[2/5] Setting up model...")
    
    net_fn = make_graph_network_sbi(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        output_dim=3,  # 3 eigenvalues
        num_flow_layers=args.num_flow_layers,
        num_bins=args.num_bins,
        flow_hidden_size=args.flow_hidden_size,
        range_min=args.range_min,
        range_max=args.range_max,
    )
    net = hk.transform(net_fn)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(args.seed)
    init_rng, train_rng = jax.random.split(rng)
    
    # Initialize parameters with a tiny dummy graph to avoid tracing a huge graph
    # Haiku only needs the feature shapes to initialize the weights.
    dummy_node_feats = jnp.zeros((1, graph.nodes.shape[1]))
    dummy_edge_feats = jnp.zeros((1, graph.edges.shape[1]))
    dummy_graph = jraph.GraphsTuple(
        nodes=dummy_node_feats,
        edges=dummy_edge_feats,
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([0], dtype=jnp.int32),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
        globals=None
    )
    dummy_theta = jnp.zeros((1, 3))
    params = net.init(init_rng, dummy_graph, theta=dummy_theta, is_training=True)
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {num_params:,}")
    
    # =====================================================================
    # 3. Optimizer Setup
    # =====================================================================
    print("\n[3/5] Setting up optimizer...")
    
    num_epochs = args.epochs
    warmup_steps = min(500, num_epochs // 2)  # Adjust warmup for short runs
    decay_steps = max(num_epochs, warmup_steps + 10)  # Ensure positive decay
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    )
    opt_state = optimizer.init(params)
    # =====================================================================
    # 4. Training Setup (Single-GPU or Multi-GPU)
    # =====================================================================
    num_devices = jax.local_device_count()
    use_multi_gpu = args.multi_gpu and num_devices > 1
    
    if use_multi_gpu:
        print(f"\n[4/5] Setting up multi-GPU (pmap) training on {num_devices} devices...")
        
        # Replicate parameters and optimizer state across devices
        replicated_params = jax.device_put_replicated(params, jax.local_devices())
        replicated_opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
        replicated_graph = jax.device_put_replicated(graph, jax.local_devices())
        replicated_targets = jax.device_put_replicated(targets, jax.local_devices())
        
        # Shard masks across devices
        train_indices = jnp.where(train_mask)[0]
        val_indices = jnp.where(val_mask)[0]
        train_shards = jnp.array_split(train_indices, num_devices)
        val_shards = jnp.array_split(val_indices, num_devices)
        sharded_train_masks = []
        sharded_val_masks = []
        for i in range(num_devices):
            m_train = jnp.zeros_like(train_mask)
            if train_shards[i].size > 0:
                m_train = m_train.at[train_shards[i]].set(True)
            sharded_train_masks.append(m_train)
            m_val = jnp.zeros_like(val_mask)
            if val_shards[i].size > 0:
                m_val = m_val.at[val_shards[i]].set(True)
            sharded_val_masks.append(m_val)
        sharded_train_masks = jax.device_put_sharded(sharded_train_masks, jax.local_devices())
        sharded_val_masks = jax.device_put_sharded(sharded_val_masks, jax.local_devices())
        
        # Loss function for pmap
        def compute_loss(params, graph, targets, mask, rng):
            mask_f = mask.astype(jnp.float32)
            log_probs, _ = net.apply(params, rng, graph, theta=targets, is_training=True)
            masked_log_probs = log_probs * mask_f
            num_masked = jnp.sum(mask_f)
            nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
            return nll, (log_probs, num_masked)

        def update_step(params, opt_state, graph, targets, mask, rng):
            step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
            (loss, (log_probs, num_masked)), grads = jax.value_and_grad(
                compute_loss, has_aux=True
            )(params, graph, targets, mask, step_rng)
            grads = jax.lax.pmean(grads, axis_name='i')
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            total_loss = jax.lax.psum(loss * num_masked, axis_name='i')
            total_count = jax.lax.psum(num_masked, axis_name='i')
            global_loss = total_loss / jnp.maximum(total_count, 1.0)
            total_log_prob = jax.lax.psum(jnp.sum(log_probs * mask.astype(jnp.float32)), axis_name='i')
            mean_log_prob = total_log_prob / jnp.maximum(total_count, 1.0)
            return new_params, new_opt_state, global_loss, mean_log_prob

        update_fn = jax.pmap(update_step, axis_name='i')

        def eval_step(params, graph, targets, mask, rng):
            step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
            log_probs, _ = net.apply(params, step_rng, graph, theta=targets, is_training=False)
            mask_f = mask.astype(jnp.float32)
            masked_log_probs = log_probs * mask_f
            num_masked = jnp.sum(mask_f)
            nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
            total_loss = jax.lax.psum(nll * num_masked, axis_name='i')
            total_count = jax.lax.psum(num_masked, axis_name='i')
            global_loss = total_loss / jnp.maximum(total_count, 1.0)
            total_log_prob = jax.lax.psum(jnp.sum(masked_log_probs), axis_name='i')
            mean_log_prob = total_log_prob / jnp.maximum(total_count, 1.0)
            return global_loss, mean_log_prob

        evaluate_fn = jax.pmap(eval_step, axis_name='i')
    else:
        print(f"\n[4/5] Setting up single-GPU training...")
        
        @jax.jit
        def update(params, opt_state, graph, targets, mask, rng):
            (train_loss, (log_probs, num_masked)), grads = jax.value_and_grad(
                nll_loss_fn, has_aux=True
            )(params, graph, targets, mask, net.apply, rng)
            mean_log_prob = jnp.sum(log_probs * mask) / jnp.maximum(num_masked, 1.0)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, train_loss, mean_log_prob
        
        @jax.jit
        def evaluate(params, graph, targets, mask, rng):
            log_probs, _ = net.apply(params, rng, graph, theta=targets, is_training=False)
            masked_log_probs = log_probs * mask
            num_masked = jnp.sum(mask)
            nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
            mean_log_prob = jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
            return nll, mean_log_prob

    # =====================================================================
    # 5. Training Loop
    # =====================================================================
    print("\n[5/5] Starting training...")
    print(f"Epochs: {num_epochs}, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"Flow: {args.num_flow_layers} layers, {args.num_bins} bins")
    print(f"Mode: {'Multi-GPU (pmap)' if use_multi_gpu else 'Single-GPU (jit)'}")
    print("-" * 70)

    current_rng = train_rng
    t0 = time.time()
    best_val_loss = float('inf')
    best_params = None
    train_losses = []
    val_losses = []
    train_log_probs = []
    val_log_probs = []
    report_every = max(1, num_epochs // 100)
    
    for epoch in range(num_epochs):
        current_rng, step_rng = jax.random.split(current_rng)
        
        if use_multi_gpu:
            step_rngs = jax.device_put_replicated(step_rng, jax.local_devices())
            replicated_params, replicated_opt_state, train_loss, train_log_prob = update_fn(
                replicated_params, replicated_opt_state,
                replicated_graph, replicated_targets, sharded_train_masks,
                step_rngs
            )
            train_losses.append(float(train_loss[0]))
            train_log_probs.append(float(train_log_prob[0]))
            
            if epoch % report_every == 0 or epoch == num_epochs - 1:
                val_loss, val_log_prob = evaluate_fn(
                    replicated_params, replicated_graph, replicated_targets,
                    sharded_val_masks, step_rngs
                )
                val_losses.append((epoch, float(val_loss[0])))
                val_log_probs.append((epoch, float(val_log_prob[0])))
                if float(val_loss[0]) < best_val_loss:
                    best_val_loss = float(val_loss[0])
                    best_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_params))
                elapsed = time.time() - t0
                print(f"Epoch {epoch:5d} | Train NLL: {train_loss[0]:.4f} | Val NLL: {val_loss[0]:.4f} | "
                      f"Train LogP: {train_log_prob[0]:.2f} | Val LogP: {val_log_prob[0]:.2f} | "
                      f"Time: {elapsed:.1f}s")
        else:
            params, opt_state, train_loss, train_log_prob = update(
                params, opt_state, graph, targets, train_mask, step_rng
            )
            train_losses.append(float(train_loss))
            train_log_probs.append(float(train_log_prob))
            
            if epoch % report_every == 0 or epoch == num_epochs - 1:
                val_loss, val_log_prob = evaluate(
                    params, graph, targets, val_mask, step_rng
                )
                val_losses.append((epoch, float(val_loss)))
                val_log_probs.append((epoch, float(val_log_prob)))
                if float(val_loss) < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_params = jax.device_get(params)
                elapsed = time.time() - t0
                print(f"Epoch {epoch:5d} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f} | "
                      f"Train LogP: {train_log_prob:.2f} | Val LogP: {val_log_prob:.2f} | "
                      f"Time: {elapsed:.1f}s")
    
    print("-" * 70)
    print(f"Training finished in {time.time() - t0:.2f}s")
    print(f"Best validation NLL: {best_val_loss:.4f}")

    
    # =====================================================================
    # Save Model
    # =====================================================================
    print("\nSaving model...")
    
    # Use best params
    if best_params is None:
        best_params = jax.device_get(params)
    
    model_filename = os.path.join(args.output_dir, f'jraph_sbi_model_seed_{args.seed}_{timestamp}.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'config': {
                'num_passes': args.num_passes,
                'latent_size': args.latent_size,
                'num_heads': args.num_heads,
                'dropout': args.dropout,
                'num_flow_layers': args.num_flow_layers,
                'num_bins': args.num_bins,
                'flow_hidden_size': args.flow_hidden_size,
                'range_min': args.range_min,
                'range_max': args.range_max,
            },
            'eigenvalue_scaler': eigenvalue_scaler,
        }, f)
    print(f"Model saved to {model_filename}")
    
    # Save training logs
    logs_filename = os.path.join(args.output_dir, f'jraph_sbi_logs_seed_{args.seed}_{timestamp}.pkl')
    with open(logs_filename, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_log_probs': train_log_probs,
            'val_log_probs': val_log_probs,
        }, f)
    print(f"Logs saved to {logs_filename}")
    
    # =====================================================================
    # Evaluation on Test Set
    # =====================================================================
    print("\nEvaluating on test set...")
    
    # Single-device evaluation
    eval_rng = jax.random.PRNGKey(args.seed + 999)
    
    @jax.jit
    def compute_test_metrics(params, graph, targets, mask, rng):
        log_probs, embeddings = net.apply(params, rng, graph, theta=targets, is_training=False)
        masked_log_probs = log_probs * mask
        nll = -jnp.sum(masked_log_probs) / jnp.maximum(jnp.sum(mask), 1.0)
        mean_log_prob = jnp.sum(masked_log_probs) / jnp.maximum(jnp.sum(mask), 1.0)
        return nll, mean_log_prob, embeddings
    
    test_nll, test_log_prob, embeddings = compute_test_metrics(
        best_params, graph, targets, test_mask, eval_rng
    )
    
    print(f"\nTest Set Results:")
    print(f"  NLL: {float(test_nll):.4f}")
    print(f"  Mean Log Prob: {float(test_log_prob):.2f}")
    
    # Save test results
    results_filename = os.path.join(args.output_dir, f'jraph_sbi_results_seed_{args.seed}_{timestamp}.txt')
    with open(results_filename, 'w') as f:
        f.write(f"SBI Pipeline Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\nModel Configuration:\n")
        f.write(f"  GNN Passes: {args.num_passes}\n")
        f.write(f"  Latent Size: {args.latent_size}\n")
        f.write(f"  Attention Heads: {args.num_heads}\n")
        f.write(f"  Flow Layers: {args.num_flow_layers}\n")
        f.write(f"  Spline Bins: {args.num_bins}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  NLL: {float(test_nll):.4f}\n")
        f.write(f"  Mean Log Prob: {float(test_log_prob):.2f}\n")
        f.write(f"\nTraining:\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Best Val NLL: {best_val_loss:.4f}\n")
    print(f"Results saved to {results_filename}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBI Pipeline for Cosmic Web Eigenvalues")
    
    # Training
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.08, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/sbi/', help='Output directory')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multi-GPU training (pmap). Default is single-GPU which is faster per-epoch.')
    
    # GNN Architecture
    parser.add_argument('--num_passes', type=int, default=8, help='Message passing iterations')
    parser.add_argument('--latent_size', type=int, default=80, help='Latent dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Flow Architecture
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers')
    parser.add_argument('--num_bins', type=int, default=8, help='Spline bins')
    parser.add_argument('--flow_hidden_size', type=int, default=128, help='Flow conditioner hidden size')
    parser.add_argument('--range_min', type=float, default=-7.0, help='Spline range minimum')
    parser.add_argument('--range_max', type=float, default=13.0, help='Spline range maximum')
    
    args = parser.parse_args()
    main(args)
