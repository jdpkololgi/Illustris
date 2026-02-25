"""
SBI Pipeline with Flowjax for Multi-GPU Training

This pipeline uses:
- Haiku GNN encoder (for graph node embeddings)
- Flowjax normalizing flow (for conditional posterior estimation)

The separation allows proper pmap parallelization across GPUs.

Usage:
    python jraph_sbi_flowjax.py [--epochs 5000] [--seed 42]
"""
import os
import sys
from pathlib import Path

# Force priority for user installed packages
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2"

import time
import pickle
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jraph
import equinox as eqx

from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal

from graph_net_models import make_gnn_encoder
from eigenvalue_transformations import increments_to_eigenvalues, samples_to_raw_eigenvalues
from tng_pipeline_paths import DEFAULT_SBI_OUTPUT_DIR, resolve_sbi_paths
from shared.resource_requirements import require_gpu_slurm


def load_cached_sbi_data(data_path: str):
    """Load cached Jraph regression targets for SBI."""
    print(f"Loading cached Jraph data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    graph = data['graph']
    targets = data['regression_targets']
    train_mask, val_mask, test_mask = data['masks']
    target_scaler = data['target_scaler']
    eigenvalues_raw = data.get('eigenvalues_raw')
    stats = data.get('stats')
    return graph, targets, train_mask, val_mask, test_mask, target_scaler, eigenvalues_raw, stats



def main(args):
    require_gpu_slurm("jraph_sbi_flowjax.py", min_gpus=1)
    print("=" * 70)
    print("SBI Pipeline: GNN + Flowjax (Multi-GPU)")
    print("=" * 70)
    
    # Use new-style PRNG keys (required for Flowjax)
    master_key = jax.random.key(args.seed)
    
    num_devices = len(jax.local_devices())
    print(f"JAX Devices: {jax.devices()}")
    print(f"Running on {num_devices} device(s).")
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Job Timestamp: {timestamp}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/6] Loading data...")
    
    use_transformed_eig = not getattr(args, 'no_transformed_eig', False)
    paths = resolve_sbi_paths(
        use_transformed_eig=use_transformed_eig,
        output_dir=args.output_dir,
    )
    args.output_dir = paths.output_dir
    
    # Select cache path based on transformation flag
    if use_transformed_eig:
        print("[Mode] Using transformed eigenvalues (v₁, Δλ₂, Δλ₃)")
    else:
        print("[Mode] Using raw eigenvalues (λ₁, λ₂, λ₃)")
    graph, targets, train_mask, val_mask, test_mask, target_scaler, eigenvalues_raw, stats = (
        load_cached_sbi_data(paths.data_path)
    )
    
    print(f"Graph stats: Nodes={graph.nodes.shape[0]}, Edges={graph.edges.shape[0]}")
    print(f"Train size: {jnp.sum(train_mask)}, Val size: {jnp.sum(val_mask)}, Test size: {jnp.sum(test_mask)}")
    print(f"Targets shape: {targets.shape}")
    if stats:
        print(f"Scaler mean: {stats.get('scaler_mean', 'N/A')}")
        print(f"Scaler std: {stats.get('scaler_std', 'N/A')}")
    
    # =========================================================================
    # 2. GNN Encoder Setup (Haiku)
    # =========================================================================
    print("\n[2/6] Setting up GNN encoder (Haiku)...")
    
    gnn_fn = make_gnn_encoder(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    gnn = hk.transform(gnn_fn)
    
    # Initialize GNN params
    key, init_key = jax.random.split(master_key)
    gnn_params = gnn.init(init_key, graph, is_training=True)
    
    gnn_param_count = sum(x.size for x in jax.tree_util.tree_leaves(gnn_params))
    print(f"GNN parameters: {gnn_param_count:,}")
    
    # =========================================================================
    # 3. Flow Setup (Flowjax/Equinox)
    # =========================================================================
    print("\n[3/6] Setting up Flow (Flowjax)...")
    
    # Base distribution: standard normal for 3 eigenvalues
    base_dist = Normal(jnp.zeros(3), jnp.ones(3))
    
    # Create conditional masked autoregressive flow
    key, flow_key = jax.random.split(key)
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=base_dist,
        cond_dim=args.latent_size,  # Conditioning on GNN embeddings
        flow_layers=args.num_flow_layers,
        nn_width=args.flow_hidden_size,
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),  # [-12, 12] to cover eigenvalue range
    )
    
    # Count flow parameters
    flow_param_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(flow, eqx.is_inexact_array)))
    print(f"Flow parameters: {flow_param_count:,}")
    print(f"Total parameters: {gnn_param_count + flow_param_count:,}")
    
    # =========================================================================
    # 4. Optimizer Setup
    # =========================================================================
    print("\n[4/6] Setting up optimizer...")
    
    num_epochs = args.epochs
    
    # Learning rate schedule
    warmup_steps = min(500, num_epochs // 10)
    decay_steps = max(num_epochs - warmup_steps, warmup_steps + 1)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    )
    
    # Combined params: {"gnn": gnn_params, "flow": flow}
    # But Flowjax models are Equinox modules, not pytrees of arrays
    # We need separate optimizers or handle them carefully
    
    gnn_opt_state = optimizer.init(gnn_params)
    flow_opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))
    
    # =========================================================================
    # 5. Training Functions
    # =========================================================================
    print("\n[5/6] Setting up parallelization...")
    
    # For Equinox modules, separate arrays from static structure
    # Static parts (functions) are captured via closure, arrays are replicated
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)
    
    # Replicate for pmap
    replicated_gnn_params = jax.device_put_replicated(gnn_params, jax.local_devices())
    replicated_gnn_opt_state = jax.device_put_replicated(gnn_opt_state, jax.local_devices())
    replicated_flow_arrays = jax.device_put_replicated(flow_arrays, jax.local_devices())
    replicated_flow_opt_state = jax.device_put_replicated(flow_opt_state, jax.local_devices())
    
    # Replicate data
    replicated_graph = jax.device_put_replicated(graph, jax.local_devices())
    replicated_targets = jax.device_put_replicated(targets, jax.local_devices())
    
    # Shard masks
    train_indices = jnp.where(train_mask)[0]
    val_indices = jnp.where(val_mask)[0]
    
    train_indices_sharded = jnp.array_split(train_indices, num_devices)
    val_indices_sharded = jnp.array_split(val_indices, num_devices)
    
    sharded_train_masks_list = []
    sharded_val_masks_list = []
    
    for i in range(num_devices):
        m_train = jnp.zeros_like(train_mask)
        if len(train_indices_sharded[i]) > 0:
            m_train = m_train.at[train_indices_sharded[i]].set(True)
        sharded_train_masks_list.append(m_train)
        
        m_val = jnp.zeros_like(val_mask)
        if len(val_indices_sharded[i]) > 0:
            m_val = m_val.at[val_indices_sharded[i]].set(True)
        sharded_val_masks_list.append(m_val)
    
    sharded_train_masks = jnp.stack(sharded_train_masks_list)
    sharded_train_masks = jax.device_put_sharded(list(sharded_train_masks), jax.local_devices())
    
    sharded_val_masks = jnp.stack(sharded_val_masks_list)
    sharded_val_masks = jax.device_put_sharded(list(sharded_val_masks), jax.local_devices())
    
    def compute_loss(gnn_params, flow_arrays, graph, targets, mask, rng, is_training=True):
        """Compute NLL loss. flow_static is captured via closure."""
        # Reconstruct flow from arrays + static
        flow_model = eqx.combine(flow_arrays, flow_static)
        
        # GNN forward pass -> embeddings
        embeddings = gnn.apply(gnn_params, rng, graph, is_training=is_training)
        
        # Flow log_prob (batched via vmap)
        batched_log_prob = jax.vmap(flow_model.log_prob)
        log_probs = batched_log_prob(targets, condition=embeddings)
        
        # Masked loss
        masked_log_probs = log_probs * mask
        num_masked = jnp.sum(mask)
        nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
        
        return nll, (log_probs, num_masked)
    
    def update(gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, graph, targets, mask, rng):
        """Single update step for both GNN and Flow."""
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Compute gradients for both GNN and Flow arrays
        def loss_fn(gnn_p, flow_arr):
            return compute_loss(gnn_p, flow_arr, graph, targets, mask, step_rng)
        
        (loss, (log_probs, num_masked)), (gnn_grads, flow_arr_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(gnn_params, flow_arrays)
        
        # Sync gradients across devices
        gnn_grads = jax.lax.pmean(gnn_grads, axis_name='i')
        flow_arr_grads = jax.lax.pmean(flow_arr_grads, axis_name='i')
        
        # Sync loss
        total_loss_part = loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        # Mean log prob metric
        total_log_prob = jax.lax.psum(jnp.sum(log_probs * mask), axis_name='i')
        mean_log_prob = total_log_prob / jnp.maximum(total_count, 1.0)
        
        # Update GNN params
        gnn_updates, new_gnn_opt_state = optimizer.update(gnn_grads, gnn_opt_state, gnn_params)
        new_gnn_params = optax.apply_updates(gnn_params, gnn_updates)
        
        # Update Flow arrays
        flow_updates, new_flow_opt_state = optimizer.update(flow_arr_grads, flow_opt_state, flow_arrays)
        new_flow_arrays = optax.apply_updates(flow_arrays, flow_updates)
        
        return new_gnn_params, new_gnn_opt_state, new_flow_arrays, new_flow_opt_state, global_loss, mean_log_prob
    
    update_fn = jax.pmap(update, axis_name='i')
    
    def evaluate(gnn_params, flow_arrays, graph, targets, mask, rng):
        """Evaluation step."""
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Reconstruct flow
        flow_model = eqx.combine(flow_arrays, flow_static)
        
        # GNN forward pass
        embeddings = gnn.apply(gnn_params, step_rng, graph, is_training=False)
        
        # Flow log_prob
        batched_log_prob = jax.vmap(flow_model.log_prob)
        log_probs = batched_log_prob(targets, condition=embeddings)
        
        # Masked metrics
        masked_log_probs = log_probs * mask
        num_masked = jnp.sum(mask)
        nll = -jnp.sum(masked_log_probs) / jnp.maximum(num_masked, 1.0)
        
        # Global metrics
        total_loss_part = nll * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        total_log_prob = jax.lax.psum(jnp.sum(masked_log_probs), axis_name='i')
        mean_log_prob = total_log_prob / jnp.maximum(total_count, 1.0)
        
        return global_loss, mean_log_prob
    
    evaluate_fn = jax.pmap(evaluate, axis_name='i')
    
    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    print("\n[6/6] Starting training...")
    print(f"Epochs: {num_epochs}, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"Flow: {args.num_flow_layers} layers, {args.num_bins} bins")
    print("-" * 70)
    
    current_rng = key
    t0 = time.time()
    
    best_val_loss = float('inf')
    best_gnn_params = None
    best_flow_arrays = None
    
    train_losses = []
    val_losses = []
    train_log_probs = []
    val_log_probs = []
    
    report_every = max(1, num_epochs // 100)
    
    for epoch in range(num_epochs):
        current_rng, step_rng = jax.random.split(current_rng)
        step_rngs = jax.device_put_replicated(step_rng, jax.local_devices())
        
        # Training step
        (replicated_gnn_params, replicated_gnn_opt_state, 
         replicated_flow_arrays, replicated_flow_opt_state,
         train_loss, train_log_prob) = update_fn(
            replicated_gnn_params, replicated_gnn_opt_state,
            replicated_flow_arrays, replicated_flow_opt_state,
            replicated_graph, replicated_targets, sharded_train_masks,
            step_rngs
        )
        
        train_losses.append(float(train_loss[0]))
        train_log_probs.append(float(train_log_prob[0]))
        
        # Validation
        if epoch % report_every == 0 or epoch == num_epochs - 1:
            val_loss, val_log_prob = evaluate_fn(
                replicated_gnn_params, replicated_flow_arrays,
                replicated_graph, replicated_targets,
                sharded_val_masks, step_rngs
            )
            
            val_losses.append((epoch, float(val_loss[0])))
            val_log_probs.append((epoch, float(val_log_prob[0])))
            
            if float(val_loss[0]) < best_val_loss:
                best_val_loss = float(val_loss[0])
                best_gnn_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_gnn_params))
                best_flow_arrays = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_flow_arrays))
            
            elapsed = time.time() - t0
            print(f"Epoch {epoch:5d} | Train NLL: {train_loss[0]:.4f} | Val NLL: {val_loss[0]:.4f} | "
                  f"Train LogP: {train_log_prob[0]:.2f} | Val LogP: {val_log_prob[0]:.2f} | "
                  f"Time: {elapsed:.1f}s")
    
    print("-" * 70)
    print(f"Training finished in {time.time() - t0:.2f}s")
    print(f"Best validation NLL: {best_val_loss:.4f}")
    
    # =========================================================================
    # Save Model
    # =========================================================================
    print("\nSaving model...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if best_gnn_params is None:
        best_gnn_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_gnn_params))
        best_flow_arrays = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_flow_arrays))
    
    # Reconstruct the full flow for saving
    best_flow = eqx.combine(best_flow_arrays, flow_static)
    
    # Save flow using equinox serialization (handles JAX functions properly)
    flow_filename = os.path.join(args.output_dir, f'flowjax_sbi_flow_seed_{args.seed}_{timestamp}.eqx')
    eqx.tree_serialise_leaves(flow_filename, best_flow)
    print(f"Flow saved to {flow_filename}")
    
    # Save GNN params and metadata with pickle (no JAX functions)
    model_filename = os.path.join(args.output_dir, f'flowjax_sbi_model_seed_{args.seed}_{timestamp}.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'gnn_params': best_gnn_params,
            'config': vars(args),
            'target_scaler': target_scaler,
            'use_transformed_eig': use_transformed_eig,
            'flow_filename': flow_filename,  # Reference to flow file
        }, f)
    print(f"Model saved to {model_filename}")
    
    # Save logs
    logs_filename = os.path.join(args.output_dir, f'flowjax_sbi_logs_seed_{args.seed}_{timestamp}.pkl')
    with open(logs_filename, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_log_probs': train_log_probs,
            'val_log_probs': val_log_probs,
        }, f)
    print(f"Logs saved to {logs_filename}")
    
    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print("\nEvaluating on test set...")
    
    test_indices = jnp.where(test_mask)[0]
    test_indices_sharded = jnp.array_split(test_indices, num_devices)
    
    sharded_test_masks_list = []
    for i in range(num_devices):
        m_test = jnp.zeros_like(test_mask)
        if len(test_indices_sharded[i]) > 0:
            m_test = m_test.at[test_indices_sharded[i]].set(True)
        sharded_test_masks_list.append(m_test)
    
    sharded_test_masks = jnp.stack(sharded_test_masks_list)
    sharded_test_masks = jax.device_put_sharded(list(sharded_test_masks), jax.local_devices())
    
    # Use best model (replicate the arrays only)
    best_gnn_replicated = jax.device_put_replicated(best_gnn_params, jax.local_devices())
    best_flow_arrays_replicated = jax.device_put_replicated(best_flow_arrays, jax.local_devices())
    
    test_rng = jax.random.key(0)
    test_rngs = jax.device_put_replicated(test_rng, jax.local_devices())
    
    test_loss, test_log_prob = evaluate_fn(
        best_gnn_replicated, best_flow_arrays_replicated,
        replicated_graph, replicated_targets,
        sharded_test_masks, test_rngs
    )
    
    print(f"\nTest Set Results:")
    print(f"  NLL: {float(test_loss[0]):.4f}")
    print(f"  Mean Log Prob: {float(test_log_prob[0]):.2f}")
    
    # =========================================================================
    # Sample from posterior and evaluate in raw eigenvalue space
    # =========================================================================
    print("\nSampling from posterior and evaluating in eigenvalue space...")
    
    # Get embeddings for test nodes using best GNN
    sample_rng = jax.random.key(123)
    test_embeddings = gnn.apply(best_gnn_params, sample_rng, graph, is_training=False)
    
    # Reconstruct flow
    best_flow = eqx.combine(best_flow_arrays, flow_static)
    
    # Sample from flow for each test node (single sample per node for point estimate)
    test_indices_np = np.array(test_indices)
    n_test = len(test_indices_np)
    
    # Get embeddings for test nodes only
    test_embeddings_subset = test_embeddings[test_indices_np]
    
    # Sample one point from posterior per test node
    sample_keys = jax.random.split(sample_rng, n_test)
    
    # Batch sample using vmap
    def sample_one(key, cond):
        return best_flow.sample(key, condition=cond)
    
    posterior_samples = jax.vmap(sample_one)(sample_keys, test_embeddings_subset)
    posterior_samples_np = np.array(posterior_samples)
    
    # Convert samples to raw eigenvalues
    samples_raw_eig = samples_to_raw_eigenvalues(posterior_samples_np, target_scaler, use_transformed_eig)
    
    # Ground truth raw eigenvalues for test set
    test_targets_raw_eig = eigenvalues_raw[test_indices_np]
    
    # Compute R² in raw eigenvalue space
    ss_res = np.sum((test_targets_raw_eig - samples_raw_eig) ** 2, axis=0)
    ss_tot = np.sum((test_targets_raw_eig - np.mean(test_targets_raw_eig, axis=0)) ** 2, axis=0)
    r2_raw = 1 - ss_res / (ss_tot + 1e-8)
    
    # Also compute metrics in scaled/transformed space
    test_targets_scaled = np.array(targets)[test_indices_np]
    ss_res_scaled = np.sum((test_targets_scaled - posterior_samples_np) ** 2, axis=0)
    ss_tot_scaled = np.sum((test_targets_scaled - np.mean(test_targets_scaled, axis=0)) ** 2, axis=0)
    r2_scaled = 1 - ss_res_scaled / (ss_tot_scaled + 1e-8)
    
    print(f"\n  Posterior Point Estimate Metrics:")
    if use_transformed_eig:
        print(f"    Transformed Space (v₁, Δλ₂, Δλ₃):")
        print(f"      R² per param: v₁={r2_scaled[0]:.4f}, Δλ₂={r2_scaled[1]:.4f}, Δλ₃={r2_scaled[2]:.4f}")
        print(f"      Mean R²: {np.mean(r2_scaled):.4f}")
    print(f"    Raw Eigenvalue Space (λ₁, λ₂, λ₃):")
    print(f"      R² per eigenvalue: λ₁={r2_raw[0]:.4f}, λ₂={r2_raw[1]:.4f}, λ₃={r2_raw[2]:.4f}")
    print(f"      Mean R²: {np.mean(r2_raw):.4f}")
    
    # Save results
    results_filename = os.path.join(args.output_dir, f'flowjax_sbi_results_seed_{args.seed}_{timestamp}.txt')
    with open(results_filename, 'w') as f:
        f.write(f"Flowjax SBI Pipeline Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Use Transformed Eigenvalues: {use_transformed_eig}\n")
        f.write(f"\nTest NLL: {float(test_loss[0]):.4f}\n")
        f.write(f"Test Mean Log Prob: {float(test_log_prob[0]):.2f}\n")
        f.write(f"Best Val NLL: {best_val_loss:.4f}\n")
        f.write(f"\nPosterior Point Estimate R² (Raw Eigenvalues):\n")
        f.write(f"  λ₁: {r2_raw[0]:.4f}\n")
        f.write(f"  λ₂: {r2_raw[1]:.4f}\n")
        f.write(f"  λ₃: {r2_raw[2]:.4f}\n")
        f.write(f"  Mean: {np.mean(r2_raw):.4f}\n")
        if use_transformed_eig:
            f.write(f"\nPosterior Point Estimate R² (Transformed Space):\n")
            f.write(f"  v₁: {r2_scaled[0]:.4f}\n")
            f.write(f"  Δλ₂: {r2_scaled[1]:.4f}\n")
            f.write(f"  Δλ₃: {r2_scaled[2]:.4f}\n")
            f.write(f"  Mean: {np.mean(r2_scaled):.4f}\n")
    print(f"Results saved to {results_filename}")
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBI Pipeline with Flowjax')
    
    # Training
    parser.add_argument('--epochs', type=int, default=7000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.08, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_SBI_OUTPUT_DIR, help='Output directory')
    
    # GNN Architecture
    parser.add_argument('--num_passes', type=int, default=8, help='Message passing iterations')
    parser.add_argument('--latent_size', type=int, default=80, help='Latent dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Flow Architecture
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers')
    parser.add_argument('--num_bins', type=int, default=8, help='Spline knots')
    parser.add_argument('--flow_hidden_size', type=int, default=128, help='Flow conditioner hidden size')
    
    # Eigenvalue transformation
    parser.add_argument('--no_transformed_eig', action='store_true',
                        help='Use raw eigenvalues instead of transformed (v₁, Δλ₂, Δλ₃)')
    
    args = parser.parse_args()
    main(args)
