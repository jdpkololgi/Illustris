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

# Avoid accidental user-site contamination (common on HPC).
# In particular, a Python 3.10 user-site can break a Python 3.11 env (NumPy/JAX ABI mismatch).
os.environ.setdefault("PYTHONNOUSERSITE", "1")
_bad_user_sites = (
    "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages",
    "/global/homes/d/dkololgi/.local/lib/python3.11/site-packages",
    "/global/u2/d/dkololgi/.local/lib/python3.10/site-packages",
    "/global/u2/d/dkololgi/.local/lib/python3.11/site-packages",
)
for _p in _bad_user_sites:
    while _p in sys.path:
        sys.path.remove(_p)

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

from shared.graph_net_models import make_gnn_encoder
from shared.eigenvalue_transformations import increments_to_eigenvalues, samples_to_raw_eigenvalues
from shared.tng_pipeline_paths import DEFAULT_SBI_OUTPUT_DIR, resolve_sbi_paths
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


# =============================================================================
# Checkpoint / resume helpers
# -----------------------------------------------------------------------------
# jraph_sbi_flowjax.py originally saved only the best model at the very end, so a
# job killed mid-training (e.g. a 4 h interactive cap) lost everything. These
# helpers write a *resumable* checkpoint (host/unreplicated arrays) atomically so
# training can continue in a fresh allocation, mirroring the jraph regression
# pipeline's periodic checkpointing.
# =============================================================================

def _unreplicate(tree):
    """Pull a pmap-replicated pytree (leading device axis) back to host arrays."""
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], tree))


def _serialise_rng(rng):
    """Serialise a typed PRNG key to a plain uint32 numpy array."""
    return np.asarray(jax.random.key_data(rng))


def _restore_rng(rng_data, seed):
    """Inverse of _serialise_rng; falls back to a fresh key on any incompatibility."""
    try:
        return jax.random.wrap_key_data(jnp.asarray(rng_data))
    except Exception:
        return jax.random.key(seed)


def save_checkpoint(path, *, epoch, gnn_params, gnn_opt_state, flow_arrays,
                    flow_opt_state, rng, best_val_loss, best_gnn_params,
                    best_flow_arrays, logs):
    """Atomically write a resumable training checkpoint.

    All array args must already be host/unreplicated (use _unreplicate for the
    pmap-replicated training state; best_* are already unreplicated). Writing to
    a .tmp sibling then os.replace makes the swap atomic, so a kill mid-write
    cannot corrupt an existing good checkpoint.
    """
    payload = {
        'epoch': int(epoch),
        'gnn_params': jax.device_get(gnn_params),
        'gnn_opt_state': jax.device_get(gnn_opt_state),
        'flow_arrays': jax.device_get(flow_arrays),
        'flow_opt_state': jax.device_get(flow_opt_state),
        'rng': _serialise_rng(rng),
        'best_val_loss': float(best_val_loss),
        'best_gnn_params': jax.device_get(best_gnn_params) if best_gnn_params is not None else None,
        'best_flow_arrays': jax.device_get(best_flow_arrays) if best_flow_arrays is not None else None,
        'logs': logs,
    }
    tmp = f"{path}.tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Load a checkpoint written by save_checkpoint (host arrays)."""
    with open(path, 'rb') as f:
        return pickle.load(f)


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
    
    # Resolve the target parameterisation: explicit --increment_mode wins,
    # else fall back to the legacy --no_transformed_eig bool.
    if getattr(args, 'increment_mode', None):
        increment_mode = args.increment_mode
    else:
        increment_mode = 'raw' if getattr(args, 'no_transformed_eig', False) else 'softplus'
    # 'use_transformed_eig' here means "targets live in an increment space" (softplus
    # OR linear) vs raw eigenvalues — controls the increment-space metric prints.
    use_transformed_eig = (increment_mode != 'raw')
    paths = resolve_sbi_paths(
        use_transformed_eig=increment_mode,
        output_dir=args.output_dir,
    )
    args.output_dir = paths.output_dir

    _mode_label = {
        'softplus': "[Mode] softplus increments (v₁, Δλ₂, Δλ₃) — ordering enforced",
        'linear':   "[Mode] linear increments (v₁, λ₂-λ₁, λ₃-λ₂) — ordering NOT enforced",
        'raw':      "[Mode] raw eigenvalues (λ₁, λ₂, λ₃) — ordering NOT enforced",
    }[increment_mode]
    print(_mode_label)
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

    # ---- checkpoint / resume setup --------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f'flowjax_sbi_checkpoint_seed_{args.seed}.pkl')
    start_epoch = 0
    resume_path = args.resume_from or (ckpt_path if getattr(args, 'resume', False) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"[resume] loading checkpoint: {resume_path}")
        ck = load_checkpoint(resume_path)
        start_epoch = int(ck['epoch']) + 1
        replicated_gnn_params = jax.device_put_replicated(ck['gnn_params'], jax.local_devices())
        replicated_gnn_opt_state = jax.device_put_replicated(ck['gnn_opt_state'], jax.local_devices())
        replicated_flow_arrays = jax.device_put_replicated(ck['flow_arrays'], jax.local_devices())
        replicated_flow_opt_state = jax.device_put_replicated(ck['flow_opt_state'], jax.local_devices())
        current_rng = _restore_rng(ck['rng'], args.seed)
        best_val_loss = ck['best_val_loss']
        best_gnn_params = ck['best_gnn_params']
        best_flow_arrays = ck['best_flow_arrays']
        _logs = ck.get('logs') or {}
        train_losses = _logs.get('train_losses', train_losses)
        val_losses = _logs.get('val_losses', val_losses)
        train_log_probs = _logs.get('train_log_probs', train_log_probs)
        val_log_probs = _logs.get('val_log_probs', val_log_probs)
        if start_epoch >= num_epochs:
            print(f"[resume] checkpoint epoch {ck['epoch']} already at/after target {num_epochs}; nothing to train.")
        else:
            print(f"[resume] continuing from epoch {start_epoch}/{num_epochs} "
                  f"(best Val NLL so far: {best_val_loss:.4f})")
    elif resume_path:
        print(f"[resume] no checkpoint at {resume_path}; starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
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

        # Periodic resumable checkpoint (atomic). Lets a fresh allocation resume
        # with --resume if the job is killed (e.g. 4 h interactive cap).
        if args.checkpoint_every and (
            (epoch + 1) % args.checkpoint_every == 0 or epoch == num_epochs - 1
        ):
            save_checkpoint(
                ckpt_path,
                epoch=epoch,
                gnn_params=_unreplicate(replicated_gnn_params),
                gnn_opt_state=_unreplicate(replicated_gnn_opt_state),
                flow_arrays=_unreplicate(replicated_flow_arrays),
                flow_opt_state=_unreplicate(replicated_flow_opt_state),
                rng=current_rng,
                best_val_loss=best_val_loss,
                best_gnn_params=best_gnn_params,
                best_flow_arrays=best_flow_arrays,
                logs={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_log_probs': train_log_probs,
                    'val_log_probs': val_log_probs,
                },
            )
            print(f"[checkpoint] epoch {epoch} -> {ckpt_path}")

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
            'increment_mode': increment_mode,  # 'softplus' | 'linear' | 'raw'
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
    
    # Sample from flow for each test node.
    # We report:
    # 1) single-sample point estimate metrics (legacy behavior)
    # 2) posterior-mean metrics from K samples/node (more stable diagnostic)
    test_indices_np = np.array(test_indices)
    n_test = len(test_indices_np)
    
    # Get embeddings for test nodes only
    test_embeddings_subset = test_embeddings[test_indices_np]
    
    n_post_samples = max(1, int(args.test_posterior_samples))
    eval_chunk_size = max(1, int(args.test_eval_chunk_size))
    print(f"  Test posterior evaluation: samples_per_node={n_post_samples}, chunk_size={eval_chunk_size}")

    point_est_chunks = []
    mean_est_chunks = []
    for start in range(0, n_test, eval_chunk_size):
        end = min(start + eval_chunk_size, n_test)
        emb_chunk = test_embeddings_subset[start:end]
        sample_rng, chunk_key = jax.random.split(sample_rng)
        node_keys = jax.random.split(chunk_key, end - start)

        # [B, K, 3]
        samples_chunk = jax.vmap(
            lambda k, cond: best_flow.sample(k, (n_post_samples,), condition=cond)
        )(node_keys, emb_chunk)

        # Legacy single-sample estimate: first sample
        point_est_chunks.append(np.asarray(samples_chunk[:, 0, :]))
        # Posterior mean estimate
        mean_est_chunks.append(np.asarray(jnp.mean(samples_chunk, axis=1)))

    posterior_point_np = np.concatenate(point_est_chunks, axis=0)
    posterior_mean_np = np.concatenate(mean_est_chunks, axis=0)
    
    # Convert samples to raw eigenvalues
    samples_raw_eig_point = samples_to_raw_eigenvalues(posterior_point_np, target_scaler, increment_mode)
    samples_raw_eig_mean = samples_to_raw_eigenvalues(posterior_mean_np, target_scaler, increment_mode)
    
    # Ground truth raw eigenvalues for test set
    test_targets_raw_eig = eigenvalues_raw[test_indices_np]
    
    # Compute R² in raw eigenvalue space
    ss_res_point = np.sum((test_targets_raw_eig - samples_raw_eig_point) ** 2, axis=0)
    ss_res_mean = np.sum((test_targets_raw_eig - samples_raw_eig_mean) ** 2, axis=0)
    ss_tot = np.sum((test_targets_raw_eig - np.mean(test_targets_raw_eig, axis=0)) ** 2, axis=0)
    r2_raw_point = 1 - ss_res_point / (ss_tot + 1e-8)
    r2_raw_mean = 1 - ss_res_mean / (ss_tot + 1e-8)
    
    # Also compute metrics in scaled/transformed space
    test_targets_scaled = np.array(targets)[test_indices_np]
    ss_res_scaled_point = np.sum((test_targets_scaled - posterior_point_np) ** 2, axis=0)
    ss_res_scaled_mean = np.sum((test_targets_scaled - posterior_mean_np) ** 2, axis=0)
    ss_tot_scaled = np.sum((test_targets_scaled - np.mean(test_targets_scaled, axis=0)) ** 2, axis=0)
    r2_scaled_point = 1 - ss_res_scaled_point / (ss_tot_scaled + 1e-8)
    r2_scaled_mean = 1 - ss_res_scaled_mean / (ss_tot_scaled + 1e-8)
    
    print(f"\n  Posterior Point Estimate Metrics (single sample):")
    if use_transformed_eig:
        print(f"    Transformed Space (v₁, Δλ₂, Δλ₃):")
        print(f"      R² per param: v₁={r2_scaled_point[0]:.4f}, Δλ₂={r2_scaled_point[1]:.4f}, Δλ₃={r2_scaled_point[2]:.4f}")
        print(f"      Mean R²: {np.mean(r2_scaled_point):.4f}")
    print(f"    Raw Eigenvalue Space (λ₁, λ₂, λ₃):")
    print(f"      R² per eigenvalue: λ₁={r2_raw_point[0]:.4f}, λ₂={r2_raw_point[1]:.4f}, λ₃={r2_raw_point[2]:.4f}")
    print(f"      Mean R²: {np.mean(r2_raw_point):.4f}")

    print(f"\n  Posterior Mean Metrics ({n_post_samples} samples/node):")
    if use_transformed_eig:
        print(f"    Transformed Space (v₁, Δλ₂, Δλ₃):")
        print(f"      R² per param: v₁={r2_scaled_mean[0]:.4f}, Δλ₂={r2_scaled_mean[1]:.4f}, Δλ₃={r2_scaled_mean[2]:.4f}")
        print(f"      Mean R²: {np.mean(r2_scaled_mean):.4f}")
    print(f"    Raw Eigenvalue Space (λ₁, λ₂, λ₃):")
    print(f"      R² per eigenvalue: λ₁={r2_raw_mean[0]:.4f}, λ₂={r2_raw_mean[1]:.4f}, λ₃={r2_raw_mean[2]:.4f}")
    print(f"      Mean R²: {np.mean(r2_raw_mean):.4f}")
    
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
        f.write(f"\nPosterior Point Estimate R² (single sample, Raw Eigenvalues):\n")
        f.write(f"  λ₁: {r2_raw_point[0]:.4f}\n")
        f.write(f"  λ₂: {r2_raw_point[1]:.4f}\n")
        f.write(f"  λ₃: {r2_raw_point[2]:.4f}\n")
        f.write(f"  Mean: {np.mean(r2_raw_point):.4f}\n")
        f.write(f"\nPosterior Mean R² ({n_post_samples} samples/node, Raw Eigenvalues):\n")
        f.write(f"  λ₁: {r2_raw_mean[0]:.4f}\n")
        f.write(f"  λ₂: {r2_raw_mean[1]:.4f}\n")
        f.write(f"  λ₃: {r2_raw_mean[2]:.4f}\n")
        f.write(f"  Mean: {np.mean(r2_raw_mean):.4f}\n")
        if use_transformed_eig:
            f.write(f"\nPosterior Point Estimate R² (single sample, Transformed Space):\n")
            f.write(f"  v₁: {r2_scaled_point[0]:.4f}\n")
            f.write(f"  Δλ₂: {r2_scaled_point[1]:.4f}\n")
            f.write(f"  Δλ₃: {r2_scaled_point[2]:.4f}\n")
            f.write(f"  Mean: {np.mean(r2_scaled_point):.4f}\n")
            f.write(f"\nPosterior Mean R² ({n_post_samples} samples/node, Transformed Space):\n")
            f.write(f"  v₁: {r2_scaled_mean[0]:.4f}\n")
            f.write(f"  Δλ₂: {r2_scaled_mean[1]:.4f}\n")
            f.write(f"  Δλ₃: {r2_scaled_mean[2]:.4f}\n")
            f.write(f"  Mean: {np.mean(r2_scaled_mean):.4f}\n")
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
    parser.add_argument('--checkpoint_every', type=int, default=250,
                        help='Write a resumable checkpoint every N epochs (0 disables). '
                             'Saved atomically to flowjax_sbi_checkpoint_seed_<seed>.pkl in output_dir.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from flowjax_sbi_checkpoint_seed_<seed>.pkl in output_dir if it exists.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Explicit checkpoint path to resume from (overrides --resume lookup).')
    
    # GNN Architecture
    parser.add_argument('--num_passes', type=int, default=8, help='Message passing iterations')
    parser.add_argument('--latent_size', type=int, default=80, help='Latent dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Flow Architecture
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers')
    parser.add_argument('--num_bins', type=int, default=8, help='Spline knots')
    parser.add_argument('--flow_hidden_size', type=int, default=128, help='Flow conditioner hidden size')
    parser.add_argument(
        '--test_posterior_samples',
        type=int,
        default=128,
        help='Number of posterior samples per test node for posterior-mean metrics.'
    )
    parser.add_argument(
        '--test_eval_chunk_size',
        type=int,
        default=2048,
        help='Chunk size (nodes) for test posterior sampling to control memory.'
    )
    
    # Eigenvalue transformation
    parser.add_argument('--no_transformed_eig', action='store_true',
                        help='Use raw eigenvalues instead of transformed (v₁, Δλ₂, Δλ₃)')
    parser.add_argument('--increment_mode', type=str, default=None,
                        choices=['softplus', 'linear', 'raw'],
                        help='Target parameterisation. Overrides --no_transformed_eig. '
                             'softplus=ordered increments (default), linear=plain increments '
                             '(λ₂-λ₁), raw=direct eigenvalues. Cache suffix: '
                             '_transformed_eig/_linear_eig/_raw_eig.')
    
    args = parser.parse_args()
    main(args)
