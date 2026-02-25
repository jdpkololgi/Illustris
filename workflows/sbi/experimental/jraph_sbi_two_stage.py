"""
Two-Stage SBI Pipeline: Separate Training of GNN Encoder and Normalizing Flow

This pipeline trains the GNN encoder and normalizing flow separately to avoid
numerical instability from joint training.

Stage 1: Train GNN encoder on eigenvalue regression task
Stage 2: Extract embeddings and train normalizing flow separately

Usage:
    python jraph_sbi_two_stage.py [--flow_backend ili] [--stage1_epochs 5000] [--stage2_epochs 5000] [--seed 42]
"""
import os
import sys
from pathlib import Path

# Force priority for user installed packages
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
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jraph

# LtU-ILI imports
try:
    import ili
    from ili.inference import InferenceRunner
    from ili.dataloaders import NumpyLoader
    from ili.utils import load_nde_sbi
    import torch
    ILI_AVAILABLE = True
except ImportError:
    print("Warning: LtU-ILI not available. Install with: pip install ltu-ili")
    ILI_AVAILABLE = False

from graph_net_models import make_gnn_encoder


def train_gnn_encoder_stage1(args, graph, targets, train_mask, val_mask, eigenvalue_scaler):
    """
    Stage 1: Train GNN encoder on eigenvalue regression task.
    
    Returns:
        trained_gnn_params: Trained GNN encoder parameters
        embeddings: Extracted embeddings for all nodes [N, latent_size]
    """
    print("=" * 70)
    print("STAGE 1: Training GNN Encoder (Regression Task)")
    print("=" * 70)
    
    master_key = jax.random.key(args.seed)
    num_devices = len(jax.local_devices())
    print(f"JAX Devices: {jax.devices()}")
    print(f"Running on {num_devices} device(s).")
    
    # =========================================================================
    # Setup GNN Encoder + Simple Decoder
    # =========================================================================
    print("\n[Stage 1] Setting up GNN encoder + decoder...")
    
    # Encoder: use make_gnn_encoder
    gnn_encoder_fn = make_gnn_encoder(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    gnn_encoder = hk.transform(gnn_encoder_fn)
    
    # Decoder: simple linear layer to map embeddings -> eigenvalues
    def decoder_fn(embeddings: jnp.ndarray) -> jnp.ndarray:
        """Simple linear decoder: embeddings -> eigenvalues."""
        return hk.Linear(3)(embeddings)  # 3 eigenvalues
    
    decoder = hk.transform(decoder_fn)
    
    # Combined model function for GNN encoder and then linear decoder for regression task
    def combined_model(graph: jraph.GraphsTuple, is_training: bool = True) -> jnp.ndarray:
        """Combined encoder + decoder."""
        embeddings = gnn_encoder_fn(graph, is_training=is_training)
        predictions = decoder_fn(embeddings)
        return predictions, embeddings
    
    combined_net = hk.transform(combined_model)
    
    # Initialize
    key, init_key = jax.random.split(master_key)
    params = combined_net.init(init_key, graph, is_training=True)
    
    # Separate encoder and decoder params
    # Note: Haiku stores params in nested dict structure
    # We'll extract them after training, for now train combined model
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")
    
    # =========================================================================
    # Optimizer Setup
    # =========================================================================
    print("\n[Stage 1] Setting up optimizer...")
    
    num_epochs_stage1 = args.stage1_epochs
    
    warmup_steps = min(500, num_epochs_stage1 // 10)
    decay_steps = max(num_epochs_stage1 - warmup_steps, warmup_steps + 1)
    
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
    opt_state = optimizer.init(params)
    
    # =========================================================================
    # Multi-GPU Setup
    # =========================================================================
    print("\n[Stage 1] Setting up parallelization...")
    
    replicated_params = jax.device_put_replicated(params, jax.local_devices())
    replicated_opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
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
    
    # =========================================================================
    # Training Functions
    # =========================================================================
    def compute_loss(params, graph, targets, mask, rng, is_training=True):
        """Compute MSE loss for regression."""
        predictions, embeddings = combined_net.apply(params, rng, graph, is_training=is_training)
        
        # MSE loss
        per_node_loss = jnp.mean(optax.l2_loss(predictions, targets), axis=-1)
        masked_loss = per_node_loss * mask
        num_masked = jnp.sum(mask)
        loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
        
        return loss, (predictions, num_masked)
    
    def update(params, opt_state, graph, targets, mask, rng):
        """Single update step."""
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        (loss, (predictions, num_masked)), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params, graph, targets, mask, step_rng)
        
        # Sync gradients
        grads = jax.lax.pmean(grads, axis_name='i')
        
        # Sync loss
        total_loss_part = loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        # Update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, global_loss
    
    update_fn = jax.pmap(update, axis_name='i')
    
    def evaluate(params, graph, targets, mask, rng):
        """Evaluation step."""
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        predictions, embeddings = combined_net.apply(params, step_rng, graph, is_training=False)
        per_node_loss = jnp.mean(optax.l2_loss(predictions, targets), axis=-1)
        masked_loss = per_node_loss * mask
        num_masked = jnp.sum(mask)
        loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
        
        # Global metrics
        total_loss_part = loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        return global_loss
    
    evaluate_fn = jax.pmap(evaluate, axis_name='i')
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n[Stage 1] Starting training...")
    print(f"Epochs: {num_epochs_stage1}, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print("-" * 70)
    
    current_rng = key
    t0 = time.time()
    
    best_val_loss = float('inf')
    best_params = None
    
    train_losses = []
    val_losses = []
    
    report_every = max(1, num_epochs_stage1 // 100)
    
    for epoch in range(num_epochs_stage1):
        current_rng, step_rng = jax.random.split(current_rng)
        step_rngs = jax.device_put_replicated(step_rng, jax.local_devices())
        
        # Training step
        replicated_params, replicated_opt_state, train_loss = update_fn(
            replicated_params, replicated_opt_state,
            replicated_graph, replicated_targets, sharded_train_masks,
            step_rngs
        )
        
        train_losses.append(float(train_loss[0]))
        
        # Validation
        if epoch % report_every == 0 or epoch == num_epochs_stage1 - 1:
            val_loss = evaluate_fn(
                replicated_params, replicated_graph, replicated_targets,
                sharded_val_masks, step_rngs
            )
            
            val_losses.append((epoch, float(val_loss[0])))
            
            if float(val_loss[0]) < best_val_loss:
                best_val_loss = float(val_loss[0])
                best_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_params))
            
            elapsed = time.time() - t0
            print(f"Epoch {epoch:5d} | Train MSE: {train_loss[0]:.6f} | Val MSE: {val_loss[0]:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    print("-" * 70)
    print(f"Stage 1 training finished in {time.time() - t0:.2f}s")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    
    if best_params is None:
        best_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_params))
    
    # =========================================================================
    # Extract Embeddings Immediately After Training
    # =========================================================================
    print("\n[Stage 1] Extracting embeddings from trained model...")
    
    # Extract embeddings from best params
    eval_rng = jax.random.key(args.seed + 1000)
    predictions, embeddings = combined_net.apply(best_params, eval_rng, graph, is_training=False)

    
    return best_params, train_losses, val_losses, embeddings

def prepare_embeddings_data(embeddings, train_mask, val_mask, test_mask):
    """
    Prepare embeddings data dictionary from extracted embeddings.
    
    Args:
        embeddings: All node embeddings [N, latent_size]
        train_mask, val_mask, test_mask: Boolean masks
    
    Returns:
        embeddings_data: Dictionary with train/val/test splits
    """
    print(f"Embeddings shape: {embeddings.shape}")

    # Validate embeddings
    if jnp.any(jnp.isnan(embeddings)):
        print("WARNING: NaN embeddings found in embeddings!")
    else:
        print("Embeddings validated successfully.")

    # Split by masks for later use
    train_emb = embeddings[train_mask]
    val_emb = embeddings[val_mask]
    test_emb = embeddings[test_mask]
    
    print(f"Train embeddings shape: {train_emb.shape}")
    print(f"Val embeddings shape: {val_emb.shape}")
    print(f"Test embeddings shape: {test_emb.shape}")
    
    print(f"[Stage 1] Embeddings extracted successfully.")

    train_emb = embeddings[train_mask]
    val_emb = embeddings[val_mask]
    test_emb = embeddings[test_mask]

    return {
        'train_embeddings': train_emb,
        'val_embeddings': val_emb,
        'test_embeddings': test_emb,
        'all_embeddings': embeddings,
    }


def train_flow_stage2_ili(args, embeddings_data, targets, train_mask, val_mask, eigenvalue_scaler):
    """
    Stage 2 (Part 2): Train normalizing flow using LtU-ILI.
    
    LtU-ILI uses InferenceRunner with NPE engine and MAF model for
    neural posterior estimation.
    """
    if not ILI_AVAILABLE:
        raise ImportError("LtU-ILI is not available. Install with: pip install ltu-ili")
    
    print("\n" + "=" * 70)
    print("STAGE 2 (Part 2): Training Normalizing Flow (LtU-ILI)")
    print("=" * 70)
    
    # Convert JAX arrays to numpy for LtU-ILI
    train_emb = np.array(embeddings_data['train_embeddings'])
    val_emb = np.array(embeddings_data['val_embeddings'])
    
    train_targets = np.array(targets[train_mask])
    val_targets = np.array(targets[val_mask])
    
    print(f"Train: {train_emb.shape[0]} samples, Val: {val_emb.shape[0]} samples")
    print(f"Embedding dimension: {train_emb.shape[1]}, Parameter dimension: {train_targets.shape[1]}")
    
    # =========================================================================
    # Set up LtU-ILI Data Loader
    # =========================================================================
    print("\n[Stage 2] Setting up data loader...")
    
    # Create data loader for LtU-ILI
    # x = observations/context (GNN embeddings)
    # theta = parameters to infer (eigenvalues)
    loader = NumpyLoader(x=train_emb, theta=train_targets)
    
    # =========================================================================
    # Define Prior over Eigenvalues
    # =========================================================================
    print("[Stage 2] Setting up prior...")
    
    # Define prior bounds based on training data range (with some margin)
    theta_min = np.min(train_targets, axis=0) - 2.0
    theta_max = np.max(train_targets, axis=0) + 2.0
    
    # Use torch for prior (LtU-ILI uses PyTorch backend for sbi)
    prior = torch.distributions.Uniform(
        low=torch.tensor(theta_min, dtype=torch.float32),
        high=torch.tensor(theta_max, dtype=torch.float32)
    )
    
    print(f"Prior bounds: low={theta_min}, high={theta_max}")
    
    # =========================================================================
    # Set up Neural Density Estimator (MAF for NPE)
    # =========================================================================
    print("[Stage 2] Setting up neural density estimator (MAF)...")
    
    # Load the neural density estimator configuration
    nets = [load_nde_sbi(
        engine='NPE',
        model='maf',
        hidden_features=args.flow_hidden_size,
        num_transforms=args.num_flow_layers,
    )]
    
    print(f"MAF config: {args.num_flow_layers} transforms, {args.flow_hidden_size} hidden features")
    
    # =========================================================================
    # Initialize InferenceRunner
    # =========================================================================
    print("[Stage 2] Initializing InferenceRunner...")
    
    runner = InferenceRunner.load(
        backend='sbi',
        engine='NPE',
        prior=prior,
        nets=nets,
        device='cpu',  # Use CPU (GPU if available via CUDA)
        train_args={'training_batch_size': getattr(args, 'batch_size', 256)},
    )
    
    print("InferenceRunner initialized with NPE engine and MAF model")
    
    # =========================================================================
    # Train the Flow
    # =========================================================================
    print("\n[Stage 2] Starting flow training...")
    print("-" * 70)
    
    t0 = time.time()
    
    # Train using InferenceRunner
    # Returns posterior and summary
    posterior, summary = runner(loader=loader)
    
    print("-" * 70)
    print(f"Stage 2 training finished in {time.time() - t0:.2f}s")
    
    # =========================================================================
    # Validate the Model
    # =========================================================================
    print("\n[Stage 2] Validating model on validation set...")
    
    # Sample from posterior for a few validation points
    num_val_samples = min(10, len(val_emb))
    for i in range(num_val_samples):
        x_val_point = torch.tensor(val_emb[i:i+1], dtype=torch.float32)
        samples = posterior.sample((100,), x=x_val_point)
        true_theta = val_targets[i]
        sample_mean = samples.mean(dim=0).numpy()
        print(f"  Val {i}: True={true_theta}, Pred Mean={sample_mean}")
    
    # =========================================================================
    # Extract Training History
    # =========================================================================
    train_losses = []
    val_losses = []
    train_log_probs = []
    val_log_probs = []
    
    # Extract from summary if available
    if summary is not None:
        if hasattr(summary, 'training_log_probs'):
            train_log_probs = summary.training_log_probs
        if hasattr(summary, 'validation_log_probs'):
            val_log_probs = summary.validation_log_probs
    
    return posterior, train_losses, val_losses, train_log_probs, val_log_probs


def main(args):
    """Main two-stage training pipeline."""
    print("=" * 70)
    print("Two-Stage SBI Pipeline: Separate GNN + Flow Training")
    print("=" * 70)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Job Timestamp: {timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[Setup] Loading data...")
    
    data_path = '/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3.pkl'
    print(f"Loading cached Jraph data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    graph = data['graph']
    targets = data['regression_targets']
    train_mask, val_mask, test_mask = data['masks']
    eigenvalue_scaler = data['eigenvalue_scaler']
    
    print(f"Graph stats: Nodes={graph.nodes.shape[0]}, Edges={graph.edges.shape[0]}")
    print(f"Train size: {jnp.sum(train_mask)}, Val size: {jnp.sum(val_mask)}, Test size: {jnp.sum(test_mask)}")
    
    # =========================================================================
    # Stage 1: Train GNN Encoder (and extract embeddings)
    # =========================================================================
    gnn_params, stage1_train_losses, stage1_val_losses, embeddings = train_gnn_encoder_stage1(
        args, graph, targets, train_mask, val_mask, eigenvalue_scaler
    )
    
    # Prepare embeddings data
    embeddings_data = prepare_embeddings_data(embeddings, train_mask, val_mask, test_mask)
    
    # Save Stage 1 results
    stage1_filename = os.path.join(args.output_dir, f'stage1_gnn_seed_{args.seed}_{timestamp}.pkl')
    with open(stage1_filename, 'wb') as f:
        pickle.dump({
            'gnn_params': gnn_params,
            'train_losses': stage1_train_losses,
            'val_losses': stage1_val_losses,
            'config': vars(args),
        }, f)
    print(f"Stage 1 model saved to {stage1_filename}")
    
    # Save embeddings
    embeddings_filename = os.path.join(args.output_dir, f'embeddings_seed_{args.seed}_{timestamp}.pkl')
    with open(embeddings_filename, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"Embeddings saved to {embeddings_filename}")
    
    # =========================================================================
    # Stage 2 (Part 2): Train Flow
    # =========================================================================
    if args.flow_backend == 'ili':
        if not ILI_AVAILABLE:
            raise ImportError("LtU-ILI is not available. Install with: pip install ltu-ili")
        
        posterior, stage2_train_losses, stage2_val_losses, stage2_train_log_probs, stage2_val_log_probs = train_flow_stage2_ili(
            args, embeddings_data, targets, train_mask, val_mask, eigenvalue_scaler
        )
        
        # Save LtU-ILI posterior (PyTorch object)
        flow_filename = os.path.join(args.output_dir, f'stage2_ili_posterior_seed_{args.seed}_{timestamp}.pkl')
        with open(flow_filename, 'wb') as f:
            pickle.dump(posterior, f)
        print(f"LtU-ILI posterior saved to {flow_filename}")
        
    elif args.flow_backend == 'flowjax':
        print("\nFlowjax backend not implemented in this version. Use --flow_backend ili")
        raise NotImplementedError("Flowjax backend not implemented. Use LtU-ILI (--flow_backend ili)")
    else:
        raise ValueError(f"Unknown flow backend: {args.flow_backend}. Use 'ili' for LtU-ILI")
    
    # =========================================================================
    # Save Final Results
    # =========================================================================
    print("\n[Saving] Saving final results...")
    
    # Save combined model info
    model_info_filename = os.path.join(args.output_dir, f'two_stage_model_seed_{args.seed}_{timestamp}.pkl')
    with open(model_info_filename, 'wb') as f:
        pickle.dump({
            'stage1_model': stage1_filename,
            'embeddings': embeddings_filename,
            'stage2_flow': flow_filename if args.flow_backend == 'ili' else None,
            'eigenvalue_scaler': eigenvalue_scaler,
            'config': vars(args),
            'stage1_train_losses': stage1_train_losses,
            'stage1_val_losses': stage1_val_losses,
            'stage2_train_losses': stage2_train_losses,
            'stage2_val_losses': stage2_val_losses,
            'stage2_train_log_probs': stage2_train_log_probs,
            'stage2_val_log_probs': stage2_val_log_probs,
        }, f)
    print(f"Model info saved to {model_info_filename}")
    
    print("\n" + "=" * 70)
    print("Two-Stage Training Complete!")
    print("=" * 70)
    if stage1_val_losses:
        print(f"Stage 1 (GNN): Best Val MSE = {min([v[1] for v in stage1_val_losses]):.6f}")
    else:
        print("Stage 1 (GNN): Training complete")
    print("Stage 2 (Flow): Training complete with LtU-ILI")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage SBI Pipeline')
    
    # Training
    parser.add_argument('--stage1_epochs', type=int, default=5000, help='Epochs for Stage 1 (GNN training)')
    parser.add_argument('--stage2_epochs', type=int, default=5000, help='Epochs for Stage 2 (Flow training)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Stage 1 (GNN)')
    parser.add_argument('--flow_lr', type=float, default=1e-3, help='Learning rate for Stage 2 (Flow)')
    parser.add_argument('--weight_decay', type=float, default=0.08, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/sbi/', help='Output directory')
    parser.add_argument('--flow_backend', type=str, default='ili', choices=['ili'], help='Flow backend: ili (LtU-ILI)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for flow training')
    
    # GNN Architecture
    parser.add_argument('--num_passes', type=int, default=8, help='Message passing iterations')
    parser.add_argument('--latent_size', type=int, default=80, help='Latent dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Flow Architecture
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers')
    parser.add_argument('--num_bins', type=int, default=8, help='Spline knots')
    parser.add_argument('--flow_hidden_size', type=int, default=128, help='Flow conditioner hidden size')
    
    args = parser.parse_args()
    main(args)

