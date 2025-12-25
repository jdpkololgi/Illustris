"""
Posterior Visualization for Flowjax SBI Pipeline

This script loads the trained GNN + Flowjax model and creates diagnostic plots
comparable to the two-stage SBI pipeline plots.

Usage:
    python plot_flowjax_posteriors.py --model_path /path/to/flowjax_sbi_model_*.pkl
"""
import os
import sys

# Force priority for user installed packages
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import equinox as eqx

from flowjax.flows import masked_autoregressive_flow, RationalQuadraticSpline
from flowjax.distributions import Normal

from graph_net_models import make_gnn_encoder

# TARP coverage tests
try:
    import tarp
    TARP_AVAILABLE = True
except ImportError:
    print("Warning: TARP not available. Install with: pip install tarp")
    TARP_AVAILABLE = False


def load_flowjax_model(model_path):
    """Load the Flowjax SBI model."""
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    gnn_params = model_info['gnn_params']
    config = model_info['config']
    eigenvalue_scaler = model_info['eigenvalue_scaler']
    flow_filename = model_info['flow_filename']
    
    print(f"Loading flow from: {flow_filename}")
    
    return gnn_params, config, eigenvalue_scaler, flow_filename


def load_data(data_path='/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3.pkl'):
    """Load the graph data and targets."""
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    graph = data['graph']
    targets = np.array(data['regression_targets'])
    train_mask, val_mask, test_mask = data['masks']
    
    return graph, targets, train_mask, val_mask, test_mask


def create_gnn_and_flow(config, flow_filename, graph, master_key):
    """Recreate the GNN encoder and load the flow."""
    # Create GNN encoder
    gnn_encoder_fn = make_gnn_encoder(
        latent_size=config['latent_size'],
        num_heads=config['num_heads'],
        num_passes=config['num_passes'],
        dropout_rate=config.get('dropout', 0.2),
    )
    
    @hk.transform
    def network(graph, is_training=True):
        return gnn_encoder_fn(graph, is_training=is_training)
    
    # Initialize GNN
    gnn_key, flow_key = jax.random.split(master_key)
    gnn = network
    
    # Create flow structure (must match training exactly)
    # Base distribution: standard normal for 3 eigenvalues
    base_dist = Normal(jnp.zeros(3), jnp.ones(3))
    
    # Create conditional masked autoregressive flow with SAME parameters as training
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=base_dist,
        cond_dim=config['latent_size'],  # Conditioning on GNN embeddings
        flow_layers=config.get('num_flow_layers', 5),
        nn_width=config.get('flow_hidden_size', 128),
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=config.get('num_bins', 8), interval=12),  # [-12, 12]
    )
    
    # Load trained flow
    flow = eqx.tree_deserialise_leaves(flow_filename, flow)
    
    return gnn, flow


def sample_posterior(flow, embedding, num_samples, key):
    """Sample from the posterior given an embedding."""
    # Flowjax sampling
    samples = flow.sample(key, (num_samples,), condition=embedding)
    return np.array(samples)


def plot_single_posterior(embedding, true_theta, idx, output_dir, 
                          flow, key, num_samples=2000, param_names=None):
    """Create a corner plot for a single posterior."""
    if param_names is None:
        param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    # Sample from posterior
    samples_np = sample_posterior(flow, embedding, num_samples, key)
    
    # Create corner plot
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Diagonal: 1D marginals
    for i in range(3):
        ax = axes[i, i]
        ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(true_theta[i], color='red', linewidth=2, label='True' if i==0 else None)
        ax.set_xlabel(param_names[i] if i == 2 else '')
        ax.set_ylabel('Density' if i == 0 else '')
        if i == 0:
            ax.legend()
    
    # Off-diagonal: 2D marginals  
    for i in range(3):
        for j in range(3):
            if i > j:
                ax = axes[i, j]
                ax.hist2d(samples_np[:, j], samples_np[:, i], bins=50, cmap='Blues')
                ax.scatter(true_theta[j], true_theta[i], color='red', s=50, marker='x', linewidths=2)
                ax.set_xlabel(param_names[j] if i == 2 else '')
                ax.set_ylabel(param_names[i] if j == 0 else '')
            elif i < j:
                axes[i, j].axis('off')
    
    plt.suptitle(f'Flowjax Posterior for Node {idx}', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'flowjax_pairplot_node_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return samples_np


def plot_posterior_comparison(embeddings, true_thetas, indices, output_dir,
                               flow, key, num_samples=1000, param_names=None):
    """Create a summary plot comparing posteriors for multiple nodes."""
    if param_names is None:
        param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    n_nodes = len(indices)
    fig, axes = plt.subplots(n_nodes, 3, figsize=(12, 3*n_nodes))
    
    if n_nodes == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        key, sample_key = jax.random.split(key)
        samples = sample_posterior(flow, embeddings[idx], num_samples, sample_key)
        true_theta = true_thetas[idx]
        
        for col in range(3):
            ax = axes[row, col]
            ax.hist(samples[:, col], bins=40, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='white')
            ax.axvline(true_theta[col], color='red', linewidth=2, linestyle='--',
                      label=f'True: {true_theta[col]:.2f}')
            
            mean_val = samples[:, col].mean()
            ax.axvline(mean_val, color='green', linewidth=2, linestyle=':',
                      label=f'Mean: {mean_val:.2f}')
            
            ax.set_xlabel(param_names[col])
            if col == 0:
                ax.set_ylabel(f'Node {idx}')
            ax.legend(fontsize=8)
    
    plt.suptitle('Flowjax Posterior Marginals Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'flowjax_posterior_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_calibration_summary(embeddings, true_thetas, output_dir,
                              flow, key, num_test=1000, num_samples=1000):
    """Create SBC-style calibration plot."""
    param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    n_points = min(num_test, len(embeddings))
    indices = np.random.choice(len(embeddings), n_points, replace=False)
    
    ranks = []
    for i, idx in enumerate(indices):
        if (i + 1) % 200 == 0:
            print(f"  Calibration sampling {i+1}/{n_points}...")
        
        key, sample_key = jax.random.split(key)
        samples = sample_posterior(flow, embeddings[idx], num_samples, sample_key)
        true_theta = true_thetas[idx]
        
        # Compute ranks
        rank = np.mean(samples < true_theta, axis=0)
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i in range(3):
        ax = axes[i]
        ax.hist(ranks[:, i], bins=20, density=True, alpha=0.7, 
               color='steelblue', edgecolor='white')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
        ax.set_xlabel(f'Rank for {param_names[i]}')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.set_title(param_names[i])
    
    plt.suptitle(f'Flowjax Calibration Check (SBC-style) - {n_points} test points', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'flowjax_calibration_check.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_tarp_coverage(embeddings, true_thetas, output_dir,
                        flow, key, num_test=500, num_samples=1000):
    """Create TARP coverage plot."""
    if not TARP_AVAILABLE:
        print("TARP not available, skipping TARP coverage test")
        return
    
    print(f"Running TARP on {min(num_test, len(embeddings))} test points...")
    
    n_points = min(num_test, len(embeddings))
    indices = np.random.choice(len(embeddings), n_points, replace=False)
    
    all_samples = []
    all_thetas = []
    
    for i, idx in enumerate(indices):
        if (i + 1) % 100 == 0:
            print(f"  Sampling {i+1}/{n_points}...")
        
        key, sample_key = jax.random.split(key)
        samples = sample_posterior(flow, embeddings[idx], num_samples, sample_key)
        all_samples.append(samples)
        all_thetas.append(true_thetas[idx])
    
    all_samples = np.array(all_samples)
    all_thetas = np.array(all_thetas)
    
    # TARP expects [n_samples, n_evals, n_params]
    samples_tarp = np.transpose(all_samples, (1, 0, 2))
    
    try:
        ecp, alpha = tarp.get_tarp_coverage(
            samples_tarp, 
            all_thetas,
            norm=True,
            bootstrap=False
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal (well-calibrated)')
        ax.plot(alpha, ecp, 'b-', linewidth=2, label='TARP coverage')
        
        ax.set_xlabel(r'Credibility Level $\alpha$', fontsize=12)
        ax.set_ylabel('Expected Coverage Probability', fontsize=12)
        ax.set_title(f'Flowjax TARP Coverage Test ({n_points} test points)', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'Above diagonal = over-confident\nBelow diagonal = under-confident',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        save_path = os.path.join(output_dir, 'flowjax_tarp_coverage.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error computing TARP coverage: {e}")


def plot_training_history(logs_path, output_dir):
    """Plot training history from logs."""
    if not os.path.exists(logs_path):
        print(f"Logs not found at {logs_path}")
        return
    
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)
    
    train_losses = logs.get('train_losses', [])
    val_losses = logs.get('val_losses', [])
    
    if not train_losses:
        print("No training history found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    epochs = range(len(train_losses))
    ax.plot(epochs, train_losses, label='Train NLL', alpha=0.7)
    
    if val_losses:
        val_epochs = [v[0] for v in val_losses]
        val_values = [v[1] for v in val_losses]
        ax.plot(val_epochs, val_values, label='Val NLL', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL Loss')
    ax.set_title('Flowjax Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'flowjax_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main(args):
    print("=" * 70)
    print("Flowjax SBI Posterior Visualization")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    gnn_params, config, eigenvalue_scaler, flow_filename = load_flowjax_model(args.model_path)
    
    # Load data
    graph, targets, train_mask, val_mask, test_mask = load_data()
    
    # Get test targets
    test_targets = targets[test_mask]
    
    print(f"\nTest set: {np.sum(test_mask)} samples")
    
    # Setup
    master_key = jax.random.key(42)
    
    # Create GNN and load flow
    gnn, flow = create_gnn_and_flow(config, flow_filename, graph, master_key)
    
    # Get embeddings for test set
    print("\n[Setup] Computing test embeddings...")
    eval_key = jax.random.key(0)
    all_embeddings = gnn.apply(gnn_params, eval_key, graph, is_training=False)
    test_embeddings = np.array(all_embeddings[test_mask])
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Plot training history
    logs_path = args.model_path.replace('_model_', '_logs_').replace('.pkl', '.pkl')
    print("\n[1/5] Plotting training history...")
    plot_training_history(logs_path, args.output_dir)
    
    # Plot individual posteriors
    print("\n[2/5] Plotting individual posteriors...")
    num_individual = min(args.num_plots, len(test_embeddings))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(test_embeddings), num_individual, replace=False)
    
    key = jax.random.key(123)
    for i, idx in enumerate(sample_indices):
        print(f"  Creating posterior plot {i+1}/{num_individual}...")
        key, plot_key = jax.random.split(key)
        plot_single_posterior(
            test_embeddings[idx],
            test_targets[idx],
            idx,
            args.output_dir,
            flow,
            plot_key,
            num_samples=args.num_samples
        )
    
    # Comparison plot
    print("\n[3/5] Plotting posterior comparison...")
    key, comp_key = jax.random.split(key)
    plot_posterior_comparison(
        test_embeddings,
        test_targets,
        sample_indices[:5],
        args.output_dir,
        flow,
        comp_key,
        num_samples=args.num_samples
    )
    
    # Calibration check
    print("\n[4/5] Running SBC-style calibration check...")
    key, cal_key = jax.random.split(key)
    plot_calibration_summary(
        test_embeddings,
        test_targets,
        args.output_dir,
        flow,
        cal_key,
        num_test=min(1000, len(test_embeddings)),
        num_samples=args.num_samples
    )
    
    # TARP coverage
    print("\n[5/5] Running TARP coverage test...")
    key, tarp_key = jax.random.split(key)
    plot_tarp_coverage(
        test_embeddings,
        test_targets,
        args.output_dir,
        flow,
        tarp_key,
        num_test=min(500, len(test_embeddings)),
        num_samples=args.num_samples
    )
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Flowjax SBI Posteriors')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the flowjax_sbi_model_*.pkl file')
    parser.add_argument('--output_dir', type=str, default='sbi_plots_flowjax',
                        help='Output directory for plots')
    parser.add_argument('--num_plots', type=int, default=5,
                        help='Number of individual posterior plots')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of posterior samples per plot')
    
    args = parser.parse_args()
    main(args)
