"""
Posterior Visualization for Two-Stage SBI Pipeline

This script loads the trained posterior from the two-stage SBI pipeline
and creates diagnostic plots including:
1. Corner plots showing 1D and 2D marginals
2. Individual posterior samples with true values
3. Calibration diagnostics (if enough samples available)

Usage:
    python plot_sbi_posteriors.py --model_path /path/to/two_stage_model_*.pkl
"""
import os
import sys

# Force priority for user installed packages
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# SBI/LtU-ILI imports
import torch
try:
    from sbi.analysis import pairplot
    SBI_PLOT_AVAILABLE = True
except ImportError:
    print("Warning: sbi.analysis not available for pairplot")
    SBI_PLOT_AVAILABLE = False

# TARP coverage tests
try:
    import tarp
    TARP_AVAILABLE = True
except ImportError:
    print("Warning: TARP not available. Install with: pip install tarp")
    TARP_AVAILABLE = False


def load_results(model_path):
    """Load the two-stage SBI model results."""
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    # Load the posterior
    posterior_path = model_info.get('stage2_flow') or model_info.get('posteriors_path')
    if posterior_path and os.path.exists(posterior_path):
        print(f"Loading posterior from: {posterior_path}")
        with open(posterior_path, 'rb') as f:
            posterior = pickle.load(f)
    else:
        posterior = None
        print("Warning: Posterior not found")
    
    # Load embeddings
    embeddings_path = model_info.get('embeddings')
    if embeddings_path and os.path.exists(embeddings_path):
        print(f"Loading embeddings from: {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
    else:
        embeddings_data = None
        print("Warning: Embeddings not found")
    
    return model_info, posterior, embeddings_data


def load_targets(data_path='/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3.pkl'):
    """Load the original targets (eigenvalues) for comparison."""
    print(f"Loading targets from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    targets = data['regression_targets']
    train_mask, val_mask, test_mask = data['masks']
    eigenvalue_scaler = data.get('eigenvalue_scaler')
    
    return targets, train_mask, val_mask, test_mask, eigenvalue_scaler


def plot_single_posterior(posterior, embedding, true_theta, idx, output_dir, 
                          num_samples=2000, param_names=None):
    """
    Create a corner plot for a single posterior given an embedding.
    
    Args:
        posterior: The trained SBI posterior
        embedding: Node embedding [latent_size]
        true_theta: True eigenvalues [3]
        idx: Index for filename
        output_dir: Directory to save plots
        num_samples: Number of posterior samples
        param_names: Parameter names for axis labels
    """
    if param_names is None:
        param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    # Convert embedding to torch tensor
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    
    # Sample from posterior
    samples = posterior.sample((num_samples,), x=x)
    samples_np = samples.squeeze().numpy()
    
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
    
    plt.suptitle(f'Posterior for Node {idx}', fontsize=14)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, f'posterior_node_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return samples_np


def plot_posterior_comparison(posterior, embeddings, true_thetas, indices, 
                              output_dir, num_samples=1000, param_names=None):
    """
    Create a summary plot comparing posteriors for multiple nodes.
    """
    if param_names is None:
        param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    n_nodes = len(indices)
    fig, axes = plt.subplots(n_nodes, 3, figsize=(12, 3*n_nodes))
    
    if n_nodes == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        x = torch.tensor(embeddings[idx:idx+1], dtype=torch.float32)
        samples = posterior.sample((num_samples,), x=x).squeeze().numpy()
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
    
    plt.suptitle('Posterior Marginals Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'posterior_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_posterior_pairplot(posterior, embedding, true_theta, idx, output_dir,
                            num_samples=5000, param_names=None):
    """
    Use sbi's pairplot for a more detailed corner plot.
    """
    if not SBI_PLOT_AVAILABLE:
        print("sbi.analysis not available, skipping pairplot")
        return
    
    if param_names is None:
        param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    samples = posterior.sample((num_samples,), x=x)
    
    # Use sbi's pairplot
    fig, axes = pairplot(
        samples,
        points=torch.tensor(true_theta).unsqueeze(0),
        labels=param_names,
        points_colors=['red'],
        figsize=(8, 8),
    )
    
    plt.suptitle(f'SBI Pairplot - Node {idx}', fontsize=14, y=1.02)
    
    save_path = os.path.join(output_dir, f'pairplot_node_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_calibration_summary(posterior, embeddings, true_thetas, 
                             output_dir, num_test=100, num_samples=1000):
    """
    Create a calibration plot to check if posteriors are well-calibrated.
    
    For each test point, we compute the rank of the true parameter
    under the posterior samples. Well-calibrated posteriors should
    produce uniformly distributed ranks.
    """
    param_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    
    # Select random test points
    n_points = min(num_test, len(embeddings))
    indices = np.random.choice(len(embeddings), n_points, replace=False)
    
    ranks = []
    for idx in indices:
        x = torch.tensor(embeddings[idx:idx+1], dtype=torch.float32)
        samples = posterior.sample((num_samples,), x=x).squeeze().numpy()
        true_theta = true_thetas[idx]
        
        # Compute ranks (proportion of samples less than true value)
        rank = np.mean(samples < true_theta, axis=0)
        ranks.append(rank)
    
    ranks = np.array(ranks)  # [n_points, 3]
    
    # Plot rank histogram for each parameter
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
    
    plt.suptitle(f'Calibration Check (SBC-style) - {n_points} test points', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'calibration_check.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_tarp_coverage(posterior, embeddings, true_thetas, output_dir, 
                        num_test=500, num_samples=1000):
    """
    Create TARP (Tests of Accuracy with Random Points) coverage plot.
    
    TARP provides a sufficient condition for posterior accuracy, unlike SBC
    which only provides a necessary condition. A well-calibrated posterior
    should have the coverage curve follow the diagonal.
    
    - Curve above diagonal = over-confident (posteriors too narrow)
    - Curve below diagonal = under-confident (posteriors too wide)
    - Curve on diagonal = well-calibrated
    """
    if not TARP_AVAILABLE:
        print("TARP not available, skipping TARP coverage test")
        return
    
    print(f"Running TARP on {min(num_test, len(embeddings))} test points...")
    
    # Select test points
    n_points = min(num_test, len(embeddings))
    indices = np.random.choice(len(embeddings), n_points, replace=False)
    
    # Collect posterior samples for each test point
    all_samples = []
    all_thetas = []
    
    for i, idx in enumerate(indices):
        if (i + 1) % 100 == 0:
            print(f"  Sampling {i+1}/{n_points}...")
        
        x = torch.tensor(embeddings[idx:idx+1], dtype=torch.float32)
        samples = posterior.sample((num_samples,), x=x).squeeze().numpy()
        all_samples.append(samples)
        all_thetas.append(true_thetas[idx])
    
    # Stack into arrays
    all_samples = np.array(all_samples)  # [n_test, n_samples, 3]
    all_thetas = np.array(all_thetas)    # [n_test, 3]
    
    # TARP expects samples of shape [n_samples, n_evals, n_params]
    samples_tarp = np.transpose(all_samples, (1, 0, 2))
    
    try:
        # Get TARP coverage (no bootstrap first to get basic result)
        ecp, alpha = tarp.get_tarp_coverage(
            samples_tarp, 
            all_thetas,
            norm=True,
            bootstrap=False
        )
        
        # Create TARP plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal (well-calibrated)')
        ax.plot(alpha, ecp, 'b-', linewidth=2, label='TARP coverage')
        
        ax.set_xlabel(r'Credibility Level $\alpha$', fontsize=12)
        ax.set_ylabel('Expected Coverage Probability', fontsize=12)
        ax.set_title(f'TARP Coverage Test ({n_points} test points)', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'Above diagonal = over-confident\nBelow diagonal = under-confident',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        save_path = os.path.join(output_dir, 'tarp_coverage.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error computing TARP coverage: {e}")
        print("Skipping TARP coverage plot")


def plot_training_history(model_info, output_dir):
    """Plot Stage 1 training history."""
    stage1_train_losses = model_info.get('stage1_train_losses', [])
    stage1_val_losses = model_info.get('stage1_val_losses', [])
    
    if not stage1_train_losses:
        print("No Stage 1 training history found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    epochs_train = range(len(stage1_train_losses))
    ax.plot(epochs_train, stage1_train_losses, label='Train MSE', alpha=0.7)
    
    if stage1_val_losses:
        val_epochs = [v[0] for v in stage1_val_losses]
        val_values = [v[1] for v in stage1_val_losses]
        ax.plot(val_epochs, val_values, label='Val MSE', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Stage 1: GNN Encoder Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'stage1_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main(args):
    """Main plotting function."""
    print("=" * 70)
    print("Two-Stage SBI Posterior Visualization")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    model_info, posterior, embeddings_data = load_results(args.model_path)
    
    if posterior is None:
        print("ERROR: Could not load posterior. Exiting.")
        return
    
    # Load targets for comparison
    targets, train_mask, val_mask, test_mask, eigenvalue_scaler = load_targets()
    targets = np.array(targets)
    
    # Get test embeddings and targets
    if embeddings_data is not None:
        test_embeddings = np.array(embeddings_data.get('test_embeddings'))
        val_embeddings = np.array(embeddings_data.get('val_embeddings'))
    else:
        print("ERROR: Could not load embeddings. Exiting.")
        return
    
    test_targets = targets[test_mask]
    val_targets = targets[val_mask]
    
    print(f"\nTest set: {len(test_embeddings)} samples")
    print(f"Val set: {len(val_embeddings)} samples")
    
    # 1. Plot training history
    print("\n[1/4] Plotting training history...")
    plot_training_history(model_info, args.output_dir)
    
    # 2. Plot individual posteriors for a few test samples
    print("\n[2/4] Plotting individual posteriors...")
    num_individual = min(args.num_plots, len(test_embeddings))
    sample_indices = np.random.choice(len(test_embeddings), num_individual, replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"  Creating posterior plot {i+1}/{num_individual}...")
        
        # Use sbi pairplot if available
        if SBI_PLOT_AVAILABLE:
            plot_posterior_pairplot(
                posterior, 
                test_embeddings[idx], 
                test_targets[idx],
                idx,
                args.output_dir,
                num_samples=args.num_samples
            )
        else:
            plot_single_posterior(
                posterior,
                test_embeddings[idx],
                test_targets[idx],
                idx,
                args.output_dir,
                num_samples=args.num_samples
            )
    
    # 3. Plot comparison figure
    print("\n[3/4] Plotting posterior comparison...")
    comparison_indices = sample_indices[:min(5, len(sample_indices))]
    plot_posterior_comparison(
        posterior, 
        test_embeddings,
        test_targets,
        comparison_indices,
        args.output_dir,
        num_samples=args.num_samples
    )
    
    # 4. Simple SBC-style calibration check
    print("\n[4/5] Running SBC-style calibration check...")
    plot_calibration_summary(
        posterior,
        test_embeddings,
        test_targets,
        args.output_dir,
        num_test=min(1000, len(test_embeddings)),
        num_samples=args.num_samples
    )
    
    # 5. TARP coverage test (more rigorous)
    print("\n[5/5] Running TARP coverage test...")
    plot_tarp_coverage(
        posterior,
        test_embeddings,
        test_targets,
        args.output_dir,
        num_test=min(500, len(test_embeddings)),
        num_samples=args.num_samples
    )
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot SBI Posteriors')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the two_stage_model_*.pkl file')
    parser.add_argument('--output_dir', type=str, default='sbi_plots',
                        help='Output directory for plots')
    parser.add_argument('--num_plots', type=int, default=5,
                        help='Number of individual posterior plots')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of posterior samples per plot')
    
    args = parser.parse_args()
    main(args)
