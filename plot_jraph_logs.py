import re
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Set style
plt.style.use('dark_background')

CLASSES = ['Void', 'Wall', 'Filament', 'Cluster']

def parse_logs(log_file):
    """Parses the training log file and determines the mode."""
    epochs, train_loss, train_metric, val_loss, val_metric = [], [], [], [], []
    mode = None
    
    # Classification pattern: Epoch X | Train Loss: Y | Train Acc: Z% | Val Loss: A | Val Acc: B%
    class_pattern = re.compile(
        r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)% \| Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)%"
    )
    # Regression pattern: Epoch X | Train Loss: Y | Train MSE: Z | Val Loss: A | Val MSE: B
    reg_pattern = re.compile(
        r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Train MSE: ([\d.]+) \| Val Loss: ([\d.]+) \| Val MSE: ([\d.]+)"
    )
    
    print(f"Reading log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            class_match = class_pattern.search(line)
            if class_match:
                mode = 'classification'
                e, tl, ta, vl, va = class_match.groups()
                epochs.append(int(e))
                train_loss.append(float(tl))
                train_metric.append(float(ta))
                val_loss.append(float(vl))
                val_metric.append(float(va))
                continue
            
            reg_match = reg_pattern.search(line)
            if reg_match:
                mode = 'regression'
                e, tl, tm, vl, vm = reg_match.groups()
                epochs.append(int(e))
                train_loss.append(float(tl))
                train_metric.append(float(tm))
                val_loss.append(float(vl))
                val_metric.append(float(vm))

    return np.array(epochs), np.array(train_loss), np.array(train_metric), np.array(val_loss), np.array(val_metric), mode

def plot_training_curves(epochs, train_loss, train_metric, val_loss, val_metric, mode, output_dir):
    """Plots training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='#FF6B6B', alpha=0.9)
    ax1.plot(epochs, val_loss, label='Val Loss', color='#4ECDC4', alpha=0.9)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    # ax1.set_ylim(0, 1)
    ax1.set_xlim(-50, max(epochs)+50)
    ax1.legend(fontsize=10)
    
    # Plot 2: Metric (Acc or MSE)
    ax2.plot(epochs, train_metric, label=f'Train {mode.capitalize()}', color='#FF6B6B', alpha=0.9)
    ax2.plot(epochs, val_metric, label=f'Val {mode.capitalize()}', color='#4ECDC4', alpha=0.9)
    ax2.set_xlabel('Epoch', fontsize=12)
    ylabel = 'Accuracy (%)' if mode == 'classification' else 'MSE'
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f'Training & Validation {mode.capitalize()}', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(-50, max(epochs)+50)
    ax2.legend(fontsize=10)
    
    # Final Stats
    if len(epochs) > 0:
        final_stats = (
            f"Final Stats (Epoch {epochs[-1]}):\n"
            f"Train {mode[:4]}: {train_metric[-1]:.4f}\n"
            f"Val {mode[:4]}:   {val_metric[-1]:.4f}"
        )
        props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
        ax2.text(0.7, 0.2, final_stats, transform=ax2.transAxes, fontsize=11,
                verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'jraph_{mode}_training_curves.png')
    plt.savefig(filename, dpi=150)
    print(f"Training curves saved to {filename}")
    plt.close(fig)

def plot_regression_parity(y_true, y_pred, output_dir):
    """Plots Predicted vs True for regression (Parity Plot)."""
    # y_true/y_pred shape: [N, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['λ1', 'λ2', 'λ3']
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    for i in range(3):
        ax = axes[i]
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        # Sample points if too many
        if len(true_i) > 10000:
            idx = np.random.choice(len(true_i), 10000, replace=False)
            t_plot = true_i[idx]
            p_plot = pred_i[idx]
        else:
            t_plot = true_i
            p_plot = pred_i
            
        ax.scatter(t_plot, p_plot, alpha=0.2, s=5, color=colors[i])
        
        # Perfect parity line
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.8)
        
        ax.set_title(f'{labels[i]} Parity Plot', fontsize=14)
        ax.set_xlabel('True Value', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        
        # Add R2
        r2 = 1 - np.sum((true_i - pred_i)**2) / (np.sum((true_i - true_i.mean())**2) + 1e-8)
        ax.text(0.1, 0.9, f'$R^2 = {r2:.4f}$', transform=ax.transAxes, fontsize=12, color='white')

    plt.suptitle("Regression Parity Plots", fontsize=16)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'jraph_regression_parity.png')
    plt.savefig(filename, dpi=300)
    print(f"Parity plots saved to {filename}")
    plt.close(fig)

def plot_eigenvalue_distributions(y_true, y_pred, output_dir):
    """Plots the distribution of eigenvalues (True vs Predicted) using smooth KDE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['λ1 (smallest)', 'λ2 (middle)', 'λ3 (largest)']
    colors_true = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    colors_pred = ['#ff9999', '#7edcd4', '#ffe066']
    
    for i in range(3):
        ax = axes[i]
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        # Plot smooth KDE
        sns.kdeplot(true_i, ax=ax, color=colors_true[i], label='True', linewidth=2, fill=True, alpha=0.4)
        sns.kdeplot(pred_i, ax=ax, color=colors_pred[i], label='Predicted', linewidth=2, linestyle='--')
        
        ax.set_title(f'{labels[i]} Distribution', fontsize=14)
        ax.set_xlabel('Eigenvalue', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        
        # Add statistics
        stats_text = f'True: μ={true_i.mean():.3f}, σ={true_i.std():.3f}\nPred: μ={pred_i.mean():.3f}, σ={pred_i.std():.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    plt.suptitle("Eigenvalue Distributions - True vs Predicted", fontsize=16)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'jraph_eigenvalue_distributions.png')
    plt.savefig(filename, dpi=300)
    print(f"Eigenvalue distributions saved to {filename}")
    plt.close(fig)

def plot_residual_distributions(y_true, y_pred, output_dir):
    """Plots the distribution of residuals (Predicted - True) for each eigenvalue."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['λ1 (smallest)', 'λ2 (middle)', 'λ3 (largest)']
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    for i in range(3):
        ax = axes[i]
        residuals = y_pred[:, i] - y_true[:, i]
        
        # Plot residual distribution
        sns.kdeplot(residuals, ax=ax, color=colors[i], linewidth=2, fill=True, alpha=0.5)
        
        # Add vertical line at zero (perfect prediction)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Error')
        
        # Add vertical line at mean residual (bias)
        mean_res = residuals.mean()
        ax.axvline(x=mean_res, color='red', linestyle='-', linewidth=1.5, alpha=0.8, label=f'Bias: {mean_res:.4f}')
        
        ax.set_title(f'{labels[i]} Residuals', fontsize=14)
        ax.set_xlabel('Residual (Predicted - True)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=9)
        
        # Add statistics
        stats_text = f'μ={mean_res:.4f}\nσ={residuals.std():.4f}\nMAE={np.abs(residuals).mean():.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    plt.suptitle("Residual Distributions", fontsize=16)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'jraph_residual_distributions.png')
    plt.savefig(filename, dpi=300)
    print(f"Residual distributions saved to {filename}")
    plt.close(fig)

def analyze_eigenvalue_ordering(y_true, y_pred, output_dir):
    """Analyzes whether eigenvalue ordering (λ1 ≤ λ2 ≤ λ3) is preserved in predictions."""
    n_samples = len(y_true)
    
    # Check ordering in true data (should be 100% ordered)
    true_ordered = (y_true[:, 0] <= y_true[:, 1]) & (y_true[:, 1] <= y_true[:, 2])
    
    # Check ordering in predictions
    pred_ordered = (y_pred[:, 0] <= y_pred[:, 1]) & (y_pred[:, 1] <= y_pred[:, 2])
    
    # Check specific violations
    violation_12 = y_pred[:, 0] > y_pred[:, 1]  # λ1 > λ2
    violation_23 = y_pred[:, 1] > y_pred[:, 2]  # λ2 > λ3
    violation_13 = y_pred[:, 0] > y_pred[:, 2]  # λ1 > λ3 (severe)
    
    # Statistics
    pct_true_ordered = 100 * np.sum(true_ordered) / n_samples
    pct_pred_ordered = 100 * np.sum(pred_ordered) / n_samples
    pct_violation_12 = 100 * np.sum(violation_12) / n_samples
    pct_violation_23 = 100 * np.sum(violation_23) / n_samples
    pct_violation_13 = 100 * np.sum(violation_13) / n_samples
    
    print("\n=== Eigenvalue Ordering Analysis ===")
    print(f"True data ordered (λ1 ≤ λ2 ≤ λ3):      {pct_true_ordered:.2f}%")
    print(f"Predictions ordered (λ1 ≤ λ2 ≤ λ3):   {pct_pred_ordered:.2f}%")
    print(f"Violations (λ1 > λ2):                  {pct_violation_12:.2f}%")
    print(f"Violations (λ2 > λ3):                  {pct_violation_23:.2f}%")
    print(f"Severe violations (λ1 > λ3):           {pct_violation_13:.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Bar chart of ordering preservation
    ax1 = axes[0]
    categories = ['Correctly\nOrdered', 'λ1 > λ2', 'λ2 > λ3', 'λ1 > λ3\n(severe)']
    values = [pct_pred_ordered, pct_violation_12, pct_violation_23, pct_violation_13]
    colors = ['#4ECDC4', '#FF6B6B', '#FFD93D', '#FF006E']
    bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Percentage of Samples (%)', fontsize=12)
    ax1.set_title('Eigenvalue Ordering Preservation', fontsize=14)
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
                ha='center', fontsize=10, color='white')
    
    # Plot 2: Scatter of λ2 - λ1 vs λ3 - λ2 (should both be positive)
    ax2 = axes[1]
    diff_12 = y_pred[:, 1] - y_pred[:, 0]  # λ2 - λ1
    diff_23 = y_pred[:, 2] - y_pred[:, 1]  # λ3 - λ2
    
    # Sample if too many points
    if n_samples > 5000:
        idx = np.random.choice(n_samples, 5000, replace=False)
        diff_12_plot = diff_12[idx]
        diff_23_plot = diff_23[idx]
    else:
        diff_12_plot = diff_12
        diff_23_plot = diff_23
    
    ax2.scatter(diff_12_plot, diff_23_plot, alpha=0.3, s=3, c='#4ECDC4')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Ordering boundary')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax2.fill_between([-2, 0], [-2, -2], [0, 0], color='red', alpha=0.1)  # Violation region
    ax2.fill_between([0, 2], [-2, -2], [0, 0], color='red', alpha=0.1)  
    ax2.fill_between([-2, 0], [0, 0], [2, 2], color='red', alpha=0.1)  
    ax2.set_xlabel('λ2 - λ1 (should be ≥ 0)', fontsize=12)
    ax2.set_ylabel('λ3 - λ2 (should be ≥ 0)', fontsize=12)
    ax2.set_title('Eigenvalue Gaps (Predictions)', fontsize=14)
    ax2.set_xlim(-1, np.percentile(diff_12, 99))
    ax2.set_ylim(-1, np.percentile(diff_23, 99))
    
    # Plot 3: Distribution of violations (magnitude)
    ax3 = axes[2]
    violation_magnitude_12 = diff_12[violation_12]  # Negative values = violations
    violation_magnitude_23 = diff_23[violation_23]
    
    if len(violation_magnitude_12) > 0:
        sns.kdeplot(-violation_magnitude_12, ax=ax3, color='#FF6B6B', label=f'λ1 > λ2 (n={len(violation_magnitude_12)})', 
                   fill=True, alpha=0.4, linewidth=2)
    if len(violation_magnitude_23) > 0:
        sns.kdeplot(-violation_magnitude_23, ax=ax3, color='#FFD93D', label=f'λ2 > λ3 (n={len(violation_magnitude_23)})', 
                   fill=True, alpha=0.4, linewidth=2)
    
    ax3.set_xlabel('Violation Magnitude', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Distribution of Ordering Violations', fontsize=14)
    ax3.legend(fontsize=10)
    
    plt.suptitle("Eigenvalue Ordering Analysis", fontsize=16)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'jraph_eigenvalue_ordering.png')
    plt.savefig(filename, dpi=300)
    print(f"Eigenvalue ordering analysis saved to {filename}")
    plt.close(fig)
    
    return pct_pred_ordered, pct_violation_12, pct_violation_23, pct_violation_13

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Plots normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cmap = sns.light_palette('#4c78a8', as_cmap=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes,
                cbar_kws={'shrink': 0.75}, annot_kws={'fontsize': 12})
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'jraph_classification_confusion_matrix.png')
    plt.savefig(filename, dpi=300)
    print(f"Confusion matrix saved to {filename}")
    plt.close(fig)

def plot_multiclass_roc(y_true, y_probs, classes, output_dir):
    """Plots ROC curves for each class (One-vs-Rest)."""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'jraph_classification_roc_curves.png')
    plt.savefig(filename, dpi=300)
    print(f"ROC curves saved to {filename}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plotting utility for Jraph Pipeline")
    parser.add_argument("--log", type=str, help="Path to training log file")
    parser.add_argument("--preds", type=str, help="Path to predictions pickle file")
    # Seed argument kept for backwards compatibility but not used in filenames
    parser.add_argument("--output_dir", type=str, default="/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    mode = None
    if args.log:
        epochs, train_loss, train_metric, val_loss, val_metric, mode = parse_logs(args.log)
        if mode:
            plot_training_curves(epochs, train_loss, train_metric, val_loss, val_metric, mode, args.output_dir)
        else:
            print("Could not determine mode from log file.")
        
    if args.preds:
        print(f"Loading predictions from {args.preds}...")
        with open(args.preds, 'rb') as f:
            data = pickle.load(f)
            
        test_mask = np.array(data['test_mask'], dtype=bool)
        
        if 'probs' in data:
            # Classification
            print("Detecting classification data in pickle...")
            probs = np.array(data['probs'])
            preds = np.array(data['preds'])
            targets = np.array(data['targets']) if 'targets' in data else np.array(data['labels'])
            
            test_probs = probs[test_mask]
            test_preds = preds[test_mask]
            test_targets = targets[test_mask]
            
            plot_confusion_matrix(test_targets, test_preds, CLASSES, args.output_dir)
            plot_multiclass_roc(test_targets, test_probs, CLASSES, args.output_dir)
            
            print("\nClassification Report:")
            print(classification_report(test_targets, test_preds, target_names=CLASSES))
        
        elif 'preds_raw' in data or 'preds_eigenvalues' in data:
            # Regression - handle both raw and transformed eigenvalue formats
            print("Detecting regression data in pickle...")
            
            # Check for transformed eigenvalue format first
            if 'preds_eigenvalues' in data:
                print("Using transformed eigenvalue format (preds_eigenvalues)...")
                preds_eig = np.array(data['preds_eigenvalues'])
                targets_eig = np.array(data['targets_eigenvalues'])
            else:
                print("Using raw eigenvalue format (preds_raw)...")
                preds_eig = np.array(data['preds_raw'])
                targets_eig = np.array(data['targets_raw'])
            
            test_preds = preds_eig[test_mask]
            test_targets = targets_eig[test_mask]
            
            plot_regression_parity(test_targets, test_preds, args.output_dir)
            plot_eigenvalue_distributions(test_targets, test_preds, args.output_dir)
            plot_residual_distributions(test_targets, test_preds, args.output_dir)
            analyze_eigenvalue_ordering(test_targets, test_preds, args.output_dir)
            
            # Print mean stats
            mse = np.mean((test_targets - test_preds)**2)
            mae = np.mean(np.abs(test_targets - test_preds))
            print(f"\nRegression Metrics (Test Set):")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")


if __name__ == "__main__":
    main()
