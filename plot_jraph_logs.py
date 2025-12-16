import re
import argparse
import pickle
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
    """Parses the training log file."""
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    
    # Regex pattern: Epoch X | Train Loss: Y | Train Acc: Z% | Val Loss: A | Val Acc: B%
    pattern = re.compile(
        r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)% \| Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)%"
    )
    
    print(f"Reading log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                e, tl, ta, vl, va = match.groups()
                epochs.append(int(e))
                train_loss.append(float(tl))
                train_acc.append(float(ta))
                val_loss.append(float(vl))
                val_acc.append(float(va))
                
    return np.array(epochs), np.array(train_loss), np.array(train_acc), np.array(val_loss), np.array(val_acc)

def plot_training_curves(epochs, train_loss, train_acc, val_loss, val_acc, seed):
    """Plots training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='#FF6B6B', alpha=0.9)
    ax1.plot(epochs, val_loss, label='Val Loss', color='#4ECDC4', alpha=0.9)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, max(epochs)+50)
    ax1.legend(fontsize=10)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, train_acc, label='Train Acc', color='#FF6B6B', alpha=0.9)
    ax2.plot(epochs, val_acc, label='Val Acc', color='#4ECDC4', alpha=0.9)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, max(epochs)+50)
    ax2.legend(fontsize=10)
    
    # Final Stats
    if len(epochs) > 0:
        final_stats = (
            f"Final Stats (Epoch {epochs[-1]}):\n"
            f"Train Acc: {train_acc[-1]:.2f}%\n"
            f"Val Acc:   {val_acc[-1]:.2f}%"
        )
        props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
        ax2.text(0.7, 0.2, final_stats, transform=ax2.transAxes, fontsize=11,
                verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    filename = f'jraph_training_curves_{seed}.png'
    plt.savefig(filename, dpi=150)
    print(f"Training curves saved to {filename}")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, classes, seed):
    """Plots normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cmap = sns.light_palette('#4c78a8', as_cmap=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes,
                cbar_kws={'shrink': 0.75}, annot_kws={'fontsize': 12})
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(f'Confusion Matrix (Seed {seed})', fontsize=16)
    plt.tight_layout()
    
    filename = f'jraph_confusion_matrix_seed_{seed}.png'
    plt.savefig(filename, dpi=300)
    print(f"Confusion matrix saved to {filename}")
    plt.close(fig)

def plot_multiclass_roc(y_true, y_probs, classes, seed):
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
    plt.title(f'ROC Curves (Seed {seed})', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f'jraph_roc_curves_seed_{seed}.png'
    plt.savefig(filename, dpi=300)
    print(f"ROC curves saved to {filename}")
    plt.close(fig)

def plot_prob_distributions(y_true, y_probs, classes, seed):
    """Plots distribution of predicted probabilities for the correct class."""
    n_classes = len(classes)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, ax in enumerate(axes):
        mask = (y_true == i)
        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, "No samples", ha='center')
            continue
            
        probs_i = y_probs[mask, i]
        
        sns.histplot(probs_i, bins=20, kde=True, ax=ax, color=colors[i], stat='density')
        ax.set_title(f'True Class: {classes[i]}', fontsize=14)
        ax.set_xlabel(f'Predicted Probability for {classes[i]}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(0, 1)
        
    plt.suptitle(f"Probability Distributions (Seed {seed})", fontsize=16)
    plt.tight_layout()
    
    filename = f'jraph_prob_dists_seed_{seed}.png'
    plt.savefig(filename, dpi=300)
    print(f"Probability distributions saved to {filename}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plotting utility for Jraph Pipeline")
    parser.add_argument("--log", type=str, help="Path to training log file")
    parser.add_argument("--preds", type=str, help="Path to predictions pickle file")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for naming files")
    
    args = parser.parse_args()
    
    if args.log:
        epochs, train_loss, train_acc, val_loss, val_acc = parse_logs(args.log)
        plot_training_curves(epochs, train_loss, train_acc, val_loss, val_acc, args.seed)
        
    if args.preds:
        print(f"Loading predictions from {args.preds}...")
        with open(args.preds, 'rb') as f:
            data = pickle.load(f)
            
        probs = data['probs']
        preds = data['preds']
        labels = data['labels']
        test_mask = data['test_mask']
        
        # Filter for Test set - assuming data saved contains full graph arrays
        # The pickle from jraph_pipeline saves: 'probs', 'preds', 'labels', 'test_mask' (all length N)
        # We need to filter them using test_mask
        
        # Convert to numpy if they are jax arrays (pickle loads them as such usually)
        # Note: They should be numpy arrays already as per jraph_pipeline saving code
        probs = np.array(probs)
        preds = np.array(preds)
        labels = np.array(labels)
        test_mask = np.array(test_mask, dtype=bool)
        
        test_probs = probs[test_mask]
        test_preds = preds[test_mask]
        test_labels = labels[test_mask]
        
        print(f"Generating plots for {len(test_labels)} test samples...")
        
        # plots
        plot_confusion_matrix(test_labels, test_preds, CLASSES, args.seed)
        plot_multiclass_roc(test_labels, test_probs, CLASSES, args.seed)
        plot_prob_distributions(test_labels, test_probs, CLASSES, args.seed)
        
        # Print report
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=CLASSES))

if __name__ == "__main__":
    main()
