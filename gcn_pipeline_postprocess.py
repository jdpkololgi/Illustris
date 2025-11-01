import os

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

from gcn_pipeline import load_data, generate_data
from gnn_models import SimpleGNN, SimpleGAT
from utils import train_gcn_full, test_gcn_full, calculate_class_weights, preprocess_features

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


# Ensure consistent font size and style across all plots
FONT_SIZE = 20
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                     'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE})

# Define a single canonical palette used across all plots
# custom_palette = {
#     'Void': '#4c78a8',     # deep teal
#     'Wall': '#a05eb5',     # violet
#     'Filament': '#76b7b2', # sky teal
#     'Cluster': '#e17c9a'   # plum pink
# }

custom_palette = {
    'Void': '#80ffdb',  # Void — mint-teal neon (distinct from blue wall)
    'Wall': '#3a86ff',  # Wall — neon blue
    'Filament': '#ff006e',  # Filament — hot pink
    'Cluster': '#ffbe0b'   # Cluster — neon yellow-orange
}
# also provide easy access by name used elsewhere
classes = ['Void (0)', 'Wall (1)', 'Filament (2)', 'Cluster (3)']
class_colors = [custom_palette['Void'], custom_palette['Wall'], custom_palette['Filament'], custom_palette['Cluster']]

# tell seaborn to use this palette for category plots
sns.set_palette(class_colors)

data, features, targets = load_data(rank=0, distributed=False)
class_weights = calculate_class_weights(targets)
model = SimpleGAT(input_dim=features.shape[1], output_dim=4, num_heads=4)
model.load_state_dict(torch.load('trained_gat_model_ddp.pth', weights_only=True))
test_gcn_full(model, data)
predicted_labels, true_labels, test_probs, _ = test_gcn_full(model, data)

cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plot confusion matrix using a cmap derived from the Cluster color for consistency
cmap = sns.light_palette(custom_palette['Void'], as_cmap=True)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes,
                 cbar_kws={'shrink': 0.75}, annot_kws={'fontsize': FONT_SIZE-2})
ax.set_xlabel('Predicted', fontsize=FONT_SIZE)
ax.set_ylabel('True', fontsize=FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
plt.tight_layout()
plt.savefig('gatplus_confusion_matrix.png', dpi=600)
plt.show()

print(cm)

training_history = pd.read_pickle('training_validation_accuracies_losses.pkl')

# plot training history with consistent colors and font sizes
fig, ax = plt.subplots(1, 2, figsize=(20, 6))  # larger so font sizes remain consistent

# choose two distinct colors from the palette for train / val curves
train_color = custom_palette['Void']
val_color = custom_palette['Cluster']

# Accuracy subplot
ax[0].plot(training_history['train_acc'], alpha=0.9, label='Training Accuracy', color=train_color, linewidth=2)
ax[0].plot(training_history['val_acc'], alpha=0.9, label='Validation Accuracy', color=val_color, linewidth=2)
ax[0].set_ylim(0, 100)
ax[0].set_xlabel('Epoch', fontsize=FONT_SIZE)
ax[0].set_ylabel('Accuracy', fontsize=FONT_SIZE)
ax[0].tick_params(labelsize=FONT_SIZE)
ax[0].legend(loc='best', fontsize=FONT_SIZE)

# Loss subplot
ax[1].plot(training_history['train_loss'], alpha=0.9, label='Training Loss', color=train_color, linewidth=2)
ax[1].plot(training_history['val_loss'], alpha=0.9, label='Validation Loss', color=val_color, linewidth=2)
ax[1].set_ylim(0, 2.5)
ax[1].set_xlabel('Epoch', fontsize=FONT_SIZE)
ax[1].set_ylabel('Loss', fontsize=FONT_SIZE)
ax[1].tick_params(labelsize=FONT_SIZE)
ax[1].legend(loc='best', fontsize=FONT_SIZE)

plt.tight_layout()
fig.savefig('training_validation_accuracies_losses.png', dpi=600)
plt.show()


stats = classification_report(true_labels, predicted_labels, target_names=classes, output_dict=True)
stats_df = pd.DataFrame(stats).transpose().drop(columns=['support'])
print(stats_df)

# KDE plots: ensure colors use the custom palette and fonts remain consistent
plt.figure(figsize=(10, 6))
sns.kdeplot(features['Mean E.L.'][targets == 0].values, label='Void', fill=True, alpha=0.5, color=custom_palette['Void'])
sns.kdeplot(features['Mean E.L.'][targets == 1].values, label='Wall', fill=True, alpha=0.5, color=custom_palette['Wall'])
sns.kdeplot(features['Mean E.L.'][targets == 2].values, label='Filament', fill=True, alpha=0.5, color=custom_palette['Filament'])
sns.kdeplot(features['Mean E.L.'][targets == 3].values, label='Cluster', fill=True, alpha=0.5, color=custom_palette['Cluster'])
plt.xlim(-4, 4)
plt.xlabel('Mean Edge Length', fontsize=FONT_SIZE)
plt.ylabel('Density', fontsize=FONT_SIZE)
plt.legend(loc='upper left', fontsize=FONT_SIZE)
plt.tick_params(labelsize=FONT_SIZE)
plt.tight_layout()
plt.savefig('mean_edge_length_distribution.png', dpi=600)
plt.show()

# Calculate mutual information

mi = mutual_info_classif(features, pd.Categorical(targets).codes, random_state=42)
mi = pd.Series(mi, index=features.columns)
mi = mi.sort_values(ascending=False)
plt.figure(figsize=(10,8))
mi.plot.bar(color=custom_palette['Void'], alpha=0.8)
plt.ylabel('Mutual Information', fontsize=FONT_SIZE)
plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
plt.title('Graph Metric Mutual Information with T-WEB Environments', fontsize=FONT_SIZE)
plt.tick_params(axis='both', labelsize=FONT_SIZE)
plt.tight_layout()
plt.savefig('mutual_info_graph_metrics.png', dpi=600, transparent=True)
plt.show()

# UMAP visualization of learned features. 3e projections colored by true and predicted labels
# Note: UMAP import is ignored in the InteractiveInput-1 file as per user instruction
try:
    from umap.umap_ import UMAP  # preferred: avoids TF entirely
except Exception:
    # last resort fallback (only if you later fix TF)
    from umap import UMAP

reducer = UMAP(random_state=42, n_components=3)
embeddings = reducer.fit_transform(data.x.numpy())

# create targets2 array mapping 0,1,2,3 to 'Void (0)', etc for legend purposes
_mapping = {0: 'Void', 1: 'Wall', 2: 'Filament', 3: 'Cluster'}
# handle torch tensors or array-like inputs

targets2 = np.array([_mapping[int(t)] for t in targets])    
# test_predictions_labels_probs.pkl
test_probs = pd.read_pickle('test_predictions_labels_probs.pkl')['probs'].detach().cpu().numpy()
# UMAP of learned features colored by true labels
node_embeddings = pd.read_pickle('node_embeddings.pkl').detach().cpu().numpy()
z = reducer.fit_transform(node_embeddings)

# UMAP of GAT embeddings colored by true labels

# UMAP of GAT embeddings colored by predicted labels
# predicted_labels2 = np.array([_mapping[int(t)] for t in predicted_labels])
# plot_umap_scatter(z, predicted_labels2, 'umap_GAT_embeddings_predicted_labels.png', custom_palette)

# select test nodes and plot only those embeddings
test_mask = data.test_mask.cpu().numpy() if isinstance(data.test_mask, torch.Tensor) else np.asarray(data.test_mask)
z_test = z[test_mask]

predicted_labels_test = np.array([_mapping[int(t)] for t in predicted_labels])  # length == z_test.shape[0]

# rows: 0 = learned features (embeddings, targets2)
#       1 = GAT embeddings (z, targets2)
#       2 = GAT embeddings (test-only) (z_test, predicted_labels_test)
row_embeddings = [embeddings, z, z_test]
row_labels = [np.asarray(targets2), np.asarray(targets2), np.asarray(predicted_labels_test)]

pairs = [(0, 1), (0, 2), (1, 2)]





# ensure each subplot gets equal physical size: keep per-panel size and build gridspec
n_rows, n_cols = 3, 3
per_panel_w, per_panel_h = 6, 6   # each panel size (inches) — adjust to taste
fig_w, fig_h = n_cols * per_panel_w, n_rows * per_panel_h

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
# place grid with explicit margins so cells have equal allocation
top, bottom, left, right = 0.95, 0.04, 0.06, 0.99
hspace, wspace = 0.25, 0.18
gs = fig.add_gridspec(n_rows, n_cols, left=left, right=right, top=top, bottom=bottom,
                      hspace=hspace, wspace=wspace)

axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

marker_size = 9
alpha_const = 0.75  # constant alpha for visibility

# prediction entropy for size (apply only to test nodes / bottom row)
entropy = -np.sum(test_probs * np.log(test_probs + 1e-12), axis=1)
normalised_entropy = entropy / np.log(test_probs.shape[1])  # in [0,1]
conf = 1 - normalised_entropy  # confidence in [0,1]
from matplotlib.cm import get_cmap
cmap_grey = get_cmap("Greys")
edgecols = cmap_grey(conf)  # darker for lower entropy
cmap_greens = get_cmap("Greens")
edgecols_green = cmap_greens(conf)  # darker for higher confidence

# map confidence to marker area (matplotlib 's' is area). tune min/max for visibility.
min_area = 6   # small visible dot
max_area = 100 # large visible dot for highest confidence
areas = min_area + conf * (max_area - min_area)  # shape == n_test_nodes







# Controls for performance with many points
max_points_per_class_top_mid = 20000   # None to disable subsampling
rasterize_top_mid = True
_rng = np.random.default_rng(42)

for row in range(n_rows):
    emb = row_embeddings[row]
    labels = row_labels[row]

    if emb.ndim != 2 or emb.shape[1] < 3:
        raise ValueError(f"embeddings for row {row} must have shape (N,3), got {emb.shape}")
    if labels.shape[0] != emb.shape[0]:
        raise ValueError(f"label length ({labels.shape[0]}) != embeddings rows ({emb.shape[0]}) for row {row}")

    for col, (x, y) in enumerate(pairs):
        ax = axes[row][col]
        ax.clear()

        # Compute subplot extents once
        xdata, ydata = emb[:, x], emb[:, y]
        xlim = np.percentile(xdata, [0, 100])
        ylim = np.percentile(ydata, [0, 100])

        for cls_name in custom_palette.keys():
            mask = (labels == cls_name)
            if not np.any(mask):
                continue

            idx = np.flatnonzero(mask)

            # Top/middle: optional per-class subsampling and rasterized scatter for speed
            if row in (0, 1) and max_points_per_class_top_mid is not None and idx.size > max_points_per_class_top_mid:
                idx = _rng.choice(idx, size=max_points_per_class_top_mid, replace=False)

            if row == 2:
                # Bottom row: keep confidence edgecolor; use current marker sizing
                ax.scatter(
                    emb[idx, x], emb[idx, y],
                    color=custom_palette[cls_name],
                    s=marker_size,               # keep or switch to `areas[idx]` if you prefer size by confidence
                    edgecolor=edgecols[idx],
                    alpha=alpha_const,
                    linewidths=0.2,
                    label=cls_name if (row == 0 and col == 0) else None
                )
            else:
                # Fast scatter for dense rows
                ax.scatter(
                    emb[idx, x], emb[idx, y],
                    color=custom_palette[cls_name],
                    s=marker_size,
                    alpha=alpha_const,
                    edgecolor='none',
                    rasterized=rasterize_top_mid,
                    label=cls_name if (row == 0 and col == 0) else None
                )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f'UMAP {x+1}', fontsize=FONT_SIZE)
        ax.set_ylabel(f'UMAP {y+1}', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        ax.set_aspect('auto')  # panels remain same physical size via gridspec

# optional: add a clearer legend where class markers are larger and confidence proxies
# reflect the use of edgecolor to indicate entropy (low/high confidence).
# Build explicit legend handles (more control than using scatter handles).
class_labels = list(custom_palette.keys())
class_colors = [custom_palette[k] for k in class_labels]

# legend marker sizes (markersize is in points, not area)
legend_class_size = 12
legend_min_size = max(6, np.sqrt(min_area) * 1.8)
legend_max_size = max(10, np.sqrt(max_area) * 1.8)

# class handles: filled marker with black edge for contrast
class_handles = [
    Line2D([0], [0],
           marker='o',
           color='#4c78a8',
           markerfacecolor=class_colors[i],
           markeredgecolor='k',
           markersize=legend_class_size,
           lw=0)
    for i in range(len(class_labels))
]

# confidence proxies: empty face (or very faint face) with edgecolor showing low/high entropy
edge_low = cmap_grey(0.7)   # light grey edge -> low confidence (high entropy)
edge_high = cmap_grey(0.05) # dark edge  -> high confidence (low entropy)

size_proxies = [
    Line2D([0], [0],
           marker='o',
           color='k',
           markerfacecolor='none',
           markeredgecolor=edge_low,
           markersize=legend_min_size,
           lw=1),
    Line2D([0], [0],
           marker='o',
           color='k',
           markerfacecolor='none',
           markeredgecolor=edge_high,
           markersize=legend_max_size,
           lw=1)
]

# Combine and draw legend centered above the figure
legend_handles = class_handles + size_proxies
legend_labels = class_labels + ['Low Entropy', 'High Entropy']

fig.legend(legend_handles,
           legend_labels,
           loc='upper center',
           ncol=len(class_labels) + 2,
           fontsize=FONT_SIZE,
           bbox_to_anchor=(0.5, 1.02),
           frameon=False)
# leave headroom for the legend
plt.tight_layout(rect=[0, 0, 1, 0.92])

# # after plotting the 3x3 subplots, add centered titles for each row
# row_titles = [
#     'Learned features: s = -0.0008477559',
#     'GAT embeddings (true labels): s = 0.10818577',
#     'GAT embeddings (predicted labels): s = 0.16120291'
# ]
# for r in range(n_rows):
#     # compute vertical center of the r-th row in figure coordinates using the same top/bottom used by gridspec
#     y_center = top - (r + 0.5) * (top - bottom) / n_rows
#     fig.text(0.5, y_center+0.15, row_titles[r],
#              ha='center', va='center', fontsize=FONT_SIZE + 2, weight='bold')

# Bottom-row standalone figure: all three UMAP projections for test galaxies
plt.style.use(['dark_background', 'science', 'no-latex'])
pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
colors = np.array([custom_palette[lbl] for lbl in predicted_labels_test])
for ax, (x, y) in zip(axes, pairs):

    # robust panel limits
    xlim = np.percentile(z_test[:, x], [1, 99])
    ylim = np.percentile(z_test[:, y], [1, 99])
    ax.scatter(
        z_test[:, x], z_test[:, y],
        c=colors,
        s=marker_size,            # size by entropy-derived areas
        edgecolor=None, # edge encodes entropy
        alpha=conf,
        linewidths=0.2,
        alpha=alpha_const,
        rasterized=True
    )
    ax.set_xlabel(f'UMAP {x+1}', fontsize=FONT_SIZE)
    ax.set_ylabel(f'UMAP {y+1}', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)

# Move legend above plots and reserve space
fig.legend(legend_handles,
           legend_labels,
           loc='upper center',
           ncol=len(class_labels) + 2,
           fontsize=FONT_SIZE,
           frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.90])


# entropy distribution by environment
fig = plt.figure(figsize=(7, 5), dpi=300)
for k, name in enumerate(['Void','Wall','Filament','Cluster']):
    mask = predicted_labels_test == name
    print(f"{name:10s}: mean entropy = {entropy[mask].mean():.3f}")
    sns.kdeplot(entropy[mask], label=name, color=class_colors[k], fill=True, alpha=0.5)
    
plt.xlabel('Prediction Entropy', fontsize=FONT_SIZE)
plt.ylabel('Density', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.tick_params(labelsize=FONT_SIZE)
plt.tight_layout()

