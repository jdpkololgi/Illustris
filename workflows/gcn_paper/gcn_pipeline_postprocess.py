import os
import builtins
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')  # force non-interactive backend for SSH runs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

from gcn_pipeline import load_data, generate_data
from gnn_models import SimpleGNN, SimpleGAT
from utils import train_gcn_full, test_gcn_full, calculate_class_weights, preprocess_features

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


@contextmanager
def _disable_tensorflow_import():
    """UMAP's parametric module imports TensorFlow, which crashes with numpy>=2 on this node."""
    original_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] == "tensorflow":
            raise ImportError("TensorFlow import blocked for UMAP to avoid numpy compatibility issues.")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _guard
    try:
        yield
    finally:
        builtins.__import__ = original_import


def _resolve_umap_class():
    with _disable_tensorflow_import():
        try:
            from umap.umap_ import UMAP as umap_cls
            return umap_cls
        except ImportError:
            try:
                from umap import UMAP as umap_cls
                return umap_cls
            except ImportError as err:
                raise ImportError(
                    "UMAP is unavailable. Install `umap-learn` in the same environment."
                ) from err


def add_density_contours(ax, points, labels, palette, xlim, ylim, grid_size=200):
    """Overlay solid KDE contours per class following Caro et al. (2024) Figure 13.
    This version draws filled contours (contourf) between the chosen level and the peak density,
    plus a stroked contour line for definition.
    """
    labels = np.asarray(labels)
    points = np.asarray(points)
    if points.shape[0] < 3:
        return

    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_positions = np.vstack([xx.ravel(), yy.ravel()])

    for cls_name, color in palette.items():
        cls_mask = labels == cls_name
        if cls_mask.sum() < 5:
            continue
        cls_points = points[cls_mask]
        try:
            kde = gaussian_kde(cls_points.T)
        except np.linalg.LinAlgError:
            continue
        density = kde(grid_positions).reshape(xx.shape)
        mean_density = float(density.mean())
        std_density = float(density.std())
        level = mean_density + std_density
        if not np.isfinite(level):
            continue
        density_max = float(density.max())
        if density_max <= 0:
            continue
        # ensure the fill interval is valid; fall back to a fraction of the max if needed
        if level >= density_max:
            level = mean_density
            if level >= density_max:
                level = 0.5 * density_max
        if level <= 0:
            continue

        # Filled contour between the threshold level and the maximum density
        ax.contourf(xx, yy, density, levels=[level, density_max], colors=[color], alpha=0.35, antialiased=True)

        # Optional: draw an outline at the threshold for clarity
        ax.contour(xx, yy, density, levels=[level], colors=[color], linewidths=1.2, alpha=0.6)


def compute_axis_limits(points, pad_fraction=0.05):
    """Return padded axis limits matching scatter extents."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    def _limits(component):
        min_val = component.min()
        max_val = component.max()
        span = max_val - min_val
        pad = span * pad_fraction if span > 0 else 1.0
        return (min_val - pad, max_val + pad)

    xlim = _limits(pts[:, 0])
    ylim = _limits(pts[:, 1])
    return xlim, ylim


# Ensure consistent font size and style across all plots
FONT_SIZE = 20
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                     'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE})

# Define a single canonical palette used across all plots
custom_palette = {
    'Void': '#4c78a8',     # deep teal
    'Wall': '#a05eb5',     # violet
    'Filament': '#76b7b2', # sky teal
    'Cluster': '#e17c9a'   # plum pink
}

# custom_palette = {
#     'Void': '#80ffdb',  # Void — mint-teal neon (distinct from blue wall)
#     'Wall': '#3a86ff',  # Wall — neon blue
#     'Filament': '#ff006e',  # Filament — hot pink
#     'Cluster': '#ffbe0b'   # Cluster — neon yellow-orange
# }
# also provide easy access by name used elsewhere
classes = ['Void (0)', 'Wall (1)', 'Filament (2)', 'Cluster (3)']
class_colors = [custom_palette['Void'], custom_palette['Wall'], custom_palette['Filament'], custom_palette['Cluster']]

# tell seaborn to use this palette for category plots
sns.set_palette(class_colors)

data, features, targets = load_data(rank=0, distributed=False)
class_weights = calculate_class_weights(targets)
model = SimpleGAT(input_dim=features.shape[1], output_dim=4, num_heads=4)
# model.load_state_dict(torch.load('trained_gat_model_ddp.pth', weights_only=True))
model.load_state_dict(torch.load('trained_gat_model_ddp_2026-01-15.pth', weights_only=True))
test_gcn_full(model, data)
predicted_labels, true_labels, test_probs, _ = test_gcn_full(model, data)

cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plot confusion matrix using a cmap derived from the Cluster color for consistency
cmap = sns.light_palette(custom_palette['Void'], as_cmap=True)
fig_cm = plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes,
                 cbar_kws={'shrink': 0.75}, annot_kws={'fontsize': FONT_SIZE-2})
ax.set_xlabel('Predicted', fontsize=FONT_SIZE)
ax.set_ylabel('True', fontsize=FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
fig_cm.tight_layout()
fig_cm.savefig('gatplus_alpha_confusion_matrix.png', dpi=600)
plt.close(fig_cm)

print(cm)

training_history = pd.read_pickle('training_validation_accuracies_losses_2026-01-15.pkl')

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
fig.savefig('training_validation_alpha_accuracies_losses.png', dpi=600)
plt.close(fig)


stats = classification_report(true_labels, predicted_labels, target_names=classes, output_dict=True)
stats_df = pd.DataFrame(stats).transpose().drop(columns=['support'])
print(stats_df)

# KDE plots: ensure colors use the custom palette and fonts remain consistent
fig_kde = plt.figure(figsize=(10, 6))
sns.kdeplot(features['Mean E.L.'][targets == 0].values, label='Void', fill=True, alpha=0.5, color=custom_palette['Void'])
sns.kdeplot(features['Mean E.L.'][targets == 1].values, label='Wall', fill=True, alpha=0.5, color=custom_palette['Wall'])
sns.kdeplot(features['Mean E.L.'][targets == 2].values, label='Filament', fill=True, alpha=0.5, color=custom_palette['Filament'])
sns.kdeplot(features['Mean E.L.'][targets == 3].values, label='Cluster', fill=True, alpha=0.5, color=custom_palette['Cluster'])
plt.xlim(-4, 4)
plt.xlabel('Mean Edge Length', fontsize=FONT_SIZE)
plt.ylabel('Density', fontsize=FONT_SIZE)
plt.legend(loc='upper left', fontsize=FONT_SIZE)
plt.tick_params(labelsize=FONT_SIZE)
fig_kde.tight_layout()
fig_kde.savefig('mean_edge_length_alpha_distribution.png', dpi=600)
plt.close(fig_kde)

# Calculate mutual information

mi = mutual_info_classif(features, pd.Categorical(targets).codes, random_state=42)
mi = pd.Series(mi, index=features.columns)
mi = mi.rename(index={'I_eig1': '$I_1$', 'I_eig2': '$I_2$', 'I_eig3': '$I_3$'})
mi = mi.sort_values(ascending=False)
fig_mi = plt.figure(figsize=(10,8))
mi.plot.bar(color=custom_palette['Void'], alpha=0.8)
plt.ylabel('Mutual Information', fontsize=FONT_SIZE)
plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
plt.title('Graph Metric Mutual Information with T-WEB Environments', fontsize=FONT_SIZE)
plt.tick_params(axis='both', labelsize=FONT_SIZE)
fig_mi.tight_layout()
fig_mi.savefig('mutual_info_graph_metrics_alpha.png', dpi=600, transparent=True)
plt.close(fig_mi)

# ── Shapley value feature importance ──────────────────────────────────────
# Direct SHAP on the GAT is prohibitively slow because every feature-mask
# evaluation requires a full forward pass through the 284k-node graph.
# Instead we train a fast *surrogate* (GradientBoosting) to mimic the GAT's
# predictions from the same input features, then use TreeExplainer for
# exact Shapley values in seconds.  Surrogate fidelity is reported so we
# can verify it faithfully represents the GAT's decision boundary.
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

feature_names = list(features.columns)
_rename = {'I_eig1': '$I_1$', 'I_eig2': '$I_2$', 'I_eig3': '$I_3$'}
display_names = [_rename.get(n, n) for n in feature_names]

# 1.  Get the GAT's predicted labels for every node
model.eval()
with torch.no_grad():
    _all_logits = model(data.x, data.edge_index, data.edge_attr)
    _gat_preds  = _all_logits.argmax(dim=1).numpy()

_train_mask = data.train_mask.numpy()
_test_mask  = data.test_mask.numpy()

X_train_surr, y_train_surr = features.values[_train_mask], _gat_preds[_train_mask]
X_test_surr,  y_test_surr  = features.values[_test_mask],  _gat_preds[_test_mask]

# 2.  Train the surrogate to replicate the GAT's predictions
#     HistGradientBoosting uses histogram binning — much faster on 200k+ samples
print("Training surrogate HistGradientBoosting classifier …")
surrogate = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
)
surrogate.fit(X_train_surr, y_train_surr)

surr_train_acc = accuracy_score(y_train_surr, surrogate.predict(X_train_surr))
surr_test_acc  = accuracy_score(y_test_surr,  surrogate.predict(X_test_surr))
print(f"Surrogate fidelity — train: {surr_train_acc:.3f}  test: {surr_test_acc:.3f}")

# 3.  TreeExplainer — exact Shapley values, fast
print("Computing SHAP values via TreeExplainer …")
explainer   = shap.TreeExplainer(surrogate)
shap_values = explainer.shap_values(X_test_surr)

# Handle different SHAP return formats:
#   - list of n_classes arrays, each (n_test, n_features)  → standard GBM
#   - single ndarray of shape (n_test, n_features, n_classes) → HistGBM
if isinstance(shap_values, list):
    shap_array = np.stack(shap_values, axis=-1)       # → (n_test, n_feat, n_class)
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_array = shap_values                          # already 3-D
else:
    shap_array = shap_values[:, :, np.newaxis]        # 2-D fallback

print(f"SHAP array shape: {shap_array.shape}  "
      f"(expected ({X_test_surr.shape[0]}, {X_test_surr.shape[1]}, 4))")

# 4a.  Bar plot — mean |SHAP| per feature (averaged over nodes & classes)
mean_abs = np.mean(np.abs(shap_array), axis=(0, 2))
order    = np.argsort(mean_abs)

fig_shap_bar = plt.figure(figsize=(10, 8))
plt.barh(np.arange(len(display_names)), mean_abs[order],
         color=custom_palette['Void'], alpha=0.8)
plt.yticks(np.arange(len(display_names)),
           [display_names[i] for i in order], fontsize=FONT_SIZE)
plt.xlabel('Mean |SHAP value|', fontsize=FONT_SIZE)
plt.title('SHAP Feature Importance for GAT Classifier', fontsize=FONT_SIZE)
plt.tick_params(axis='both', labelsize=FONT_SIZE)
fig_shap_bar.tight_layout()
fig_shap_bar.savefig('shap_bar_graph_metrics_alpha.png', dpi=600, transparent=True)
plt.close(fig_shap_bar)

# 4b.  Beeswarm plot per class
class_names_shap = ['Void', 'Wall', 'Filament', 'Cluster']
for c_idx, c_name in enumerate(class_names_shap):
    explanation = shap.Explanation(
        values=shap_array[:, :, c_idx],
        base_values=explainer.expected_value[c_idx] * np.ones(shap_array.shape[0]),
        data=X_test_surr,
        feature_names=display_names,
    )
    fig_bee = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)
    plt.title(f'SHAP Beeswarm — {c_name}', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    fig_bee.tight_layout()
    fig_bee.savefig(f'shap_beeswarm_{c_name.lower()}_alpha.png', dpi=600, transparent=True)
    plt.close(fig_bee)

print("SHAP plots saved.")

# UMAP visualization of learned features. 3e projections colored by true and predicted labels
UMAP = _resolve_umap_class()

reducer = UMAP(random_state=42, n_components=3)
embeddings = reducer.fit_transform(data.x.numpy())

# create targets2 array mapping 0,1,2,3 to 'Void (0)', etc for legend purposes
_mapping = {0: 'Void', 1: 'Wall', 2: 'Filament', 3: 'Cluster'}
# handle torch tensors or array-like inputs

targets2 = np.array([_mapping[int(t)] for t in targets])    
# test_predictions_labels_probs.pkl
test_probs = pd.read_pickle('test_predictions_labels_probs_2026-01-15.pkl')['probs'].detach().cpu().numpy()
# UMAP of learned features colored by true labels
node_embeddings = pd.read_pickle('node_embeddings_2026-01-15.pkl').detach().cpu().numpy()
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

fig_grid = plt.figure(figsize=(fig_w, fig_h), dpi=300)
# place grid with explicit margins so cells have equal allocation
top, bottom, left, right = 0.95, 0.04, 0.06, 0.99
hspace, wspace = 0.25, 0.18
gs = fig_grid.add_gridspec(n_rows, n_cols, left=left, right=right, top=top, bottom=bottom,
                           hspace=hspace, wspace=wspace)

axes = [[fig_grid.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

marker_size = 9
alpha_const = 0.75  # constant alpha for visibility

# prediction entropy for size (apply only to test nodes / bottom row)
entropy = -np.sum(test_probs * np.log(test_probs + 1e-12), axis=1)
normalised_entropy = entropy / np.log(test_probs.shape[1])  # in [0,1]
conf = 1 - normalised_entropy  # confidence in [0,1]
from matplotlib.cm import get_cmap
cmap_grey = get_cmap("Greys")
cmap_grey_r = get_cmap("Greys_r")
edgecols = cmap_grey(conf)  # darker for lower entropy
edgecols_r = cmap_grey_r(conf)  # darker for higher entropy
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

fig_grid.legend(legend_handles,
                legend_labels,
                loc='upper center',
                ncol=len(class_labels) + 2,
                fontsize=FONT_SIZE,
                bbox_to_anchor=(0.5, 1.02),
                frameon=False)
# leave headroom for the legend
fig_grid.tight_layout(rect=[0, 0, 1, 0.92])
plt.close(fig_grid)

# # after plotting the 3x3 subplots, add centered titles for each row
# row_titles = [
#     'Learned features: s = -0.0008477559',
#     'GAT embeddings (true labels): s = 0.10818577',
#     'GAT embeddings (predicted labels): s = 0.16120291'
# ]
# for r in range(n_rows):
#     # compute vertical center of the r-th row in figure coordinates using the same top/bottom used by gridspec
#     y_center = top - (r + 0.5) * (top - bottom) / n_rows
#     fig_grid.text(0.5, y_center+0.15, row_titles[r],
#              ha='center', va='center', fontsize=FONT_SIZE + 2, weight='bold')

# Bottom-row standalone figure: all three UMAP projections for test galaxies
# plt.style.use(['dark_background', 'science', 'no-latex'])
plt.style.use(['science', 'no-latex'])
from scipy.ndimage import gaussian_filter


pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = np.array([custom_palette[lbl] for lbl in predicted_labels_test])
for ax, (x, y) in zip(axes, pairs):

    plane = z_test[:, [x, y]]
    xlim, ylim = compute_axis_limits(plane)
    ax.scatter(
        plane[:, 0], plane[:, 1],
        c=colors,
        s=marker_size,            # size by entropy-derived areas
        edgecolor=edgecols, # edge encodes entropy
        linewidths=0.2,
        alpha=0.8,  # alpha inversely proportional to confidence
        rasterized=True
    )
    ax.set_xlabel(f'UMAP {x+1}', fontsize=FONT_SIZE)
    ax.set_ylabel(f'UMAP {y+1}', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.set_facecolor('none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    add_density_contours(ax, plane, predicted_labels_test, custom_palette, xlim, ylim)

# Move legend above plots and reserve space
fig.legend(legend_handles,
           legend_labels,
           loc='upper center',
           ncol=len(class_labels) + 2,
           fontsize=FONT_SIZE,
           frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('umap_gat_embeddings_test_predictions.pdf', dpi=600, transparent=True)
plt.close(fig)

# entropy distribution by environment
fig_entropy = plt.figure(figsize=(18, 12))
for k, name in enumerate(['Void','Wall','Filament','Cluster']):
    mask = predicted_labels_test == name
    print(f"{name:10s}: mean entropy = {entropy[mask].mean():.3f}")
    sns.kdeplot(entropy[mask], label=name, color=class_colors[k], fill=True, alpha=0.5)
    
plt.xlabel('Prediction Entropy', fontsize=FONT_SIZE)
plt.ylabel('Density', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.tick_params(labelsize=FONT_SIZE)
fig_entropy.tight_layout()
plt.close(fig_entropy)

pairs = [(0, 1), (0, 2), (1, 2)]
single_color = 'r'  # neutral color for entropy row

fig_comb, axes_comb = plt.subplots(2, 3, figsize=(18, 12), dpi=300)

# Row 0: color by predicted class (uniform size)
colors = np.array([custom_palette[lbl] for lbl in predicted_labels_test])
for ax, (x, y) in zip(axes_comb[0], pairs):
    plane = z_test[:, [x, y]]
    xlim, ylim = compute_axis_limits(plane)
    # ax.scatter(
    #     plane[:, 0], plane[:, 1],
    #     c=colors,
    #     s=marker_size,
    #     edgecolor='none',
    #     alpha=alpha_const,
    #     rasterized=True
    # )
    ax.set_xlabel(f'UMAP {x+1}', fontsize=20)
    ax.set_ylabel(f'UMAP {y+1}', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_facecolor('none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    add_density_contours(ax, plane, predicted_labels_test, custom_palette, xlim, ylim)

# Row 1: single color, size encodes entropy (areas already computed) + entropy contours
for ax, (x, y) in zip(axes_comb[1], pairs):
    plane = z_test[:, [x, y]]
    xlim, ylim = compute_axis_limits(plane)

    # scatter: single color, size by entropy-derived areas
    ax.scatter(
        plane[:, 0], plane[:, 1],
        color='white',
        s=marker_size,               # size encodes entropy (higher entropy -> larger area if you used conf, invert if needed)
        edgecolor=edgecols_r,
        alpha=0.3,
        rasterized=True
    )


    ax.set_xlabel(f'UMAP {x+1}', fontsize=20)
    ax.set_ylabel(f'UMAP {y+1}', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_facecolor('none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# # Row titles
# fig_comb.text(0.5, 0.97, 'Test galaxies: predicted classes', ha='center', va='top',
#               fontsize=20+2, weight='bold')
# fig_comb.text(0.5, 0.49, 'Test galaxies: entropy (marker size)', ha='center', va='top',
#               fontsize=20+2, weight='bold')

# Legends (classes + entropy size proxies) — use square markers to match contour fills
class_labels = list(custom_palette.keys())
class_colors = [custom_palette[k] for k in class_labels]
class_handles = [
    Line2D([0], [0], marker='s', linestyle='None',
           markerfacecolor=class_colors[i], markeredgecolor='k',
           markersize=14)
    for i in range(len(class_labels))
]

entropy_handles = [
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor=edge_high,
           alpha=alpha_const, markersize=legend_class_size),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor=edge_low,
           alpha=alpha_const, markersize=legend_class_size),
]
entropy_labels = ['Low entropy', 'High entropy']

combined_handles = class_handles + entropy_handles
combined_labels = class_labels + entropy_labels

fig_comb.legend(combined_handles, combined_labels, ncol=len(class_labels) + 2,
                fontsize=20, loc='upper center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('umap_gat_embeddings_alpha_test_predictions.pdf', dpi=600)
plt.close(fig_comb)
