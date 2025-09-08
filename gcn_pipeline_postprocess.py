import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt

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
custom_palette = {
    'Void': '#4c78a8',     # deep teal
    'Wall': '#a05eb5',     # violet
    'Filament': '#76b7b2', # sky teal
    'Cluster': '#e17c9a'   # plum pink
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
import umap
reducer = umap.UMAP(random_state=42, n_components=3)
embeddings = reducer.fit_transform(data.x.numpy())

# create targets2 array mapping 0,1,2,3 to 'Void (0)', etc for legend purposes
_mapping = {0: 'Void', 1: 'Wall', 2: 'Filament', 3: 'Cluster'}
# handle torch tensors or array-like inputs

targets2 = np.array([_mapping[int(t)] for t in targets])

fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
    # iterate over class names (keys) so masking and color lookup align
    for cls_name in custom_palette.keys():
        mask = targets2 == cls_name
        ax[i].scatter(
            embeddings[mask, x], embeddings[mask, y],
            color=custom_palette[cls_name],
            s=1, alpha=0.5, label=cls_name if i == 0 else None
        )
    ax[i].set_xlabel(f'UMAP {x+1}', fontsize=FONT_SIZE)
    ax[i].set_ylabel(f'UMAP {y+1}', fontsize=FONT_SIZE)
    ax[i].tick_params(axis='both', labelsize=FONT_SIZE)

# Collect handles/labels from the first subplot and place a single horizontal legend above all subplots
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=FONT_SIZE, bbox_to_anchor=(0.5, 1.03), markerscale=10)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('umap_learned_features_true_labels.png', dpi=600)