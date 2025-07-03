import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Network_stats import network

# Import custom modules
from gnn_models import SimpleGNN, SimpleGAT
from utils import train_gcn_full, test_gcn_full, calculate_class_weights, preprocess_features

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_data(masscut=1e9):
    """
    Load and preprocess data for GCN training.
    """
    print("Loading data...")
    testcat = network(masscut=masscut)
    testcat.cweb_classify(xyzplot=False)
    testcat.network_stats_delaunay(buffer=False) # For GNN models removing buffer region can only be done after pyg geom object created
    
    # Get the final dataset after all processing
    features = testcat.data.iloc[:, :-1]
    targets = testcat.data.iloc[:, -1]

    # Scale features
    features = preprocess_features(features)

    # Convert to PyTorch Geometric Data object
    netx_geom = from_networkx(testcat.subhalo_delauany_network(xyzplot=False), group_edge_attrs='all')
    netx_geom.x = torch.tensor(features.values, dtype=torch.float32)
    netx_geom.y = torch.tensor(targets.values, dtype=torch.long)
    print(netx_geom.num_nodes, netx_geom.num_edges, netx_geom.num_node_features, netx_geom.num_edge_features)

    # removing 10Mpc buffer region from training to prevent edge galaxies biasing training
    buffered_indices_for_mask = np.where((netx_geom.pos[:,0]>10) & (netx_geom.pos[:,0]<290) & (netx_geom.pos[:,1]>10) & (netx_geom.pos[:,1]<290) & (netx_geom.pos[:,2]>10) & (netx_geom.pos[:,2]<290))[0]
    anti_buffered_indices_for_mask = np.where((netx_geom.pos[:,0]<10) | (netx_geom.pos[:,0]>290) | (netx_geom.pos[:,1]<10) | (netx_geom.pos[:,1]>290) | (netx_geom.pos[:,2]<10) | (netx_geom.pos[:,2]>290))[0]
    print(len(buffered_indices_for_mask), len(anti_buffered_indices_for_mask), len(buffered_indices_for_mask)+len(anti_buffered_indices_for_mask))
    
    # Now perform train/test split on the final dataset
    train_x, test_x, train_y, test_y = train_test_split(features.iloc[buffered_indices_for_mask], targets.iloc[buffered_indices_for_mask], test_size=0.3, random_state=42, stratify=targets.iloc[buffered_indices_for_mask])
    valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=0.3, random_state=42, stratify=test_y)

    # Create masks of lenth num_nodes
    train_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)
    valid_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)
    test_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)

    # set indices of train/valid/test to True
    train_mask[train_x.index.values] = True
    valid_mask[valid_x.index.values] = True
    test_mask[test_x.index.values] = True

    print(
        'number counts for each class for train/valid/test sets',
        train_y.value_counts(), 
        valid_y.value_counts(), 
        test_y.value_counts()
    )

    netx_geom.train_mask = train_mask
    netx_geom.val_mask = valid_mask
    netx_geom.test_mask = test_mask

    print('PyG Data object populted: ', netx_geom)

    return netx_geom, features, targets

def train_and_evaluate(model, data, class_weights, num_epochs=3000, lr=3e-3):
    """
    Train and evaluate the GCN model.
    """
    model = model.to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=200, threshold=1.5e-3, cooldown=50, min_lr=5e-4)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        loss, val_loss_epoch, acc, val_acc_epoch, _ = train_gcn_full(model, data, optimizer, criterion)
        train_loss.append(loss.item())
        val_loss.append(val_loss_epoch.item())
        train_acc.append(acc)
        val_acc.append(val_acc_epoch)

        scheduler.step(val_loss_epoch.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss:.4f} - Val Loss: {val_loss_epoch:.4f} - Train Acc: {acc:.2f}% - Val Acc: {val_acc_epoch:.2f}%")

    predicted, labels, probs, _ = test_gcn_full(model, data)
    return train_loss, val_loss, train_acc, val_acc, predicted, labels, probs

if __name__ == "__main__":
    # Load data
    data, features, targets = load_data()

    # Calculate class weights
    class_weights = calculate_class_weights(targets)

    # Initialize model
    model = SimpleGAT(input_dim=features.shape[1], output_dim=4, num_heads=4)

    # Train and evaluate
    train_loss, val_loss, train_acc, val_acc, predicted, labels, probs = train_and_evaluate(
        model, data, class_weights, num_epochs=3000, lr=3e-3
    )

    # Save model
    torch.save(model.state_dict(), 'trained_gat_model.pth')
    print("Model saved as 'trained_gat_model.pth'")