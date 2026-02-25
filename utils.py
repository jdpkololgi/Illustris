import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import PowerTransformer
from sklearn.utils.class_weight import compute_class_weight

def preprocess_features(features, save_scaler=False):
    """
    Scale features using PowerTransformer.
    """
    scaler = PowerTransformer(method='box-cox')
    # Add epsilon to handle zero or negative values as done for DESI
    features_scales = scaler.fit_transform(features + 1e-6)

    if save_scaler:
        torch.save(scaler, 'features_scaler.pkl')
        
    return pd.DataFrame(features_scales, index=features.index, columns=features.columns)

def calculate_class_weights(targets):
    """
    Calculate class weights for imbalanced datasets.
    """
    targets_copy = np.copy(targets)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets_copy), y=targets_copy)
    return torch.tensor(class_weights, dtype=torch.float32)

# Create scaler without device specification for DDP compatibility
scaler = GradScaler()

def train_gcn_full(model, data, optimizer, criterion):
    """
    Train GCN model in full-batch mode with mixed precision and distributed training compatibility.
    """
    model.train()
    optimizer.zero_grad()

    # Mixed precision training
    with autocast('cuda'):
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
    
    scaler.scale(loss).backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()

    # Calculate training accuracy
    train_probs = output[data.train_mask]
    _, predicted = torch.max(train_probs, 1)
    correct = (predicted == data.y[data.train_mask]).sum().item()
    total = data.y[data.train_mask].size(0)
    train_acc = 100 * correct / total if total > 0 else 0

    # Validation accuracy and loss
    model.eval()
    with torch.no_grad():
        val_output = model(data.x, data.edge_index, data.edge_attr)
        val_loss = criterion(val_output[data.val_mask], data.y[data.val_mask])
        val_probs = val_output[data.val_mask]
        _, val_predicted = torch.max(val_probs, 1)
        val_correct = (val_predicted == data.y[data.val_mask]).sum().item()
        val_total = data.y[data.val_mask].size(0)
        val_acc = 100 * val_correct / val_total

    model.train()
    return loss, val_loss, train_acc, val_acc, None

def test_gcn_full(model, data):
    """
    Test GCN model in full-batch mode.
    """
    model.eval()
    with torch.no_grad():
        output, embeddings = model(data.x, data.edge_index, data.edge_attr, return_embeddings=True)
        embeddings = embeddings # Move embeddings to CPU for further processing
        test_probs = F.softmax(output[data.test_mask], dim=1)
        _, predicted = torch.max(test_probs, 1)
        correct = (predicted == data.y[data.test_mask]).sum().item()
        total = data.y[data.test_mask].size(0)
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return predicted, data.y[data.test_mask], test_probs, embeddings