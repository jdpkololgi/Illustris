import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.class_weight import compute_class_weight

def preprocess_features(features):
    """
    Scale features using PowerTransformer.
    """
    scaler = PowerTransformer(method='box-cox')
    return pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

def calculate_class_weights(targets):
    """
    Calculate class weights for imbalanced datasets.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
    return torch.tensor(class_weights, dtype=torch.float32)

def train_gcn_full(model, data, optimizer, criterion):
    """
    Train GCN model in full-batch mode.
    """
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_probs = output[data.train_mask]
    _, predicted = torch.max(train_probs, 1)
    correct = (predicted == data.y[data.train_mask]).sum().item()
    total = data.y[data.train_mask].size(0)
    train_acc = 100 * correct / total

    model.eval()
    with torch.no_grad():
        val_output = model(data.x, data.edge_index, data.edge_attr)
        val_loss = criterion(val_output[data.val_mask], data.y[data.val_mask])
        val_probs = val_output[data.val_mask]
        _, val_predicted = torch.max(val_probs, 1)
        val_correct = (val_predicted == data.y[data.val_mask]).sum().item()
        val_total = data.y[data.val_mask].size(0)
        val_acc = 100 * val_correct / val_total

    return loss, val_loss, train_acc, val_acc, None

def test_gcn_full(model, data):
    """
    Test GCN model in full-batch mode.
    """
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr)
        test_probs = F.softmax(output[data.test_mask], dim=1)
        _, predicted = torch.max(test_probs, 1)
        correct = (predicted == data.y[data.test_mask]).sum().item()
        total = data.y[data.test_mask].size(0)
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return predicted, data.y[data.test_mask], test_probs, None