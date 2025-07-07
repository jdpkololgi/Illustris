import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

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

scaler = GradScaler('cuda')

def train_gcn_full(model, data, optimizer, criterion):
    """
    Train GCN model in full-batch mode.
    """

    model.train()
    optimizer.zero_grad()

    # output = model(data.x, data.edge_index, data.edge_attr)
    # loss = criterion(output[data.train_mask], data.y[data.train_mask])
    
    # loss.backward()
    # optimizer.step()

    with autocast('cuda'):
        output = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    
    scaler.scale(loss).backward() # scale the loss for mixed precision training
    scaler.step(optimizer) # step the optimizer
    scaler.update() # update the scaler

    # Check gradients are flowing and are not none
    # total_norm = 0
    # for p in model.parameters():
    #     if p.grad is not None:
    #         total_norm += p.grad.data.norm(2).item()**2
    # total_norm = total_norm**0.5
    # print(f"Grad norm: {total_norm:.4e}")

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

    model.train()  # Set model back to train mode
    return loss, val_loss, train_acc, val_acc, None
#####################   

# def train_gcn_full(model, data, optimiser, criterion, return_attention=False, num_heads=1):
#     '''
#     Function to train the GNN model on the full graph
#     Args:
#         model (nn.Module): GNN model
#         data (Data): Data object containing the graph data
#         optimiser (torch.optim.Optimizer): Optimiser for the model
#         criterion (nn.Module): Loss function
#         return_attention (bool): Whether to return the attention weights or not
#         num_heads (int): Number of attention heads for GAT model
#     Returns:
#         loss (float): Training loss
#         val_loss (float): Validation loss
#         train_acc (float): Training accuracy
#         val_acc (float): Validation accuracy
#         val_attention_data (list): Attention weights for each layer if return_attention is True
#     '''

#     # Identify GNN type
#     is_gat = isinstance(model, SimpleGAT)

#     # Training loop
#     model.train()
#     optimiser.zero_grad()
    
#     with autocast():
#         if is_gat and return_attention:
#             output, _ = model(data.x, data.edge_index, data.edge_attr, return_attention=True)
#         else:
#             output = model(data.x, data.edge_index, data.edge_attr)
#         loss = criterion(output[data.train_mask], data.y[data.train_mask])
#     # output = model(data.x, data.edge_index, data.edge_attr) # remember to pass the edge_attr as we are weighting the edges by their distance

#     scaler.scale(loss).backward() # scale the loss for mixed precision training
#     scaler.step(optimiser) # update the weights based on the loss
#     scaler.update() # update the scaler for mixed precision training

#     # loss.backward()
#     # optimiser.step() # update the weights based on the loss!

#     train_probs = output[data.train_mask]
#     _, predicted = torch.max(train_probs, 1)
#     correct = (predicted == data.y[data.train_mask]).sum().item()
#     total = data.y[data.train_mask].size(0)

#     # validation accuracy and loss
#     model.eval()
#     with torch.no_grad():
#         if is_gat and return_attention:
#             output, val_attention_data = model(data.x, data.edge_index, data.edge_attr, return_attention=True)
#         else:
#             output = model(data.x, data.edge_index, data.edge_attr)

#         val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
#         val_probs = output[data.val_mask]
#         _, val_predicted = torch.max(val_probs, 1)
#         val_correct = (val_predicted == data.y[data.val_mask]).sum().item()
#         val_total = data.y[data.val_mask].size(0)

#     model.train()
#     val_acc = 100 * val_correct / val_total
#     train_acc = 100 * correct / total

#     # print(f'Training Loss: {loss.item():.4f} - Training Accuracy: {100 * correct / total:.2f}% - Validation Loss: {val_loss.item():.4f} - Validation Accuracy: {100 * val_correct / val_total:.2f}%')
#     return loss, val_loss, train_acc, val_acc, val_attention_data if is_gat and return_attention else None 

#####################

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