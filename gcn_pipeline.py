from copyreg import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

def setup_ddp(rank, world_size):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if rank == 0:
        print(f"Setting up distributed training with {world_size} GPUs...")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"Process group initialized. Using device: cuda:{rank}")

def cleanup_ddp():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def load_data(masscut=1e9, cache_path=None, rank=0, distributed=True):
    """
    Load and preprocess data for GCN training.
    Caches the processed data to a file to avoid re-computation.
    Only rank 0 loads and processes the data, then broadcasts to other ranks if distributed=True.
    """
    if cache_path is None:
        cache_path = f"processed_gcn_data_mc{masscut:.0e}.pt"

    # Only rank 0 loads/processes the data (or in single GPU mode)
    if rank == 0 or not distributed:
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                data, features, targets = torch.load(cache_path, weights_only=False)
                print("Cached data loaded successfully.")
            except Exception as e:
                print(f"Could not load cache file: {e}. Re-generating data.")
                data, features, targets = generate_data(masscut, cache_path)
        else:
            data, features, targets = generate_data(masscut, cache_path)
    else:
        # Other ranks wait and will receive data via broadcast
        data, features, targets = None, None, None
    
    # Only broadcast if distributed training is enabled
    if distributed:
        # Broadcast data from rank 0 to all other ranks
        if rank == 0:
            # Prepare data for broadcasting
            broadcast_data = [data, features, targets]
        else:
            broadcast_data = [None, None, None]
        
        # Use object list broadcasting for complex objects
        dist.broadcast_object_list(broadcast_data, src=0)
        data, features, targets = broadcast_data
    
    return data, features, targets

def generate_data(masscut, cache_path):
    """Generate the graph data (only called by rank 0)."""
    print("Loading data...")
    testcat = network(masscut=masscut, from_DESI=False)
    testcat.cweb_classify(xyzplot=False)
    testcat.network_stats_delaunay(buffer=False)
    
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
    buffered_indices_for_mask = np.where((netx_geom.pos[:,0]>10) & (netx_geom.pos[:,0]<290) & 
                                       (netx_geom.pos[:,1]>10) & (netx_geom.pos[:,1]<290) & 
                                       (netx_geom.pos[:,2]>10) & (netx_geom.pos[:,2]<290))[0]
    anti_buffered_indices_for_mask = np.where((netx_geom.pos[:,0]<10) | (netx_geom.pos[:,0]>290) | 
                                            (netx_geom.pos[:,1]<10) | (netx_geom.pos[:,1]>290) | 
                                            (netx_geom.pos[:,2]<10) | (netx_geom.pos[:,2]>290))[0]
    print(len(buffered_indices_for_mask), len(anti_buffered_indices_for_mask), 
          len(buffered_indices_for_mask)+len(anti_buffered_indices_for_mask))
    
    # Now perform train/test split on the final dataset
    train_x, test_x, train_y, test_y = train_test_split(
        features.iloc[buffered_indices_for_mask], 
        targets.iloc[buffered_indices_for_mask], 
        test_size=0.3, random_state=42, 
        stratify=targets.iloc[buffered_indices_for_mask]
    )
    valid_x, test_x, valid_y, test_y = train_test_split(
        test_x, test_y, test_size=0.3, random_state=42, stratify=test_y
    )

    # Create masks of length num_nodes
    train_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)
    valid_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)
    test_mask = torch.zeros(len(netx_geom.y), dtype=torch.bool)

    # set indices of train/valid/test to True
    train_mask[train_x.index.values] = True
    valid_mask[valid_x.index.values] = True
    test_mask[test_x.index.values] = True

    print('number counts for each class for train/valid/test sets',
          train_y.value_counts(), 
          valid_y.value_counts(), 
          test_y.value_counts())

    netx_geom.train_mask = train_mask
    netx_geom.val_mask = valid_mask
    netx_geom.test_mask = test_mask

    print('PyG Data object populated: ', netx_geom)

    print(f"Saving processed data to {cache_path}...")
    torch.save((netx_geom, features, targets), cache_path)
    print("Data saved.")

    return netx_geom, features, targets

def create_distributed_sampler(data, rank, world_size):
    """Create a distributed sampler for training data."""
    # Get training indices
    train_indices = torch.where(data.train_mask)[0]
    
    # Calculate indices for this rank
    num_samples = len(train_indices)
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    
    if rank == world_size - 1:  # Last rank gets remaining samples
        end_idx = num_samples
    else:
        end_idx = start_idx + samples_per_rank
    
    # Get subset of training indices for this rank
    rank_train_indices = train_indices[start_idx:end_idx]
    
    # Create new mask for this rank
    rank_train_mask = torch.zeros_like(data.train_mask)
    rank_train_mask[rank_train_indices] = True
    
    return rank_train_mask

def train_and_evaluate(model, data, class_weights, rank, world_size, num_epochs=500, lr=3e-3):
    """
    Train and evaluate the GCN model with distributed training.
    """
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    data = data.to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create distributed training mask
    distributed_train_mask = create_distributed_sampler(data, rank, world_size)
    data.train_mask = distributed_train_mask
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=200, 
                                threshold=1.5e-3, cooldown=50, min_lr=5e-4)

    full_train_loss = []
    full_val_loss = []
    full_train_acc = []
    full_val_acc = []

    # Track the initial learning rate
    last_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        train_loss, val_loss, train_acc, val_acc, _ = train_gcn_full(ddp_model, data, optimizer, criterion)
        
        # Synchronize metrics across all ranks
        train_loss_tensor = torch.tensor(train_loss.item()).to(device)
        val_loss_tensor = torch.tensor(val_loss.item()).to(device)
        train_acc_tensor = torch.tensor(train_acc).to(device)
        val_acc_tensor = torch.tensor(val_acc).to(device)
        
        # Average metrics across all ranks
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)
        
        full_train_loss.append(train_loss_tensor.item())
        full_val_loss.append(val_loss_tensor.item())
        full_train_acc.append(train_acc_tensor.item())
        full_val_acc.append(val_acc_tensor.item())

        scheduler.step(val_loss_tensor.item())

        # Check for learning rate change
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr and rank == 0:
            print(f"[Epoch {epoch+1}] Learning rate changed from {last_lr:.6e} → {current_lr:.6e}")
            last_lr = current_lr

        if (epoch+1) % 100 == 0 and rank == 0:
            print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB | "
                  f"Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss_tensor.item():.4f} - '
                  f'Validation Loss: {val_loss_tensor.item():.4f} - '
                  f'Training Accuracy: {train_acc_tensor.item():.2f}% - '
                  f'Validation Accuracy: {val_acc_tensor.item():.2f}%')

    # Test only on rank 0 to avoid duplicated output
    if rank == 0:
        # Use the full test mask for evaluation
        data.train_mask = torch.zeros_like(data.train_mask)  # Reset train mask for testing
        predicted, labels, probs, embeddings = test_gcn_full(ddp_model.module, data)
        return model, full_train_loss, full_val_loss, full_train_acc, full_val_acc, predicted, labels, probs, embeddings
    else:
        return model, full_train_loss, full_val_loss, full_train_acc, full_val_acc, None, None, None

def main(rank, world_size, num_epochs):
    """Main training function for each process."""
    setup_ddp(rank, world_size)
    
    # Load data (only rank 0 loads, others receive via broadcast)
    data, features, targets = load_data(rank=rank, distributed=True)
    
    # Calculate class weights (same on all ranks)
    class_weights = calculate_class_weights(targets)

    # Debug info only from rank 0
    if rank == 0:
        print("Class weights:", class_weights)
        print("Features shape:", features.shape)
        print("Feature columns:", features.columns.tolist())

    # Initialize model
    model = SimpleGAT(input_dim=features.shape[1], output_dim=4, num_heads=4)

    # Train and evaluate
    results = train_and_evaluate(
        model, data, class_weights, rank, world_size, num_epochs=num_epochs, lr=3e-3
    )

    # Save model only from rank 0
    if rank == 0:
        model, train_loss, val_loss, train_acc, val_acc, predicted, labels, probs, embeddings = results
        
        # Save losses as a pickle file
        import pickle
        with open('training_validation_accuracies_losses.pkl', 'wb') as f:
            pickle.dump({'train_loss': train_loss, 'val_loss': val_loss,
                          'train_acc': train_acc, 'val_acc': val_acc}, f)
        print("Training and validation losses and accuracies saved to 'training_validation_accuracies_losses.pkl'")

        torch.save(model.state_dict(), 'trained_gat_model_ddp.pth')
        print("Model saved as 'trained_gat_model_ddp.pth'")

        with open('test_predictions_labels_probs.pkl', 'wb') as f:
            pickle.dump({'predicted': predicted, 'labels': labels, 'probs': probs}, f)
        print("Test predictions, labels, and probabilities saved to 'test_predictions_labels_probs.pkl'")

        # Save node embeddings
        with open('node_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Node embeddings saved to 'node_embeddings.pkl'")

    cleanup_ddp()

if __name__ == "__main__":
    # Automatically detect number of available GPUs
    world_size = torch.cuda.device_count()
    num_epochs = 8000 # 15000 # Set number of epochs for training
    print(f"Detected {world_size} GPUs. Starting distributed training...")
    
    if world_size < 2:
        print("Warning: Only 1 GPU detected. Running single-GPU training...")
        # Fall back to single GPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
        # Load data without distributed operations
        data, features, targets = load_data(rank=0, distributed=False)
        class_weights = calculate_class_weights(targets)
        
        # Initialize model
        model = SimpleGAT(input_dim=features.shape[1], output_dim=4, num_heads=4)
        model = model.to(device)
        data = data.to(device)
        
        # Single GPU training
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=200, 
                                    threshold=1.5e-3, cooldown=50, min_lr=5e-4)
        
        for epoch in range(num_epochs):
            train_loss, val_loss, train_acc, val_acc, _ = train_gcn_full(model, data, optimizer, criterion)
            scheduler.step(val_loss.item())
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1}/num_epochs - Training Loss: {train_loss:.4f} - '
                      f'Validation Loss: {val_loss:.4f} - Training Accuracy: {train_acc:.2f}% - '
                      f'Validation Accuracy: {val_acc:.2f}%')
        
        predicted, labels, probs, embeddings = test_gcn_full(model, data)
        torch.save(model.state_dict(), 'trained_gat_model.pth')
        print("Model saved as 'trained_gat_model.pth'")
    else:
        # Multi-GPU training
        print(f"Starting multi-GPU training on {world_size} GPUs...")
        mp.spawn(main, args=(world_size,num_epochs,), nprocs=world_size, join=True)