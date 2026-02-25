import os
import sys
from pathlib import Path

# Force priority for user installed packages (must be FIRST to avoid NumPy version conflicts)
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
# Remove user_site if it exists anywhere in sys.path
while user_site in sys.path:
    sys.path.remove(user_site)
# Insert at the very beginning
sys.path.insert(0, user_site)

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import time
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
import torch # Used for compatibility with existing data structures if needed
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

from Network_stats import network

import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax
from sklearn.metrics import classification_report

from graph_net_models import make_graph_network
from eigenvalue_transformations import eigenvalues_to_shape_params, shape_params_to_eigenvalues, compute_shape_param_statistics, eigenvalues_to_increments, increments_to_eigenvalues
from tng_pipeline_paths import DEFAULT_JRAPH_OUTPUT_DIR, resolve_pipeline_paths
# Set up JAX to use 64-bit precision if needed, though 32 is usually fine for ML
# jax.config.update("jax_enable_x64", True)

def convert_pyg_to_jraph(pyg_data):
    """
    Convert PyTorch Geometric Data object to Jraph GraphsTuple.
    """
    print("Converting PyG Data to Jraph GraphsTuple...")
    
    # Nodes
    # pym_data.x is [N, F]
    node_features = jnp.array(pyg_data.x.numpy(), dtype=jnp.float32)
    
    # Edges
    # pyg_data.edge_index is [2, E]
    # jraph expects senders, receivers
    senders = jnp.array(pyg_data.edge_index[0].numpy(), dtype=jnp.int32)
    receivers = jnp.array(pyg_data.edge_index[1].numpy(), dtype=jnp.int32)
    
    # Edge Features
    # pyg_data.edge_attr is [E, F_edge] or [E]
    if pyg_data.edge_attr is not None:
        edge_attr = pyg_data.edge_attr.numpy()
        # If it's 1D, reshape to [E, 1]
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr.reshape(-1, 1)
        edge_features = jnp.array(edge_attr, dtype=jnp.float32)
    else:
        # Default edge features if missing
        edge_features = jnp.ones((len(senders), 1), dtype=jnp.float32)
        
    # Labels
    # pyg_data.y
    labels = jnp.array(pyg_data.y.numpy(), dtype=jnp.int32)
    
    # Masks (train_mask, val_mask, test_mask are bool tensors in pyg_data)
    train_mask = jnp.array(pyg_data.train_mask.numpy(), dtype=bool)
    val_mask = jnp.array(pyg_data.val_mask.numpy(), dtype=bool)
    test_mask = jnp.array(pyg_data.test_mask.numpy(), dtype=bool)
    
    # Graph Tuple
    n_node = jnp.array([node_features.shape[0]])
    n_edge = jnp.array([senders.shape[0]])
    
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=None
    )
    
    return graph, labels, (train_mask, val_mask, test_mask)

def preprocess_features(features):
    """
    Scale features using PowerTransformer.
    """
    scaler = PowerTransformer(method='box-cox')
    return pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

def generate_data(masscut, cache_path, version='v2', use_transformed_eig=True):
    """Generate Jraph data from scratch using Network_stats."""
    print(f"Generating data using {version}...")
    testcat = network(masscut=masscut, from_DESI=False)
    testcat.cweb_classify(xyzplot=False)
    
    netx = None
    if version == 'v2':
        # v2 returns the graph and computes 'edge_features'
        netx = testcat.network_stats_delaunay2(buffer=False)
    else:
        # v1 does not return the graph, need to retrieve it
        testcat.network_stats_delaunay(buffer=False)
        netx = testcat.subhalo_delauany_network(xyzplot=False)
        
    # Get node features
    # features = testcat.data.iloc[:, :-1]
    # targets = testcat.data.iloc[:, -1]
    # Note: testcat.data includes features and 'Target'.
    features = testcat.data.drop(columns=['Target'])
    targets = testcat.data['Target']
    continuous_targets = [testcat.eig1, testcat.eig2, testcat.eig3]
    
    # Preprocess features
    features = preprocess_features(features)
    node_features_array = features.values.astype(np.float32) # [N, F_node]
    
    # Construct Edges and Edge Features
    senders = []
    receivers = []
    edge_feats = []
    
    if version == 'v2':
        # v2 has detailed edge features in testcat.edge_features dataframe
        # Encoded as (u, v) -> row
        # Since Jraph/GNNs require directed edges for message passing, we must explicitly 
        # represent the undirected graph as bidirectional: both u->v and v->u must exist.
        
        edge_df = testcat.edge_features
        # edge_df columns include: ['edge_length', 'x_dir', 'y_dir', 'z_dir', 'density_contrast']
        
        # Helper to get features and ensure every undirected edge becomes two directed edges
        # We iterate netx edges to ensure valid connections
        for u, v in netx.edges():
            # Check edge direction in the dataframe index (could be (u,v) or (v,u))
            edge_cols = ['edge_length', 'x_dir', 'y_dir', 'z_dir', 'density_contrast']
            if (u, v) in edge_df.index:
                f = edge_df.loc[(u,v)][edge_cols]
                
                # --- Forward Edge u -> v ---
                # Use features directly as calculated
                senders.append(u)
                receivers.append(v)
                edge_feats.append(f.values) # [F_edge]
                
                # --- Backward Edge v -> u ---
                # We must transform directional features to be relative to 'v' looking at 'u'
                senders.append(v)
                receivers.append(u)
                
                f_back = f.copy()
                
                # Flip 3D directional vectors (vector v->u is -(vector u->v))
                f_back['x_dir'] *= -1
                f_back['y_dir'] *= -1
                f_back['z_dir'] *= -1
                
                # Invert Density Contrast
                # Contrast u->v is rho_v / rho_u
                # Contrast v->u should be rho_u / rho_v = 1 / (rho_v / rho_u)
                # We add a small epsilon 1e-6 (in previous steps logic, or just handle safe div)
                f_back['density_contrast'] = 1.0 / (f_back['density_contrast']) 
                
                edge_feats.append(f_back.values)
                
            elif (v, u) in edge_df.index:
                # The edge exists in DF but stored as (v,u). 
                # This means existing row is v -> u properties.
                f = edge_df.loc[(v,u)][edge_cols]
                
                # --- Reverse of Storage (u -> v) ---
                # We are constructing u->v, but 'f' is v->u. So we Flip 'f'.
                senders.append(u)
                receivers.append(v)
                
                f_rev = f.copy()
                f_rev['x_dir'] *= -1
                f_rev['y_dir'] *= -1
                f_rev['z_dir'] *= -1
                f_rev['density_contrast'] = 1.0 / (f_rev['density_contrast'])
                edge_feats.append(f_rev.values)
                
                # --- Direct from Storage (v -> u) ---
                # 'f' is already v->u
                senders.append(v)
                receivers.append(u)
                edge_feats.append(f.values)
                
    else:
        # v1: only 'length' attribute in graph
        for u, v in netx.edges():
            l = netx[u][v]['length']
            # u->v
            senders.append(u)
            receivers.append(v)
            edge_feats.append([l])
            # v->u
            senders.append(v)
            receivers.append(u)
            edge_feats.append([l])
            
    # Convert to JNP arrays
    senders = jnp.array(senders, dtype=jnp.int32)
    receivers = jnp.array(receivers, dtype=jnp.int32)
    # Convert to JNP arrays
    # Preprocess Edge Features (Scaling)
    # Convert list to numpy array first
    edge_feats_np = np.array(edge_feats, dtype=np.float32)
    
    if version == 'v2':
        print(f"Post-processing V2 Edge Features (shape {edge_feats_np.shape})...")
        print(f"Raw Stats: Mean={np.mean(edge_feats_np, axis=0)}, Std={np.std(edge_feats_np, axis=0)}")
        print(f"Raw Range: Min={np.min(edge_feats_np, axis=0)}, Max={np.max(edge_feats_np, axis=0)}")
        
        # Columns: [Length, x_dir, y_dir, z_dir, DensityContrast]
        # 1. Log transform Length (idx 0) and DensityContrast (idx 4)
        # Add epsilon for safety, though they should be > 0
        edge_feats_np[:, 0] = np.log(np.maximum(edge_feats_np[:, 0], 1e-6))
        edge_feats_np[:, 4] = np.log(np.maximum(edge_feats_np[:, 4], 1e-6))
        
        # 2. Standard Scale ONLY length (0) and density (4)
        # This preserves the unit vector nature of x,y,z (1,2,3)
        scaler_edge = StandardScaler()
        edge_feats_np[:, [0, 4]] = scaler_edge.fit_transform(edge_feats_np[:, [0, 4]])
        
        print(f"Scaled Stats: Mean={np.mean(edge_feats_np, axis=0)}, Std={np.std(edge_feats_np, axis=0)}")
        
    edge_features_array = jnp.array(edge_feats_np, dtype=jnp.float32)
    node_features_array = jnp.array(node_features_array, dtype=jnp.float32)
    
    # Graph Tuple
    n_node = jnp.array([len(netx.nodes)])
    n_edge = jnp.array([len(senders)])
    
    graph = jraph.GraphsTuple(
        nodes=node_features_array,
        edges=edge_features_array,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=None
    )
    
    # Masks logic (Buffering)
    # Replicate gcn_pipeline logic
    pos = np.zeros((len(netx.nodes), 3))
    for i in netx.nodes():
        pos[i] = netx.nodes[i]['pos']
        
    buffered_indices = np.where((pos[:,0]>10) & (pos[:,0]<290) & 
                                (pos[:,1]>10) & (pos[:,1]<290) & 
                                (pos[:,2]>10) & (pos[:,2]<290))[0]
                                
    # Creates masks
    # Stratified split on buffered nodes
    # targets is pandas Series?
    targets_buffered = targets.iloc[buffered_indices]
    features_buffered = features.iloc[buffered_indices] # Just for split consistency
    
    # indices relative to dataframe (which matches node indices 0..N-1)
    train_idx, test_idx = train_test_split(
        buffered_indices,
        test_size=0.3, random_state=42, 
        stratify=targets_buffered
    )
    
    # Validation split
    # Should split 'test_idx' further? gcn_pipeline:
    # valid_x, test_x, ... = train_test_split(test_x, ..., test_size=0.3)
    # So gcn_pipeline splits the "test" portion (30% of total) into Val and Test?
    # No, gcn_pipeline:
    # train_x, test_x ... = split(..., test_size=0.3) => Train=70%, Remainder=30%
    # valid_x, test_x ... = split(test_x, ..., test_size=0.3) => Val=70% of Remainder, Test=30% of Remainder
    # Total: Train=70%, Val=21%, Test=9%
    
    # Wait, splitting indices is safer.
    # Get targets for the 'test' set (remainder)
    targets_remainder = targets.iloc[test_idx]
    
    valid_idx, true_test_idx = train_test_split(
        test_idx,
        test_size=0.3, random_state=42,
        stratify=targets_remainder
    )
    
    train_mask = np.zeros(len(netx.nodes), dtype=bool)
    val_mask = np.zeros(len(netx.nodes), dtype=bool)
    test_mask = np.zeros(len(netx.nodes), dtype=bool)
    
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[true_test_idx] = True
    
    # Convert masks and labels to jnp
    masks = (jnp.array(train_mask), jnp.array(val_mask), jnp.array(test_mask))
    classification_labels = jnp.array(targets.values, dtype=jnp.int32)
    
    # Prepare regression targets
    # Stack eigenvalues: [N, 3]
    eigenvalues_raw = np.stack([testcat.eig1, testcat.eig2, testcat.eig3], axis=-1).astype(np.float64)
    
    # Always use StandardScaler for regression targets to ensure balanced loss
    target_scaler = StandardScaler()
    
    if use_transformed_eig:
        # Transform to shape parameters
        transformed_eig = eigenvalues_to_increments(eigenvalues_raw)
        
        # Fit scaler on training set shape parameters
        target_scaler.fit(transformed_eig[train_idx])
        transformed_eig_scaled = target_scaler.transform(transformed_eig)
        regression_targets = jnp.array(transformed_eig_scaled, dtype=jnp.float64)
        
        # Calculate bounds for bounded activations (on the scaled targets)
        # Update stats to reflect the SCALED ranges
        scaled_min = np.min(transformed_eig_scaled[train_idx], axis=0)
        scaled_max = np.max(transformed_eig_scaled[train_idx], axis=0)
        stats = {
            'v1_min_scaled': float(scaled_min[0]),
            'v1_max_scaled': float(scaled_max[0]),
            'target_min': scaled_min.tolist(),
            'target_max': scaled_max.tolist(),
            'scaler_mean': target_scaler.mean_.tolist(),
            'scaler_std': target_scaler.scale_.tolist()
        }

        print(f"\nTraining Set Stats:")
        print(f"  Means: {np.mean(transformed_eig_scaled[train_idx], axis=0)}")
        print(f"  Stds:  {np.std(transformed_eig_scaled[train_idx], axis=0)}")
        print(f"\nTransformed Eigenvalue Stats:")
        print(f" Means: {stats['scaler_mean']}")
        print(f" Stds: {stats['scaler_std']}")
        print(f" Min: {stats['target_min']}")
        print(f" Max: {stats['target_max']}")

        # Save to cache
        print(f"Saving generated data to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'graph': graph, 
                'classification_labels': classification_labels, 
                'regression_targets': regression_targets,
                'target_scaler': target_scaler,
                'stats': stats,
                'eigenvalues_raw': eigenvalues_raw,
                'masks': masks
            }, f)
    
        # Always return everything as a 7-tuple
        return graph, classification_labels, regression_targets, stats, target_scaler, eigenvalues_raw, masks
    
    else:
        # Scale raw eigenvalues
        target_scaler.fit(eigenvalues_raw[train_idx])
        eigenvalues_scaled = target_scaler.transform(eigenvalues_raw)
        regression_targets = jnp.array(eigenvalues_scaled, dtype=jnp.float32)
    
        print(f"Eigenvalue stats (raw): mean={np.mean(eigenvalues_raw, axis=0)}")
        print(f"Eigenvalue stats (scaled): mean={np.mean(eigenvalues_scaled, axis=0)}")
    
        # Save to cache
        print(f"Saving generated data to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'graph': graph,
                'classification_labels': classification_labels,
                'regression_targets': regression_targets,
                'target_scaler': target_scaler,
                'eigenvalues_raw': eigenvalues_raw,
                'masks': masks
            }, f)
            
        return graph, classification_labels, regression_targets, None, target_scaler, eigenvalues_raw, masks

def load_data(
    masscut=1e9,
    use_v2=True,
    prediction_mode='classification',
    use_transformed_eig=True,
    cache_dir=None,
):
    """
    Load data from cache if available, otherwise generate it.
    Can switch between v1 and v2 stats.
    
    Args:
        masscut: Mass cutoff for halos
        use_v2: Use v2 edge features
        prediction_mode: 'classification' or 'regression'
    
    Returns:
        For classification: graph, targets, None, None, masks
        For regression: graph, targets, stats, eigenvalues_raw, masks
    """
    version = 'v2' if use_v2 else 'v1'
    paths = resolve_pipeline_paths(
        masscut=masscut,
        use_v2=use_v2,
        use_transformed_eig=use_transformed_eig,
        cache_dir=cache_dir,
    )
    cache_path = paths.cache_path

    if os.path.exists(cache_path):
        print(f"Loading cached Jraph data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            
            # Universal fallback for all regression modes
            if prediction_mode == 'regression':
                targets = data.get('regression_targets')
                stats = data.get('stats')
                scaler = data.get('target_scaler')
                raw_eig = data.get('eigenvalues_raw')
                return data['graph'], targets, stats, scaler, raw_eig, data['masks']
            else:
                # Classification
                return data['graph'], data['classification_labels'], None, None, None, data['masks']
            
    # Generate fresh
    # generate_data returns: graph, labels, reg_targets, stats, scaler, raw_eig, masks
    res = generate_data(masscut, cache_path, version=version, use_transformed_eig=use_transformed_eig)
    
    if prediction_mode == 'classification':
        return res[0], res[1], None, None, None, res[6]
    else:
        return res[0], res[2], res[3], res[4], res[5], res[6]


def calculate_class_weights(targets):
    """Calculate class weights using sklearn."""
    try:
        classes = np.unique(targets)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=targets)
        return jnp.array(weights, dtype=jnp.float32)
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}")
        return jnp.ones(4, dtype=jnp.float32)



#########################################################################
# Main
#########################################################################
def main(args):
    print(f"JAX Devices: {jax.devices()}")
    num_devices = jax.local_device_count()
    print(f"Running on {num_devices} devices.")
    print(f"Prediction Mode: {args.prediction_mode}")
    
    # generate unique timestamp
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Job Timestamp: {timestamp}")

    # 1. Resolve paths and load data
    masscut = 1e9
    use_transformed_eig = not getattr(args, 'no_transformed_eig', False)
    paths = resolve_pipeline_paths(
        masscut=masscut,
        use_v2=True,
        use_transformed_eig=use_transformed_eig,
        output_dir=args.output_dir,
    )
    args.output_dir = paths.output_dir

    graph, targets, stats, target_scaler, eigenvalues_raw, masks = load_data(
        masscut=masscut,
        use_v2=True,
        prediction_mode=args.prediction_mode,
        use_transformed_eig=use_transformed_eig,
        cache_dir=paths.cache_dir,
    )
    train_mask, val_mask, test_mask = masks

    # Class weights only for classification
    if args.prediction_mode == 'classification':
        class_weights = calculate_class_weights(np.array(targets))
        output_dim = 4  # 4 cosmic web classes
        stats = None
        eigenvalue_scaler = None
    else:
        class_weights = None  # Not used in regression
        output_dim = 3  # 3 transformed eigenvalues or raw eigenvalues

        if use_transformed_eig:
            # Using transformed eigenvalues (v₁, Δλ₂, Δλ₃) - standard scored
            print("\n[Regression Mode] Using transformed eigenvalues (v₁, Δλ₂, Δλ₃)")
        else:
            # Using raw scaled eigenvalues
            stats = None
            print("\n[Regression Mode] Using raw eigenvalues (λ₁, λ₂, λ₃)")

    print(f"Graph stats: Nodes={graph.n_node[0]}, Edges={graph.n_edge[0]}")
    print(f"Train size: {jnp.sum(train_mask)}, Val size: {jnp.sum(val_mask)}, Test size: {jnp.sum(test_mask)}")

    # 2. Model Setup
    # Initialize network with simple linear output
    net_fn = make_graph_network(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        output_dim=output_dim,
    )
    net = hk.transform(net_fn) # hk.transform converts a function using Haiku modules into an (init, apply) pair.
    
    # Init params (single device for init)
    rng = jax.random.PRNGKey(args.seed)
    # We need a dummy graph with correct shape for initialization
    # Jraph graphs can be batched/padded, but here we use the full graph logic.
    params = net.init(rng, graph, is_training=True)
    
    # 3. Optimizer
    # Similar schedule to gcn_pipeline if possible, or simple AdamW
    lr = 1e-3
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(params)
    
    # 4. Replicate Data to Devices for PMAP
    # We replicate the GRAPH and PARAMS across devices.
    # We SHARD the MASKS so each device computes loss on a subset of nodes.
    
    # Replicate params
    replicated_params = jax.device_put_replicated(params, jax.local_devices())
    
    # 3. Optimizer
    # GCN uses 3e-3. We'll use a schedule.
    num_epochs = args.epochs
    warmup_steps = min(500, num_epochs // 2)  # Ensure warmup doesn't exceed half the epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=args.lr, # Match GCN peak
        warmup_steps=warmup_steps,
        decay_steps=max(1, num_epochs - warmup_steps),  # Ensure positive decay steps
        end_value=1e-4
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)
    
    # Replicate optimizer state
    replicated_opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    
    # Replicate graph (Data Parallelism usually shards data, but for Transductive on One Graph
    # where the graph fits in memory, replicating the graph structure is common 
    # so every node can see its neighbors. We only split the *supervision* signal).
    # stack along first dimension (device dim)
    # jraph definitions are not easily stackable with jnp.stack directly if contents are variable length,
    # we replicate the graph and shard the MASK).
    # Since we have only 1 graph, we just expand it to [num_devices, ...]
    replicated_graph = jax.device_put_replicated(graph, jax.local_devices())
    
    # Replicate targets (labels for classification, eigenvalues for regression)
    replicated_targets = jax.device_put_replicated(targets, jax.local_devices())
    
    # Replicate class weights (None for regression)
    if class_weights is not None:
        replicated_class_weights = jax.device_put_replicated(class_weights, jax.local_devices())
    else:
        # Dummy weights for regression (not used but needed for function signature)
        replicated_class_weights = jax.device_put_replicated(jnp.ones(4, dtype=jnp.float32), jax.local_devices())

    # Shard the Train Mask
    # We want to split train_mask indices among devices
    # Create N unique masks.
    num_devices = len(jax.local_devices())
    train_indices = jnp.where(train_mask)[0]
    # split indices
    train_indices_sharded = jnp.array_split(train_indices, num_devices)
    
    # Create masks for each shard
    # This must be a static array per device, so we stack them.
    sharded_train_masks_list = []
    sharded_val_masks_list = [] # For parallel validation
    
    # Full masks
    full_val_mask = val_mask
    val_indices = jnp.where(full_val_mask)[0]
    val_indices_sharded = jnp.array_split(val_indices, num_devices)

    for i in range(num_devices):
        # Train mask shard
        m_train = jnp.zeros_like(train_mask)
        if len(train_indices_sharded[i]) > 0:
            m_train = m_train.at[train_indices_sharded[i]].set(True)
        sharded_train_masks_list.append(m_train)

        # Val mask shard
        m_val = jnp.zeros_like(full_val_mask)
        if len(val_indices_sharded[i]) > 0:
            m_val = m_val.at[val_indices_sharded[i]].set(True)
        sharded_val_masks_list.append(m_val)
        
    sharded_train_masks = jnp.stack(sharded_train_masks_list) # [Devices, Nodes]
    sharded_train_masks = jax.device_put_sharded(list(sharded_train_masks), jax.local_devices())

    sharded_val_masks = jnp.stack(sharded_val_masks_list)
    sharded_val_masks = jax.device_put_sharded(list(sharded_val_masks), jax.local_devices())

    # 4. Training Functions
    # Mode-aware loss function
    def loss_fn(params, graph, targets, mask, net_apply, rng, class_weights, prediction_mode, label_smoothing=0.1):
        # Pass is_training=True for Dropout
        outputs = net_apply(params, rng, graph, is_training=True).nodes
        
        if prediction_mode == 'classification':
            num_classes = 4
            # Cross Entropy Loss
            # outputs: [N, C], targets: [N] (class indices)
            labels_one_hot = jax.nn.one_hot(targets, num_classes=num_classes)
            smoothed_labels = optax.smooth_labels(labels_one_hot, alpha=label_smoothing)
            per_node_loss = optax.softmax_cross_entropy(outputs, smoothed_labels)
            
            # Apply class weights
            weights = jnp.take(class_weights, targets)
            weighted_loss = per_node_loss * weights
            
            # Mask
            masked_loss = weighted_loss * mask
        else:
            # Regression: MSE loss
            # outputs: [N, 3], targets: [N, 3] (eigenvalues)
            per_node_loss = jnp.mean(optax.l2_loss(outputs, targets), axis=-1)  # Mean over 3 eigenvalues
            masked_loss = per_node_loss * mask
        
        # Mean over masked nodes
        num_masked = jnp.sum(mask)
        loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
        
        return loss, (outputs, num_masked)

    # Update Function (mode-aware)
    def update(params, opt_state, graph, targets, mask, rng, class_weights, prediction_mode):
        # rng mix
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Gradients
        (train_loss, (outputs, num_masked)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, graph, targets, mask, net.apply, step_rng, class_weights, prediction_mode
        )
        
        # Sync gradients across devices (average)
        grads = jax.lax.pmean(grads, axis_name='i')
        
        # Sync Loss metrics for reporting
        total_loss_part = train_loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        # Calculate metric (Accuracy for classification, R² for regression)
        if prediction_mode == 'classification':
            preds = jnp.argmax(outputs, axis=-1)
            correct = (preds == targets) & mask
            total_correct = jax.lax.psum(jnp.sum(correct), axis_name='i')
            global_metric = total_correct / jnp.maximum(total_count, 1.0)
        else:
            # R² score for regression (approximation across devices)
            # R² = 1 - SS_res / SS_tot
            # For simplicity, we report MSE here; R² needs global mean
            # We'll compute MSE as the metric
            mse_per_node = jnp.mean(optax.l2_loss(outputs, targets), axis=-1)
            masked_mse = mse_per_node * mask
            total_mse = jax.lax.psum(jnp.sum(masked_mse), axis_name='i')
            global_metric = total_mse / jnp.maximum(total_count, 1.0)  # Mean MSE
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, global_loss, global_metric
        
    update_fn = jax.pmap(update, axis_name='i', static_broadcasted_argnums=(7,))  # prediction_mode is static

    def evaluate(params, graph, targets, mask, rng, class_weights, prediction_mode):
        # We can evaluate in parallel too
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Pass is_training=False to disable Dropout
        outputs = net.apply(params, step_rng, graph, is_training=False).nodes
        
        if prediction_mode == 'classification':
            labels_one_hot = jax.nn.one_hot(targets, num_classes=4)
            per_node_loss = optax.softmax_cross_entropy(outputs, labels_one_hot)
            weights = jnp.take(class_weights, targets)
            masked_loss = per_node_loss * weights * mask
            
            num_masked = jnp.sum(mask)
            loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
            
            # Accuracy
            preds = jnp.argmax(outputs, axis=-1)
            correct = (preds == targets) & mask
            total_correct = jax.lax.psum(jnp.sum(correct), axis_name='i')
            total_mask = jax.lax.psum(jnp.sum(mask), axis_name='i')
            metric = total_correct / jnp.maximum(total_mask, 1.0)
        else:
            # Regression: MSE
            per_node_loss = jnp.mean(optax.l2_loss(outputs, targets), axis=-1)
            masked_loss = per_node_loss * mask
            
            num_masked = jnp.sum(mask)
            loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
            
            # MSE as metric
            total_mse = jax.lax.psum(jnp.sum(masked_loss), axis_name='i')
            total_mask = jax.lax.psum(jnp.sum(mask), axis_name='i')
            metric = total_mse / jnp.maximum(total_mask, 1.0)
        
        # Global loss logic
        total_loss_part = loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        return global_loss, metric
        
    evaluate_fn = jax.pmap(evaluate, axis_name='i', static_broadcasted_argnums=(6,))  # prediction_mode is static

    # 5. Training Loop
    # num_epochs defined above
    report_every = 1
    
    # Random keys for each step (replicated)
    # We can just pass a single key and fold inside, 
    # but strictly we should shard RNGs if we want different randomness per device (dropout etc)
    # Here we fold inside 'update' with axis index, so single key is fine to start.
    
    current_rng = jax.random.PRNGKey(0)
    
    print("Starting training...")
    t0 = time.time()
    
    for epoch in range(num_epochs):
        current_rng, step_rng = jax.random.split(current_rng)
        # Replicate RNG
        step_rngs = jax.device_put_replicated(step_rng, jax.local_devices())
        
        replicated_params, replicated_opt_state, train_loss, train_metric = update_fn(
            replicated_params, replicated_opt_state, 
            replicated_graph, replicated_targets, sharded_train_masks, 
            step_rngs, replicated_class_weights, args.prediction_mode
        )
        
        # Validation
        if epoch % report_every == 0:
            val_loss, val_metric = evaluate_fn(
                replicated_params, replicated_graph, replicated_targets, 
                sharded_val_masks, step_rngs, replicated_class_weights, args.prediction_mode
            )
            # take first device result (they are identical due to pmean/psum)
            if args.prediction_mode == 'classification':
                print(f"Epoch {epoch} | Train Loss: {train_loss[0]:.4f} | Train Acc: {train_metric[0]*100:.3f}% | Val Loss: {val_loss[0]:.4f} | Val Acc: {val_metric[0]*100:.3f}%")
            else:
                print(f"Epoch {epoch} | Train Loss: {train_loss[0]:.4f} | Train MSE: {train_metric[0]:.6f} | Val Loss: {val_loss[0]:.4f} | Val MSE: {val_metric[0]:.6f}")

    print(f"Training finished in {time.time() - t0:.2f}s")
    
    # Save Model
    # Take params from first device
    final_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_params))
    
    # Save model
    print("Saving model...")
    import pickle
    save_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_model_seed_{args.seed}_{timestamp}.pkl')
    with open(save_filename, 'wb') as f:
        pickle.dump(final_params, f)
    print(f"Model saved to {save_filename}")
    
    # 6. Evaluation & Final Predictions
    print("Generating final predictions...")
    
    # Define prediction function (no gradients, no dropout)
    @jax.jit
    def predict(params, graph, rng):
        return net.apply(params, rng, graph, is_training=False).nodes

    # Use final_params (on CPU/Host or first device)
    # Re-use the graph (we need a single graph instance, not replicated)
    # 'graph' variable from earlier is the original single graph.
    
    # Predict
    eval_rng = jax.random.PRNGKey(args.seed + 999)
    outputs = predict(final_params, graph, eval_rng)
    test_mask_np = np.array(test_mask)
    
    if args.prediction_mode == 'classification':
        # Classification: Confusion matrix and classification report
        probs = jax.nn.softmax(outputs, axis=-1)
        preds = jnp.argmax(outputs, axis=-1)
        
        # Convert to numpy for sklearn
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        
        # Filter for Test set
        test_preds = preds_np[test_mask_np]
        test_targets = targets_np[test_mask_np]
        
        # Classification Report
        classes = ['Void', 'Wall', 'Filament', 'Cluster']
        print("\nClassification Report:")
        report = classification_report(test_targets, test_preds, target_names=classes)
        print(report)
        report_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_report_seed_{args.seed}_{timestamp}.txt')
        with open(report_filename, 'w') as f:
            f.write(report)
            
        # Save Predictions
        preds_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_predictions_seed_{args.seed}_{timestamp}.pkl')
        preds_data = {
            'probs': probs, # All nodes
            'preds': preds, # All nodes
            'targets': targets, # All nodes
            'test_mask': test_mask
        }
    else:
        # Regression: MSE, R², inverse transform
        preds_output = np.array(outputs)  # Model outputs 
        targets_output = np.array(targets)  # Targets in same space as model outputs

        if use_transformed_eig:
            # Model predicted transformed eigenvalues
            preds_scaled = preds_output
            targets_scaled = targets_output

            # Inverse transform to get raw eigenvalues
            if target_scaler is not None:
                preds_transformed_eig = target_scaler.inverse_transform(preds_scaled)
                targets_transformed_eig = target_scaler.inverse_transform(targets_scaled)
            else:
                preds_transformed_eig = preds_scaled
                targets_transformed_eig = targets_scaled

            # Convert to eigenvalues for physical interpretation
            preds_eigenvalues = increments_to_eigenvalues(preds_transformed_eig)
            targets_eigenvalues = eigenvalues_raw  # Already have raw eigenvalues

            # Filter for test set
            test_preds_shape = preds_transformed_eig[test_mask_np]
            test_targets_shape = targets_transformed_eig[test_mask_np]
            test_preds_eig = preds_eigenvalues[test_mask_np]
            test_targets_eig = targets_eigenvalues[test_mask_np]

            # Compute metrics in shape parameter space
            mse_shape = np.mean((test_preds_shape - test_targets_shape) ** 2)
            mae_shape = np.mean(np.abs(test_preds_shape - test_targets_shape))

            ss_res_shape = np.sum((test_targets_shape - test_preds_shape) ** 2, axis=0)
            ss_tot_shape = np.sum((test_targets_shape - np.mean(test_targets_shape, axis=0)) ** 2, axis=0)
            r2_shape = 1 - ss_res_shape / (ss_tot_shape + 1e-8)

            # Compute metrics in eigenvalue space (physical interpretation)
            mse_eig = np.mean((test_preds_eig - test_targets_eig) ** 2)
            mae_eig = np.mean(np.abs(test_preds_eig - test_targets_eig))

            ss_res_eig = np.sum((test_targets_eig - test_preds_eig) ** 2, axis=0)
            ss_tot_eig = np.sum((test_targets_eig - np.mean(test_targets_eig, axis=0)) ** 2, axis=0)
            r2_eig = 1 - ss_res_eig / (ss_tot_eig + 1e-8)

            print(f"\nRegression Metrics (Test Set):")
            print(f"\n  Transformed Eigenvalue Space (v₁, Δλ₂, Δλ₃):")
            print(f"    MSE: {mse_shape:.6f}")
            print(f"    MAE: {mae_shape:.6f}")
            print(f"    R² per param: v₁={r2_shape[0]:.4f}, Δλ₂={r2_shape[1]:.4f}, Δλ₃={r2_shape[2]:.4f}")
            print(f"    Mean R²: {np.mean(r2_shape):.4f}")
            print(f"\n  Eigenvalue Space (λ₁, λ₂, λ₃) - Physical:")
            print(f"    MSE: {mse_eig:.6f}")
            print(f"    MAE: {mae_eig:.6f}")
            print(f"    R² per eigenvalue: λ₁={r2_eig[0]:.4f}, λ₂={r2_eig[1]:.4f}, λ₃={r2_eig[2]:.4f}")
            print(f"    Mean R²: {np.mean(r2_eig):.4f}")

            # Save report
            report_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_transformed_eig_report_seed_{args.seed}_{timestamp}.txt')
            with open(report_filename, 'w') as f:
                f.write("Transformed Eigenvalue Regression Results\n")
                f.write("=" * 50 + "\n\n")
                f.write("Transformed Eigenvalue Space (v₁, Δλ₂, Δλ₃):\n")
                f.write(f"  MSE: {mse_shape:.6f}\n")
                f.write(f"  MAE: {mae_shape:.6f}\n")
                f.write(f"  R² per param: v₁={r2_shape[0]:.4f}, Δλ₂={r2_shape[1]:.4f}, Δλ₃={r2_shape[2]:.4f}\n")
                f.write(f"  Mean R²: {np.mean(r2_shape):.4f}\n\n")
                f.write("Eigenvalue Space (λ₁, λ₂, λ₃) - Physical:\n")
                f.write(f"  MSE: {mse_eig:.6f}\n")
                f.write(f"  MAE: {mae_eig:.6f}\n")
                f.write(f"  R² per eigenvalue: λ₁={r2_eig[0]:.4f}, λ₂={r2_eig[1]:.4f}, λ₃={r2_eig[2]:.4f}\n")
                f.write(f"  Mean R²: {np.mean(r2_eig):.4f}\n")
            print(f"Report saved to {report_filename}")

            # Save Predictions
            preds_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_transformed_eig_predictions_seed_{args.seed}_{timestamp}.pkl')
            preds_data = {
                'preds_transformed_eig': preds_transformed_eig,  # All nodes (I₁, e, p)
                'targets_transformed_eig': targets_transformed_eig,  # All nodes (I₁, e, p)
                'preds_eigenvalues': preds_eigenvalues,  # All nodes (λ₁, λ₂, λ₃)
                'targets_eigenvalues': targets_eigenvalues,  # All nodes (λ₁, λ₂, λ₃)
                'test_mask': test_mask,
                'stats': stats,
                'use_transformed_eig': True
            }
        else:
            # Legacy path: scaled eigenvalues
            preds_scaled = preds_output
            targets_scaled = targets_output

            # Inverse transform to get raw eigenvalues
            if target_scaler is not None:
                preds_raw = target_scaler.inverse_transform(preds_scaled)
                targets_raw = target_scaler.inverse_transform(targets_scaled)
            else:
                preds_raw = preds_scaled
                targets_raw = targets_scaled

            # Filter for Test set
            test_preds = preds_raw[test_mask_np]
            test_targets = targets_raw[test_mask_np]

            # Compute metrics
            mse = np.mean((test_preds - test_targets) ** 2)
            mae = np.mean(np.abs(test_preds - test_targets))

            # R² per eigenvalue
            ss_res = np.sum((test_targets - test_preds) ** 2, axis=0)
            ss_tot = np.sum((test_targets - np.mean(test_targets, axis=0)) ** 2, axis=0)
            r2_per_eig = 1 - ss_res / (ss_tot + 1e-8)

            print(f"\nRegression Metrics (Test Set):")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R² per eigenvalue: λ₁={r2_per_eig[0]:.4f}, λ₂={r2_per_eig[1]:.4f}, λ₃={r2_per_eig[2]:.4f}")
            print(f"  Mean R²: {np.mean(r2_per_eig):.4f}")

            # Save report
            report_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_report_seed_{args.seed}_{timestamp}.txt')
            with open(report_filename, 'w') as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R² per eigenvalue: λ₁={r2_per_eig[0]:.4f}, λ₂={r2_per_eig[1]:.4f}, λ₃={r2_per_eig[2]:.4f}\n")
                f.write(f"Mean R²: {np.mean(r2_per_eig):.4f}\n")
            print(f"Report saved to {report_filename}")

            # Save Predictions
            preds_filename = os.path.join(args.output_dir, f'jraph_{args.prediction_mode}_predictions_seed_{args.seed}_{timestamp}.pkl')
            preds_data = {
                'preds_scaled': preds_scaled, # All nodes (scaled)
                'preds_raw': preds_raw, # All nodes (raw eigenvalues)
                'targets_scaled': targets_scaled, # All nodes (scaled)
                'targets_raw': targets_raw, # All nodes (raw eigenvalues)
                'test_mask': test_mask,
                'eigenvalue_scaler': eigenvalue_scaler,
                'use_transformed_eig': False
            }
    
    with open(preds_filename, 'wb') as f:
        pickle.dump(preds_data, f)
    print(f"Predictions saved to {preds_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    
    # Model Hparams
    parser.add_argument("--latent_size", type=int, default=80)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_passes", type=int, default=8) # Used to be 4, increasing for more context
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-3) # Used to be 3e-3
    parser.add_argument("--prediction_mode", type=str, default="regression",
                        choices=["classification", "regression"],
                        help="Prediction mode: 'classification' for cosmic web classes, 'regression' for eigenvalues")
    parser.add_argument("--no_transformed_eig", action="store_true",
                        help="Disable softplus transformed eigenvalues to train on increments instead of eigenvalues")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_JRAPH_OUTPUT_DIR,
                        help="Directory to save models and predictions")
       
    args = parser.parse_args()
    
    # Pass args to main (we need to modify main signature or use globals, 
    # better to refactor main to accept args)
    # Since main is large, I'll pass args as a simple object or refactor main locally.
    # Actually, simplest is to just use 'args' global since main is called in __name__ == main
    # But I should update main definition to accept kwargs.
    
    main(args)
