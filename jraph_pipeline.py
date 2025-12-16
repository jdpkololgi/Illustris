import os
import sys

# Force priority for user installed packages to resolve numpy/scipy/astropy conflicts
user_site = "/global/homes/d/dkololgi/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

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

def generate_data(masscut, cache_path, version='v2'):
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
    labels = jnp.array(targets.values, dtype=jnp.int32)
    
    # Save to cache
    print(f"Saving generated data to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({'graph': graph, 'labels': labels, 'masks': masks}, f)
    
    return graph, labels, masks

def load_data(masscut=1e9, use_v2=True):
    """
    Load data from cache if available, otherwise generate it.
    Can switch between v1 and v2 stats.
    """
    version = 'v2' if use_v2 else 'v1'
    cache_path = f"processed_jraph_data_mc{masscut:.0e}_{version}_scaled_2.pkl"
    pyg_cache_path = f"processed_gcn_data_mc{masscut:.0e}.pt"

    if os.path.exists(cache_path):
        print(f"Loading cached Jraph data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            return data['graph'], data['labels'], data['masks']
            
    # Fallback for V1: Try loading PyG cache if exists
    if not use_v2 and os.path.exists(pyg_cache_path):
        print(f"Loading cached PyG data from {pyg_cache_path}...")
        try:
            data_tuple = torch.load(pyg_cache_path, weights_only=False)
            netx_geom = data_tuple[0]
            # Convert
            return convert_pyg_to_jraph(netx_geom)
        except Exception as e:
            print(f"Failed to load PyG cache: {e}. Generating fresh.")
            
    # Generate fresh
    return generate_data(masscut, cache_path, version=version)


def calculate_class_weights(targets):
    """Calculate class weights using sklearn."""
    try:
        classes = np.unique(targets)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=targets)
        return jnp.array(weights, dtype=jnp.float32)
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}")
        return jnp.ones(4, dtype=jnp.float32)





def main(args):
    print(f"JAX Devices: {jax.devices()}")
    num_devices = jax.local_device_count()
    print(f"Running on {num_devices} devices.")
    
    # generate unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Job Timestamp: {timestamp}")

    # 1. Load Data
    masscut = 1e9
    graph, labels, masks = load_data(masscut=masscut, use_v2=True)
    train_mask, val_mask, test_mask = masks
    
    class_weights = calculate_class_weights(np.array(labels))
    
    print(f"Graph stats: Nodes={graph.n_node[0]}, Edges={graph.n_edge[0]}")
    print(f"Train size: {jnp.sum(train_mask)}, Val size: {jnp.sum(val_mask)}, Test size: {jnp.sum(test_mask)}")
    
    # 2. Model Setup
    # Initialize network
    # make_graph_network returns a function that takes a graph
    # Reduced passes to 2 to prevent OOM on single GPU - RESTORED to 4 for Compute Node
    net_fn = make_graph_network(
        num_passes=args.num_passes, # Used to be 4, increasing for more context 
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
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
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=args.lr, # Match GCN peak
        warmup_steps=500,
        decay_steps=num_epochs,
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
    
    # Replicate labels
    replicated_labels = jax.device_put_replicated(labels, jax.local_devices())
    replicated_class_weights = jax.device_put_replicated(class_weights, jax.local_devices())

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
    def loss_fn(params, graph, labels, mask, net_apply, rng, class_weights, label_smoothing=0.1):
        num_classes = 4
        # Pass is_training=True for Dropout
        logits = net_apply(params, rng, graph, is_training=True).nodes
        
        # Cross Entropy Loss
        # logits: [N, C], labels: [N]
        # one_hot labels
        labels_one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        smoothed_labels = optax.smooth_labels(labels_one_hot, alpha=label_smoothing)
        # optax loss
        per_node_loss = optax.softmax_cross_entropy(logits, smoothed_labels)
        
        # Apply class weights
        weights = jnp.take(class_weights, labels)
        weighted_loss = per_node_loss * weights
        
        # Mask
        masked_loss = weighted_loss * mask
        
        # Mean over masked nodes
        # Avoid division by zero
        num_masked = jnp.sum(mask)
        loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
        
        return loss, (logits, num_masked)

    # Update Function
    def update(params, opt_state, graph, labels, mask, rng, class_weights):
        # rng mix
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Gradients
        (train_loss, (logits, num_masked)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, graph, labels, mask, net.apply, step_rng, class_weights
        )
        
        # Sync gradients across devices (average)
        grads = jax.lax.pmean(grads, axis_name='i')
        
        # Sync Loss metrics for reporting
        # We sum the loss*count from each device and divide by total count
        total_loss_part = train_loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        # Calculate Training Accuracy
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == labels) & mask
        
        total_correct = jax.lax.psum(jnp.sum(correct), axis_name='i')
        # total_count matches total_mask roughly, but num_masked is float from loss_fn?
        # loss_fn returns num_masked = jnp.sum(mask)
        # So total_count is correct denominator.
        global_acc = total_correct / jnp.maximum(total_count, 1.0)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, global_loss, global_acc
        
    update = jax.pmap(update, axis_name='i')

    def evaluate(params, graph, labels, mask, rng, class_weights):
        # We can evaluate in parallel too
        step_rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))
        
        # Pass is_training=False to disable Dropout
        # We need to manually invoke the model here because loss_fn assumes training?
        # Re-implementing evaluating part of loss_fn without gradients:
        
        logits = net.apply(params, step_rng, graph, is_training=False).nodes
        
        labels_one_hot = jax.nn.one_hot(labels, num_classes=4)
        
        labels_one_hot = jax.nn.one_hot(labels, num_classes=4)
        
        # Cross Entropy Logic
        per_node_loss = optax.softmax_cross_entropy(logits, labels_one_hot)
        weights = jnp.take(class_weights, labels)
        masked_loss = per_node_loss * weights * mask
        
        num_masked = jnp.sum(mask)
        loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)
        
        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == labels) & mask
        
        total_correct = jax.lax.psum(jnp.sum(correct), axis_name='i')
        total_mask = jax.lax.psum(jnp.sum(mask), axis_name='i')
        
        accuracy = total_correct / jnp.maximum(total_mask, 1.0)
        
        # Global loss logic
        total_loss_part = loss * num_masked
        total_count = jax.lax.psum(num_masked, axis_name='i')
        global_loss = jax.lax.psum(total_loss_part, axis_name='i') / jnp.maximum(total_count, 1.0)
        
        return global_loss, accuracy
        
    evaluate = jax.pmap(evaluate, axis_name='i')

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
        
        replicated_params, replicated_opt_state, train_loss, train_acc = update(
            replicated_params, replicated_opt_state, 
            replicated_graph, replicated_labels, sharded_train_masks, 
            step_rngs, replicated_class_weights
        )
        
        # Validation
        if epoch % report_every == 0:
            val_loss, val_acc = evaluate(
                replicated_params, replicated_graph, replicated_labels, 
                sharded_val_masks, step_rngs, replicated_class_weights
            )
            # take first device result (they are identical due to pmean/psum)
            print(f"Epoch {epoch} | Train Loss: {train_loss[0]:.4f} | Train Acc: {train_acc[0]*100:.3f}% | Val Loss: {val_loss[0]:.4f} | Val Acc: {val_acc[0]*100:.3f}%")

    print(f"Training finished in {time.time() - t0:.2f}s")
    
    # Save Model
    # Take params from first device
    final_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], replicated_params))
    
    # Save model
    # Save model
    print("Saving model...")
    import pickle
    save_filename = f'jraph_model_seed_{args.seed}_{timestamp}.pkl'
    with open(save_filename, 'wb') as f:
        pickle.dump(final_params, f)
    print(f"Model saved to {save_filename}")
    
    # 6. Evaluation & Confusion Matrix
    print("Generating predictions for confusion matrix...")
    
    # Define prediction function (no gradients, no dropout)
    @jax.jit
    def predict(params, graph, rng):
        return net.apply(params, rng, graph, is_training=False).nodes

    # Use final_params (on CPU/Host or first device)
    # Re-use the graph (we need a single graph instance, not replicated)
    # 'graph' variable from earlier is the original single graph.
    
    # Predict
    eval_rng = jax.random.PRNGKey(args.seed + 999)
    logits = predict(final_params, graph, eval_rng)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    
    # Convert to numpy for sklearn
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    test_mask_np = np.array(test_mask)
    
    # Filter for Test set
    test_preds = preds_np[test_mask_np]
    test_labels = labels_np[test_mask_np]
    
    # Plot Confusion Matrix
    classes = ['Void', 'Wall', 'Filament', 'Cluster']
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=classes)
    print(report)
    with open(f'jraph_classification_report_seed_{args.seed}_{timestamp}.txt', 'w') as f:
        f.write(report)
        
    # Save Predictions
    preds_filename = f'jraph_predictions_seed_{args.seed}_{timestamp}.pkl'
    preds_data = {
        'probs': probs, # All nodes
        'preds': preds, # All nodes
        'labels': labels, # All nodes
        'test_mask': test_mask
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
       
    args = parser.parse_args()
    
    # Pass args to main (we need to modify main signature or use globals, 
    # better to refactor main to accept args)
    # Since main is large, I'll pass args as a simple object or refactor main locally.
    # Actually, simplest is to just use 'args' global since main is called in __name__ == main
    # But I should update main definition to accept kwargs.
    
    main(args)
