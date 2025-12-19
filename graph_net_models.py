'''
Creating graph net models Battaglia et al 2018 https://arxiv.org/abs/1806.01261 using jraph
'''
import functools
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
from typing import Any, Callable, Dict, List, Optional, Tuple
import distrax

# Hyperparameters
# Removed globals in favor of arguments

def make_graph_network(
    num_passes: int = 4,
    latent_size: int = 80,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    output_dim: int = 4 # set to 3 for eigenvalue regression
) -> Callable:
    """
    Creates a GraphNetwork with Multi-Head Attention.
    """
    head_dim = latent_size // num_heads
    
    def _network(graph: jraph.GraphsTuple, is_training: bool = True) -> jraph.GraphsTuple:
        # NOTE: We define update functions inside here to capture 'is_training' and params
        
        @jraph.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.relu,
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size), 
            ])
            return net(feats)
        
        @jraph.concatenated_args
        def attention_logit_fn(feats: jnp.ndarray) -> jnp.ndarray:
             # Attention mechanism
             head_logits = []
             for i in range(num_heads):
                 head_net = hk.Sequential([
                     hk.Linear(latent_size // num_heads, name = f'head_{i}_l1'),
                     jax.nn.relu,
                     hk.Linear(latent_size // num_heads, name = f'head_{i}_l2'),
                     jax.nn.relu,
                     hk.Linear(1, name = f'head_{i}_l3'),
                 ])
                 head_logits.append(head_net(feats))
             return jnp.concatenate(head_logits, axis=-1)

        def attention_reduce_fn(edges: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            num_edges = edges.shape[0]
            edges_per_head = edges.reshape(num_edges, num_heads, head_dim)
            weighted = edges_per_head * weights[:, :, None]
            return weighted.reshape(num_edges, latent_size)

        gn = jraph.GraphNetwork(
            update_edge_fn=edge_update_fn,       
            update_node_fn=node_update_fn,
            attention_logit_fn=attention_logit_fn,
            attention_normalize_fn=jraph.segment_softmax,
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
        )

        # Encoder
        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(latent_size),
            embed_node_fn=hk.Linear(latent_size),
        )
        graph = embedder(graph)

        # Message Passing Steps
        for _ in range(num_passes):
            # Gradient Checkpointing used here
            updated_graph = hk.remat(gn)(graph)

            # Residual connections
            graph = graph._replace(
                nodes=graph.nodes + updated_graph.nodes,
                edges=graph.edges + updated_graph.edges,
            )
        
        # Decoder
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=hk.Linear(output_dim) 
        )
        return decoder(graph)

    return _network


def make_gnn_encoder(
    num_passes: int = 4,
    latent_size: int = 80,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
) -> Callable:
    """
    Creates a GNN Encoder that outputs node embeddings only.
    
    This is designed to work with external flow libraries (e.g., Flowjax)
    for multi-GPU training compatibility.
    
    Args:
        num_passes: Number of message passing iterations
        latent_size: Dimension of node embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout probability during training
    
    Returns:
        A Haiku-transformable function that returns node embeddings [N, latent_size].
    """
    head_dim = latent_size // num_heads
    
    def _network(graph: jraph.GraphsTuple, is_training: bool = True) -> jnp.ndarray:
        """
        Forward pass returning node embeddings.
        
        Args:
            graph: Input graph with node/edge features
            is_training: Whether we're training (enables dropout)
        
        Returns:
            embeddings: Node embeddings [N, latent_size]
        """
        
        @jraph.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size), 
            ])
            return net(feats)
        
        @jraph.concatenated_args
        def attention_logit_fn(feats: jnp.ndarray) -> jnp.ndarray:
            head_logits = []
            for i in range(num_heads):
                head_net = hk.Sequential([
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l1'),
                    jax.nn.relu,
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l2'),
                    jax.nn.relu,
                    hk.Linear(1, name=f'head_{i}_l3'),
                ])
                head_logits.append(head_net(feats))
            return jnp.concatenate(head_logits, axis=-1)

        def attention_reduce_fn(edges: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            num_edges = edges.shape[0]
            edges_per_head = edges.reshape(num_edges, num_heads, head_dim)
            weighted = edges_per_head * weights[:, :, None]
            return weighted.reshape(num_edges, latent_size)

        gn = jraph.GraphNetwork(
            update_edge_fn=edge_update_fn,       
            update_node_fn=node_update_fn,
            attention_logit_fn=attention_logit_fn,
            attention_normalize_fn=jraph.segment_softmax,
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
        )

        # Encoder
        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(latent_size),
            embed_node_fn=hk.Linear(latent_size),
        )
        graph = embedder(graph)

        # Message Passing Steps
        for _ in range(num_passes):
            updated_graph = hk.remat(gn)(graph)
            graph = graph._replace(
                nodes=graph.nodes + updated_graph.nodes,
                edges=graph.edges + updated_graph.edges,
            )
        
        # Return node embeddings (no decoder)
        return graph.nodes

    return _network


def make_graph_network_sbi(
    num_passes: int = 4,
    latent_size: int = 80,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    # Flow-specific parameters
    output_dim: int = 3,          # Number of parameters to infer (3 eigenvalues)
    num_flow_layers: int = 5,     # Depth of the normalizing flow
    num_bins: int = 8,            # Spline bins (more = more expressive)
    flow_hidden_size: int = 128,  # Hidden layer size in conditioner
    range_min: float = -5.0,      # Spline support range (for scaled eigenvalues)
    range_max: float = 5.0,
) -> Callable:
    """
    Creates a GraphNetwork encoder + Conditional Spline Flow for SBI.
    
    This model learns the posterior distribution p(θ | graph) where θ are the
    eigenvalues (λ₁, λ₂, λ₃) for each node in the cosmic web graph.
    
    Architecture:
        Input Graph → GNN Encoder → Node Embeddings [N, latent_size]
                                          ↓
                            Conditional Spline Flow
                                          ↓
                            log p(θ_i | embedding_i) for each node
    
    Args:
        num_passes: Number of message passing iterations
        latent_size: Dimension of node embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout probability during training
        output_dim: Number of parameters to infer (3 for eigenvalues)
        num_flow_layers: Number of spline transformation layers
        num_bins: Number of bins for rational quadratic spline
        flow_hidden_size: Hidden layer size in conditioner networks
        range_min: Minimum value of spline support
        range_max: Maximum value of spline support
    
    Returns:
        A Haiku-transformable function.
    """
    head_dim = latent_size // num_heads
    
    def _network(
        graph: jraph.GraphsTuple, 
        theta: Optional[jnp.ndarray] = None,  # [N, output_dim] or None
        is_training: bool = True
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
        """
        Forward pass of the SBI model.
        
        Args:
            graph: Input graph with node/edge features
            theta: Target parameters (eigenvalues). Shape: [N, output_dim]
                   Required during training for computing log_prob.
                   If None, returns only embeddings for sampling.
            is_training: Whether we're training (enables dropout)
        
        Returns:
            log_prob: Log probability of theta for each node [N] (None if theta not provided)
            embeddings: Node embeddings after message passing [N, latent_size]
        """
        
        # ===== GNN ENCODER (same architecture as make_graph_network) =====
        @jraph.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size, name='edge_l1'),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='edge_ln'),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='edge_l2'),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='edge_l3'),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size, name='node_l1'),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='node_ln'),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='node_l2'),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='node_l3'),
            ])
            return net(feats)

        @jraph.concatenated_args
        def attention_logit_fn(feats: jnp.ndarray) -> jnp.ndarray:
            head_logits = []
            for i in range(num_heads):
                head_net = hk.Sequential([
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l1'),
                    jax.nn.relu,
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l2'),
                    jax.nn.relu,
                    hk.Linear(1, name=f'head_{i}_l3'),
                ])
                head_logits.append(head_net(feats))
            return jnp.concatenate(head_logits, axis=-1)

        def attention_reduce_fn(edges: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            num_edges_local = edges.shape[0]
            edges_per_head = edges.reshape(num_edges_local, num_heads, head_dim)
            weighted = edges_per_head * weights[:, :, None]
            return weighted.reshape(num_edges_local, latent_size)

        gn = jraph.GraphNetwork(
            update_edge_fn=edge_update_fn,
            update_node_fn=node_update_fn,
            attention_logit_fn=attention_logit_fn,
            attention_normalize_fn=jraph.segment_softmax,
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
        )

        # Encoder: embed initial features
        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(latent_size, name='embed_edge'),
            embed_node_fn=hk.Linear(latent_size, name='embed_node'),
        )
        graph = embedder(graph)

        # Message passing with residual connections
        for pass_idx in range(num_passes):
            with hk.experimental.name_scope(f'pass_{pass_idx}'):
                updated_graph = hk.remat(gn)(graph)
                graph = graph._replace(
                    nodes=graph.nodes + updated_graph.nodes,
                    edges=graph.edges + updated_graph.edges,
                )

        # Node embeddings after message passing: [N, latent_size]
        embeddings = graph.nodes
        
        # ===== CONDITIONAL SPLINE FLOW (replaces linear decoder) =====
        # We use a coupling-style flow where each layer transforms all dimensions
        # conditioned on the node embedding.
        
        # Number of parameters needed for rational quadratic spline per dimension:
        # distrax expects params of shape [..., 3 * num_bins + 1]
        # which it internally splits into bin widths, heights, and knot slopes
        params_per_dim = 3 * num_bins + 1
        total_flow_params = output_dim * params_per_dim
        
        def conditioner_fn(embedding: jnp.ndarray, layer_idx: int) -> jnp.ndarray:
            """
            Conditioner network: maps node embedding to spline parameters.
            
            Args:
                embedding: Node embedding [latent_size]
                layer_idx: Which flow layer (for unique parameter naming)
            
            Returns:
                Spline parameters [output_dim, params_per_dim]
            """
            net = hk.Sequential([
                hk.Linear(flow_hidden_size, name=f'flow_{layer_idx}_cond_l1'),
                jax.nn.relu,
                hk.Linear(flow_hidden_size, name=f'flow_{layer_idx}_cond_l2'),
                jax.nn.relu,
                hk.Linear(total_flow_params, 
                          w_init=hk.initializers.Constant(0.0),
                          b_init=hk.initializers.Constant(0.0),
                          name=f'flow_{layer_idx}_cond_out'),
            ])
            # Reshape to [output_dim, params_per_dim]
            return net(embedding).reshape(output_dim, params_per_dim)

        def compute_log_prob_single(embedding: jnp.ndarray, theta_single: jnp.ndarray) -> jnp.ndarray:
            """
            Compute log p(theta | embedding) for a single node.
            
            Uses jax.lax.fori_loop for pmap compatibility.
            
            Args:
                embedding: [latent_size]
                theta_single: [output_dim]
            
            Returns:
                log_prob: scalar
            """
            # Get all conditioner outputs upfront (one per layer)
            # This ensures Haiku sees all parameters during tracing
            all_spline_params = []
            for layer_idx in range(num_flow_layers):
                all_spline_params.append(conditioner_fn(embedding, layer_idx))
            all_spline_params = jnp.stack(all_spline_params)  # [num_flow_layers, output_dim, params_per_dim]
            
            def apply_inverse_dim(params_dim, z_dim):
                """Apply inverse spline to a single dimension."""
                spline = distrax.RationalQuadraticSpline(
                    params_dim, range_min=range_min, range_max=range_max
                )
                z_new, log_det = spline.inverse_and_log_det(z_dim)
                return z_new, log_det
            
            def inverse_layer_body(layer_idx_from_end, carry):
                """Body function for jax.lax.fori_loop over layers."""
                z, log_det_total = carry
                # Convert to forward layer index (we go backward through layers)
                layer_idx = num_flow_layers - 1 - layer_idx_from_end
                
                # Get spline params for this layer: [output_dim, params_per_dim]
                spline_params = all_spline_params[layer_idx]
                
                # Apply inverse spline to all dimensions in parallel
                z_new, log_dets = jax.vmap(apply_inverse_dim)(spline_params, z)
                log_det_layer = jnp.sum(log_dets)
                
                # Apply permutation (except after last layer going backward, i.e., first forward layer)
                perm = jnp.roll(jnp.arange(output_dim), -1)
                z_new = jax.lax.cond(
                    layer_idx > 0,
                    lambda x: x[perm],
                    lambda x: x,
                    z_new
                )
                
                return (z_new, log_det_total + log_det_layer)
            
            # Initialize and run loop
            init_carry = (theta_single, 0.0)
            z_final, log_det_total = jax.lax.fori_loop(
                0, num_flow_layers, inverse_layer_body, init_carry
            )
            
            # Base distribution log prob: standard normal
            base_log_prob = -0.5 * jnp.sum(z_final ** 2) - 0.5 * output_dim * jnp.log(2 * jnp.pi)
            
            return base_log_prob + log_det_total

        def sample_single(embedding: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
            """
            Sample theta ~ p(theta | embedding) for a single node.
            
            Uses jax.lax.fori_loop for pmap compatibility.
            
            Args:
                embedding: [latent_size]
                rng_key: Random key for sampling
            
            Returns:
                theta_sample: [output_dim]
            """
            # Get all conditioner outputs upfront
            all_spline_params = []
            for layer_idx in range(num_flow_layers):
                all_spline_params.append(conditioner_fn(embedding, layer_idx))
            all_spline_params = jnp.stack(all_spline_params)  # [num_flow_layers, output_dim, params_per_dim]
            
            def apply_forward_dim(params_dim, z_dim):
                """Apply forward spline to a single dimension."""
                spline = distrax.RationalQuadraticSpline(
                    params_dim, range_min=range_min, range_max=range_max
                )
                return spline.forward(z_dim)
            
            def forward_layer_body(layer_idx, carry):
                """Body function for jax.lax.fori_loop over layers."""
                z = carry
                
                # Get spline params for this layer
                spline_params = all_spline_params[layer_idx]
                
                # Apply forward spline to all dimensions in parallel
                z_new = jax.vmap(apply_forward_dim)(spline_params, z)
                
                # Apply permutation (except after last layer)
                perm = jnp.roll(jnp.arange(output_dim), 1)
                z_new = jax.lax.cond(
                    layer_idx < num_flow_layers - 1,
                    lambda x: x[perm],
                    lambda x: x,
                    z_new
                )
                
                return z_new
            
            # Sample from base distribution
            z = jax.random.normal(rng_key, shape=(output_dim,))
            
            # Run forward loop
            theta = jax.lax.fori_loop(0, num_flow_layers, forward_layer_body, z)
            
            return theta

        # ===== COMPUTE OUTPUTS =====
        if theta is not None:
            # Training mode: compute log p(θ_i | embedding_i) for each node
            log_probs = jax.vmap(compute_log_prob_single)(embeddings, theta)  # [N]
            return log_probs, embeddings
        else:
            # Inference mode: return embeddings and sampling function
            # The sampling function will be created in the pipeline
            return None, embeddings
    
    # Attach helper for sampling (can be called after getting embeddings)
    def sample_posterior(
        embeddings: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        num_samples: int = 1000
    ) -> jnp.ndarray:
        """
        Sample from posterior for all nodes.
        
        Note: This function needs to be called within the same Haiku context
        as the forward pass to access the same parameters.
        
        Args:
            embeddings: Node embeddings [N, latent_size]
            rng_key: Random key
            num_samples: Number of samples per node
        
        Returns:
            samples: [N, num_samples, output_dim]
        """
        num_nodes = embeddings.shape[0]
        
        # Generate keys for each node and sample
        keys = jax.random.split(rng_key, num_nodes * num_samples)
        keys = keys.reshape(num_nodes, num_samples, 2)
        
        def sample_node(embedding, node_keys):
            return jax.vmap(lambda k: sample_single(embedding, k))(node_keys)
        
        # This needs sample_single defined in scope - for now return placeholder
        # Full implementation requires refactoring to share the sampling logic
        raise NotImplementedError("Use the sampling function in the pipeline instead")
    
    return _network