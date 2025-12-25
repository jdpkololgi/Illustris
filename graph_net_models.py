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
    output_dim: int = 4, 
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
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.gelu,
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.gelu,
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
                     jax.nn.gelu,
                     hk.Linear(latent_size // num_heads, name = f'head_{i}_l2'),
                     jax.nn.gelu,
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
        
        # Decoder: simple linear layer for all modes
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
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.gelu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size),
                jax.nn.gelu,
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
                    jax.nn.gelu,
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l2'),
                    jax.nn.gelu,
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
    """
    head_dim = latent_size // num_heads
    
    def _network(
        graph: jraph.GraphsTuple, 
        theta: Optional[jnp.ndarray] = None,  # [N, output_dim] or None
        is_training: bool = True
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
        
        # ===== GNN ENCODER =====
        @jraph.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size, name='edge_l1'),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='edge_ln'),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='edge_l2'),
                jax.nn.gelu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='edge_l3'),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(latent_size, name='node_l1'),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='node_ln'),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(latent_size, name='node_l2'),
                jax.nn.gelu,
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
                    jax.nn.gelu,
                    hk.Linear(latent_size // num_heads, name=f'head_{i}_l2'),
                    jax.nn.gelu,
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

        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=hk.Linear(latent_size, name='embed_edge'),
            embed_node_fn=hk.Linear(latent_size, name='embed_node'),
        )
        graph = embedder(graph)

        for pass_idx in range(num_passes):
            with hk.experimental.name_scope(f'pass_{pass_idx}'):
                updated_graph = hk.remat(gn)(graph)
                graph = graph._replace(
                    nodes=graph.nodes + updated_graph.nodes,
                    edges=graph.edges + updated_graph.edges,
                )

        embeddings = graph.nodes
        
        # NOTE: Parameter-based flow omitted for brevity since flowjax is used externally
        return None, embeddings
    
    return _network