'''
Creating graph net models Battaglia et al 2018 https://arxiv.org/abs/1806.01261 using jraph
'''
import functools
import jax
import jax.numpy as jnp
import jraph
import haiku as hk
from typing import Any, Callable, Dict, List, Optional, Tuple

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