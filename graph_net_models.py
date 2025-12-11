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
LATENT_SIZE = 80
NUM_HEADS = 8
HEAD_DIM = LATENT_SIZE // NUM_HEADS
NUM_CLASSES = 4


def make_graph_network(num_passes: int = 4) -> Callable:
    """
    Creates a GraphNetwork with Multi-Head Attention (8 Heads).
    Features:
    - Latent Size: 80
    - Dropout: 0.1 (if is_training=True)
    - Gradient Checkpointing: Yes
    """
    
    def _network(graph: jraph.GraphsTuple, is_training: bool = True) -> jraph.GraphsTuple:
        # Define Dropout Rate
        dropout_rate = 0.1

        # NOTE: We define update functions inside here to capture 'is_training'
        
        @jraph.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(LATENT_SIZE),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(LATENT_SIZE),
                jax.nn.relu,
                # Dropout
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(LATENT_SIZE),
            ])
            return net(feats)

        @jraph.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([
                hk.Linear(LATENT_SIZE),
                jax.nn.relu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(LATENT_SIZE),
                jax.nn.relu,
                lambda x: hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x,
                hk.Linear(LATENT_SIZE), 
            ])
            return net(feats)
        
        @jraph.concatenated_args
        def attention_logit_fn(feats: jnp.ndarray) -> jnp.ndarray:
             # Attention mechanism
             head_logits = []
             for i in range(NUM_HEADS):
                 head_net = hk.Sequential([
                     hk.Linear(LATENT_SIZE // NUM_HEADS, name = f'head_{i}_l1'),
                     jax.nn.relu,
                     hk.Linear(LATENT_SIZE // NUM_HEADS, name = f'head_{i}_l2'),
                     jax.nn.relu,
                     hk.Linear(1, name = f'head_{i}_l3'),
                 ])
                 head_logits.append(head_net(feats))
             return jnp.concatenate(head_logits, axis=-1)

        def attention_reduce_fn(edges: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            num_edges = edges.shape[0]
            edges_per_head = edges.reshape(num_edges, NUM_HEADS, HEAD_DIM)
            weighted = edges_per_head * weights[:, :, None]
            return weighted.reshape(num_edges, LATENT_SIZE)

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
            embed_edge_fn=hk.Linear(LATENT_SIZE),
            embed_node_fn=hk.Linear(LATENT_SIZE),
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
            embed_node_fn=hk.Linear(NUM_CLASSES) 
        )
        return decoder(graph)

    return _network