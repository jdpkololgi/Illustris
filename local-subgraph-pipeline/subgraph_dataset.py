"""
Local subgraph (ego-graph) extraction utilities for Phase A.

This module is intentionally independent from the existing transductive pipeline:
- It consumes an already-built global `jraph.GraphsTuple` (from cache)
- It produces many small per-center ego-graphs for inductive training

Notes
-----
The cached graph does not currently include xyz positions, so the "locality" here is
defined in *graph hop distance* (k-hop neighborhood) rather than metric radius.
For the Delaunay-based graph, hop distance is still a reasonable proxy for locality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np
import jax.numpy as jnp
import jraph


@dataclass(frozen=True)
class CsrEdges:
    """CSR structure storing outgoing edge indices per node."""

    n_node: int
    indptr: np.ndarray  # shape [n_node + 1], int64
    edge_indices: np.ndarray  # shape [n_edge], int64

    def outgoing_edges(self, node: int) -> np.ndarray:
        start = int(self.indptr[node])
        end = int(self.indptr[node + 1])
        return self.edge_indices[start:end]


def build_outgoing_edge_csr(senders: np.ndarray, n_node: int) -> CsrEdges:
    """
    Build CSR over nodes -> outgoing edge indices.

    Args:
        senders: [E] sender node ids (int32/int64)
        n_node: number of nodes
    """
    senders = np.asarray(senders, dtype=np.int64)
    counts = np.bincount(senders, minlength=n_node).astype(np.int64)
    indptr = np.zeros(n_node + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    # Stable edge index placement by sender buckets
    edge_indices = np.empty_like(senders, dtype=np.int64)
    cursor = indptr[:-1].copy()
    for e, s in enumerate(senders):
        pos = cursor[s]
        edge_indices[pos] = e
        cursor[s] += 1

    return CsrEdges(n_node=n_node, indptr=indptr, edge_indices=edge_indices)


def k_hop_nodes(
    csr: CsrEdges,
    receivers: np.ndarray,
    center: int,
    k_hops: int,
    max_nodes: int,
) -> np.ndarray:
    """
    Collect nodes in the k-hop out-neighborhood from `center` with a hard cap.

    Returns:
        nodes: 1D np.ndarray of global node ids, with `center` guaranteed first.
    """
    if k_hops < 0:
        raise ValueError("k_hops must be >= 0")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")

    visited = np.zeros(csr.n_node, dtype=bool)
    visited[center] = True
    nodes = [int(center)]

    frontier = [int(center)]
    receivers = np.asarray(receivers, dtype=np.int64)

    for _ in range(k_hops):
        if not frontier:
            break
        next_frontier: list[int] = []
        for u in frontier:
            for eidx in csr.outgoing_edges(u):
                v = int(receivers[eidx])
                if not visited[v]:
                    visited[v] = True
                    nodes.append(v)
                    next_frontier.append(v)
                    if len(nodes) >= max_nodes:
                        return np.asarray(nodes, dtype=np.int64)
        frontier = next_frontier

    return np.asarray(nodes, dtype=np.int64)


def induced_subgraph_from_nodes(
    graph: jraph.GraphsTuple,
    csr: CsrEdges,
    center: int,
    nodes_global: np.ndarray,
    max_edges: int,
) -> jraph.GraphsTuple:
    """
    Build an induced directed subgraph restricted to `nodes_global`.

    Convention: The center node is remapped to index 0.
    """
    nodes_global = np.asarray(nodes_global, dtype=np.int64)
    if nodes_global[0] != center:
        raise ValueError("nodes_global[0] must be the center")
    if max_edges < 1:
        raise ValueError("max_edges must be >= 1")

    senders = np.asarray(graph.senders, dtype=np.int64)
    receivers = np.asarray(graph.receivers, dtype=np.int64)
    edge_feats = np.asarray(graph.edges)

    # Remap globals -> locals without allocating an O(N) array
    node_to_local = {int(g): int(i) for i, g in enumerate(nodes_global.tolist())}
    node_set = set(node_to_local.keys())

    sub_senders: list[int] = []
    sub_receivers: list[int] = []
    sub_edges: list[np.ndarray] = []

    for u_global in nodes_global:
        u_local = node_to_local[int(u_global)]
        for eidx in csr.outgoing_edges(int(u_global)):
            v_global = int(receivers[eidx])
            if v_global not in node_set:
                continue
            v_local = node_to_local[v_global]
            sub_senders.append(u_local)
            sub_receivers.append(v_local)
            sub_edges.append(edge_feats[eidx])
            if len(sub_senders) >= max_edges:
                break
        if len(sub_senders) >= max_edges:
            break

    if len(sub_senders) == 0:
        # Degenerate: no edges. Create an empty edge set of the correct feature dim.
        edge_dim = int(edge_feats.shape[-1]) if edge_feats.ndim == 2 else 1
        sub_edges_arr = np.zeros((0, edge_dim), dtype=edge_feats.dtype)
    else:
        sub_edges_arr = np.stack(sub_edges, axis=0)

    sub_graph = jraph.GraphsTuple(
        nodes=jnp.asarray(np.asarray(graph.nodes)[nodes_global]),
        edges=jnp.asarray(sub_edges_arr),
        senders=jnp.asarray(np.asarray(sub_senders, dtype=np.int32)),
        receivers=jnp.asarray(np.asarray(sub_receivers, dtype=np.int32)),
        n_node=jnp.asarray(np.asarray([nodes_global.shape[0]], dtype=np.int32)),
        n_edge=jnp.asarray(np.asarray([len(sub_senders)], dtype=np.int32)),
        globals=jnp.zeros((1, 1), dtype=jnp.float32),  # placeholder, batchable
    )
    return sub_graph


def padded_induced_subgraph_from_nodes(
    graph: jraph.GraphsTuple,
    csr: CsrEdges,
    center: int,
    nodes_global: np.ndarray,
    max_nodes: int,
    max_edges: int,
) -> jraph.GraphsTuple:
    """
    Build a fixed-shape induced subgraph with padding.

    Why this exists
    ---------------
    Variable node/edge counts per batch lead to *many* distinct XLA compile shapes,
    which can OOM LLVM over multiple epochs. This function pads each ego-graph to
    static shapes so training can be reliably jitted / compiled once.

    Padding strategy
    ----------------
    - Reserve local node id 0 as a **dummy node**.
    - Real nodes are mapped to local ids 1..R (center node is local id 1).
    - Pad node features with zeros up to `max_nodes + 1` (including dummy).
    - Pad edges up to `max_edges`; padded edges are (0 -> 0) with zero features,
      so they only interact with the dummy node and cannot contaminate the real center.
    """
    nodes_global = np.asarray(nodes_global, dtype=np.int64)
    if nodes_global[0] != center:
        raise ValueError("nodes_global[0] must be the center")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")
    if max_edges < 1:
        raise ValueError("max_edges must be >= 1")

    # Limit real nodes to max_nodes (center is included)
    nodes_global = nodes_global[:max_nodes]
    real_n = int(nodes_global.shape[0])

    node_feats = np.asarray(graph.nodes)
    edge_feats = np.asarray(graph.edges)
    receivers = np.asarray(graph.receivers, dtype=np.int64)

    node_dim = int(node_feats.shape[1]) if node_feats.ndim == 2 else 1
    edge_dim = int(edge_feats.shape[1]) if edge_feats.ndim == 2 else 1

    # Node features padded (dummy at 0)
    padded_nodes = np.zeros((max_nodes + 1, node_dim), dtype=node_feats.dtype)
    padded_nodes[1 : real_n + 1] = node_feats[nodes_global]

    # Remap globals -> locals (shift by +1 because of dummy)
    node_to_local = {int(g): int(i + 1) for i, g in enumerate(nodes_global.tolist())}
    node_set = set(node_to_local.keys())

    sub_senders = np.zeros((max_edges,), dtype=np.int32)
    sub_receivers = np.zeros((max_edges,), dtype=np.int32)
    sub_edges = np.zeros((max_edges, edge_dim), dtype=edge_feats.dtype)

    m = 0
    for u_global in nodes_global:
        u_local = node_to_local[int(u_global)]
        for eidx in csr.outgoing_edges(int(u_global)):
            v_global = int(receivers[eidx])
            if v_global not in node_set:
                continue
            v_local = node_to_local[v_global]
            sub_senders[m] = np.int32(u_local)
            sub_receivers[m] = np.int32(v_local)
            sub_edges[m] = edge_feats[eidx]
            m += 1
            if m >= max_edges:
                break
        if m >= max_edges:
            break

    # Static-shape GraphsTuple (counts are fixed for XLA shape stability)
    sub_graph = jraph.GraphsTuple(
        nodes=jnp.asarray(padded_nodes),
        edges=jnp.asarray(sub_edges),
        senders=jnp.asarray(sub_senders),
        receivers=jnp.asarray(sub_receivers),
        n_node=jnp.asarray(np.asarray([max_nodes + 1], dtype=np.int32)),
        n_edge=jnp.asarray(np.asarray([max_edges], dtype=np.int32)),
        globals=jnp.zeros((1, 1), dtype=jnp.float32),
    )
    return sub_graph


@dataclass
class SubgraphBuilder:
    """Reusable builder to avoid rebuilding CSR / converting arrays each batch."""

    graph: jraph.GraphsTuple
    csr: CsrEdges
    receivers: np.ndarray
    targets: np.ndarray

    @classmethod
    def from_cache_arrays(cls, graph: jraph.GraphsTuple, targets: np.ndarray) -> "SubgraphBuilder":
        n_node_total = int(graph.nodes.shape[0])
        csr = build_outgoing_edge_csr(np.asarray(graph.senders), n_node_total)
        receivers = np.asarray(graph.receivers, dtype=np.int64)
        targets = np.asarray(targets)
        return cls(graph=graph, csr=csr, receivers=receivers, targets=targets)

    def batch(
        self,
        center_nodes: Sequence[int],
        k_hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        graphs = []
        thetas = []
        for c in center_nodes:
            nodes_global = k_hop_nodes(self.csr, self.receivers, int(c), k_hops=k_hops, max_nodes=max_nodes)
            sg = induced_subgraph_from_nodes(self.graph, self.csr, int(c), nodes_global, max_edges=max_edges)
            graphs.append(sg)
            thetas.append(self.targets[int(c)])
        batched = jraph.batch(graphs)
        return batched, jnp.asarray(np.stack(thetas, axis=0))


@dataclass
class PaddedSubgraphBuilder:
    """Builder that pads every ego-graph to static shapes (stable XLA compilation)."""

    graph: jraph.GraphsTuple
    csr: CsrEdges
    receivers: np.ndarray
    targets: np.ndarray

    @classmethod
    def from_cache_arrays(cls, graph: jraph.GraphsTuple, targets: np.ndarray) -> "PaddedSubgraphBuilder":
        n_node_total = int(graph.nodes.shape[0])
        csr = build_outgoing_edge_csr(np.asarray(graph.senders), n_node_total)
        receivers = np.asarray(graph.receivers, dtype=np.int64)
        targets = np.asarray(targets)
        return cls(graph=graph, csr=csr, receivers=receivers, targets=targets)

    def batch(
        self,
        center_nodes: Sequence[int],
        k_hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        graphs = []
        thetas = []
        for c in center_nodes:
            nodes_global = k_hop_nodes(self.csr, self.receivers, int(c), k_hops=k_hops, max_nodes=max_nodes)
            sg = padded_induced_subgraph_from_nodes(
                self.graph, self.csr, int(c), nodes_global, max_nodes=max_nodes, max_edges=max_edges
            )
            graphs.append(sg)
            thetas.append(self.targets[int(c)])
        batched = jraph.batch(graphs)
        return batched, jnp.asarray(np.stack(thetas, axis=0))


def iter_center_batches(
    centers: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[np.ndarray]:
    centers = np.asarray(centers, dtype=np.int64)
    if shuffle:
        centers = centers.copy()
        rng.shuffle(centers)
    for i in range(0, len(centers), batch_size):
        yield centers[i : i + batch_size]


