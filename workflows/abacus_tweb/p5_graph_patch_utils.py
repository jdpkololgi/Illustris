#!/usr/bin/env python3
"""Pure NumPy utilities for exact P5 canonical-graph patch views.

The P2b graph is stored once as undirected canonical pairs.  Production
GraphNet inputs expand every pair into the two directed messages used by the
existing Jraph cache builder.  Reverse dependency traversal is therefore the
incident-edge traversal of the canonical undirected graph.  No graph metric is
ever recomputed inside a patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GraphPatch:
    """Unpadded exact graph view for one core or core subdivision."""

    core_id: int
    fold: int
    num_passes: int
    dependency_hops: int
    parent_node_id: np.ndarray
    node_features: np.ndarray
    union_edge_id: np.ndarray
    edge_features: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    authoritative_core_mask: np.ndarray
    loss_mask: np.ndarray

    @property
    def n_node(self) -> int:
        return int(len(self.parent_node_id))

    @property
    def n_edge(self) -> int:
        return int(len(self.senders))


def reverse_edge_features(edge_features: np.ndarray) -> np.ndarray:
    """Apply the established P2/Jraph reverse-edge convention."""
    rev = np.asarray(edge_features, dtype=np.float32).copy()
    rev[:, 1:4] *= -1.0
    rev[:, 4] = 1.0 / np.maximum(rev[:, 4], np.float32(1e-6))
    return rev


def make_bidirectional(
    pairs: np.ndarray, edge_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand canonical pair order to originals followed by reverses."""
    pairs = np.asarray(pairs)
    edge_features = np.asarray(edge_features, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (E,2), got {pairs.shape}")
    if len(edge_features) != len(pairs):
        raise ValueError("edge feature count does not match pair count")
    senders = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    receivers = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)
    attrs = np.concatenate([edge_features, reverse_edge_features(edge_features)])
    return senders, receivers, attrs


def reverse_k_hop_directed(
    core_nodes: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    num_passes: int,
) -> np.ndarray:
    """Reference reverse dependency traversal for an arbitrary directed graph."""
    if num_passes < 0:
        raise ValueError("num_passes must be non-negative")
    nodes = np.unique(np.asarray(core_nodes, dtype=np.int64))
    senders = np.asarray(senders, dtype=np.int64)
    receivers = np.asarray(receivers, dtype=np.int64)
    frontier = nodes
    for _ in range(num_passes):
        incoming = np.isin(receivers, frontier, assume_unique=False)
        candidates = np.unique(senders[incoming])
        new = np.setdiff1d(candidates, nodes, assume_unique=True)
        if not len(new):
            break
        nodes = np.union1d(nodes, new)
        frontier = new
    return nodes


def _incident_ids(
    nodes: np.ndarray, offsets: np.ndarray, incident_edge_id: np.ndarray
) -> np.ndarray:
    chunks = [
        np.asarray(incident_edge_id[int(offsets[node]):int(offsets[node + 1])])
        for node in np.asarray(nodes, dtype=np.int64)
        if int(offsets[node + 1]) > int(offsets[node])
    ]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks).astype(np.int64, copy=False))


def reverse_k_hop_from_incident_csr(
    core_nodes: np.ndarray,
    offsets: np.ndarray,
    incident_edge_id: np.ndarray,
    union_pairs: np.ndarray,
    num_passes: int,
) -> np.ndarray:
    """Exact reverse traversal for the bidirectional P2b union graph."""
    if num_passes < 0:
        raise ValueError("num_passes must be non-negative")
    nodes = np.unique(np.asarray(core_nodes, dtype=np.int64))
    frontier = nodes
    for _ in range(num_passes):
        edge_ids = _incident_ids(frontier, offsets, incident_edge_id)
        if not len(edge_ids):
            break
        endpoints = np.unique(np.asarray(union_pairs[edge_ids]).reshape(-1))
        new = np.setdiff1d(endpoints, nodes, assume_unique=True)
        if not len(new):
            break
        nodes = np.union1d(nodes, new)
        frontier = new
    return nodes.astype(np.int64, copy=False)


def induced_union_edges(
    nodes: np.ndarray,
    offsets: np.ndarray,
    incident_edge_id: np.ndarray,
    union_pairs: np.ndarray,
) -> np.ndarray:
    """Return canonical edge IDs whose two endpoints are in ``nodes``."""
    nodes = np.asarray(nodes, dtype=np.int64)
    candidates = _incident_ids(nodes, offsets, incident_edge_id)
    if not len(candidates):
        return candidates
    pairs = np.asarray(union_pairs[candidates], dtype=np.int64)
    left = np.searchsorted(nodes, pairs[:, 0])
    right = np.searchsorted(nodes, pairs[:, 1])
    valid = (
        (left < len(nodes))
        & (right < len(nodes))
        & (nodes[np.minimum(left, len(nodes) - 1)] == pairs[:, 0])
        & (nodes[np.minimum(right, len(nodes) - 1)] == pairs[:, 1])
    )
    return candidates[valid]


def assemble_patch(
    *,
    core_id: int,
    fold: int,
    core_parent_ids: np.ndarray,
    loss_parent_ids: np.ndarray,
    num_passes: int,
    dependency_hops: int,
    node_features: np.ndarray,
    union_pairs: np.ndarray,
    union_edge_features: np.ndarray,
    offsets: np.ndarray,
    incident_edge_id: np.ndarray,
) -> GraphPatch:
    """Assemble an exact, untruncated view by immutable global node ID."""
    core_parent_ids = np.unique(np.asarray(core_parent_ids, dtype=np.int64))
    loss_parent_ids = np.unique(np.asarray(loss_parent_ids, dtype=np.int64))
    if not len(core_parent_ids):
        raise ValueError("cannot build a patch without authoritative core nodes")
    if np.setdiff1d(loss_parent_ids, core_parent_ids).size:
        raise ValueError("loss nodes must be a subset of authoritative core nodes")

    dependency_hops = int(dependency_hops)
    nodes = reverse_k_hop_from_incident_csr(
        core_parent_ids, offsets, incident_edge_id, union_pairs, dependency_hops
    )
    edge_ids = induced_union_edges(nodes, offsets, incident_edge_id, union_pairs)
    pairs = np.asarray(union_pairs[edge_ids], dtype=np.int64)
    local_u = np.searchsorted(nodes, pairs[:, 0]).astype(np.int32)
    local_v = np.searchsorted(nodes, pairs[:, 1]).astype(np.int32)
    local_pairs = np.column_stack([local_u, local_v])
    senders, receivers, attrs = make_bidirectional(
        local_pairs, np.asarray(union_edge_features[edge_ids], dtype=np.float32)
    )
    canonical_edge_ids = np.concatenate([edge_ids, edge_ids]).astype(np.int64)
    core_mask = np.isin(nodes, core_parent_ids, assume_unique=True)
    loss_mask = np.isin(nodes, loss_parent_ids, assume_unique=True)
    return GraphPatch(
        core_id=int(core_id),
        fold=int(fold),
        num_passes=int(num_passes),
        dependency_hops=dependency_hops,
        parent_node_id=nodes,
        node_features=np.asarray(node_features[nodes], dtype=np.float32),
        union_edge_id=canonical_edge_ids,
        edge_features=attrs,
        senders=senders,
        receivers=receivers,
        authoritative_core_mask=core_mask,
        loss_mask=loss_mask,
    )


def next_power_of_two(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum), 1)
    return 1 << (value - 1).bit_length()


def bucket_shape(patch: GraphPatch) -> tuple[int, int]:
    """Power-of-two XLA bucket with room for an isolated padding graph."""
    n_node = next_power_of_two(patch.n_node + 1)
    n_edge = next_power_of_two(patch.n_edge + 1)
    return n_node, n_edge


def pad_patch(
    patch: GraphPatch, bucket_nodes: int | None = None, bucket_edges: int | None = None
) -> dict[str, np.ndarray]:
    """Pad without truncation; padded edges are isolated from real nodes."""
    auto_nodes, auto_edges = bucket_shape(patch)
    bucket_nodes = auto_nodes if bucket_nodes is None else int(bucket_nodes)
    bucket_edges = auto_edges if bucket_edges is None else int(bucket_edges)
    if bucket_nodes <= patch.n_node or bucket_edges < patch.n_edge:
        raise ValueError("bucket cannot truncate nodes or edges and needs one dummy node")
    nodes = np.zeros((bucket_nodes, patch.node_features.shape[1]), dtype=np.float32)
    edges = np.zeros((bucket_edges, patch.edge_features.shape[1]), dtype=np.float32)
    senders = np.full(bucket_edges, patch.n_node, dtype=np.int32)
    receivers = np.full(bucket_edges, patch.n_node, dtype=np.int32)
    parent = np.full(bucket_nodes, -1, dtype=np.int64)
    union_edge = np.full(bucket_edges, -1, dtype=np.int64)
    node_mask = np.zeros(bucket_nodes, dtype=bool)
    edge_mask = np.zeros(bucket_edges, dtype=bool)
    core_mask = np.zeros(bucket_nodes, dtype=bool)
    loss_mask = np.zeros(bucket_nodes, dtype=bool)
    nodes[:patch.n_node] = patch.node_features
    edges[:patch.n_edge] = patch.edge_features
    senders[:patch.n_edge] = patch.senders
    receivers[:patch.n_edge] = patch.receivers
    parent[:patch.n_node] = patch.parent_node_id
    union_edge[:patch.n_edge] = patch.union_edge_id
    node_mask[:patch.n_node] = True
    edge_mask[:patch.n_edge] = True
    core_mask[:patch.n_node] = patch.authoritative_core_mask
    loss_mask[:patch.n_node] = patch.loss_mask
    return {
        "nodes": nodes, "edges": edges, "senders": senders,
        "receivers": receivers, "parent_node_id": parent,
        "union_edge_id": union_edge, "node_mask": node_mask,
        "edge_mask": edge_mask, "authoritative_core_mask": core_mask,
        "loss_mask": loss_mask,
        "n_node": np.asarray([patch.n_node, bucket_nodes - patch.n_node], dtype=np.int32),
        "n_edge": np.asarray([patch.n_edge, bucket_edges - patch.n_edge], dtype=np.int32),
    }


class CanonicalGraphPatchAdapter:
    """Lazy reader for the immutable P5 graph/CSR products."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.node_features = np.load(self.root / "node_features.npy", mmap_mode="r")
        self.union_pairs = np.load(self.root / "union_pairs.npy", mmap_mode="r")
        self.union_edge_features = np.load(
            self.root / "union_edge_features.npy", mmap_mode="r"
        )
        self.offsets = np.load(self.root / "incident_offsets.npy", mmap_mode="r")
        self.incident_edge_id = np.load(
            self.root / "incident_edge_id.npy", mmap_mode="r"
        )
        self.core_offsets = np.load(self.root / "core_active_offsets.npy", mmap_mode="r")
        self.core_parent = np.load(self.root / "core_active_parent.npy", mmap_mode="r")
        self.core_eligible = np.load(self.root / "core_active_eligible.npy", mmap_mode="r")
        self.core_safe2hop = np.load(self.root / "core_active_safe2hop.npy", mmap_mode="r")
        self.core_safe4hop = np.load(self.root / "core_active_safe4hop.npy", mmap_mode="r")
        self.core_fold = np.load(self.root / "core_fold.npy", mmap_mode="r")
        self.core_cap = np.load(self.root / "core_cap.npy", mmap_mode="r")

    def core_nodes(self, core_id: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = int(self.core_offsets[core_id]), int(self.core_offsets[core_id + 1])
        parent = np.asarray(self.core_parent[start:stop], dtype=np.int64)
        eligible = np.asarray(self.core_eligible[start:stop], dtype=bool)
        return parent[eligible], np.flatnonzero(eligible)

    def extract(
        self,
        core_id: int,
        num_passes: int,
        core_parent_ids: np.ndarray | None = None,
        dependency_hops_per_pass: int = 2,
    ) -> GraphPatch:
        start, stop = int(self.core_offsets[core_id]), int(self.core_offsets[core_id + 1])
        parent = np.asarray(self.core_parent[start:stop], dtype=np.int64)
        eligible = np.asarray(self.core_eligible[start:stop], dtype=bool)
        dependency_hops = int(num_passes) * int(dependency_hops_per_pass)
        if dependency_hops not in (2, 4):
            raise ValueError("strict P4 support is registered only for 2 or 4 graph hops")
        safe = np.asarray(
            self.core_safe2hop[start:stop] if dependency_hops == 2 else self.core_safe4hop[start:stop],
            dtype=bool,
        )
        all_core = parent[eligible]
        if core_parent_ids is not None:
            selected = np.unique(np.asarray(core_parent_ids, dtype=np.int64))
            if np.setdiff1d(selected, all_core).size:
                raise ValueError("subdivision contains non-authoritative core IDs")
            all_core = selected
        safe_lookup_ids = parent[eligible & safe]
        loss_ids = np.intersect1d(all_core, safe_lookup_ids, assume_unique=False)
        return assemble_patch(
            core_id=core_id, fold=int(self.core_fold[core_id]),
            core_parent_ids=all_core, loss_parent_ids=loss_ids,
            num_passes=num_passes, dependency_hops=dependency_hops, node_features=self.node_features,
            union_pairs=self.union_pairs, union_edge_features=self.union_edge_features,
            offsets=self.offsets, incident_edge_id=self.incident_edge_id,
        )

    def subdivide_exact(
        self,
        core_id: int,
        num_passes: int,
        max_nodes: int,
        max_edges: int,
        dependency_hops_per_pass: int = 2,
    ) -> list[GraphPatch]:
        """Recursively subdivide core loss ownership; never truncate context."""
        core, _ = self.core_nodes(core_id)
        queue = [np.sort(core)]
        output: list[GraphPatch] = []
        while queue:
            selected = queue.pop(0)
            patch = self.extract(
                core_id, num_passes, selected,
                dependency_hops_per_pass=dependency_hops_per_pass,
            )
            if (
                (patch.n_node <= max_nodes and patch.n_edge <= max_edges)
                or len(selected) == 1
            ):
                output.append(patch)
                continue
            split = len(selected) // 2
            queue.extend([selected[:split], selected[split:]])
        return output


def core_prediction_map(patches: Iterable[GraphPatch], values: Iterable[np.ndarray]) -> dict[int, np.ndarray]:
    """Collect subdivision outputs and reject duplicated core ownership."""
    result: dict[int, np.ndarray] = {}
    for patch, prediction in zip(patches, values):
        prediction = np.asarray(prediction)
        for parent, value in zip(
            patch.parent_node_id[patch.authoritative_core_mask],
            prediction[patch.authoritative_core_mask],
        ):
            key = int(parent)
            if key in result:
                raise ValueError(f"duplicate core prediction for parent node {key}")
            result[key] = np.asarray(value)
    return result
