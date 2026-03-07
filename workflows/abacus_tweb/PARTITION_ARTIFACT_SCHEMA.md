# Abacus Partition Artifact Schema

This document defines the partition-batch format used to train SBI GNN models
without loading the full Abacus graph into each GPU.

## Inputs

- SBI-ready cache produced by `build_abacus_sbi_cache.py` with keys:
  - `graph` (`jraph.GraphsTuple`)
  - `regression_targets`
  - `masks` (`train_mask`, `val_mask`, `test_mask`)
  - `target_scaler`, `stats`, `eigenvalues_raw`

## Manifest file

JSON file: `partition_manifest.json`

Top-level fields:

- `schema_version`: integer
- `source_cache_path`: absolute path
- `num_passes`: integer hint used when building halo
- `halo_hops`: integer used by builder
- `core_partition_size`: integer
- `edge_selection_chunk_size`: integer
- `n_nodes_global`, `n_edges_global`: integers
- `target_dtype`, `feature_dtype`: strings
- `adaptive_core_size`: bool
- `min_core_nodes`, `max_core_nodes`: integers
- `target_total_nodes`, `target_edges`: integers (0 means disabled)
- `partitions`: array of partition entries

Partition entry fields:

- `partition_id`: string
- `split`: one of `train|val|test`
- `file`: relative filename for partition NPZ
- `n_core_nodes`, `n_halo_nodes`, `n_total_nodes`, `n_edges`: integers
- `oversized_budget`: bool (true if emitted despite unmet adaptive budget)

## Partition NPZ files

Each partition file stores:

- `global_node_ids`: int64, shape `[n_total_nodes]` (sorted ascending)
- `core_mask_local`: bool, shape `[n_total_nodes]`
- `x`: float32, shape `[n_total_nodes, node_feat_dim]`
- `targets`: float32/float64, shape `[n_total_nodes, 3]`
- `edge_index`: int32, shape `[2, n_edges]` local indexing
- `edge_attr`: float32, shape `[n_edges, edge_feat_dim]`
- `split_code`: int8 scalar (`0=train`, `1=val`, `2=test`)

Optional:

- `global_edge_ids`: int64, shape `[n_edges]` (for debug/provenance)

## Training semantics

- Message passing runs on all local nodes (core + halo).
- Loss/metrics are computed only on `core_mask_local`.
- No train/val/test mixing within a partition file.

## Notes

- `halo_hops=0` is valid and yields core-only induced subgraphs.
- For very large graphs, edge induction should be chunked to bound CPU memory.

