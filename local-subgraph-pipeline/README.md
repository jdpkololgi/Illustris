# Local Subgraph Pipeline

This directory contains an independent Phase A pilot for training an integrated
GNN + FlowJAX conditional normalizing flow on batched ego-graphs extracted from
a cached global TNG300 graph.

## Intent

The main Jraph/SBI workflows are transductive: they train on one full cached
graph. This pilot instead samples many small k-hop neighborhoods around center
nodes, making it a candidate path for inductive experiments and for comparing
local graph context against full-graph message passing.

## Entry Points

- `train_flowjax_subgraphs.py`: trains the local-subgraph GNN + FlowJAX model
  from a global cache containing `graph`, `regression_targets`, `masks`,
  `target_scaler`, and `eigenvalues_raw`.
- `eval_local_subgraph.py`: evaluates a trained local-subgraph checkpoint on a
  subset of held-out test center nodes.
- `subgraph_dataset.py`: builds k-hop ego-graphs and padded batched
  `jraph.GraphsTuple` inputs.
- `tng_positions.py`: optional TNG position loading and edge-direction
  alignment checks.

## Usage Sketch

```bash
python local-subgraph-pipeline/train_flowjax_subgraphs.py \
  --cache_path "/path/to/processed_jraph_data.pkl" \
  --output_dir "/path/to/local_subgraph_outputs" \
  --k_hops 2 \
  --max_nodes 256 \
  --max_edges 2048
```

```bash
python local-subgraph-pipeline/eval_local_subgraph.py \
  --model_pkl "/path/to/checkpoint.pkl" \
  --cache_path "/path/to/processed_jraph_data.pkl" \
  --num_test 1024
```

## Constraints And Current Status

- Locality is graph-hop based, not metric-radius based, because the cached
  graph does not currently carry xyz coordinates.
- The center node is kept at a stable local index so the model can extract the
  center embedding after batching.
- This pipeline is experimental and intentionally separate from
  `workflows/sbi/jraph_sbi_flowjax.py`.
- The scripts currently use legacy top-level import names for shared model and
  eigenvalue helpers. If running from a checkout that only exposes modules under
  `shared/`, update the import path or provide compatibility shims before using
  this pilot.
