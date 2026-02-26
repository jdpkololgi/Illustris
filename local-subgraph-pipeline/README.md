# Local Subgraph Pipeline (Phase A: TNG300 Pilot)

This directory contains an **independent** pipeline that trains an **integrated** GNN+Flowjax conditional normalizing flow on **batched ego-graphs** (local subgraphs) extracted from the cached TNG300 Delaunay graph.

Design constraints:
- **Does not modify** existing transductive pipelines (e.g. `jraph_sbi_flowjax.py`).
- Reuses shared modules (e.g. `graph_net_models.py`, `eigenvalue_transformations.py`) via imports.

Entry points:
- `train_flowjax_subgraphs.py`: end-to-end training on local subgraphs



