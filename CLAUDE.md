# CLAUDE.md

This file provides guidance to automation agents working in this repository.

## Project Overview

This repository implements machine learning pipelines for inferring cosmic web
structure from galaxy observables in IllustrisTNG and Abacus mock catalogs. The
main targets are local density-Hessian eigenvalues and derived T-Web classes
for voids, walls, filaments, and clusters.

Start with:

- `README.md` for repository orientation.
- `ACTIVE_WORKFLOWS.md` for the current canonical entrypoint list.
- `RUNBOOK.md` for Perlmutter commands, path overrides, and troubleshooting.

## Running Jobs On NERSC Perlmutter

Production workflows generally run through SLURM with `cosmic_env`. Some Abacus
graph-feature jobs use a RAPIDS/cuGraph environment; check the workflow SLURM
script before assuming a single Python environment.

### JAX/Jraph Regression Pipeline

```bash
sbatch workflows/jraph/submit_jraph.slurm
python workflows/jraph/jraph_pipeline.py --prediction_mode regression --use_shape_params --epochs 10000
```

### PyTorch GCN Paper Pipeline

```bash
sbatch workflows/gcn_paper/submit_gcn.slurm
python workflows/gcn_paper/gcn_pipeline.py --help
```

### TNG / Full-Graph SBI FlowJAX

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

### Abacus Partitioned SBI FlowJAX

```bash
sbatch workflows/abacus_tweb/submit_build_partitions_adaptive.slurm
sbatch workflows/sbi/submit_sbi_partitioned_data_parallel.slurm
```

### Key Environment Setup

```bash
conda activate cosmic_env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
```

## Architecture

### Main Pipeline Approaches

1. **Abacus T-Web and mock graph pipeline** (`workflows/abacus_tweb/`): builds
   slabwise T-Web outputs, annotates DESI/Abacus CutSky mocks via host-halo
   linkage, constructs alpha/Delaunay graph artifacts, computes graph features,
   and builds partitioned SBI caches.
2. **Regression** (`workflows/jraph/jraph_pipeline.py`): JAX/Jraph
   GraphNetwork predicting eigenvalues or transformed shape/derivative targets.
3. **SBI** (`workflows/sbi/`): GNN encoder plus FlowJAX normalizing flow for
   posterior estimation; `jraph_sbi_flowjax_partitioned.py` is the Abacus-scale
   path.
4. **Classification** (`workflows/gcn_paper/gcn_pipeline.py`): PyTorch/Torch
   Geometric GCN/GAT workflow for 4-class T-Web classification.

### Key Modules

| Module | Purpose |
| --- | --- |
| `shared/graph_net_models.py` | JAX GraphNetwork and encoder helpers. |
| `shared/eigenvalue_transformations.py` | Physics target transformations for eigenvalues, invariants, and derivative targets. |
| `shared/config_paths.py` | Environment-variable driven Perlmutter and scratch paths. |
| `shared/tng_pipeline_paths.py` | TNG/Jraph/SBI cache and output path resolution. |
| `shared/resource_requirements.py` | Runtime guards for CPU/GPU SLURM allocations. |
| `shared/sbi_cache_schema.py` | Cache-schema helpers used by tests and SBI paths. |
| `workflows/gcn_paper/gnn_models.py` | PyTorch GCN/GAT model definitions. |
| `workflows/gcn_paper/Utilities.py` | TNG data loading and graph construction for the paper workflow. |
| `workflows/gcn_paper/Network_stats.py` | Graph feature extraction and T-Web classification utilities. |

### Physics Targets

The regression/SBI stack supports multiple target representations:

- Raw ordered Hessian eigenvalues: `lambda1`, `lambda2`, `lambda3`.
- Shape/invariant representations such as trace, ellipticity, and prolateness.
- Abacus cache targets may include transformed eigenvalue increments and
  derivative columns when available.

Use `--use_shape_params` in the Jraph regression path when shape parameters are
desired. For Abacus SBI caches, inspect `build_abacus_sbi_cache.py --help` and
the generated cache metadata to confirm whether raw, transformed, or
three-target-only labels were written.

### Data Flow

1. Load IllustrisTNG subhalos or Abacus/DESI CutSky mock galaxies.
2. Assign or load T-Web Hessian eigenvalues.
3. Construct graph topology via Delaunay, MST, alpha-complex, or partitioned
   subgraphs depending on workflow.
4. Extract node and edge features.
5. Train regression, classification, or conditional density models.

### Caching

Canonical cache and output roots are resolved by `shared/config_paths.py`.
Override with `TNG_CANONICAL_CACHE_ROOT`, `TNG_CANONICAL_OUTPUT_ROOT`,
`TNG_JRAPH_CACHE_DIR`, `TNG_SBI_CACHE_DIR`, and related variables rather than
hard-coding new scratch paths.

## Local Subgraph Pipeline

The `local-subgraph-pipeline/` directory contains an independent inductive pilot
that trains on batched ego-graphs extracted from cached TNG graph data. It is
separate from the transductive full-graph Jraph/SBI paths.

## Testing

For lightweight local validation, run:

```bash
python -m unittest discover -s tests/phase4
```

These tests do not replace Perlmutter-scale scientific validation, but they are
useful for catching broken entrypoints, cache-schema drift, and eigenvalue
transformation regressions.
