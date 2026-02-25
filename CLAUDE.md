# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements machine learning pipelines for inferring **cosmic web structure** from galaxy observables in the IllustrisTNG cosmological simulation. The cosmic web is characterized by eigenvalues of the local density Hessian matrix, which classify regions into voids, walls, filaments, and clusters.

## Running Jobs on NERSC Perlmutter

All pipelines run on NERSC's Perlmutter supercomputer using SLURM and the `cosmic_env` conda environment.

### JAX/Jraph Regression Pipeline
```bash
sbatch submit_jraph.slurm
# Or directly:
srun python jraph_pipeline.py --prediction_mode regression --use_shape_params --epochs 10000
```

### PyTorch Classification Pipeline
```bash
sbatch submit_gcn.slurm
# Uses mp.spawn() for 4-GPU distributed data parallel training
```

### SBI (Simulation-Based Inference) Pipeline
```bash
sbatch submit_sbi_flowjax.slurm
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

### Three Main Pipeline Approaches

1. **Classification** (`gcn_pipeline.py`) - PyTorch GCN/GAT with DDP for 4-class T-Web classification
2. **Regression** (`jraph_pipeline.py`) - JAX/Jraph GraphNetwork predicting eigenvalues or shape parameters
3. **SBI** (`jraph_sbi_flowjax.py`, `jraph_sbi_two_stage.py`) - GNN encoder + Flowjax normalizing flow for posterior estimation

### Key Modules

| Module | Purpose |
|--------|---------|
| `graph_net_models.py` | JAX GraphNetwork with multi-head attention, bounded activations |
| `gnn_models.py` | PyTorch GCN and GAT models |
| `eigenvalue_transformations.py` | Physics transformations: eigenvalues ↔ shape parameters (I₁, e, p) |
| `Utilities.py` | TNG data loading, Delaunay/MST/alpha-complex graph construction |
| `Network_stats.py` | Graph feature extraction, T-Web classification |
| `utils.py` | PyTorch training utilities, class weighting, UMAP plotting |

### Physics: Shape Parameters

The pipeline supports two target representations:
- **Raw eigenvalues** (λ₁, λ₂, λ₃): Ordered Hessian eigenvalues
- **Shape parameters** (I₁, e, p): Rotationally invariant representation
  - I₁ = trace (overall strength)
  - e = ellipticity (deviation from sphericity)
  - p = prolateness (prolate vs oblate)

Use `--use_shape_params` flag for shape parameter mode (recommended for regression).

### Data Flow

1. Load IllustrisTNG subhalos from HDF5 files
2. Construct graphs via Delaunay triangulation, MST, or alpha-complex
3. Extract node features (stellar mass, velocity, gas fraction, etc.) and edge features
4. Compute Hessian eigenvalues from smoothed density field
5. Train GNN to predict eigenvalues/shape parameters or posterior distributions

### Caching

Processed data is cached at `/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/` to avoid expensive recomputation:
- `processed_jraph_data_mc1e+09_v2_scaled_3_*.pkl` - Graph data with different representations

## Local Subgraph Pipeline

The `local-subgraph-pipeline/` directory contains an independent inductive pipeline that trains on batched ego-graphs:
- `train_flowjax_subgraphs.py` - End-to-end training on local subgraphs
- Enables generalization to unseen graphs (not transductive)

## Framework Usage

- **JAX ecosystem** (Jraph, Haiku, Optax, Flowjax): Production pipelines for regression and SBI
- **PyTorch** (Torch Geometric): Classification pipeline with multi-GPU DDP
- **Flowjax/Distrax**: Normalizing flows for conditional density estimation

## SLURM Configuration

Standard job setup: 1 node, 4 GPUs, 128 CPUs, `desi` account, `regular` QOS
- Logs: `/pscratch/sd/d/dkololgi/logs/` or `logs/`
