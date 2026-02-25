# GCN Paper Workflow

This folder contains the paper-critical legacy GCN workflow that was previously spread across repository root scripts.

## Canonical entrypoints

- `gcn_pipeline.py`
- `gcn_pipeline_postprocess.py`
- `postprocessing.py`
- `submit_gcn.slurm`

## Core modules

- `Model.py`
- `Model_classes.py`
- `Network_stats.py`
- `Utilities.py`
- `gnn_models.py`
- `MLP_optuna_optimisation.py`

## Compatibility

Root-level script names are currently compatibility shims and forward to these canonical paths.
