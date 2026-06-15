# TNG/Illustris Cosmic Web Workflows

This repository contains research pipelines for predicting and inferring cosmic
web structure from galaxy observables in IllustrisTNG and Abacus mock catalogs.
The core labels are T-Web Hessian eigenvalues and derived environment classes
for voids, walls, filaments, and clusters.

## Where To Start

- `ACTIVE_WORKFLOWS.md` is the current index of supported entrypoints.
- `RUNBOOK.md` has NERSC Perlmutter launch commands, environment notes, and
  common operational pitfalls.
- `workflows/abacus_tweb/README.md` explains the Abacus CutSky to T-Web to graph
  cache chain.
- `workflows/sbi/README.md` explains the FlowJAX SBI trainers and launchers.
- `workflows/gcn_paper/README.md` covers the paper-critical PyTorch GCN path.
- `local-subgraph-pipeline/README.md` covers the independent ego-graph pilot.

## Repository Layout

- `workflows/abacus_tweb/`: Abacus T-Web generation, CutSky annotation, graph
  construction, graph features, SBI cache builders, and staged-mock helpers.
- `workflows/jraph/`: JAX/Jraph regression, tuning, checkpoint evaluation, and
  ensemble utilities for TNG-style graph training.
- `workflows/sbi/`: FlowJAX conditional density estimation for TNG/full-graph
  caches and the current Abacus wedge-subvolume NPE path. The older
  partitioned Abacus trainer remains for reference/debugging only.
- `workflows/gcn_paper/`: PyTorch/Torch Geometric classification pipeline used
  for paper reproduction.
- `shared/`: reusable model, path, resource, cache-schema, graph-construction,
  and eigenvalue-transformation helpers.
- `legacy/`, `archive/shims/`, `to-delete/`: retained references from workflow
  migration. Prefer canonical paths under `workflows/` for new runs.
- `tests/phase4/`: lightweight smoke tests for entrypoints, cache schema, and
  eigenvalue compatibility.

## Operational Context

Most production workflows are written for NERSC Perlmutter and expect
`cosmic_env` unless a SLURM script documents a more specific environment, such
as the RAPIDS/cuGraph graph-feature job. Large graph construction, partition
building, and training jobs should run through SLURM rather than login nodes.

For path overrides and scratch layout, see `RUNBOOK.md` and the constants in
`shared/config_paths.py`.
