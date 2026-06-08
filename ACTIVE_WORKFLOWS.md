# Active Workflow Index

This file is the quick reference for what to run in this repository now. For
Perlmutter commands and operational details, see `RUNBOOK.md`.

## Canonical Layout

- `workflows/abacus_tweb/` for Abacus slab T-Web generation, CutSky annotation,
  graph construction, graph features, SBI cache construction, and staged-mock
  helpers.
- `workflows/visualization/abacus_tweb/` for exploratory Abacus/T-Web notebooks
  and HTML visualizations.
- `workflows/jraph/` for JAX/Jraph regression, tuning, checkpoint evaluation,
  diagnostics, and ensembles.
- `workflows/sbi/` for FlowJAX SBI trainers, including the partition-aware
  Abacus-scale path.
- `workflows/sbi/experimental/` for the optional two-stage SBI prototype.
- `workflows/gcn_paper/` for the paper-critical PyTorch GCN workflow.
- `shared/` for reusable model, transformation, path, resource, and cache-schema
  modules.
- `legacy/`, `archive/shims/`, and `to-delete/` for retired or reference-only
  migration artifacts.

## Active

- Abacus slab + MPI T-Web:
  - `workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm`
  - `workflows/abacus_tweb/abacus_cactus_tweb.py`
  - `workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py`
- Abacus mock graph + features:
  - `workflows/abacus_tweb/submit_abacus_graph_cpu.slurm`
  - `workflows/abacus_tweb/build_abacus_graph.py`
  - `workflows/abacus_tweb/submit_abacus_graph_features_cpu.slurm`
  - `workflows/abacus_tweb/submit_abacus_graph_features_cugraph.slurm`
  - `workflows/abacus_tweb/abacus_graph_features.py`
  - `workflows/abacus_tweb/abacus_graph_features_cugraph.py`
- Abacus SBI cache + partitions:
  - `workflows/abacus_tweb/build_abacus_sbi_cache.py`
  - `workflows/abacus_tweb/submit_build_partitions_adaptive.slurm`
  - `workflows/abacus_tweb/build_abacus_partition_batches.py`
  - `workflows/abacus_tweb/PARTITION_ARTIFACT_SCHEMA.md`
- SBI FlowJAX:
  - `workflows/sbi/jraph_sbi_flowjax.py` for TNG/full-graph scale.
  - `workflows/sbi/jraph_sbi_flowjax_partitioned.py` for Abacus partition
    artifacts.
  - `workflows/sbi/submit_sbi_partitioned_data_parallel.slurm`
  - `workflows/sbi/submit_sbi_partitioned_data_parallel_multinode.slurm`
  - `workflows/sbi/plot_flowjax_posteriors_partitioned.py`
- Jraph training baseline:
  - `workflows/jraph/jraph_pipeline.py`
  - `workflows/jraph/submit_jraph.slurm`
  - `workflows/jraph/hyperparameter_tuning.py`
  - `workflows/jraph/train_ensemble.slurm`
- GCN paper workflow:
  - `workflows/gcn_paper/gcn_pipeline.py`
  - `workflows/gcn_paper/gcn_pipeline_postprocess.py`
  - `workflows/gcn_paper/postprocessing.py`
  - `workflows/gcn_paper/submit_gcn.slurm`

## Experimental And Diagnostic

- `workflows/sbi/experimental/jraph_sbi_two_stage.py`
- `workflows/sbi/submit_sbi_overfit_tiny.slurm`
- `workflows/sbi/benchmark_partition_data_parallel.py`
- `workflows/jraph/experimental/reproduce_error.py`
- `workflows/jraph/debug_eig_order.py`
- `workflows/gcn_paper/experimental/Illustris_cactus.py`
- `local-subgraph-pipeline/*`

## Legacy

- `legacy/abacus_process_particles.py` (use
  `workflows/abacus_tweb/abacus_process_particles2.py`)
- `legacy/sbi/jraph_sbi_pipeline.py` (legacy SBI path)
- `legacy/sbi/jraph_sbi_flowjax_two_stage.py` (retired overlap; optional
  two-stage path is `workflows/sbi/experimental/jraph_sbi_two_stage.py`)
- `legacy/gcn_paper/getting_started.py` (legacy onboarding/demo script)
- `to-delete/workflows/abacus_tweb/annotate_cutsky_with_tweb.py` (superseded by
  host-halo linked `annotate_cutsky_with_tweb_eigs.py`)

## Known Issues And Audit Notes

- Abacus label quality depends on host-halo linkage, not naive sky-coordinate
  inversion. Start label-alignment debugging with
  `workflows/abacus_tweb/ABACUS_TWEB_AUDIT_FINDINGS.md`.
- Partitioned SBI alignment checks and current learning diagnostics are tracked
  in `workflows/sbi/ABACUS_SBI_DEBUG_STRATEGY.md`.
- Root-level compatibility shims exist for some historical imports and scripts,
  but new runs and docs should use canonical `workflows/...` and `shared/...`
  paths.
