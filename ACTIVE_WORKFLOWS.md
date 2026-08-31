# Active Workflow Index

This file is the quick reference for what to run in this repository now. For
Perlmutter commands and operational details, see `RUNBOOK.md`.

The **current Abacus VAC posterior** is P12-A (FMPE on OOF U-PATCH base
predictions plus deployable response). The older wedge-graph FlowJAX NPE path
remains in the inventory below but is not the production uncertainty model.

## Canonical Layout

- `workflows/abacus_tweb/` for Abacus slab T-Web generation, CutSky annotation,
  graph construction, graph features, SBI cache construction, and staged-mock
  helpers.
- `workflows/visualization/abacus_tweb/` for exploratory Abacus/T-Web notebooks
  and HTML visualizations.
- `workflows/jraph/` for JAX/Jraph regression, tuning, checkpoint evaluation,
  diagnostics, and ensembles.
- `workflows/sbi/` for P12-A FMPE (current Abacus VAC posterior) and FlowJAX
  SBI trainers. Wedge-subvolume FlowJAX NPE is the older Abacus graph path;
  partitioned FlowJAX is legacy/reference.
- `workflows/sbi/experimental/` for the optional two-stage SBI prototype.
- `workflows/gcn_paper/` for the paper-critical PyTorch GCN workflow.
- `shared/` for reusable model, transformation, path, resource, and cache-schema
  modules.
- `legacy/`, `archive/shims/`, and `to-delete/` for retired or reference-only
  migration artifacts.

## Active

- Generalisable GraphWeb P12-A posterior (current VAC uncertainty model):
  - `workflows/sbi/p12_prepare_crossfit_contracts.py`
  - `workflows/sbi/p12_export_unet_summaries.py`
  - `workflows/sbi/p12_prepare_base_response_dataset.py`
  - `workflows/sbi/p12_train_base_response_fmpe.py`
  - `workflows/sbi/p12_calibration_diagnostics.py`
  - `workflows/sbi/p12_affine_calibration_canary.py` (challenger only; rejected)
  - `workflows/sbi/p12_width_information_diagnostics.py`
  - `workflows/sbi/run_p12a_posterior_interactive.sh`
- P11 factorial observation views (JEPA substrate; not a posterior):
  - `configs/p11_factorial_views_v1.json`
  - `workflows/abacus_tweb/p11_prepare_factorial_view_sources.py`
  - `workflows/abacus_tweb/p11_build_factorial_view_counts.py`
  - `workflows/abacus_tweb/run_p11_factorial_view_counts_interactive.sh`
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
- Generalisable-GraphWeb canonical fields:
  - `workflows/abacus_tweb/p3a_audit_units.py`
  - `workflows/abacus_tweb/p3a_canary_parity.py`
  - `workflows/abacus_tweb/p3a_build_canonical_fields.py`
  - `workflows/abacus_tweb/p3a_postbuild_validate.py`
- Abacus SBI cache + wedge subvolumes:
  - `workflows/abacus_tweb/build_abacus_sbi_cache.py`
  - `workflows/abacus_tweb/subset_abacus_graph_wedge_for_sbi.py`
  - `workflows/abacus_tweb/subset_cugraph_metrics_for_wedge.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_truth_npz.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_variants.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_sbi_cache.py`
- SBI FlowJAX (TNG / older Abacus wedge-graph NPE; not P12-A):
  - `workflows/sbi/jraph_sbi_flowjax.py` for TNG/full-graph caches and Abacus
    wedge-subvolume caches.
  - `workflows/sbi/plot_flowjax_posteriors.py`
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
- `workflows/abacus_tweb/build_abacus_partition_batches.py`,
  `workflows/abacus_tweb/submit_build_partitions_adaptive.slurm`, and
  `workflows/abacus_tweb/PARTITION_ARTIFACT_SCHEMA.md` (partitioned Abacus
  cache artifacts; superseded by wedge subvolumes for graph-NPE work; the
  current VAC posterior is P12-A, not partitions)
- `workflows/sbi/jraph_sbi_flowjax_partitioned.py`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel.slurm`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel_multinode.slurm`, and
  `workflows/sbi/plot_flowjax_posteriors_partitioned.py` (legacy partitioned
  FlowJAX diagnostics)
- `to-delete/workflows/abacus_tweb/annotate_cutsky_with_tweb.py` (superseded by
  host-halo linked `annotate_cutsky_with_tweb_eigs.py`)

## Known Issues And Audit Notes

- P12-A is technically complete (`P12A_COMPLETE.json`) but **not** exactly
  calibrated in the sparsest redshift shell. The uncorrected posterior is the
  production model. Do not adopt the affine correction; do not write
  `P12A_CALIBRATION_PASS.json` by hand. Evidence:
  `docs/evidence/p12/` and `SCIENCE_LOG.md` (2026-08-30).
- `ph001` is sealed across P11 and P12. `ph000` is excluded from P11 factorial
  views only (catalogue nesting), not from P10/P12 training.
- Abacus label quality depends on host-halo linkage, not naive sky-coordinate
  inversion. Start label-alignment debugging with
  `workflows/abacus_tweb/ABACUS_TWEB_AUDIT_FINDINGS.md`.
- Partitioned SBI alignment checks and current learning diagnostics are tracked
  in `workflows/sbi/ABACUS_SBI_DEBUG_STRATEGY.md`.
- Root-level compatibility shims exist for some historical imports and scripts,
  but new runs and docs should use canonical `workflows/...` and `shared/...`
  paths.
