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
- `workflows/sbi/` for posterior workflows. The current Abacus VAC posterior is
  P12-A FMPE on leakage-safe OOF U-PATCH predictions plus P3b-R response
  covariates. Graph-cache FlowJAX NPE (TNG/wedge) and partitioned FlowJAX remain
  as older graph-NPE stacks.
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
- Generalisable-GraphWeb canonical fields:
  - `workflows/abacus_tweb/p3a_audit_units.py`
  - `workflows/abacus_tweb/p3a_canary_parity.py`
  - `workflows/abacus_tweb/p3a_build_canonical_fields.py`
  - `workflows/abacus_tweb/p3a_postbuild_validate.py`
- P12-A VAC posterior (current Abacus production candidate):
  - Frozen candidate: `docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json`
  - Fit/export: `workflows/sbi/p12_prepare_crossfit_contracts.py`,
    `workflows/sbi/p12_export_unet_summaries.py`,
    `workflows/sbi/p12_prepare_base_response_dataset.py`,
    `workflows/sbi/p12_train_base_response_fmpe.py`,
    `workflows/sbi/run_p12a_posterior_interactive.sh`
  - Blind opening: `workflows/sbi/p12a_open_blind.py`,
    `workflows/sbi/p12a_authorized_truth.py`,
    `workflows/sbi/submit_p12a_ph001_truth_chain.sh`,
    `workflows/sbi/submit_p12a_ph001_postopen_chain.sh`
  - Precision recovery: `workflows/sbi/p12a_compact_precision_recovery.py`,
    `workflows/sbi/diagnose_p12a_compact_closure.py`,
    `workflows/sbi/submit_p12a_ph001_precision_recovery_chain.sh`
  - Evaluation: `workflows/sbi/p12a_evaluate_blind.py`,
    `workflows/sbi/p12a_blind_energy_score.py`,
    `workflows/sbi/p12a_plot_blind_evaluation.py`
- Abacus graph-NPE cache + wedge subvolumes (older graph stack; not the VAC
  posterior):
  - `workflows/abacus_tweb/build_abacus_sbi_cache.py`
  - `workflows/abacus_tweb/subset_abacus_graph_wedge_for_sbi.py`
  - `workflows/abacus_tweb/subset_cugraph_metrics_for_wedge.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_truth_npz.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_variants.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_sbi_cache.py`
- SBI FlowJAX (TNG/full-graph and older Abacus wedge caches):
  - `workflows/sbi/jraph_sbi_flowjax.py`
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

- P12-F3-D2 field diffusion (parallel, non-blocking; `ph001` stays sealed):
  - `configs/p12f3_d2_diffusion_v1.json`
  - `workflows/sbi/run_p12f3_d2_in_allocation.sh`
  - `workflows/sbi/p12f3_d2_train.py`, `p12f3_d2_select.py`, `p12f3_d2_confirm.py`
  - `workflows/sbi/dispatch_p12f3_d2_after_primary.py`
  - Selected arm is frozen `modern_base4`; attention remains unlicensed.
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
  cache artifacts; superseded first by wedge subvolumes, then by P12-A for
  the current Abacus VAC posterior)
- `workflows/sbi/jraph_sbi_flowjax_partitioned.py`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel.slurm`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel_multinode.slurm`, and
  `workflows/sbi/plot_flowjax_posteriors_partitioned.py` (legacy partitioned
  FlowJAX diagnostics)
- `to-delete/workflows/abacus_tweb/annotate_cutsky_with_tweb.py` (superseded by
  host-halo linked `annotate_cutsky_with_tweb_eigs.py`)

## Known Issues And Audit Notes

- P12-A is the production posterior candidate; `P12A_CALIBRATION_PASS.json` is
  still absent. Do not revive the rejected affine map or treat the trainer's
  in-script `calibration_pass` field as the scientific verdict. Blind
  acceptance is pending the immutable ph001 report; P13/Loa is not authorized.
- Compact-truth validation must compare stored eigenvalues against native
  CACTUS `CWEB` labels with a counted float32-threshold boundary exception,
  not a reconstructed class from `float32` eigenvalues alone. See
  `workflows/abacus_tweb/p10_target_contract.py` and
  `docs/evidence/p12/p12a_blind_opening_20260905/`.
- Abacus label quality depends on host-halo linkage, not naive sky-coordinate
  inversion. Start label-alignment debugging with
  `workflows/abacus_tweb/ABACUS_TWEB_AUDIT_FINDINGS.md`.
- Partitioned SBI alignment checks and older graph-NPE diagnostics are tracked
  in `workflows/sbi/ABACUS_SBI_DEBUG_STRATEGY.md`.
- Root-level compatibility shims exist for some historical imports and scripts,
  but new runs and docs should use canonical `workflows/...` and `shared/...`
  paths.
