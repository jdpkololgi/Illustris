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
- `workflows/sbi/` for FlowJAX SBI trainers. Current Abacus-scale SBI uses
  wedge-subvolume caches with the full-graph NPE trainer; partitioned FlowJAX is
  retained as legacy/reference.
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
- Generalisable-GraphWeb P8 spatial-transfer screen and recovery:
  - Short screens (immutable): `p8_prepare_deterministic.py`,
    `p8_prepare_graph_features.py`, `p8_classical_fullcap.py`,
    `p8_train_graph_patch.py`, `p8_train_unet_patch.py`,
    `p8_audit_training_adequacy.py`, `p8_summarize_screens.py`
  - Exposure-aware recovery / extension: `p8_epoch_training.py`,
    `p8_train_patch_recovery.py`, `p8_audit_recovery_run.py`
  - Evidence plots: `plot_p8_recovery_curves.py`,
    `plot_p8_recovery_parity.py`, `plot_p8_recovery_visuals.py`,
    `plot_p8_rotation2_eval.py`, `plot_all_models_parity.py`
  - P9 residual complementarity diagnostic:
    `p9_residual_complementarity_audit.py`
  - Current status: two-rotation `recovery_v1` complete (U-PATCH mean
    0.5035 clears both registered bars; G-PATCH fails the classical
    supported-shell bar). Primary artifacts are immutable.
    `convergence_extension_v1` on rotations 0 and 2 finished with
    `NOT_CONVERGED_EXTENSION_CAP` (no material late gain). Same-phase
    evidence only — P10 remains the production-transfer gate. See
    `RUNBOOK.md` and `docs/plan_generalisable_graphweb_vac.md` §P8/P9.
- Generalisable-GraphWeb P8.8 multitracer (BGS_FAINT context):
  - Catalogues / fields / selection:
    `p8_build_multitracer_catalogues.py`, `p8_build_multitracer_fields.py`,
    `p8_refit_multitracer_selection.py`, `p8_build_multitracer_control_fields.py`
  - Graph handoff: `p8_build_multitracer_graph_adapter.py`,
    `p8_prepare_multitracer_graph_features.py`,
    `p8_validate_multitracer_global_graph.py`,
    `p8_validate_multitracer_graph_patch.py`
  - Training / closeout: `p8_train_multitracer_unet_patch.py`,
    `p8_train_multitracer_graph_patch.py`, `p8_smoke_multitracer_null.py`,
    `p8_evaluate_multitracer_controls.py`, `p8_close_multitracer_mt4.py`,
    `run_p8_multitracer_recovery_supervisor.sh`
  - Current status: MT4 closed — same-phase two-rotation information
    `PASS_SAME_PHASE_TWO_ROTATIONS`, but current multitracer U-PATCH
    adoption is `NO_GO_FOLD_INSTABILITY_AND_ROTATION_2_SAFEGUARDS`.
    Decision: `docs/evidence/p8/multitracer_mt4_decision.json`. Next
    registered model is P8.9 `U-DENSITY-PHYS-v1`, not MT5.
- Generalisable-GraphWeb P8.9 density-first target pipeline:
  - Coordinate preflight: `p8_density_target_alignment.py`,
    `run_p8_density_target_alignment.sh`
  - Cap-aligned `delta_R7` targets: `p8_build_density_targets.py`,
    `run_p8_build_density_targets.sh`
  - Exact-owner output tiling: `p8_build_field_output_tiling.py`,
    `run_p8_build_field_output_tiling.sh`
  - Current status: coordinate gate **PASS** (`Z_COSMO` + origin
    `(-1000,-1000,-1000) Mpc/h`). First target/tiling runtime products
    are **superseded** (observer-Mpc vs periodic-Mpc/h unit bug; shell
    radii mixed Mpc with Mpc/h). Corrected rebuild + trace/tensor
    closure required before any `U-DENSITY-PHYS-v1` training. No density
    trainer entrypoint exists yet.
- Second-gen staged mocks (ph000 helpers):
  - `workflows/abacus_tweb/secondgen_mocks/ph000/README.md`
  - Preservation evidence: `docs/evidence/p0s/`
- Abacus SBI cache + wedge subvolumes:
  - `workflows/abacus_tweb/build_abacus_sbi_cache.py`
  - `workflows/abacus_tweb/subset_abacus_graph_wedge_for_sbi.py`
  - `workflows/abacus_tweb/subset_cugraph_metrics_for_wedge.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_truth_npz.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_variants.py`
  - `workflows/abacus_tweb/build_staged_mock_wedge_sbi_cache.py`
- SBI FlowJAX:
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
  cache artifacts; superseded by wedge subvolumes for current SBI work)
- `workflows/sbi/jraph_sbi_flowjax_partitioned.py`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel.slurm`,
  `workflows/sbi/submit_sbi_partitioned_data_parallel_multinode.slurm`, and
  `workflows/sbi/plot_flowjax_posteriors_partitioned.py` (legacy partitioned
  FlowJAX diagnostics)
- `to-delete/workflows/abacus_tweb/annotate_cutsky_with_tweb.py` (superseded by
  host-halo linked `annotate_cutsky_with_tweb_eigs.py`)

## Known Issues And Audit Notes

- Abacus label quality depends on host-halo linkage, not naive sky-coordinate
  inversion. Start label-alignment debugging with
  `workflows/abacus_tweb/ABACUS_TWEB_AUDIT_FINDINGS.md`.
- Partitioned SBI alignment checks and current learning diagnostics are tracked
  in `workflows/sbi/ABACUS_SBI_DEBUG_STRATEGY.md`.
- P8 recovery `--resume` freezes the checkpoint Git revision and CLI contract:
  resume from a detached worktree at that SHA with identical arguments rather
  than weakening the guard. Clear inherited DESI `PYTHONPATH`/`PYTHONHOME`
  before launching interactive recovery jobs.
- P8.9 target construction maps observer-frame P3 centres (comoving Mpc) into
  the periodic T-web cube with
  `(observer_xyz_Mpc * h + origin_Mpc/h) modulo 2000 Mpc/h`. Do not add
  observer Mpc to an Mpc/h origin. Pre-correction target HDF5 files and the
  first tiling manifest must not be used for training or science.
- Root-level compatibility shims exist for some historical imports and scripts,
  but new runs and docs should use canonical `workflows/...` and `shared/...`
  paths.
