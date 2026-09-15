# Active Workflow Index

This file is the quick reference for what to run in this repository now. For
Perlmutter commands and operational details, see `RUNBOOK.md`.

## Posterior programme status — 2026-09-08

- P12-A's registered blind evaluation is green; its frozen artifacts remain the
  per-galaxy production anchor. Do not reopen ph001 or start P13 without authority.
- P12-B pilot and follow-up are complete research workflows, not rerun requests:
  `docs/plan_p12b_unet_representation_pilot.md` and
  `docs/plan_p12b_representation_followup_v1.md` link the evidence and stop gates.
- D2 seed42 passed; the licensed seed314159 chain is submitted. Its training
  wrapper is `workflows/sbi/submit_p12f3_d2_replication_fullwall.slurm`;
  `workflows/sbi/submit_p12f3_d2_native_support.slurm` is the amended evaluation
  entrypoint. The immutable 467f442 worktree owns original science code.
  Do not relaunch legacy unamended evaluators or alter frozen source hashes.
- E2E metadata/analytic entrypoints remain `e2e_field_prepare_data.py` and
  `e2e_field_calibration_fixture.py` under `workflows/sbi/`. Actual data build:
  `e2e_field_build_products.py` (screening), `e2e_field_native_reference.py`
  (independent physical audit), `e2e_field_dataset.py` (packaging/guarded reader),
  `e2e_field_package_native_truth.py` (native reference shards), and
  `e2e_field_validate_products.py` (full-size engineering checks).
  `e2e_field_archive_preparation.py` archives bounded metadata/receipts only.
  The completed September-8 build is engineering-ready, not science-released;
  `training_ready=false`. See `docs/e2e_field_data_build_20260908.md` for the
  immutable evidence, Scratch products, target-convention caveat and remaining
  gates. These are not automatic rerun or training instructions.
  `e2e_field_error_budget.py` implements the completed training-only physical
  attribution audit; `e2e_field_summarize_error_budget.py` produces bounded
  metadata summaries/archives. Both are under `workflows/sbi/`. Findings and
  numerical-versus-topology limitations are in
  `docs/e2e_field_error_budget_20260908.md`; neither entrypoint licenses fits.
  The September-10 wide384_f4 research pipeline is
  `python -m workflows.sbi.e2e_wide_pipeline` with `prepare`, `train`, `sample`
  and `diagnose` subcommands. Its train-only reader and matched CFM/DIFF models
  are `e2e_wide_data.py` / `e2e_wide_models.py`; config is
  `configs/e2e_wide_pipeline_v1.json`. All real-array work requires approved
  compute. Approved job 58196582 passed normalization/full-size GPU engineering
  checks and was released; no science fit or research canary was started. See
  `docs/e2e_wide_gpu_smoke_20260911.md` for the verified run and
  `docs/e2e_wide_pipeline_20260910.md` for usage, caps and scientific boundaries.
  Subsequent approved research job 58196924 completed all four 192-update fits.
  September-14 evaluation job 58305867 completed and was released: 864 fields
  and 4,608 fixed-noise probes. Entrypoints `e2e_wide_evaluate.py` and
  `e2e_wide_evaluation_report.py` retain paired checkpoint/sampler diagnostics.
  The canary is not converged; support/power discrepancies remain. Further
  training or holdout access is not authorized. See
  `docs/e2e_wide_evaluation_20260914.md`; no calibration or release claim.
  The subsequent approved `e2e_wide_continue.py` extension completed 192->384
  updates for all four fits, with exact real-checkpoint resume parity. Job
  58309454 also completed the same draw evaluation plus
  `e2e_wide_continuation_assess.py` diagnostic gates/conditioning controls and
  was released. All registered gates fail despite support/topology improvement;
  power shape and convergence remain unresolved. No automatic continuation,
  model revision or holdout access. Contract/results:
  `docs/e2e_wide_continuation_20260914.md`.
  The approved September 15 `e2e_wide_denoising_audit.py` diagnosis completed
  as 58352703 (released), with no training or held-out reads. Fine-stage
  near-clean high-k noise cancellation fails; true coarse improves only the
  largest-scale deficit. Report/phase summaries: `e2e_wide_denoising_report.py`
  and `docs/e2e_wide_denoising_audit_20260915.md`. Further training/validation
  remains separately authorized, not an automatic follow-on.
  The subsequent approved `e2e_fine_learning_test.py` experiment completed as
  58358688 (released): six 256-update diagnostic fine fits, all near-clean
  capability checks fail. Partial noise-removal learning does not justify
  promotion or another full E2E extension. Source/config/report:
  `configs/e2e_fine_learning_20260915.json`, `e2e_fine_learning_report.py`,
  `docs/e2e_fine_learning_20260915.md`. These branches are not production models.
  The subsequent approved `e2e_edm_ablation.py` experiment completed as58361744
  (released): six512-update diffusion-only branches; all capability gates fail.
  Noise exposure improves denoising; log-noise FiLM partly helps. Optimizer reset,
  clip10 and the tested dilated bottleneck do not establish a remedy. Original
  E2E stays paused at384. Model/config/report: `e2e_edm_ablation_models.py`,
  `configs/e2e_edm_ablation_20260915.json`, `e2e_edm_ablation_report.py`,
  `docs/e2e_edm_ablation_20260915.md`. No arm promoted.
  The approved `e2e_fixed_noise_test.py` follow-on completed as58363476 (released):
  eight512-update fixed-ratio fits. The field-unit residual U-Net passes fitted
  capability at .05/.2 and transfer at .2; .05 transfer reconstruction error
  still fails despite noise cancellation. `e2e_fixed_noise_response.py` localizes
  a dominant noise-even distortion closely matching clean-input error; no
  sole architectural/conditioning cause established. Config/model/report:
  `configs/e2e_fixed_noise_20260915.json`, `e2e_fixed_noise_models.py`,
  `e2e_fixed_noise_report.py`, `docs/e2e_fixed_noise_20260915.md`.
  Original full E2E remains paused at384. Approved joint-noise comparison
  `e2e_multinoise_test.py` and separately approved skip-only contrast
  `e2e_skip_path_test.py` completed as58368502 (released; all steps0:0).
  Config/model/report/response: `configs/e2e_multinoise_20260915.json`,
  `e2e_multinoise_models.py`, `e2e_multinoise_report.py`,
  `e2e_multinoise_response.py`, `docs/e2e_multinoise_20260915.md`.
  `configs/e2e_skip_path_20260915.json`, `e2e_skip_path_report.py`.
  Nine3,072-update fits total. Near-identity+FiLM U-Net is best, but .05 transfer
  distortion still fails two phases; U-Net/wavelet pass .2, transformer does not
  help. No full-gate/convergence/calibration claim or production promotion.
  Diversity/global-normalization isolation is running as58373492 (twoGPUs):
  `e2e_diversity_norm.py`, `e2e_diversity_norm_report.py`,
  `configs/e2e_diversity_norm_20260915.json`,
  `docs/e2e_diversity_norm_20260915.md`. Sixteen matched3,072-update fits;
  3/6/9/15 non-overlapping training regions, two fixed global scalers, two seeds.
  Fixed12 transfer regions; exact0 identity and positive-nominal-noise clean probes.
  No held-out data, internal GroupNorm change, or E2E resumption.
- Root-level `*_supervisor_v2_20260904.sh` scripts are historical operational
  records, not current launch instructions. Frozen production uses Slurm.

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
- Root-level compatibility shims exist for some historical imports and scripts,
  but new runs and docs should use canonical `workflows/...` and `shared/...`
  paths.
