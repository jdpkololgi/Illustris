# SBI Workflows

This directory contains posterior-estimation workflows for cosmic-web
eigenvalue targets.

**Current Abacus VAC posterior:** P12-A FMPE on leakage-safe OOF U-PATCH
predictions plus P3b-R response covariates. Graph-cache FlowJAX NPE remains
for TNG and older Abacus wedge diagnostics. Partitioned FlowJAX is legacy.

## Which Path To Use

| Use case | Entrypoint | Notes |
| --- | --- | --- |
| Abacus VAC posterior (current) | `p12_train_base_response_fmpe.py`, `run_p12a_posterior_interactive.sh` | Frozen as `docs/evidence/p12/P12A_PRODUCTION_CANDIDATE_FROZEN.json`. |
| P12-A ph001 blind opening | `p12a_open_blind.py`, `p12a_authorized_truth.py` | One shared opening; `open_count=1` is consumed before truth is read. |
| P12-A compact-truth recovery | `p12a_compact_precision_recovery.py` | Authorized one-row float32 threshold exception only. |
| P12-A blind evaluation | `p12a_evaluate_blind.py`, `p12a_plot_blind_evaluation.py` | Cannot fit, open truth, or change gates. |
| P12-F3-D2 field diffusion | `run_p12f3_d2_in_allocation.sh` | Parallel ph006 experiment; `ph001` stays sealed. |
| TNG/full-graph cache | `jraph_sbi_flowjax.py` | Older graph-NPE stack. |
| Abacus wedge-subvolume cache | `jraph_sbi_flowjax.py` | Older graph-NPE stack, not the VAC posterior. |
| Posterior plots, full graph/wedge | `plot_flowjax_posteriors.py` | Uses saved model outputs from the full-graph trainer. |
| Abacus partition artifacts | `jraph_sbi_flowjax_partitioned.py` | Legacy partitioned experiment. |
| Two-stage prototype | `experimental/jraph_sbi_two_stage.py` | Optional experimental path. |

## P12-A FMPE Posterior

Estimand:

```text
q(lambda_ordered | U_PATCH_R0_epoch20_prediction, P3b-R response, H_fid)
```

The encoder is the five-phase U-PATCH R0 checkpoint (epoch 20). The posterior
is explicitly response-conditioned; the encoder is not. Target coordinates are
ordered softplus increments from `p12_prepare_base_response_dataset.py`
(`softplus_coordinates`). Conversion to physical `(lambda1, lambda2, lambda3)`
is evaluation-only.

OOF summaries (`p12_export_unet_summaries.py`) refuse `ph001` and refuse any
phase listed in the checkpoint's training phases. Dataset builder
`p12_prepare_base_response_dataset.py` likewise refuses `ph001`. Feature names:

```text
base_lambda1, base_lambda2, base_lambda3, redshift, log_ntilde_mpc3,
cap_ngc, log1p_random_support_boundary_distance_mpc
```

Scratch roots default to `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/`.
Completion marker: `p12a_base_response_v1/fmpe_seed42/P12A_COMPLETE.json`.

The supervisor `run_p12a_posterior_interactive.sh` stops at that marker. It
does not run `p12_calibration_diagnostics.py`, the affine canary, or the
width diagnostic. GPU is required; `--dataloader-workers` must remain `0`.

`P12A_CALIBRATION_PASS.json` is still absent. The later 50k-row physical
TARP audit passes the registered 0.05 gate, but the sparsest shell retains a
lambda2/lambda3 residual. The affine challenger in
`p12_affine_calibration_canary.py` was rejected; promotion requires every
`selection_gates()` check, including spatial-block proper log-score
improvement. Do not revive that map.

## P12-A Blind Opening And Evaluation

```text
freeze truth-free ph001 predictions
  -> authorize (P12_BLIND_OPEN_AUTHORIZED.json, open_count=1)
  -> isolated truth chain (particle_b -> density -> tweb -> annotation -> compact)
  -> finalize (P12_BLIND_OPENED.json)
  -> energy score -> evaluate -> plot
  -> STOP (P13/Loa needs separate authorization)
```

Contract: `docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json`. Isolated
truth root: `/pscratch/sd/d/dkololgi/abacus/p12_blind_truth/ph001/p12a_v1`.

```bash
python -m workflows.sbi.p12a_open_blind authorize --help
bash workflows/sbi/submit_p12a_ph001_truth_chain.sh
bash workflows/sbi/submit_p12a_ph001_postopen_chain.sh
```

Four-GPU truth-free sampling: `submit_p12a_blind_export.slurm` with
`sbi==0.26.1`. Markers are `O_EXCL`; exclusive chain claims refuse duplicate
submission.

Compact-join validation must use `stored_class_consistency()` so a single
`float32(0.2)` boundary row is a counted exception, not a reconstructed-class
failure. Authorized recovery:

```bash
python -m workflows.sbi.p12a_compact_precision_recovery validate
bash workflows/sbi/submit_p12a_ph001_precision_recovery_chain.sh
```

The recovery path does not edit the frozen original builder, does not score
posteriors, and does not increment `open_count`.

## P12-F3-D2 (parallel)

P12-F v1/v2 closed with no field finalist. D2 is a hard-budget ph006
diffusion funnel on pinned commit `467f442`. Selected arm: `modern_base4`.
Confirmation passed internally; attention is unlicensed. This is not a
P12-A blocker and must not open `ph001`.

```bash
bash workflows/sbi/run_p12f3_d2_in_allocation.sh test
bash workflows/sbi/run_p12f3_d2_in_allocation.sh science modern_base4
```

The four-hour science wrapper
`submit_p12f3_d2_training_fullwall.slurm` is hash-pinned by
`dispatch_p12f3_d2_after_primary.py`. Confirmation allocations need >1 h.

## Older Abacus Wedge Inputs

Build wedge caches in `workflows/abacus_tweb/`:

```text
subset_abacus_graph_wedge_for_sbi.py
  -> subset_cugraph_metrics_for_wedge.py
  -> build_abacus_sbi_cache.py
  -> jraph_sbi_flowjax.py
```

The full-graph trainer currently resolves its input through
`shared/tng_pipeline_paths.py`. For a wedge run, place or symlink the desired
cache under `TNG_SBI_CACHE_DIR` using the expected transformed-cache name:

```bash
export TNG_SBI_CACHE_DIR="/path/to/wedge_sbi_cache_dir"
python workflows/sbi/jraph_sbi_flowjax.py --epochs 1000 --output_dir "/path/to/out"
```

By default the expected cache file is:

```text
$TNG_SBI_CACHE_DIR/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl
```

Use `--no_transformed_eig` only for explicit raw-eigenvalue ablations; that mode
expects the `_raw_eig.pkl` cache name instead.

## Target Convention

The current SBI target is the ordered softplus-increment representation from
`shared/eigenvalue_transformations.py`: `lambda1` is the anchor and the
`lambda2 - lambda1` and `lambda3 - lambda2` gaps are encoded as non-negative
increments. The model trains and samples in increment space; conversion back to
physical `(lambda1, lambda2, lambda3)` is for evaluation and plotting. Do not
replace this with an unconstrained direct three-eigenvalue head unless the run is
an intentional ablation.

## Launchers

P12-A and D2 launchers are listed above. There is no tracked production
`submit_sbi_flowjax.slurm` for the older wedge NPE path; run
`jraph_sbi_flowjax.py` inside a GPU allocation.

Tracked SLURM scripts for graph-NPE / partition diagnostics:

| SLURM script | Purpose |
| --- | --- |
| `submit_p12a_blind_export.slurm` | Four-GPU truth-free P12-A posterior export. |
| `submit_p12a_ph001_truth_chain.sh` | Isolated ph001 particle/density/T-web/annotation/compact chain. |
| `submit_p12a_ph001_postopen_chain.sh` | Finalize, energy score, evaluate, plot. |
| `submit_p12a_ph001_precision_recovery_chain.sh` | Authorized compact-truth recovery plus post-open dispatch. |
| `submit_p12f3_d2_training_fullwall.slurm` | Hash-pinned D2 science wrapper (pinned `467f442` worktree). |
| `submit_sbi_partitioned_data_parallel.slurm` | Legacy single-node, four-GPU partitioned SBI training. |
| `submit_sbi_partitioned_data_parallel_multinode.slurm` | Legacy multi-node partitioned SBI training. |
| `submit_sbi_overfit_tiny.slurm` | Tiny overfit diagnostic for partitioned SBI. |

## Legacy Partition Notes

The partitioned trainer requires:

- `--partition-manifest`: path to `partition_manifest.json` generated by
  `workflows/abacus_tweb/build_abacus_partition_batches.py`.
- `--sbi-cache-path`: original SBI cache pickle generated by
  `workflows/abacus_tweb/build_abacus_sbi_cache.py`; this carries scaler and
  raw-eigenvalue metadata.
- `--output-dir`: model checkpoints, logs, and metrics output directory.

Partition semantics are documented in
`workflows/abacus_tweb/PARTITION_ARTIFACT_SCHEMA.md`. `ABACUS_SBI_DEBUG_STRATEGY.md`
records the row-order checks and overfit diagnostics that motivated moving away
from this path for later Abacus graph-NPE work. The current VAC posterior is
P12-A, documented above.
