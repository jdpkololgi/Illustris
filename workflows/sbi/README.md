# FlowJAX SBI Workflows

This directory contains conditional density estimation workflows for cosmic web
targets. There are two separate stacks:

1. **P12-A FMPE** — current Abacus VAC posterior: per-galaxy ordered-eigenvalue
   uncertainty on OOF U-PATCH base predictions plus deployable response.
2. **FlowJAX NPE** — GNN encoder plus FlowJAX flow on TNG/full-graph caches and
   older Abacus wedge-subvolume graphs.

Do not mix their caches, checkpoints, or calibration claims.

## P12-A VAC Posterior (current Abacus uncertainty model)

P12-A estimates

```text
q(λ | λ̂_U-PATCH^OOF, z, ñ(z), cap, d_random-support, H_fid)
```

It is an amortized per-galaxy posterior under the fiducial cosmology, not a
jointly coherent tidal field. Targets are ordered softplus increments from the
same convention as `shared/eigenvalue_transformations.py`. The 32-d U-PATCH
fold latents are **excluded**: their coordinate systems are not aligned across
leave-one-phase-out checkpoints.

### Entrypoints

| Stage | Script | Notes |
| --- | --- | --- |
| OOF contracts | `p12_prepare_crossfit_contracts.py` | Leave-one-phase-out U-PATCH contracts. |
| OOF summaries | `p12_export_unet_summaries.py` | GPU; refuses in-sample and `ph001`. |
| Dataset | `p12_prepare_base_response_dataset.py` | Default 2e6 train / 6e5 ph006 rows; sqrt-count shell sampling with natural-volume weights. |
| FMPE fit | `p12_train_base_response_fmpe.py` | GPU; `--dataloader-workers` must be 0. |
| Calibration audit | `p12_calibration_diagnostics.py` | Physical ranks, spatial-block bootstrap, TARP. Folds 0–1 reserved. |
| Affine challenger | `p12_affine_calibration_canary.py` | Per-shell offset+scale in scaled softplus space. **Rejected 2026-08-30.** |
| Width vs information | `p12_width_information_diagnostics.py` | Frozen audit draws; BRIGHT neighbour counts are external. |
| Supervisor | `run_p12a_posterior_interactive.sh` | Login watcher; GPU `salloc` through `P12A_COMPLETE.json`; flock against duplicates. Does not run audit/canary/width. |

Default roots:

```text
/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1
```

Feature vector, in order: `base_lambda1`, `base_lambda2`, `base_lambda3`,
`redshift`, `log_ntilde_mpc3`, `cap_ngc`,
`log1p_random_support_boundary_distance_mpc`.

Training phases: `ph000`, `ph002`–`ph005`. Selection: `ph006` (folds 0–1
calibration, 2–4 selection). Sealed: `ph001`.

### Status And Markers

`P12A_COMPLETE.json` means the fit ran. It is **not** a calibration pass.
`P12A_CALIBRATION_PASS.json` is correctly absent. Keep the uncorrected
posterior. The affine canary writes `P12A_AFFINE_CORRECTION_REJECTED.json`
when any gate fails (crossfit stability, out-of-fit proper log score, R²,
coverage, TARP). Do not adopt a correction because histograms look flatter.

Repository evidence: `docs/evidence/p12/`. Operational detail: `RUNBOOK.md`.

Launch help:

```bash
python workflows/sbi/p12_train_base_response_fmpe.py --help
python workflows/sbi/p12_calibration_diagnostics.py --help
python workflows/sbi/p12_width_information_diagnostics.py --help
```

## Which FlowJAX Trainer To Use

| Use case | Entrypoint | Notes |
| --- | --- | --- |
| TNG/full-graph cache | `jraph_sbi_flowjax.py` | Loads one SBI cache in memory and trains/evaluates the baseline FlowJAX NPE model. |
| Abacus wedge-subvolume cache | `jraph_sbi_flowjax.py` | Older Abacus graph-NPE path: one RA/Dec/z wedge graph from `workflows/abacus_tweb/`. Not P12-A. |
| Posterior plots, full graph/wedge | `plot_flowjax_posteriors.py` | Uses saved model outputs from the full-graph trainer. |
| Abacus partition artifacts | `jraph_sbi_flowjax_partitioned.py` | Legacy partitioned experiment; keep for audit/debugging, not new production runs. |
| Posterior plots, partitioned | `plot_flowjax_posteriors_partitioned.py` | Legacy diagnostics for partitioned checkpoints. |
| Data-parallel benchmark | `benchmark_partition_data_parallel.py` | Measures legacy partition collation/loading behavior. |
| Tensor sharding prototype | `prototype_tensor_sharding_fullgraph.py` | Experimental full-graph sharding prototype. |
| Two-stage prototype | `experimental/jraph_sbi_two_stage.py` | Optional experimental path, not the primary Abacus run. |

## Current Abacus Wedge Inputs (older graph NPE)

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

The FlowJAX (and P12-A) target is the ordered softplus-increment representation
from `shared/eigenvalue_transformations.py`: `lambda1` is the anchor and the
`lambda2 - lambda1` and `lambda3 - lambda2` gaps are encoded as non-negative
increments. The model trains and samples in increment space; conversion back to
physical `(lambda1, lambda2, lambda3)` is for evaluation and plotting. Do not
replace this with an unconstrained direct three-eigenvalue head unless the run is
an intentional ablation.

## Launchers

P12-A uses the interactive GPU supervisor `run_p12a_posterior_interactive.sh`
(see `RUNBOOK.md`). There is no tracked production `submit_sbi_flowjax.slurm`
for the older Abacus wedge NPE; run that trainer inside an appropriate GPU
allocation:

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

Tracked SLURM scripts in this directory are mostly legacy partition diagnostics
or experiments:

| SLURM script | Purpose |
| --- | --- |
| `submit_sbi_partitioned_data_parallel.slurm` | Legacy single-node, four-GPU partitioned SBI training. |
| `submit_sbi_partitioned_data_parallel_multinode.slurm` | Legacy multi-node partitioned SBI training with JAX distributed initialization. |
| `submit_sbi_overfit_tiny.slurm` | Tiny overfit diagnostic for partitioned SBI. |
| `submit_partition_data_parallel_benchmark.slurm` | Partition loading and data-parallel benchmark. |
| `submit_plot_flowjax_posteriors_partitioned.slurm` | Posterior plotting for partitioned checkpoints. |
| `submit_sbi_stageB_4node.slurm` | Stage-B multi-node experiment. |
| `submit_tensor_sharding_prototype.slurm` | Experimental tensor-sharding prototype. |

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
from this path for current Abacus SBI work.
